"""Run one full pass through the private training split on Apple Silicon."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

RUN = Path(__file__).resolve().parent
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(RUN / 'cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.train import build_optimizer, parse_args
from mlx_vlm.trainer import Trainer, prepare_model_for_training
from mlx_vlm.trainer.datasets.loading import load as load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('full',))
    parser.parse_args()
    steps, accumulation = 98, 4
    output = RUN / 'full'
    output.mkdir(exist_ok=True)
    adapter = output / 'adapter.safetensors'
    if adapter.exists():
        raise FileExistsError(adapter)
    options = [
        '--model', str(RUN / 'model'), '--data', str(RUN / 'data'),
        '--algorithm', 'sft', '--train-on-completions', '--batch-size', '1',
        '--gradient-accumulation-steps', str(accumulation), '--iters', str(steps),
        '--optimizer', 'adamw', '--learning-rate', '0.0001', '--lora-rank', '4',
        '--lora-alpha', '8', '--seed', '6000', '--grad-checkpoint', '--cache-size', '0',
        '--max-seq-length', '768', '--steps-per-report', '10',
        '--steps-per-eval', str(steps), '--val-batches', '-1',
        '--steps-per-save', str(steps), '--adapter-file', str(adapter),
    ]
    args, settings = parse_args(options, task='vlm')
    (output / 'settings.json').write_text(json.dumps({'argv': options}, indent=2) + '\n')
    print('Loading verified model for full training', flush=True)
    mx.random.seed(args.seed)
    model, processor = load(args.model, trust_remote_code=False)
    model = prepare_model_for_training(model, train_type='lora', lora_rank=args.lora_rank,
                                       lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    rows = load_dataset(args.data)
    assert (len(rows['train']), len(rows['validation'])) == (98, 33)
    trainer = Trainer(model, settings, optimizer=build_optimizer(args, steps, accumulation),
                      task='vlm', algorithm='sft', train_dataset=rows['train'],
                      eval_dataset=rows['validation'], processing_class=processor,
                      dataset_config=args.dataset_config, seed=args.seed)
    before = trainer.evaluate()
    print('Validation before training:', json.dumps(before), flush=True)
    (output / 'baseline.json').write_text(json.dumps(before, indent=2) + '\n')
    mx.reset_peak_memory()
    started = time.perf_counter()
    metrics = trainer.train()
    peak = mx.get_peak_memory()
    assert metrics['iteration'] == steps
    assert metrics['optimizer_step'] == math.ceil(steps / accumulation)
    assert math.isfinite(metrics['train_loss']) and math.isfinite(metrics['val_loss'])
    assert adapter.stat().st_size > 0 and (output / 'adapter_config.json').is_file()
    report = {'stage': 'full', 'train_rows': 98, 'validation_rows': 33, 'pre_training_validation': before,
              'metrics': metrics, 'history': trainer.log_history, 'peak_mlx_memory_bytes': peak,
              'adapter_bytes': adapter.stat().st_size,
              'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
              'training_seconds': time.perf_counter() - started}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Completed full training, validation loss {metrics['val_loss']:.6f}, peak MLX {peak / 10**9:.2f} GB", flush=True)


if __name__ == '__main__':
    main()
