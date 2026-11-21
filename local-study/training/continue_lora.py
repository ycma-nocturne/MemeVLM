"""Continue the local LoRA with a fresh optimizer and score the same validation split."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LOCAL_RUN = PROJECT / 'training/runs/mlx-qwen3vl-2b'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('epoch', type=int, choices=(2, 3))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    label = f'epoch-{args.epoch:02}'
    parent = LOCAL_RUN / 'full' if args.epoch == 2 else HERE / 'epoch-02'
    previous = read(parent / 'summary.json')
    previous_loss = previous['metrics']['val_loss']
    parent_weights = parent / 'adapter.safetensors'
    parent_hash = sha(parent_weights)
    if parent_hash != previous['adapter_sha256']:
        raise ValueError('Parent adapter SHA-256 differs from the completed run')
    if args.epoch == 3:
        plan = read(HERE / 'selection-plan.json')['epoch_03']
        if previous['pre_training_validation']['val_loss'] - previous_loss < plan['run_if_epoch_02_loss_reduction_at_least']:
            raise ValueError('Validation has reached the recorded stopping threshold')
    choices = read(HERE / 'selection-plan.json')[f'epoch_{args.epoch:02}']
    data = LOCAL_RUN / 'data'
    provenance = read(LOCAL_RUN / 'provenance.json')
    for name, digest in provenance['data'].items():
        if sha(data / name) != digest:
            raise ValueError(f'Training data changed: {name}')
    for item in read(data / 'images-receipt.json'):
        if sha(data / item['image']) != item['output_sha256']:
            raise ValueError(f"Training image changed: {item['image']}")
    if args.preflight:
        print(json.dumps({'epoch': args.epoch, 'prior_validation_loss': previous_loss,
                          'parent_sha256': parent_hash, 'learning_rate': choices['learning_rate']}))
        return

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HOME', str(LOCAL_RUN / 'cache'))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.train import build_optimizer, parse_args
    from mlx_vlm.trainer import Trainer, prepare_model_for_training
    from mlx_vlm.trainer.datasets.loading import load as load_dataset

    output = HERE / label
    output.mkdir(exist_ok=True)
    adapter = output / 'adapter.safetensors'
    if adapter.exists():
        raise FileExistsError(adapter)
    options = [
        '--model', str(LOCAL_RUN / 'model'), '--data', str(data), '--algorithm', 'sft',
        '--train-on-completions', '--batch-size', '1', '--gradient-accumulation-steps', '4',
        '--iters', '98', '--optimizer', 'adamw', '--learning-rate', str(choices['learning_rate']),
        '--lora-rank', '4', '--lora-alpha', '8', '--seed', str(choices['seed']),
        '--grad-checkpoint', '--cache-size', '0', '--max-seq-length', '768',
        '--steps-per-report', '10', '--steps-per-eval', '98', '--val-batches', '-1',
        '--steps-per-save', '98', '--adapter-file', str(adapter),
        '--resume-adapter-file', str(parent_weights),
    ]
    cli, settings = parse_args(options, task='vlm')
    save(output / 'settings.json', {'argv': options, 'parent_adapter_sha256': parent_hash})
    mx.random.seed(cli.seed)
    print(f'Loading parent for epoch {args.epoch}', flush=True)
    model, processor = load(cli.model, trust_remote_code=False)
    model = prepare_model_for_training(model, train_type='lora', checkpoint_path=str(parent_weights))
    rows = load_dataset(cli.data)
    if (len(rows['train']), len(rows['validation'])) != (98, 33):
        raise ValueError('Training or validation row count changed')
    trainer = Trainer(model, settings, optimizer=build_optimizer(cli, 98, 4), task='vlm',
                      algorithm='sft', train_dataset=rows['train'], eval_dataset=rows['validation'],
                      processing_class=processor, dataset_config=cli.dataset_config, seed=cli.seed)
    baseline = trainer.evaluate()
    save(output / 'baseline.json', baseline)
    print('Validation before continuation:', json.dumps(baseline), flush=True)
    if baseline['val_num_sequences'] != 33 or abs(baseline['val_loss'] - previous_loss) > 1e-4:
        raise ValueError('Restored parent does not reproduce its saved validation results')
    mx.reset_peak_memory()
    started = time.perf_counter()
    metrics = trainer.train()
    expected = (98, 25, 98, 33, baseline['val_num_tokens'])
    keys = ('iteration', 'optimizer_step', 'total_sequences', 'val_num_sequences', 'val_num_tokens')
    if tuple(metrics.get(key) for key in keys) != expected:
        raise ValueError('Training or validation denominator differs from the planned cohort')
    if not math.isfinite(metrics['val_loss']) or not adapter.is_file() or not (output / 'adapter_config.json').is_file():
        raise ValueError('Missing adapter or non-finite validation loss')
    if sha(parent_weights) != parent_hash:
        raise ValueError('Parent adapter was modified')
    summary = {'epoch': args.epoch, 'pre_training_validation': baseline, 'metrics': metrics,
               'history': trainer.log_history, 'peak_mlx_memory_bytes': mx.get_peak_memory(),
               'training_seconds': time.perf_counter() - started, 'adapter_bytes': adapter.stat().st_size,
               'adapter_sha256': sha(adapter), 'parent_adapter_sha256': parent_hash}
    save(output / 'summary.json', summary)
    print(f"Completed {label}, validation loss {metrics['val_loss']:.6f}", flush=True)


if __name__ == '__main__':
    main()
