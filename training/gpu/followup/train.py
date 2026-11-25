"""Train and save the first-epoch checkpoints for the real-MemeQA follow-up."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path


ARMS = {
    "baseline": ("baseline", 1e-4),
    "real": ("real-mix", 1e-4),
    "real-low": ("real-mix", 5e-5),
}
SEEDS = (42, 43, 44)
STEPS = (25, 50, 75, 97)
PLANNED_STEPS = 291
BASELINE_TRAIN = "f80caf8213d1c908bf1122c03e117493b0e65000f31de5dc92e1da25c84c9bf2"
VALIDATION = "c681a64926db71d4c17cf29963b331c93cad952b790cd3d851f0f9bda70d5986"
LOCATION_KEYS = {"model_name_or_path", "dataset_dir", "media_dir", "output_dir"}


def in_root(root: Path, path: Path, *, exists: bool = False) -> Path:
    resolved = path.resolve(strict=exists)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"path must be inside the scoped directory: {path}")
    return resolved


def configure(args):
    root = args.root.resolve(strict=True)
    work = in_root(root, args.work, exists=True)
    project = in_root(root, root / "project", exists=True)
    model = in_root(root, root / "model", exists=True)
    sys.path.insert(0, str(project))
    from training.gpu import launch, prepare_data

    dataset_name, lr = ARMS[args.arm]
    dataset = in_root(work, work / "prepared" / dataset_name, exists=True)
    manifest, dataset_sha = prepare_data.verify_dataset(dataset)
    selection_path = in_root(work, work / "prepared" / "selection-receipt.json", exists=True)
    selection_raw = selection_path.read_bytes()
    selection = json.loads(selection_raw)
    selected_key = "baseline_dataset_sha256" if dataset_name == "baseline" else "real_mix_dataset_sha256"
    if (selection[selected_key], selection["shared_validation_sha256"]) != (dataset_sha, VALIDATION):
        raise ValueError("the trainer export differs from the declared data selection")
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    if (manifest["profile"], train["rows"], validation["rows"]) != ("bilingual", 770, 193):
        raise ValueError("expected 770 training rows and the common 193-row validation split")
    files = manifest["files"]
    if files["validation.json"] != VALIDATION:
        raise ValueError("validation differs from the original 193-row export")
    if (files["train.json"] == BASELINE_TRAIN) != (args.arm == "baseline"):
        raise ValueError("training rows do not match the assigned arm")

    output = in_root(work, work / "runs" / f"{args.arm}-{args.seed}")
    config = launch.make_config(dataset, model, output / "trainer")
    config.update(seed=args.seed, data_seed=args.seed, learning_rate=lr,
                  eval_strategy="steps", save_strategy="steps", eval_steps=25, save_steps=25,
                  save_total_limit=len(STEPS), load_best_model_at_end=False)
    config.pop("metric_for_best_model")
    config.pop("greater_is_better")
    portable = {k: v for k, v in config.items() if k not in LOCATION_KEYS}
    binding = {
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataset_sha256": dataset_sha,
        "data_selection_sha256": prepare_data.sha256(selection_raw),
        "train_sha256": files["train.json"],
        "validation_sha256": VALIDATION,
        "train_rows": train["rows"],
        "validation_rows": validation["rows"],
        "model_id": launch.MODEL_ID,
        "model_revision": launch.MODEL_REVISION,
        "model_files": launch.MODEL_FILES,
        "config_sha256": prepare_data.sha256(prepare_data.json_bytes(portable)),
        "runner_sha256": prepare_data.file_sha256(Path(__file__)),
        "learning_rate": lr,
        "planned_optimizer_steps": PLANNED_STEPS,
        "stop_step": STEPS[-1],
        "checkpoint_steps": list(STEPS),
    }
    return launch, prepare_data, config, binding, model, output


def checkpoint_receipt(output, binding, data, launch):
    trainer = output / "trainer"
    checkpoints = {p.name for p in trainer.glob("checkpoint-*") if p.is_dir()}
    if checkpoints != {f"checkpoint-{step}" for step in STEPS}:
        raise ValueError("checkpoint inventory differs from the declared schedule")
    final_state = json.loads((trainer / "checkpoint-97" / "trainer_state.json").read_bytes())
    if (final_state["global_step"], final_state["max_steps"], final_state["epoch"]) != (97, 291, 1.0):
        raise ValueError("trainer did not stop at the first epoch of the 291-step schedule")
    events = [row for row in final_state["log_history"] if "eval_loss" in row]
    if tuple(row["step"] for row in events) != STEPS or not all(math.isfinite(row["eval_loss"]) for row in events):
        raise ValueError("checkpoint validation losses are missing or incomplete")
    result = {}
    for row in events:
        step = row["step"]
        adapter = in_root(output, trainer / f"checkpoint-{step}", exists=True)
        state = adapter / "trainer_state.json"
        if json.loads(state.read_bytes())["global_step"] != step:
            raise ValueError(f"checkpoint step differs from trainer state: {step}")
        result[str(step)] = {
            "adapter_dir": adapter.relative_to(output).as_posix(),
            "adapter_sha256": data.file_sha256(adapter / launch.ADAPTER),
            "adapter_config_sha256": data.file_sha256(adapter / launch.ADAPTER_CONFIG),
            "trainer_state_sha256": data.file_sha256(state),
            "sft_validation_loss": row["eval_loss"],
        }
    return {**binding, "optimizer_steps": STEPS[-1], "checkpoints": result}


def train(args, launch, data, config, binding, model, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU training runs within an allocated Slurm job")
    if os.environ.get("PYTHONHASHSEED") != str(binding["seed"]):
        raise ValueError("set PYTHONHASHSEED to the training seed before starting Python")
    launch.verify_model(model)
    binding = {**binding, "dependencies": launch._runtime()}
    manifest = output / "run-manifest.json"
    receipt = output / "run-receipt.json"
    resolved = output / "run.yaml"
    if not args.resume:
        output.mkdir(parents=True)
    with (output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if receipt.exists():
                raise ValueError("training receipt already exists; use verify")
            if json.loads(manifest.read_bytes()) != binding or json.loads(resolved.read_bytes()) != config:
                raise ValueError("resume requires the original run bindings and config")
            checkpoint = launch.latest_checkpoint(output / "trainer")
            if checkpoint.name != "checkpoint-97":
                run_training({**config, "resume_from_checkpoint": str(checkpoint)})
        else:
            launch._write_json(manifest, binding)
            launch._write_json(resolved, config)
            run_training(config)
        result = checkpoint_receipt(output, binding, data, launch)
        launch._write_json(receipt, result)
    return result


def run_training(config):
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    class FirstEpoch(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps != PLANNED_STEPS:
                raise ValueError("the original 291-step scheduler budget has changed")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == STEPS[-1]:
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

    run_exp(args=config.copy(), callbacks=[FirstEpoch()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "verify"))
    parser.add_argument("--root", required=True, type=Path, help="Scoped project directory")
    parser.add_argument("--work", required=True, type=Path, help="Follow-up directory within root")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.command != "train":
        parser.error("--resume is available for train")
    try:
        launch, data, config, binding, model, output = configure(args)
        if args.command == "plan":
            result = {"binding": binding, "config": config}
        elif args.command == "train":
            result = train(args, launch, data, config, binding, model, output)
        else:
            saved = json.loads((output / "run-manifest.json").read_bytes())
            if any(saved.get(key) != value for key, value in binding.items()):
                raise ValueError("current inputs differ from the saved run")
            result = checkpoint_receipt(output, saved, data, launch)
            if result != json.loads((output / "run-receipt.json").read_bytes()):
                raise ValueError("saved receipt differs from the checkpoint files")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"train: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
