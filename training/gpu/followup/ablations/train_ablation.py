"""Train one single-factor ablation of the real-MemeQA recipe."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as followup


ARMS = {
    "batch4": {"gradient_accumulation_steps": 4},
    "rank4": {"lora_rank": 4, "lora_alpha": 8},
    "rank16": {"lora_rank": 16, "lora_alpha": 32},
    "alpha8": {"lora_alpha": 8},
}


def portable_digest(config, data):
    portable = {key: value for key, value in config.items() if key not in followup.LOCATION_KEYS}
    return data.sha256(data.json_bytes(portable))


def configure(args):
    real = argparse.Namespace(root=args.root, work=args.work, arm="real", seed=args.seed)
    launch, data, config, binding, model, _ = followup.configure(real)
    work = args.work.resolve(strict=True)
    output = followup.in_root(work, work / "ablations" / "runs" / f"{args.arm}-{args.seed}")
    config.update(ARMS[args.arm], output_dir=str(output / "trainer"))
    batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    end = math.ceil(binding["train_rows"] / batch)
    interval = 50 if args.arm == "batch4" else 25
    steps = [interval, 2 * interval, 3 * interval, end]
    if steps != sorted(set(steps)):
        raise ValueError("the checkpoint schedule exceeds the first epoch")
    config.update(eval_steps=interval, save_steps=interval, save_total_limit=len(steps))
    planned = int(end * config["num_train_epochs"])
    binding = {
        **binding,
        "arm": args.arm,
        "real_config_sha256": binding["config_sha256"],
        "parent_runner_sha256": binding["runner_sha256"],
        "config_sha256": portable_digest(config, data),
        "runner_sha256": data.file_sha256(Path(__file__)),
        "effective_batch_size": batch,
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_scale": config["lora_alpha"] / config["lora_rank"],
        "planned_optimizer_steps": planned,
        "stop_step": end,
        "checkpoint_steps": steps,
    }
    return launch, data, config, binding, model, output


def checkpoint_receipt(output, binding, data, launch):
    trainer = output / "trainer"
    steps = binding["checkpoint_steps"]
    end = binding["stop_step"]
    if {p.name for p in trainer.glob("checkpoint-*") if p.is_dir()} != {f"checkpoint-{n}" for n in steps}:
        raise ValueError("checkpoint inventory differs from the declared schedule")
    last = followup.in_root(output, trainer / f"checkpoint-{end}", exists=True)
    state = json.loads((last / "trainer_state.json").read_bytes())
    if (state["global_step"], state["max_steps"], state["epoch"]) != (
        end, binding["planned_optimizer_steps"], 1.0
    ):
        raise ValueError("training did not stop after one epoch at the declared scheduler budget")
    events = [row for row in state["log_history"] if "eval_loss" in row]
    if [row["step"] for row in events] != steps or not all(math.isfinite(row["eval_loss"]) for row in events):
        raise ValueError("checkpoint validation losses are missing or incomplete")
    checkpoints = {}
    for row in events:
        step = row["step"]
        adapter = followup.in_root(output, trainer / f"checkpoint-{step}", exists=True)
        saved = adapter / "trainer_state.json"
        if json.loads(saved.read_bytes())["global_step"] != step:
            raise ValueError(f"checkpoint step differs from trainer state: {step}")
        adapter_config = adapter / launch.ADAPTER_CONFIG
        peft = json.loads(adapter_config.read_bytes())
        if (peft["r"], peft["lora_alpha"], peft.get("use_rslora", False)) != (
            binding["lora_rank"], binding["lora_alpha"], False
        ):
            raise ValueError(f"adapter rank or scaling differs from the declared arm: {step}")
        checkpoints[str(step)] = {
            "adapter_dir": adapter.relative_to(output).as_posix(),
            "adapter_sha256": data.file_sha256(adapter / launch.ADAPTER),
            "adapter_config_sha256": data.file_sha256(adapter_config),
            "trainer_state_sha256": data.file_sha256(saved),
            "sft_validation_loss": row["eval_loss"],
        }
    return {**binding, "optimizer_steps": end, "checkpoints": checkpoints}


def run_training(config, binding):
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    class FirstEpoch(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps != binding["planned_optimizer_steps"]:
                raise ValueError("the planned three-epoch scheduler budget has changed")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == binding["stop_step"]:
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return control

    run_exp(args=config.copy(), callbacks=[FirstEpoch()])


def train(args, launch, data, config, binding, model, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU training runs within an allocated Slurm job")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("set PYTHONHASHSEED to the training seed before starting Python")
    runtime = launch._runtime()
    launch.verify_model(model)
    binding = {**binding, "dependencies": runtime}
    manifest = output / "run-manifest.json"
    resolved = output / "run.yaml"
    receipt = output / "run-receipt.json"
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
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            if step not in binding["checkpoint_steps"]:
                raise ValueError("resume checkpoint is outside the declared schedule")
            if step < binding["stop_step"]:
                run_training({**config, "resume_from_checkpoint": str(checkpoint)}, binding)
        else:
            launch._write_json(manifest, binding)
            launch._write_json(resolved, config)
            run_training(config, binding)
        result = checkpoint_receipt(output, binding, data, launch)
        launch._write_json(receipt, result)
    return result


def verify(launch, data, binding, output):
    saved = json.loads((output / "run-manifest.json").read_bytes())
    if any(saved.get(key) != value for key, value in binding.items()):
        raise ValueError("current inputs differ from the saved run")
    config = json.loads((output / "run.yaml").read_bytes())
    if portable_digest(config, data) != binding["config_sha256"]:
        raise ValueError("resolved configuration differs from the saved run")
    result = checkpoint_receipt(output, saved, data, launch)
    if result != json.loads((output / "run-receipt.json").read_bytes()):
        raise ValueError("saved receipt differs from the checkpoint files")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=followup.SEEDS, required=True)
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
            result = verify(launch, data, binding, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"train_ablation: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
