"""Prepare, launch and verify the matched 4B replay and seed controls."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


BILINGUAL = "8c8973952c8eb68367d56673c97ecaf5c1a76e96e0ac1cf094d96e365d8ad2a3"
CHINESE = "8debd9dc07ccc2cc06b035e10fb8cbb993d625346a0c32e0ae4d44e5f8110252"
CHINESE_REPLAY = "eebd6a9f13663133464be3cf62d0c35500776cf53a52f4ff160786aeeda5d884"
VALIDATION = "c681a64926db71d4c17cf29963b331c93cad952b790cd3d851f0f9bda70d5986"
REFERENCE_CONFIG = "2886018920c9d9aa1682b78a0d5a92669e2b98ed74a5ee2801abb304f0ef87eb"
STEPS = (97, 194, 291)
REPLAY = {"source_dataset_sha256": CHINESE, "validation_dataset_sha256": BILINGUAL,
          "unique_train_rows": 98, "rows_per_epoch": 770, "copies_per_row": [7, 8],
          "extra_copy_seed": 6000, "extra_copies": 84}
LOCATION_KEYS = {"model_name_or_path", "dataset_dir", "media_dir", "output_dir"}


def read(path):
    return json.loads(path.read_bytes())


def prepare_replay(data, bilingual, chinese, output):
    mixed, mixed_digest = data.verify_dataset(bilingual)
    baseline, baseline_digest = data.verify_dataset(chinese)
    if (mixed_digest, baseline_digest) != (BILINGUAL, CHINESE):
        raise ValueError("prepare requires the original matched release exports")
    if mixed["files"]["validation.json"] != VALIDATION or mixed["sources"] != baseline["sources"]:
        raise ValueError("source exports differ from the fixed study")
    source = read(chinese / "train.json")
    if len(source) != 98 or len({data.json_bytes(row) for row in source}) != 98:
        raise ValueError("the Chinese source must contain exactly 98 unique training records")
    extras = random.Random(REPLAY["extra_copy_seed"]).sample(range(98), REPLAY["extra_copies"])
    rows = source * 7 + [source[index] for index in extras]
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".chinese-replay-") as tmp:
        staged = Path(tmp) / "dataset"
        (staged / "images").mkdir(parents=True)
        for name in ("dataset_info.json", "validation.json"):
            shutil.copyfile(bilingual / name, staged / name)
        (staged / "train.json").write_bytes(data.json_bytes(rows))
        for root, records in ((chinese, source), (bilingual, read(bilingual / "validation.json"))):
            for name in {row["images"][0] for row in records}:
                shutil.copyfile(root / name, staged / name)
        result = {**mixed, "profile": "chinese", "replay": REPLAY,
                  "splits": {**mixed["splits"], "train": {**baseline["splits"]["train"],
                             "rows": 770, "by_language": {"zh": 770}}},
                  "files": {path.relative_to(staged).as_posix(): data.file_sha256(path)
                            for path in staged.rglob("*") if path.is_file()}}
        (staged / data.MANIFEST).write_bytes(data.json_bytes(result))
        _, digest = verify_data(data, staged, "chinese-replay")
        staged.rename(output)
    return {"dataset_sha256": digest, "train_rows": 770, "unique_train_rows": 98,
            "validation_rows": 193, "validation_sha256": VALIDATION}


def verify_data(data, dataset, arm):
    manifest, digest = data.verify_dataset(dataset)
    splits = manifest["splits"]
    if (splits["train"]["rows"], splits["validation"]["rows"],
            manifest["files"]["validation.json"]) != (770, 193, VALIDATION):
        raise ValueError("expected 770 training rows and the exact 193-row bilingual validation split")
    if arm == "bilingual":
        if digest != BILINGUAL:
            raise ValueError("bilingual repeats must use the original seed-42 SFT export")
    else:
        counts = Counter(data.json_bytes(row) for row in read(dataset / "train.json"))
        if (digest != CHINESE_REPLAY or manifest["profile"] != "chinese" or manifest.get("replay") != REPLAY or
                splits["train"]["by_language"] != {"zh": 770} or Counter(counts.values()) != {7: 14, 8: 84}):
            raise ValueError("expected the fixed 98-row Chinese replay cohort")
    return manifest, digest


def config_for(launch, data, dataset, model, output, seed):
    config = launch.make_config(dataset, model, output / "trainer")
    portable = {key: value for key, value in config.items() if key not in LOCATION_KEYS}
    if data.sha256(data.json_bytes(portable)) != REFERENCE_CONFIG:
        raise ValueError("training recipe differs from the recorded seed-42 run")
    config.update(seed=seed, data_seed=seed)
    digest = data.sha256(data.json_bytes({key: value for key, value in config.items() if key not in LOCATION_KEYS}))
    return config, digest


def verify_schedule(launch, output):
    trainer = output / "trainer"
    state = read(trainer / "trainer_state.json")
    if tuple(state[key] for key in ("global_step", "max_steps", "num_train_epochs", "train_batch_size")) != (291, 291, 3, 1):
        raise ValueError("trainer did not complete the fixed 291-step budget")
    events = [row for row in state["log_history"] if "eval_loss" in row]
    if [(row["step"], row["epoch"]) for row in events] != list(zip(STEPS, range(1, 4))):
        raise ValueError("validation did not occur at the fixed epoch boundaries")
    if not all(math.isfinite(row["eval_loss"]) for row in events):
        raise ValueError("validation loss must be finite at all three boundaries")
    checkpoints = {p.name for p in trainer.glob("checkpoint-*") if p.is_dir()}
    if checkpoints != {f"checkpoint-{step}" for step in STEPS}:
        raise ValueError("saved checkpoint steps do not match the evaluation schedule")
    for step in STEPS:
        path = trainer / f"checkpoint-{step}"
        if read(path / "trainer_state.json")["global_step"] != step or any(
            (path / name).stat().st_size == 0 for name in (launch.ADAPTER, launch.ADAPTER_CONFIG)
        ):
            raise ValueError(f"checkpoint-{step} is incomplete")
    best = min(events, key=lambda row: row["eval_loss"])
    if (Path(state["best_model_checkpoint"]).name != f"checkpoint-{best['step']}" or
            state["best_metric"] != best["eval_loss"]):
        raise ValueError("trainer selection differs from the best fixed validation checkpoint")
    return [{key: row[key] for key in ("step", "epoch", "eval_loss")} for row in events]


def run(args, launch, data, manifest, digest, config, config_digest):
    output = args.output.resolve()
    model = args.model.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    if output.is_relative_to(dataset) or output.is_relative_to(model):
        raise ValueError("place run outputs outside the input and model paths")
    binding = {"mode": "train", "profile": manifest["profile"], "dataset_sha256": digest,
               "model_id": launch.MODEL_ID, "model_revision": launch.MODEL_REVISION,
               "model_files": launch.verify_model(model), "config_sha256": config_digest,
               "validation_rows": 193, "dependencies": launch._runtime(), "control_arm": args.arm,
               "seed": args.seed, "data_seed": args.seed, "validation_sha256": VALIDATION,
               "expected_steps": list(STEPS), "control_launcher_sha256": data.file_sha256(Path(__file__))}
    resume = args.command == "resume"
    if not resume:
        output.mkdir(parents=True)
    with (output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if (output / launch.SELECTED).exists():
                raise ValueError("the run already has its selected adapter")
            if read(output / launch.BINDING) != binding:
                raise ValueError("resume requires the original run recipe and inputs")
            original = read(output / "run.yaml")
            if original != config:
                raise ValueError("resume requires the original absolute input and output paths")
            config["resume_from_checkpoint"] = str(launch.latest_checkpoint(output / "trainer"))
        else:
            launch._write_json(output / launch.BINDING, binding)
        resolved = output / ("resume.yaml" if resume else "run.yaml")
        launch._write_json(resolved, config)
        env = os.environ | {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "FORCE_TORCHRUN": "0",
                            "USE_MCA": "0", "USE_V1": "0", "USE_RAY": "0", "WANDB_MODE": "disabled",
                            "PYTHONHASHSEED": "42"}
        subprocess.run([sys.executable, "-m", "llamafactory.cli", "train", str(resolved)], env=env, check=True)
        evaluations = verify_schedule(launch, output)
        receipt = launch.selected_adapter(output, binding)
        launch._write_json(output / launch.SELECTED, receipt)
    return {"evaluation_checkpoints": evaluations, "selected": receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("bilingual", "chinese", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan", "train", "resume", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--arm", choices=("bilingual", "chinese-replay"), required=True)
        command.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
        for flag in ("dataset", "output", "model"):
            command.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve(strict=True)))
    from training.gpu import launch, prepare_data as data

    if args.command == "prepare":
        result = prepare_replay(data, args.bilingual, args.chinese, args.output)
    else:
        if args.arm == "bilingual" and args.seed == 42 and args.command in {"train", "resume"}:
            parser.error("reuse the completed seed-42 bilingual run")
        manifest, digest = verify_data(data, args.dataset, args.arm)
        output = args.output.resolve()
        config, config_digest = config_for(launch, data, args.dataset.resolve(), args.model.resolve(), output, args.seed)
        if args.command == "plan":
            result = {"arm": args.arm, "seed": args.seed, "config": config, "config_sha256": config_digest,
                      "dataset_sha256": digest, "validation_sha256": VALIDATION, "expected_steps": list(STEPS)}
        elif args.command == "verify":
            receipt = launch._verify_run(output, manifest, digest)
            if (receipt["config_sha256"] != config_digest or
                    receipt.get("seed", 42) != args.seed or receipt.get("control_arm", "bilingual") != args.arm):
                raise ValueError("receipt differs from the requested control")
            result = {"evaluation_checkpoints": verify_schedule(launch, output), "selected": receipt}
        else:
            result = run(args, launch, data, manifest, digest, config, config_digest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
