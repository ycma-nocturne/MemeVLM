"""Train the combined batch-4 and rank-16 real-MemeQA recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
import train_ablation as ablation


def configure(args):
    anchor = argparse.Namespace(root=args.root, work=args.work, arm="batch4", seed=args.seed)
    launch, data, config, binding, model, _ = ablation.configure(anchor)
    work = args.work.resolve(strict=True)
    output = ablation.followup.in_root(work, work / "confirmation" / "runs" / f"combined-{args.seed}")
    config.update(lora_rank=16, lora_alpha=32, output_dir=str(output / "trainer"))
    binding = {
        **binding,
        "arm": "combined",
        "batch4_config_sha256": binding["config_sha256"],
        "ablation_runner_sha256": binding["runner_sha256"],
        "config_sha256": ablation.portable_digest(config, data),
        "runner_sha256": data.file_sha256(Path(__file__)),
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_scale": config["lora_alpha"] / config["lora_rank"],
    }
    return launch, data, config, binding, model, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", choices=("combined",), default="combined")
    parser.add_argument("--seed", type=int, choices=ablation.followup.SEEDS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.command != "train":
        parser.error("--resume is available for train")
    try:
        launch, data, config, binding, model, output = configure(args)
        if args.command == "plan":
            result = {"binding": binding, "config": config}
        elif args.command == "train":
            result = ablation.train(args, launch, data, config, binding, model, output)
        else:
            result = ablation.verify(launch, data, binding, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"train_combined: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
