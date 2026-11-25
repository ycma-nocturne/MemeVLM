"""Freeze the selected checkpoints and run the existing MemeQA inference policy."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys


ARMS = ("batch4", "rank4", "rank16", "alpha8")
SEEDS = (42, 43, 44)


def steps(arm):
    return (50, 100, 150, 193) if arm == "batch4" else (25, 50, 75, 97)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def select(work):
    native = work / "prepared/native-dev"
    outputs = work / "ablations"
    data_binding = read(work / "prepared/selection-receipt.json")
    manifest_path = native / "manifest.json"
    manifest = read(manifest_path)
    labels = lines(native / "gold.jsonl")
    gold = {row["id"]: row["answer"] for row in labels}
    inputs = lines(native / "inputs.jsonl")
    if len(labels) != 96 or len(inputs) != 96 or len(gold) != 96 or set(gold) != {r["id"] for r in inputs}:
        raise ValueError("Invalid native development questions or labels")
    hashes = {"inputs_sha256": digest(native / "inputs.jsonl"), "gold_sha256": digest(native / "gold.jsonl"),
              "manifest_sha256": digest(manifest_path)}
    if any(manifest[key] != hashes[key] for key in ("inputs_sha256", "gold_sha256")):
        raise ValueError("Development inputs or labels differ from native manifest")
    if hashes["manifest_sha256"] != data_binding["native_dev_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen experiment")
    results, chosen = {}, {}
    for arm in ARMS:
        results[arm], chosen[arm] = {}, {}
        for seed in SEEDS:
            run = f"{arm}-{seed}"
            receipt_path = outputs / "runs" / run / "run-receipt.json"
            receipt = read(receipt_path)
            if tuple(receipt["checkpoint_steps"]) != steps(arm):
                raise ValueError(f"Checkpoint schedule mismatch: {run}")
            results[arm][str(seed)] = entries = []
            for step in steps(arm):
                directory = outputs / "dev-runs" / run / str(step)
                binding_path = directory / "binding.json"
                binding = read(binding_path)
                completion = read(directory / "completion.json")
                ledger = directory / "responses.jsonl"
                for key, expected in {"arm": arm, "seed": seed, "step": step, "split": "dev",
                                      "training_receipt_sha256": digest(receipt_path),
                                      "inputs_sha256": hashes["inputs_sha256"],
                                      "manifest_sha256": hashes["manifest_sha256"]}.items():
                    if binding[key] != expected:
                        raise ValueError(f"Invalid development binding {key}: {run}/{step}")
                if digest(binding_path) != completion["binding_sha256"] or digest(ledger) != completion["ledger_sha256"]:
                    raise ValueError(f"Invalid development ledger: {run}/{step}")
                rows = lines(ledger)
                if len(rows) != 96 or {row["id"] for row in rows} != set(gold):
                    raise ValueError(f"Question set differs from frozen development: {run}/{step}")
                entries.append({"step": step, "correct": sum(row["choice"] == gold[row["id"]] for row in rows),
                                "adapter": binding["adapter"], "ledger_sha256": completion["ledger_sha256"],
                                "binding_sha256": completion["binding_sha256"]})
            chosen[arm][str(seed)] = max(entries, key=lambda row: (row["correct"], -row["step"]))
    mean_accuracy = {arm: sum(value["correct"] for value in seeds.values()) / (96 * len(SEEDS))
                     for arm, seeds in chosen.items()}
    result = {"checkpoints": chosen, "scores": results, "mean_accuracy": mean_accuracy,
              "test_arms": list(ARMS), "dev": hashes, "questions_per_seed": 96}
    path = outputs / "selection.json"
    content = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Checkpoint selection is already frozen")
    else:
        with path.open("x") as handle:
            handle.write(content)
    print(json.dumps(mean_accuracy, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dev", "select", "test"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    work = args.work.resolve()
    if not work.is_relative_to(args.root.resolve()):
        parser.error("Follow-up outputs must stay within the experiment root")
    if args.action == "select":
        select(work)
    else:
        if args.arm is None or args.seed is None:
            parser.error("Predictions require an arm and a seed")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from predict import run_prediction
        args.split = args.action
        run_prediction(args, subdirectory="ablations")


if __name__ == "__main__":
    main()
