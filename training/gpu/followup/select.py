"""Select checkpoints and the real-data learning rate from development predictions."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path


SEEDS = (42, 43, 44)
STEPS = (25, 50, 75, 97)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stage", choices=("data", "lr"), required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    data = work / "prepared/native-dev"
    inputs = data / "inputs.jsonl"
    labels = data / "gold.jsonl"
    manifest = data / "manifest.json"
    gold = {row["id"]: row["answer"] for row in rows(labels)}
    if len(gold) != 96 or set(gold) != {row["id"] for row in rows(inputs)}:
        raise ValueError("Development questions and answers do not match")
    hashes = {"inputs_sha256": digest(inputs), "gold_sha256": digest(labels),
              "manifest_sha256": digest(manifest)}
    dev_manifest = json.loads(manifest.read_text())
    if any(hashes[key] != dev_manifest[key] for key in ("inputs_sha256", "gold_sha256")):
        raise ValueError("Development questions or labels differ from the frozen manifest")
    arms = ("baseline", "real", "real-low") if args.stage == "lr" else ("baseline", "real")
    score_rows, checkpoints, totals = {}, {}, {}
    for arm in arms:
        score_rows[arm], checkpoints[arm] = {}, {}
        for seed in SEEDS:
            key = str(seed)
            score_rows[arm][key] = entries = []
            for step in STEPS:
                path = work / "dev-runs" / f"{arm}-{seed}" / str(step)
                predictions = path / "responses.jsonl"
                records = rows(predictions)
                binding_path = path / "binding.json"
                binding = json.loads(binding_path.read_text())
                completion = json.loads((path / "completion.json").read_text())
                if (binding["arm"], binding["seed"], binding["step"], binding["split"]) != (arm, seed, step, "dev"):
                    raise ValueError(f"Development predictions are mislabelled: {path}")
                if any(binding[field] != hashes[field] for field in ("inputs_sha256", "manifest_sha256")):
                    raise ValueError(f"Development inputs changed: {path}")
                if completion["ledger_sha256"] != digest(predictions) or completion["binding_sha256"] != digest(binding_path):
                    raise ValueError(f"Development artifact disagrees with its receipt: {path}")
                if len(records) != len(gold) or {row["id"] for row in records} != set(gold):
                    raise ValueError(f"Development prediction IDs differ from gold: {path}")
                entries.append({"step": step, "correct": sum(row["choice"] == gold[row["id"]] for row in records),
                                "ledger_sha256": completion["ledger_sha256"],
                                "binding_sha256": completion["binding_sha256"], "adapter": binding["adapter"]})
            checkpoints[arm][key] = max(entries, key=lambda row: (row["correct"], -row["step"]))
        totals[arm] = sum(row["correct"] for row in checkpoints[arm].values())
    total_questions = len(gold) * len(SEEDS)
    report = {"stage": args.stage, "dev": hashes, "questions_per_seed": len(gold), "scores": score_rows,
              "checkpoints": checkpoints, "mean_accuracy": {arm: total / total_questions for arm, total in totals.items()},
              "real_minus_baseline_pp": 100 * (totals["real"] - totals["baseline"]) / total_questions}
    target = work / "data-dev-comparison.json"
    if args.stage == "lr":
        prior = json.loads(target.read_text())
        for arm in ("baseline", "real"):
            if prior["scores"][arm] != report["scores"][arm] or prior["dev"] != report["dev"]:
                raise ValueError("The data-comparison evidence changed after the learning-rate comparison")
        selected = "real-low" if totals["real-low"] > totals["real"] else "real"
        report["selected_by_seed"] = {str(seed): selected for seed in SEEDS}
        report["data_comparison_sha256"] = digest(target)
        target = work / "selection.json"
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text() != content:
            raise ValueError(f"Selection is already frozen: {target}")
    else:
        with target.open("x") as handle:
            handle.write(content)
    print(json.dumps(report["mean_accuracy"], sort_keys=True))


if __name__ == "__main__":
    main()
