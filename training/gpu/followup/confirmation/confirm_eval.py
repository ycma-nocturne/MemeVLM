"""Freeze combined and inherited checkpoints, then apply the pinned MemeQA policy."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

ARMS = ("real", "batch4", "combined")
SEEDS = (42, 43, 44)
STEPS = (50, 100, 150, 193)
HOLDOUT_MANIFEST = "410011f7f9aa6253165d876805904971f1b97d209c8cfec07ec5433073164942"
REFERENCES = {
    "real": ("", "eb88e6832c4ee54745fad716d487f1dde5973bc1574a268b29e750fdbfc99553"),
    "batch4": ("ablations", "a87200ec1c0d4eb247f30570afcd526e09f2c5a2814de9d3a60aad43b563427a"),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_input_table(directory, expected_manifest):
    manifest_path = directory / "manifest.json"
    if digest(manifest_path) != expected_manifest:
        raise ValueError("Evaluation manifest differs from its freeze")
    manifest = read(manifest_path)
    hashes = {"manifest_sha256": expected_manifest, "inputs_sha256": digest(directory / "inputs.jsonl"),
              "gold_sha256": digest(directory / "gold.jsonl")}
    if any(hashes[key] != manifest[key] for key in ("inputs_sha256", "gold_sha256")):
        raise ValueError("Question or answer bytes differ from the manifest")
    questions, answers = rows(directory / "inputs.jsonl"), rows(directory / "gold.jsonl")
    ids = {row["id"] for row in questions}
    gold = {row["id"]: row["answer"] for row in answers}
    if len(questions) != len(ids) or len(answers) != len(ids) or ids != gold.keys():
        raise ValueError("Question and answer IDs differ")
    sizes = len(questions), len({row["group_id"] for row in questions})
    if sizes != (manifest["n_questions"], manifest["n_images"]):
        raise ValueError("Question or image census differs from the manifest")
    return {**hashes, "questions": sizes[0], "images": sizes[1]}, gold


def select(work):
    output = work / "confirmation"
    held, _ = verify_input_table(output / "prepared", HOLDOUT_MANIFEST)
    data_selection = read(work / "prepared/selection-receipt.json")
    dev, gold = verify_input_table(work / "prepared/native-dev", data_selection["native_dev_manifest_sha256"])
    if (held["questions"], held["images"], len(gold)) != (433, 77, 96):
        raise ValueError("Unexpected cohort dimensions")
    selected, all_results = {}, {}
    for arm, (source, signature) in REFERENCES.items():
        path = work / source / "selection.json"
        if digest(path) != signature:
            raise ValueError(f"Inherited checkpoint selection changed: {arm}")
        selected[arm] = read(path)["checkpoints"][arm]

    selected["combined"] = {}
    for seed in SEEDS:
        name = f"combined-{seed}"
        record = output / "runs" / name / "run-receipt.json"
        receipt = read(record)
        if tuple(receipt["checkpoint_steps"]) != STEPS:
            raise ValueError(f"Checkpoint schedule changed: {name}")
        all_results[str(seed)] = results = []
        for step in STEPS:
            directory = output / "dev-runs" / name / str(step)
            binding_path, ledger = directory / "binding.json", directory / "responses.jsonl"
            binding, completion = read(binding_path), read(directory / "completion.json")
            expected = {"arm": "combined", "seed": seed, "step": step, "split": "dev", "inputs_sha256": dev["inputs_sha256"],
                        "manifest_sha256": dev["manifest_sha256"], "training_receipt_sha256": digest(record)}
            if any(binding[key] != value for key, value in expected.items()):
                raise ValueError(f"Development output differs from frozen inputs: {name}/{step}")
            if digest(ledger) != completion["ledger_sha256"] or digest(binding_path) != completion["binding_sha256"]:
                raise ValueError(f"Development ledger differs from its receipt: {name}/{step}")
            predictions = rows(ledger)
            if len(predictions) != len(gold) or {row["id"] for row in predictions} != gold.keys():
                raise ValueError(f"Development prediction IDs changed: {name}/{step}")
            results.append({"step": step, "correct": sum(row["choice"] == gold[row["id"]] for row in predictions),
                            "adapter": binding["adapter"], "ledger_sha256": completion["ledger_sha256"],
                            "binding_sha256": completion["binding_sha256"]})
        selected["combined"][str(seed)] = max(results, key=lambda row: (row["correct"], -row["step"]))
    summary = {"test_arms": list(ARMS), "checkpoints": selected, "scores": {"combined": all_results},
               "holdout": held, "dev": dev, "questions_per_seed": len(gold)}
    content = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    path = output / "selection.json"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Checkpoint selection is already frozen")
    else:
        with path.open("x") as stream:
            stream.write(content)
    print(json.dumps({arm: {seed: row["step"] for seed, row in seeds.items()} for arm, seeds in selected.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "dev", "test"))
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
        return
    if not args.arm or args.seed is None or (args.action == "dev" and args.arm != "combined"):
        parser.error("Prediction requires an arm and seed; only the combined arm uses new development inference")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from predict import run_prediction
    args.split = args.action
    inherited = REFERENCES[args.arm][0] if args.arm in REFERENCES else None
    run_prediction(args, subdirectory="confirmation", independent_holdout=True, source_subdirectory=inherited)


if __name__ == "__main__":
    main()
