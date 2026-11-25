"""Export confirmation-run choices, bindings, and receipt-verified training histories."""
import argparse
import gzip
import json
from pathlib import Path
import sys

from confirm_eval import ARMS, SEEDS, STEPS, REFERENCES, digest, read, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    if not work.is_relative_to(args.root.resolve()):
        raise ValueError("Work directory must stay within the experiment root")
    sys.path.insert(0, str(work / "ablations"))
    from export import choices, verified_parameter_count, verified_state

    study = work / "confirmation"
    selection_path = study / "selection.json"
    selection = read(selection_path)
    selection_sha = digest(selection_path)
    test_ids = [row["id"] for row in rows(study / "prepared/inputs.jsonl")]
    dev_ids = [row["id"] for row in rows(work / "prepared/native-dev/inputs.jsonl")]
    training, test, dev, states, counts = {}, {}, {}, {}, {}
    for arm in ARMS:
        source = work / REFERENCES[arm][0] if arm in REFERENCES else study
        for seed in SEEDS:
            name = f"{arm}-{seed}"
            run = source / "runs" / name
            receipt_path = run / "run-receipt.json"
            training[name] = receipt = read(receipt_path)
            selected = selection["checkpoints"][arm][str(seed)]
            counts[name] = verified_parameter_count(run, receipt["checkpoints"][str(selected["step"])])
            test[name] = choices(study / "test-runs" / name, test_ids)
            if test[name]["binding"]["selection_sha256"] != selection_sha:
                raise ValueError(f"Selection binding changed for {name}")
            if arm == "combined":
                states[name] = verified_state(run, receipt)
                for step in STEPS:
                    dev[f"{name}/{step}"] = choices(study / "dev-runs" / name / str(step), dev_ids)
    result = {"job_ids": [args.job], "selection": selection, "selection_sha256": selection_sha,
              "holdout": selection["holdout"], "test_ids": test_ids, "dev_ids": dev_ids,
              "training": training, "trainer_states": states, "adapter_parameters": counts, "test": test, "dev": dev}
    path = study / "compact-results.json.gz"
    content = gzip.compress(json.dumps(result, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Existing export differs from verified results")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    print("Archived three new training runs, 12 development checkpoints and nine holdout ledgers", flush=True)


if __name__ == "__main__":
    main()
