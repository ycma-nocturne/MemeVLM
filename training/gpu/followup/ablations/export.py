"""Export verified training histories and compact choice ledgers for local analysis."""
import argparse
import gzip
import json
from math import prod
from pathlib import Path

from evaluate import ARMS, SEEDS, digest, lines, read, steps


def choices(path, ids):
    ledger_path = path / "responses.jsonl"
    binding_path = path / "binding.json"
    completion = read(path / "completion.json")
    if completion["ledger_sha256"] != digest(ledger_path) or completion["binding_sha256"] != digest(binding_path):
        raise ValueError(f"Invalid evaluation receipt: {path}")
    rows = lines(ledger_path)
    lookup = {row["id"]: row["choice"] for row in rows}
    if len(rows) != len(ids) or lookup.keys() != set(ids):
        raise ValueError(f"Invalid evaluation roster: {path}")
    return {"choices": "".join(lookup[key] or "-" for key in ids), "binding": read(binding_path),
            "completion": completion, "hardware": read(path / "hardware.json")}


def verified_state(directory, receipt):
    final = receipt["checkpoints"][str(receipt["stop_step"])]
    path = directory / final["adapter_dir"] / "trainer_state.json"
    if digest(path) != final["trainer_state_sha256"]:
        raise ValueError(f"Invalid trainer state: {directory}")
    frozen = read(path)
    updated = directory / "trainer/trainer_state.json"
    if updated.is_file():
        complete = read(updated)
        if complete["global_step"] != frozen["global_step"] or complete["log_history"][:len(frozen["log_history"])] != frozen["log_history"]:
            raise ValueError(f"Final trainer history differs from checkpoint: {directory}")
        return complete
    return frozen


def count_adapter(path):
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size <= 10_000_000:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(size))
    return sum(prod(info["shape"]) for key, info in header.items() if key != "__metadata__")


def verified_parameter_count(run, checkpoint):
    path = run / checkpoint["adapter_dir"] / "adapter_model.safetensors"
    if digest(path) != checkpoint["adapter_sha256"]:
        raise ValueError(f"Adapter digest mismatch: {path}")
    return count_adapter(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    if not work.is_relative_to(root):
        raise ValueError("Work directory must stay inside the experiment root")
    study = work / "ablations"
    selection_path = study / "selection.json"
    selection = read(selection_path)
    selection_sha = digest(selection_path)
    dev_ids = [row["id"] for row in lines(work / "prepared/native-dev/inputs.jsonl")]
    test_ids = [row["id"] for row in lines(root / "final-expansion/native/inputs.jsonl") if not row["pilot_group"]]
    if (len(dev_ids), len(test_ids)) != (96, 816):
        raise ValueError("Unexpected question census")
    training, states, prior, counts, prior_counts, dev, test = {}, {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}-{seed}"
            path = study / "runs" / name
            training[name] = receipt = read(path / "run-receipt.json")
            states[name] = verified_state(path, receipt)
            selected = selection["checkpoints"][arm][str(seed)]
            checkpoint = receipt["checkpoints"][str(selected["step"])]
            counts[name] = verified_parameter_count(path, checkpoint)
            test[name] = choices(study / "test-runs" / name, test_ids)
            if test[name]["binding"]["selection_sha256"] != selection_sha:
                raise ValueError(f"Test result has a different selection: {name}")
            for step in steps(arm):
                dev[f"{name}/{step}"] = choices(study / "dev-runs" / name / str(step), dev_ids)
    previous_selection = read(work / "selection.json")
    for arm in ("baseline", "real", "real-low"):
        for seed in SEEDS:
            name = f"{arm}-{seed}"
            path = work / "runs" / name
            receipt = read(path / "run-receipt.json")
            prior[name] = verified_state(path, receipt)
            step = str(previous_selection["checkpoints"][arm][str(seed)]["step"])
            prior_counts[name] = verified_parameter_count(path, receipt["checkpoints"][step])
    report = {"job_ids": [args.job], "selection": selection, "selection_sha256": selection_sha,
              "test_ids": test_ids, "dev_ids": dev_ids, "training": training, "adapter_parameters": counts,
              "trainer_states": states, "previous_trainer_states": prior,
              "previous_adapter_parameters": prior_counts, "test": test, "dev": dev}
    path = study / "compact-results.json.gz"
    encoded = gzip.compress(json.dumps(report, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Export already exists with different content")
    else:
        with path.open("xb") as handle:
            handle.write(encoded)
    print(f"Archived {len(training)} trained adapters, {len(dev)} development and {len(test)} test ledgers", flush=True)


if __name__ == "__main__":
    main()
