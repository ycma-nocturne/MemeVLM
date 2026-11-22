"""Verify the frozen MemeQA artifacts and recompute the reported scores."""
import argparse
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path

from run_memeqa import answer, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    root = args.evidence
    freeze = json.loads((root / "freeze.json").read_text())
    scores = json.loads((root / "scores.json").read_text())

    def verify(name, expected):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Digest mismatch for {name}")

    verify("freeze.json", scores["freeze_sha256"])
    verify("gold.jsonl", freeze["gold_sha256"])
    verify("selection-manifest.json", freeze["selection_manifest_sha256"])
    verify("run_memeqa.py", freeze["runner_sha256"])

    gold_rows = lines(root / "gold.jsonl")
    gold = {row["id"]: row["answer"] for row in gold_rows}
    if len(gold) != len(gold_rows) or len(gold) != freeze["planned_per_arm"]:
        raise ValueError("Gold IDs must be unique and match the planned sample size")
    all_correct = {}
    for arm, expected in scores["per_arm"].items():
        name = arm + ".jsonl"
        verify(name, scores["ledger_sha256"][arm])
        responses = lines(root / name)
        if len(responses) != len(gold) or {row["id"] for row in responses} != gold.keys():
            raise ValueError(f"Unmatched case IDs in {name}")
        choices = {row["id"]: answer(row) for row in responses}
        correct = {key: choice == gold[key] for key, choice in choices.items()}
        result = {"generated": len(responses), "valid": sum(x is not None for x in choices.values()),
                  "correct": sum(correct.values())}
        if result != expected:
            raise ValueError(f"Recomputed choice accuracy differs for {arm}")
        all_correct[arm] = correct

    paired = Counter((all_correct["base"][key], all_correct["adapter"][key]) for key in gold)
    gains, losses = paired[False, True], paired[True, False]
    n = gains + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)
    result = {"both_correct": paired[True, True], "base_only": losses, "adapter_only": gains,
              "both_incorrect": paired[False, False], "two_sided_exact_p": p}
    if result != scores["paired"]:
        raise ValueError("Recomputed paired totals differ")
    print(json.dumps({"per_arm": scores["per_arm"], "paired": result}, sort_keys=True))


if __name__ == "__main__":
    main()

