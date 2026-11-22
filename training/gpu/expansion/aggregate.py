"""Recompute expanded benchmark results from verified paired response ledgers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
import sys


ARMS = ("base", "bilingual-42", "bilingual-43", "bilingual-44", "chinese-42", "chinese-43", "chinese-44")
BASE, BILINGUAL, CHINESE = (0,), (1, 2, 3), (4, 5, 6)
CONTRASTS = {"bilingual_minus_base": (BILINGUAL, BASE), "chinese_minus_base": (CHINESE, BASE),
             "bilingual_minus_chinese": (BILINGUAL, CHINESE)}


def metrics(rows, draws, seed):
    """Rows are one cluster per image, with a shared question count and per-arm sums."""
    den = sum(row[0] for row in rows)
    counts = [sum(row[i + 1] for row in rows) for i in range(len(ARMS))]
    rates = [count / den for count in counts]
    out = {"images": len(rows), "questions": den,
           "per_arm": {arm: {"correct": n, "denominator": den, "accuracy": rate}
                       for arm, n, rate in zip(ARMS, counts, rates)},
           "by_profile": {}}
    for profile, cols in (("bilingual", BILINGUAL), ("chinese", CHINESE)):
        values = [rates[c] for c in cols]
        out["by_profile"][profile] = {"mean": mean(values), "seed_sample_std": stdev(values)}
    rng = random.Random(seed)
    samples = {name: [] for name in CONTRASTS}
    for _ in range(draws):
        selected = rng.choices(rows, k=len(rows))
        totals = [sum(col) for col in zip(*selected)]
        for name, (a, b) in CONTRASTS.items():
            samples[name].append((mean(totals[c + 1] for c in a) - mean(totals[c + 1] for c in b)) / totals[0])
    out["paired_contrasts"] = {}
    for name, (a, b) in CONTRASTS.items():
        values = sorted(samples[name])
        out["paired_contrasts"][name] = {
            "delta": mean(rates[c] for c in a) - mean(rates[c] for c in b),
            "paired_image_bootstrap_ci95": [values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    exp = root / "final-expansion"
    sys.path.insert(0, str(root / "project/code"))
    sys.path.insert(0, str(root / "project/training/gpu/eval"))
    sys.path.insert(0, str(exp))
    import materials
    import run_eval
    from run_memeqa import answer, digest, lines

    plan_path = exp / "protocol.json"
    plan = json.loads(plan_path.read_text())
    assert digest(Path(__file__)) == plan["code"]["aggregate.py"]
    assert [arm["name"] for arm in plan["arms"]] == list(ARMS)
    prior_path = root / "evaluation/freeze.json"
    prior = json.loads(prior_path.read_text())
    assert digest(prior_path) == plan["prior_freeze_sha256"]
    cohorts, hashes = materials.compile_release(root / "evaluation/english-corpus")
    cases = cohorts["test"]
    assert len(cases) == 96
    assert run_eval.release_identity(cohorts, hashes) == {k: v for k, v in prior.items() if k != "training"}

    def synthetic(path, arm):
        frozen = json.loads((path / "freeze.json").read_text())
        assert {k: v for k, v in frozen.items() if k != "training"} == {
            k: v for k, v in prior.items() if k != "training"}
        assert frozen["training"]["model_files"] == prior["training"]["model_files"]
        result = path / "runs/test" / arm
        run = json.loads((result / "run.json").read_text())
        assert run["identity"] == run_eval.arm_identity(digest(path / "freeze.json"), "test", arm, cases)
        if arm == "adapter":
            entry = current["adapter_run"]
            selected = root / entry / "selected-adapter.json"
            assert frozen["training"]["selected_adapter_receipt_sha256"] == digest(selected)
        _, outcomes = run_eval.read_events(result / "events.jsonl", cases)
        assert len(outcomes) == len(cases) and all(row["status"] == "ok" for row in outcomes.values())
        rows = materials.score_rows(cases, outcomes)
        return rows, run["hardware"], digest(result / "events.jsonl")

    synth, synth_hashes = {}, {}
    for current in plan["arms"]:
        name = current["name"]
        original = name in ("base", "bilingual-42")
        path = root / "evaluation" if original else exp / "synthetic-runs" / name
        arm = "base" if name == "base" else "adapter"
        rows, hardware, ledger_sha = synthetic(path, arm)
        synth[name] = rows
        synth_hashes[name] = {"sha256": ledger_sha, "hardware": hardware}
    synthetic_groups = defaultdict(list)
    for index, item in enumerate(cases):
        synthetic_groups[item["group_id"]].append(index)
    assert len(synthetic_groups) == 32 and all(len(indices) == 3 for indices in synthetic_groups.values())
    triple_rows = [(1, *(sum(synth[arm][i]["score"]["joint_correct"] for i in indexes) == 3 for arm in ARMS))
                   for indexes in synthetic_groups.values()]
    synth_summary = metrics(triple_rows, plan["bootstrap"]["draws"], plan["bootstrap"]["seed"])
    synth_summary["per_arm_details"] = {
        arm: {"valid": sum(row["score"]["status"] == "ok" for row in records),
              "cue_correct": sum(row["score"]["cue_correct"] for row in records),
              "joint_correct": sum(row["score"]["joint_correct"] for row in records),
              "target_edit_correct": sum(row["score"]["joint_correct"] for row in records if row["variant"] == "edited")}
        for arm, records in synth.items()}

    data = exp / "native"
    for filename, key in [("inputs.jsonl", "inputs_sha256"), ("gold.jsonl", "gold_sha256"),
                          ("manifest.json", "manifest_sha256")]:
        assert digest(data / filename) == plan["native"][key]
    inputs = lines(data / "inputs.jsonl")
    ids = {row["id"] for row in inputs}
    gold = {row["id"]: row["answer"] for row in lines(data / "gold.jsonl")}
    assert ids == set(gold) and len(ids) == plan["native"]["questions"]
    natural_responses, natural_valid, natural_hashes = {}, {}, {}
    for current in plan["arms"]:
        arm = current["name"]
        path = exp / "native-runs" / arm
        completion = json.loads((path / "completion.json").read_text())
        assert completion["cases"] == len(ids) and completion["arm"] == arm
        binding = json.loads((path / "binding.json").read_text())
        assert digest(path / "binding.json") == completion["binding_sha256"]
        assert binding["protocol_sha256"] == digest(plan_path)
        assert binding["generation_policy"] == prior["policy"]
        if current["adapter_run"]:
            assert binding["adaptation"]["selected_receipt_sha256"] == digest(
                root / current["adapter_run"] / "selected-adapter.json")
        assert digest(path / "responses.jsonl") == completion["ledger_sha256"]
        records = lines(path / "responses.jsonl")
        assert len(records) == len(ids) and {row["id"] for row in records} == ids
        rendered = {row["id"]: answer(run_eval.checked_response(row)) for row in records}
        assert all(rendered[row["id"]] == row["choice"] for row in records)
        natural_valid[arm] = {case: choice is not None for case, choice in rendered.items()}
        natural_responses[arm] = {case: choice == gold[case] for case, choice in rendered.items()}
        assert sum(natural_valid[arm].values()) == completion["valid"]
        natural_hashes[arm] = {"sha256": completion["ledger_sha256"], "hardware": completion["hardware"]}

    groups = defaultdict(list)
    for case in inputs:
        groups[case["group_id"]].append(case)
    cohorts = {"all_eligible": list(groups.values()),
               "unseen_pilot_images": [group for group in groups.values() if not group[0]["pilot_group"]]}
    native_summary = {}
    for label, cohort in cohorts.items():
        rows = [(len(group), *(sum(natural_responses[arm][case["id"]] for case in group) for arm in ARMS))
                for group in cohort]
        totals = metrics(rows, plan["bootstrap"]["draws"], plan["bootstrap"]["seed"])
        totals["per_arm_valid"] = {
            arm: sum(natural_valid[arm][case["id"]] for group in cohort for case in group) for arm in ARMS}
        native_summary[label] = totals

    output = {"protocol_sha256": digest(plan_path), "synthetic_strict_triples": synth_summary,
              "native_memeqa": native_summary, "synthetic_ledgers": synth_hashes, "native_ledgers": natural_hashes}
    run_eval.write_new(exp / "full-results.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

