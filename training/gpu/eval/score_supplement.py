"""Replay fixed supplementary analyses on the two English test ledgers."""
import hashlib
import json
from math import comb
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "project/training/gpu/eval"))
import materials
import run_eval


OUTER_FENCE = re.compile(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", re.IGNORECASE | re.DOTALL)


def normalize(event):
    text = event.get("text")
    if event["status"] != "ok" or not isinstance(text, str):
        return event
    match = OUTER_FENCE.fullmatch(text)
    return dict(event, text=match[1]) if match and "```" not in match[1] else event


def valid(row):
    return row["score"] is not None and row["score"]["status"] == "ok"


def arm_diagnostics(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], {})[row["variant"]] = row
    correct_flip = correct_original = preserved = 0
    for group in groups.values():
        original = group["original"]["score"] or {}
        edited = group["edited"]["score"] or {}
        sham = group["sham"]["score"] or {}
        if original.get("relation_correct"):
            correct_original += 1
            correct_flip += bool(edited.get("relation_correct"))
            preserved += (sham.get("status") == "ok"
                          and sham.get("semantic_relation") == original.get("semantic_relation"))
    return {"correct_original_and_edit_relation": [correct_flip, len(groups)],
            "sham_relation_preserved_given_correct_original": [preserved, correct_original]}


def summarize(scored):
    summary = materials.summarize(scored)
    shared = [(base, adapter) for base, adapter in zip(scored["base"], scored["adapter"])
              if valid(base) and valid(adapter)]
    for name in ("all", *materials.VARIANTS):
        pairs = [(a, b) for a, b in shared if name == "all" or a["variant"] == name]
        summary.setdefault("shared_valid_cases", {})[name] = {
            "denominator": len(pairs),
            "base_joint_correct": sum(base["score"]["joint_correct"] for base, _ in pairs),
            "adapter_joint_correct": sum(adapter["score"]["joint_correct"] for _, adapter in pairs),
        }
    shared_ids = {base["id"] for base, _ in shared}
    shared_groups = {row["group_id"] for row in scored["base"]
                     if all(row["group_id"] + "/" + variant in shared_ids for variant in materials.VARIANTS)}
    summary["shared_valid_triples"] = (
        materials.summarize({arm: [row for row in rows if row["group_id"] in shared_groups]
                             for arm, rows in scored.items()})["paired_triples"]
        if shared_groups else {"both_correct": 0, "base_only": 0, "adapter_only": 0, "both_incorrect": 0})
    summary["shared_valid_triples"]["denominator"] = len(shared_groups)
    discordant = summary["paired_triples"]
    losses, gains = discordant["base_only"], discordant["adapter_only"]
    n = losses + gains
    discordant["two_sided_exact_p"] = (
        min(1.0, 2 * sum(comb(n, count) for count in range(min(losses, gains) + 1)) / 2**n)
        if n else 1.0)
    for arm, rows in scored.items():
        summary[arm].update(arm_diagnostics(rows))
    return summary


def main():
    plan = json.loads((ROOT / "analysis-plan.json").read_text())
    assert plan["supplement_code_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cohorts, _ = materials.compile_release(ROOT / "english-corpus")
    cases = cohorts["test"]
    paths = {arm: ROOT / "runs/test" / arm / "events.jsonl" for arm in run_eval.ARMS}
    events = {arm: run_eval.read_events(path, cases)[1] for arm, path in paths.items()}
    for arm in events:
        assert len(events[arm]) == len(cases) and all(row["status"] == "ok" for row in events[arm].values())
    normalized = {arm: {key: normalize(row) for key, row in payload.items()}
                  for arm, payload in events.items()}
    report = {
        "analysis_plan_sha256": materials.sha(ROOT / "analysis-plan.json"),
        "freeze_sha256": materials.sha(ROOT / "freeze.json"),
        "raw_events_sha256": {arm: materials.sha(path) for arm, path in paths.items()},
        "strict": summarize({arm: materials.score_rows(cases, payload) for arm, payload in events.items()}),
        "outer_fence_only": summarize({arm: materials.score_rows(cases, payload)
                                      for arm, payload in normalized.items()}),
        "outer_fences_removed": {arm: sum(row != normalized[arm][key] for key, row in payload.items())
                                 for arm, payload in events.items()},
    }
    run_eval.write_new(ROOT / "supplementary-scores.json", report)


if __name__ == "__main__":
    main()
