#!/usr/bin/env python3
"""Export only aggregate explanation metrics from verified study bundles."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from meme_study.reference_execution import load_scores
from meme_study.reference_protocol import CONDITIONS, PROTOCOL_ID

MODELS = ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")
DIMENSIONS = "VIKR"
CONTRASTS = ("image_only_minus_ocr_only", "hinted_minus_image_only", "hinted_minus_ocr_only")
METRIC_FIELDS = ("estimate", "denominator", "n_expected", "n_unavailable", "n_clusters", "ci95")
RUN_STATUS = {"ok", "failed", "interrupted", "missing", "pending"}
JUDGE_STATUS = {"ok", "failed", "interrupted", "missing", "not_applicable"}


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def metric(data):
    out = {field: data[field] for field in METRIC_FIELDS}
    n, expected, unavailable, clusters = (out[k] for k in METRIC_FIELDS[1:5])
    if (any(type(x) is not int for x in (n, expected, unavailable, clusters))
            or not 0 <= n <= expected or unavailable != expected - n or clusters != n):
        raise ValueError("invalid metric family counts")
    point, interval = out["estimate"], out["ci95"]
    if ((n == 0 and point is not None) or (n > 0 and
            (type(point) not in (int, float) or not math.isfinite(point)))):
        raise ValueError("invalid metric estimate")
    if interval is not None and (n < 2 or not isinstance(interval, list) or len(interval) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
            or interval[0] > interval[1]):
        raise ValueError("invalid metric bootstrap interval")
    return out


def failure_category(reason):
    text = (reason or "").casefold()
    if "truncat" in text:
        return "truncated"
    if "evidence id" in text:
        return "evidence_id"
    if "verbatim" in text or "span" in text or "candidate text" in text:
        return "verbatim_span"
    if (("line " in text and "column " in text) or "duplicate json key" in text
            or "non-json constant" in text):
        return "invalid_json"
    if "item" in text or "exactly" in text or "verdict" in text or "schema" in text:
        return "checklist_or_schema"
    return "other_unavailable"


def counts(rows):
    run, judge, scoring, reasons, profiles = (Counter() for _ in range(5))
    for row in rows:
        if row["run_status"] not in RUN_STATUS or row["judge_status"] not in JUDGE_STATUS:
            raise ValueError("unexpected execution status")
        run[row["run_status"]] += 1
        judge[row["judge_status"]] += 1
        score = row["score"]
        status = score["status"] if score else "not_scored"
        if status not in {"ok", "unavailable", "not_scored"}:
            raise ValueError("unexpected explanation score status")
        scoring[status] += 1
        if status == "unavailable":
            reasons[failure_category(score["failure_reason"])] += 1
        if row.get("judge_profile_sha256"):
            profile = row["judge_profile_sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", profile):
                raise ValueError("invalid judge profile SHA-256")
            profiles[profile] += 1
    return {
        "planned_explanations": len(rows), "candidate_run_status": dict(run),
        "candidate_truncated": sum(row["candidate_truncated"] for row in rows),
        "judge_run_status": dict(judge), "score_status": dict(scoring),
        "unavailable_reasons": dict(reasons), "judge_profile_sha256s": dict(profiles),
    }


def model_metrics(model):
    return {
        "condition_differences": {
            contrast: {d: metric(model["condition_differences"][contrast][d]) for d in DIMENSIONS}
            for contrast in CONTRASTS
        },
        "hint_gains": {key: metric(model["hint_gains"][key]) for key in ("K", "R", "R_minus_K")},
        "conditions": {
            condition: {
                "dimensions": {d: metric(model["conditions"][condition]["dimensions"][d])
                               for d in DIMENSIONS},
                "valid_score_coverage": metric(model["conditions"][condition]["valid_score_coverage"]),
            } for condition in CONDITIONS
        },
    }


def aggregate(scores, analysis, split):
    score_dir, analysis_dir = (p.resolve(strict=True) for p in (scores, analysis))
    score_rows = load_scores(score_dir)
    score_provenance = read(score_dir / "provenance.json")
    analysis_provenance = read(analysis_dir / "provenance.json")
    if (analysis_provenance["scores_sha256"] != score_provenance["artifact_sha256"]["scored.jsonl"]
            or analysis_provenance["analysis_sha256"] != file_hash(analysis_dir / "analysis.json")):
        raise ValueError("analysis differs from the verified scores")
    report = read(analysis_dir / "analysis.json")
    if report["protocol_id"] != PROTOCOL_ID or report["bootstrap_unit"] != "family_id":
        raise ValueError("analysis does not use the reference protocol's family bootstrap")
    rows = [row for row in score_rows if row["split"] == split and row["task_kind"] == "explanation"]
    expected = 20 if split == "dev" else 40
    models = report["splits"][split]["explanation"]["models"]
    if set(models) != set(MODELS) or len(rows) != expected * len(MODELS) * len(CONDITIONS):
        raise ValueError("score bundle does not cover both models' full planned split")
    by_model = {}
    for name in MODELS:
        subset = [row for row in rows if row["model"] == name]
        by_condition = {c: counts([row for row in subset if row["condition"] == c]) for c in CONDITIONS}
        if any(group["planned_explanations"] != expected for group in by_condition.values()):
            raise ValueError("model/condition has the wrong number of planned families")
        public = model_metrics(models[name])
        for condition in CONDITIONS:
            valid = public["conditions"][condition]["valid_score_coverage"]
            ok = by_condition[condition]["score_status"].get("ok", 0)
            if valid["denominator"] != expected or not math.isclose(valid["estimate"], ok / expected):
                raise ValueError("score status count does not match analyzed availability")
        public["counts_by_condition"] = by_condition
        public["ocr_subgroups"] = {key: model_metrics(models[name]["ocr_subgroups"][key])
                                   for key in ("available", "unavailable")}
        by_model[name] = public
    return {
        "protocol_id": PROTOCOL_ID, "schema_version": report["schema_version"],
        "primary_split": report["primary_split"], "bootstrap_unit": report["bootstrap_unit"],
        "seed": report["seed"], "draws": report["draws"],
        "input_hashes": {"scored_sha256": analysis_provenance["scores_sha256"],
                         "analysis_sha256": analysis_provenance["analysis_sha256"],
                         "prepared_sha256": score_provenance["prepared_sha256"],
                         "ledger_sha256": score_provenance["ledger_sha256"]},
        "splits": {split: {"explanation": {"counts": counts(rows), "models": by_model}}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    args = parser.parse_args()
    try:
        report = aggregate(args.scores, args.analysis, args.split)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
