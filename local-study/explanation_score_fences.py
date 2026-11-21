"""Replay Task A scores after removing one outer JSON fence from judge text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import explanation_study as original
from meme_study import reference_analysis, reference_protocol, reference_scoring
from meme_study.data import content_hash
from meme_study.reference_execution import _execution_lock, _write_bundle, load_prepared
from meme_study.reference_runner import _LEDGER


SCOPE = "local_mlx_2b_task_a_outer_json_fence"
RULE = "strip_sole_outer_json_fence_v1"
OUTER = re.compile(r"[ \t\r\n]*\x60{3}json[ \t]*\r?\n(?P<body>.*?)\r?\n\x60{3}[ \t\r\n]*",
                   re.DOTALL)
INNER = re.compile(r"^[ \t]*\x60{3}", re.MULTILINE)


def strip_json_fence(text):
    """Remove a complete JSON wrapper and leave the captured body unchanged."""
    match = OUTER.fullmatch(text)
    if match is None or INNER.search(match["body"]):
        return text
    return match["body"]


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize_score(candidate, checklist, judge_text, truncated, strict):
    normalized = strip_json_fence(judge_text)
    changed = normalized != judge_text
    score = reference_scoring.score_checklist(candidate, checklist, normalized,
                                             truncated=truncated) if changed else strict
    return score, {
        "rule": RULE,
        "changed": changed,
        "raw_judge_text_sha256": hash_bytes(judge_text.encode("utf-8")),
        "normalized_judge_text_sha256": hash_bytes(normalized.encode("utf-8")),
        "strict_status": strict["status"],
        "strict_failure_reason": strict["failure_reason"],
    }


def score_rows(study, snapshot, split):
    rows = original.score_rows(study, snapshot, split)
    jobs = {job["job_id"]: job for job in study.judge_jobs(split, snapshot, require_candidates=False)}
    for row in rows:
        strict = row["score"]
        if strict is None:
            continue
        judge = original.bound_output(jobs[row["judge_job_id"]], snapshot)
        score, trace = normalize_score(row["candidate_text"], row["reference"]["checklist"],
                                       judge["text"], judge["truncated"], strict)
        row["score"] = score
        row["judge_normalization"] = trace
    return rows


def rule_binding(study, original_freeze_sha):
    return {
        "scope": SCOPE,
        "scoring_rule": RULE,
        "prepared_sha256": study.prepared,
        "compiled_protocol_sha256": study.frozen_protocol,
        "explanation_freeze_sha256": original_freeze_sha,
        "code_sha256": {
            "supplemental_scorer": original.sha(__file__),
            "explanation_runner": original.sha(original.__file__),
            "exact_checklist_scorer": original.sha(reference_scoring.__file__),
            "analysis": original.sha(reference_analysis.__file__),
        },
    }


def check_scoring_freeze(path, binding, ledger_path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("binding") != binding:
        raise ValueError("supplemental scoring freeze differs from current scoring inputs")
    prefix = value.get("dev_ledger_prefix")
    if not isinstance(prefix, dict) or type(prefix.get("bytes")) is not int or prefix["bytes"] < 0:
        raise ValueError("supplemental scoring freeze has an invalid ledger prefix")
    count = prefix["bytes"]
    actual = b""
    if count:
        with ledger_path.open("rb") as stream:
            actual = stream.read(count)
    if len(actual) != count or hash_bytes(actual) != prefix.get("sha256"):
        raise ValueError("ledger differs from the supplemental scoring freeze prefix")
    return hash_bytes(raw)


def summarize(rows, study, ledger_sha, split):
    traces = [row["judge_normalization"] for row in rows if "judge_normalization" in row]
    return {
        "scope": SCOPE, "scoring_rule": RULE, "split": split, "prepared_sha256": study.prepared,
        "ledger_sha256": ledger_sha, "planned_explanations": len(rows),
        "candidate_status": dict(Counter(row["run_status"] for row in rows)),
        "judge_status": dict(Counter(row["judge_status"] for row in rows)),
        "strict_score_status": dict(Counter(
            row.get("judge_normalization", {}).get("strict_status", "missing") for row in rows)),
        "normalized_score_status": dict(Counter(
            (row["score"] or {}).get("status", "missing") for row in rows)),
        "sole_json_fences_removed": sum(trace["changed"] for trace in traces),
        "strict_unavailable_normalized_ok": sum(
            row["judge_normalization"]["strict_status"] == "unavailable"
            and row["score"]["status"] == "ok" for row in rows if "judge_normalization" in row),
        "terminal_coverage": all(row["run_status"] in {"ok", "failed", "interrupted"}
                                 and row["judge_status"] not in {"missing", "pending"} for row in rows),
    }


def run(args):
    held_out = args.command == "freeze" or args.split == "test"
    if held_out and (args.freeze_receipt is None or args.scoring_freeze is None):
        raise ValueError("test requires the explanation freeze and supplemental scoring freeze paths")
    if args.command == "score" and held_out and args.allow_partial:
        raise ValueError("held-out supplemental scoring requires terminal coverage")
    note = args.test_judge_note.read_text(encoding="utf-8") if args.test_judge_note else None
    study = original.ExplanationStudy(load_prepared(args.prepared), note)
    ledger_path = args.runs / _LEDGER
    with _execution_lock(args.runs):
        snapshot = original.read_ledger(study, args.runs)
        a_freeze_sha = original.locked_test(study, snapshot, args.freeze_receipt)[1] if held_out else None
        binding = rule_binding(study, a_freeze_sha)
        if args.command == "freeze":
            if any(entry[1]["split"] == "test" for entry in snapshot["jobs"].values()):
                raise ValueError("freeze supplemental scoring before any test jobs enter the ledger")
            existing = ledger_path.read_bytes() if ledger_path.is_file() else b""
            original.write_new(args.scoring_freeze, {
                "binding": binding,
                "dev_ledger_prefix": {"bytes": len(existing), "sha256": hash_bytes(existing)},
            })
            return {"scope": SCOPE, "scoring_freeze_sha256": original.sha(args.scoring_freeze)}

        scoring_freeze_sha = check_scoring_freeze(
            args.scoring_freeze, binding, ledger_path) if held_out else None
        rows = score_rows(study, snapshot, args.split)
        summary = summarize(rows, study, snapshot["file_sha256"], args.split)
        if not summary["terminal_coverage"] and not args.allow_partial:
            raise ValueError("selected split is incomplete. A partial export is available for development")
        summary["allow_partial"] = args.allow_partial
        report = reference_analysis.analyze_reference_scores(rows)
        report.update(scope=SCOPE, primary_split=args.split, scoring_rule=RULE)
        artifacts = {
            "scored.jsonl": b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                                allow_nan=False) + "\n").encode() for row in rows),
            "summary.json": original.json_bytes(summary),
            "analysis.json": original.json_bytes(report),
        }
        artifacts["provenance.json"] = original.json_bytes({
            **binding, "protocol_id": reference_protocol.PROTOCOL_ID, "source_scope": original.SCOPE,
            "ledger_sha256": snapshot["file_sha256"], "scoring_freeze_sha256": scoring_freeze_sha,
            "artifact_sha256": {name: hash_bytes(raw) for name, raw in artifacts.items()},
        })
        _write_bundle(args.output, artifacts)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "runs"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-receipt", "scoring-freeze", "test-judge-note"):
        parser.add_argument("--" + name, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze", help="Bind this scoring rule before any test dispatch")
    score = commands.add_parser("score", help="Replay exact scoring with the supplemental rule")
    score.add_argument("--split", choices=("dev", "test"), required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--allow-partial", action="store_true", help="Allow a partial development export")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
