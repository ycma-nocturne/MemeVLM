#!/usr/bin/env python3
"""Render a LaTeX result fragment from sanitized reference coverage JSON."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import sys

MODELS = {"4B": "Qwen/Qwen3-VL-4B-Instruct", "8B": "Qwen/Qwen3-VL-8B-Instruct"}
CONDITIONS = ("image_only", "hinted", "ocr_only")
IMAGE = "image_only_minus_ocr_only"
HINT = ("K", "R", "R_minus_K")


def pct(value):
    value = Decimal.from_float(100 * value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{value:+.1f}" if value else "0.0"


def has_estimate(metric):
    n, total = metric["denominator"], metric["n_expected"]
    point, interval = metric["estimate"], metric["ci95"]
    if type(n) is not int or type(total) is not int or not 0 <= n <= total:
        raise ValueError("invalid family denominator")
    if (n and (type(point) not in (int, float) or not math.isfinite(point))) or (not n and point is not None):
        raise ValueError("point estimate disagrees with the family denominator")
    if interval is not None and (n < 2 or not isinstance(interval, list) or len(interval) != 2
                                 or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
                                 or interval[0] > interval[1]):
        raise ValueError("invalid family-bootstrap interval")
    return bool(n)


def effect_line(size, metrics, labels, interval_keys, expected):
    eligible = {key: has_estimate(value) for key, value in metrics.items()}
    counts = [(obj["denominator"], obj["n_expected"]) for obj in metrics.values()]
    if any(planned != expected for _, planned in counts):
        raise ValueError(f"{size} contrast requires {expected} eligible or planned families")
    if not any(eligible.values()):
        planned = counts[0][1]
        return f"{size} had 0/{planned} usable families."
    common = len(set(counts)) == 1
    pieces = []
    for key, obj in metrics.items():
        value = rf"${pct(obj['estimate'])}$" if eligible[key] else r"\textsc{na}"
        sample = "" if common else f" ({obj['denominator']}/{obj['n_expected']})"
        pieces.append(f"{labels[key]}~{value}{sample}")
    shared = f" ({counts[0][0]}/{counts[0][1]} families)" if common else ""
    intervals = []
    for key in interval_keys:
        ci = metrics[key]["ci95"]
        if ci is not None:
            intervals.append(rf"{labels[key]}~$[{pct(ci[0])}, {pct(ci[1])}]$")
    details = r", 95\% CI " + ", ".join(intervals) if intervals else ""
    return f"For {size}, " + ", ".join(pieces) + f"{shared}{details}."


def counts_line(counts, models, split, *, replay):
    total = counts["planned_explanations"]
    returned = counts["judge_run_status"].get("ok", 0)
    valid = counts["score_status"].get("ok", 0)
    details = []
    for size, name in MODELS.items():
        groups = [models[name]["counts_by_condition"][key] for key in CONDITIONS]
        planned = sum(row["planned_explanations"] for row in groups)
        usable = sum(row["score_status"].get("ok", 0) for row in groups)
        details.append((size, usable, planned))
    if sum(group[1] for group in details) != valid or sum(group[2] for group in details) != total:
        raise ValueError("condition counts do not sum to the reported totals")
    if not 0 <= valid <= returned <= total:
        raise ValueError("inconsistent judge execution and scoring counts")
    phase = "held-out" if split == "test" else "development"
    parts = " and ".join(f"{usable}/{planned} for {size}" for size, usable, planned in details)
    if replay:
        if split != "test" or (returned, total) != (240, 240):
            raise ValueError("the citation-cap replay requires all 240 stored test judgments")
        return (f"We rescored all {returned} stored judgments with zero GPU calls. "
                f"{valid}/{total} passed the relaxed rule ({parts}).")
    result = (f"The judge returned {returned}/{total} planned {phase} judgments. "
              f"{valid}/{total} passed the frozen rule ({parts}).")
    if split == "test":
        reasons = counts["unavailable_reasons"]
        failures = [reasons[key] for key in
                    ("evidence_id", "invalid_json", "truncated", "checklist_or_schema")]
        if tuple(failures) != (105, 27, 12, 5) or sum(failures) != returned - valid:
            raise ValueError("the rejection counts differ from the frozen test diagnosis")
        citation, invalid_json, judge_cap, checklist = failures
        result += (f" Rejections came from the citation-count rule ({citation}), malformed JSON "
                   f"({invalid_json}), the judge token cap ({judge_cap}), or missing checklist items ({checklist}).")
    return result


def render(data, split, *, replay=False):
    if data["protocol_id"] != "reference_reuse_v1" or data["bootstrap_unit"] != "family_id":
        raise ValueError("expected sanitized family-paired reference coverage")
    explanation = data["splits"][split]["explanation"]
    models = explanation["models"]
    opening = counts_line(explanation["counts"], models, split, replay=replay)
    ocr_eligible, hint_planned = (35, 40) if split == "test" else (18, 20)
    image, hint = [], []
    has_pairs = False
    hint_labels = {"K": r"$\Delta K$", "R": r"$\Delta R$", "R_minus_K": r"$\Delta R-\Delta K$"}
    for size, name in MODELS.items():
        report = models[name]
        contrast = report["ocr_subgroups"]["available"]["condition_differences"][IMAGE]
        keys = "R" if replay else "VIKR"
        image.append(effect_line(size, {k: contrast[k] for k in keys}, {k: k for k in keys},
                                 ("R",) if replay else ("V", "R"), ocr_eligible))
        gains = {key: report["hint_gains"][key] for key in HINT}
        hint.append(effect_line(size, gains, hint_labels, HINT, hint_planned))
        has_pairs |= any(has_estimate(value) for value in (*contrast.values(), *gains.values()))
    scale = ""
    if has_pairs:
        scale = r"Differences are percentage points with paired 95\% family-bootstrap intervals."
    if replay:
        image_header = r"\textbf{Reasoning coverage with image versus OCR.} "
        hint_header = r"\textbf{Adding a factual hint.} "
    else:
        image_header = (r"\textbf{Image versus source OCR.} "
                        f"Counts use {ocr_eligible} transcript-eligible families as the denominator. ")
        hint_header = (r"\textbf{Adding a factual hint.} "
                       f"Counts use all {hint_planned} planned families as the denominator. ")
        if split == "test":
            four = models[MODELS["4B"]]
            image_valid = four["counts_by_condition"]["image_only"]["score_status"].get("ok", 0)
            hint_valid = four["counts_by_condition"]["hinted"]["score_status"].get("ok", 0)
            if (image_valid, hint_valid) != (8, 3) or any(four["hint_gains"][key]["denominator"] for key in HINT):
                raise ValueError("4B matched hint counts differ from the frozen test diagnosis")
            hint_header += f"The {image_valid} scorable 4B image families and the {hint_valid} hinted families were disjoint. "
    return "\n\n".join(part for part in (opening + (" " + scale if scale else ""),
                                         image_header + " ".join(image), hint_header + " ".join(hint)) if part) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path, help="sanitized aggregate JSON")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--replay", action="store_true", help="render post-hoc citation-cap replay results")
    args = parser.parse_args()
    try:
        sys.stdout.write(render(json.loads(args.aggregate.read_bytes()), args.split, replay=args.replay))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
