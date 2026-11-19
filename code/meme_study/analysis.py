"""Paired analysis of supplied scored core rows, using only the standard library.

Callers select the intended split/condition before workflow comparisons. Invalid
model outputs remain rows with zero scores. Missing or duplicated pairs are errors.
"""
import math
import random
from collections import defaultdict


_IDENTITY = ("case_id", "family_id", "template_group", "split", "variant", "background")


def _ratio(num, den):
    return {"num": num, "den": den, "rate": num / den if den else None}


def _core_rows(rows, model, workflows):
    selected = []
    seen = set()
    for row in rows:
        if row.get("control") != "core" or row.get("model") != model or row.get("workflow") not in workflows:
            continue
        if any(not isinstance(row.get(key), str) or not row[key] for key in _IDENTITY):
            raise ValueError("core row has missing or invalid identity metadata")
        if type(row.get("interpretation_correct")) is not int or row["interpretation_correct"] not in (0, 1):
            raise ValueError("interpretation_correct must be integer 0 or 1")
        key = (row["case_id"], row["workflow"])
        if key in seen:
            raise ValueError(f"duplicate scored row: {key}")
        seen.add(key)
        selected.append(row)
    return selected


def _index(rows, keys):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in indexed:
            raise ValueError(f"duplicate pairing key: {key}")
        indexed[key] = row
    return indexed


def _matching(left, right, fields):
    if any(left[field] != right[field] for field in fields):
        raise ValueError("paired rows disagree on identity metadata")


def paired_background(rows, model, workflow, note, variant="original"):
    """Return repair/harm counts against this workflow's own none baseline."""
    if note not in ("relevant", "irrelevant", "false"):
        raise ValueError("note must be relevant, irrelevant, or false")
    if variant not in ("original", "edited", "sham") or (variant != "original" and note != "relevant"):
        raise ValueError("variant/background combination is outside the study design")
    selected = [r for r in _core_rows(rows, model, {workflow}) if r["variant"] == variant]
    keys = ("split", "family_id", "variant")
    baseline = _index((r for r in selected if r["background"] == "none"), keys)
    treatment = _index((r for r in selected if r["background"] == note), keys)
    if baseline.keys() != treatment.keys():
        raise ValueError("missing baseline or background pair")
    repair_num = repair_den = harm_num = harm_den = 0
    for key in sorted(baseline):
        before, after = baseline[key], treatment[key]
        _matching(before, after, ("family_id", "template_group", "split", "variant"))
        if before["interpretation_correct"]:
            harm_den += 1
            harm_num += 1 - after["interpretation_correct"]
        else:
            repair_den += 1
            repair_num += after["interpretation_correct"]
    return {"repair": _ratio(repair_num, repair_den),
            "harm": _ratio(harm_num, harm_den), "n_pairs": len(baseline)}


def paired_workflows(rows, model, workflow_a="V", workflow_b="R"):
    """Return per-case interpretation differences A minus B, retaining clusters."""
    if workflow_a == workflow_b or any(w not in ("D", "R", "V") for w in (workflow_a, workflow_b)):
        raise ValueError("comparison requires two different D/R/V workflows")
    selected = _core_rows(rows, model, {workflow_a, workflow_b})
    a = _index((r for r in selected if r["workflow"] == workflow_a), ("case_id",))
    b = _index((r for r in selected if r["workflow"] == workflow_b), ("case_id",))
    if a.keys() != b.keys():
        raise ValueError("missing workflow pair")
    pairs = []
    for key in sorted(a):
        _matching(a[key], b[key], _IDENTITY)
        pair = {field: a[key][field] for field in _IDENTITY}
        pair.update(model=model, workflow_a=workflow_a, workflow_b=workflow_b,
                    difference=a[key]["interpretation_correct"] - b[key]["interpretation_correct"])
        pairs.append(pair)
    return pairs


def _percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (position - low) * (sorted_values[high] - sorted_values[low])


def cluster_bootstrap(pairs, seed=6000, repeats=10000):
    """Bootstrap the row-mean difference by resampling whole template groups.

    Each draw retains every row of a selected group, including repeated draws.
    Uses a 95% percentile interval with linear interpolation. Fewer than two
    clusters yield no interval; an empty input also has no point estimate.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    clusters = defaultdict(list)
    for pair in pairs:
        group, difference = pair.get("template_group"), pair.get("difference")
        if not isinstance(group, str) or not group:
            raise ValueError("pair requires a nonempty template_group")
        if type(difference) not in (int, float) or not math.isfinite(difference):
            raise ValueError("pair difference must be a finite number")
        clusters[group].append(difference)
    totals = {group: (sum(values), len(values)) for group, values in clusters.items()}
    n_pairs = sum(count for _, count in totals.values())
    result = {"estimate": sum(total for total, _ in totals.values()) / n_pairs if n_pairs else None,
              "ci": None, "n_clusters": len(clusters), "n_pairs": n_pairs,
              "seed": seed, "repeats": repeats}
    if len(clusters) < 2:
        return result
    rng = random.Random(seed)
    names = sorted(clusters)
    estimates = []
    for _ in range(repeats):
        numerator = denominator = 0
        for _ in names:
            total, count = totals[rng.choice(names)]
            numerator += total
            denominator += count
        estimates.append(numerator / denominator)
    estimates.sort()
    result["ci"] = [_percentile(estimates, .025), _percentile(estimates, .975)]
    return result


def edit_consistency(rows, model, workflow, background="none"):
    """Require correct original+edited, original+sham, and all three images."""
    if background not in ("none", "relevant"):
        raise ValueError("edit analysis only supports none or relevant background")
    selected = [r for r in _core_rows(rows, model, {workflow}) if r["background"] == background]
    indexed = _index(selected, ("split", "family_id", "variant"))
    editable = {(r["split"], r["family_id"]) for r in selected if r["variant"] in ("edited", "sham")}
    edited_num = sham_num = triple_num = 0
    for split, family in sorted(editable):
        keys = [(split, family, variant) for variant in ("original", "edited", "sham")]
        if any(key not in indexed for key in keys):
            raise ValueError("editable family lacks original, edited, or sham pair")
        original, edited, sham = (indexed[key] for key in keys)
        for other in (edited, sham):
            _matching(original, other, ("family_id", "template_group", "split", "background"))
        c, e, s = (row["interpretation_correct"] for row in (original, edited, sham))
        edited_num += c * e
        sham_num += c * s
        triple_num += c * e * s
    count = len(editable)
    return {"original_edited": _ratio(edited_num, count),
            "original_sham": _ratio(sham_num, count),
            "triple": _ratio(triple_num, count)}
