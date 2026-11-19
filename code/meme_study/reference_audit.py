"""Export blank forms for a two-rater audit of 30 explanations.

Select candidates by stable job IDs and predefined strata. Retain eligible
outputs with judge failures, empty text, or truncation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
import unicodedata

from .data import content_hash
from .freeze import _publish


QUOTAS = {"dev": {"image_only": 2, "hinted": 2, "ocr_only": 1},
          "test": {"image_only": 4, "hinted": 3, "ocr_only": 3}}
RATING_FIELDS = ("audit_id", "reviewer", "V_coverage", "I_coverage", "K_coverage",
                 "R_coverage", "grounded_relation", "invented_claim", "notes")
_PUBLIC_FIELDS = ("family_id", "split", "condition", "image_path", "candidate_text",
                  "candidate_truncated", "reference_image_path", "reference", "ocr_text", "hint_text")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _image_path(value):
    if value is not None and (not isinstance(value, str) or not value.strip()
            or "\\" in value or PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts):
        raise ValueError("image path must remain relative to the material data root")


def _validate(rows, model_ids, seed):
    if (not isinstance(model_ids, list) or len(model_ids) != 2
            or any(not isinstance(m, str) or not m.strip() for m in model_ids)
            or len(set(model_ids)) != 2):
        raise ValueError("the 30-output audit requires exactly two distinct model IDs")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(rows, list) or not rows:
        raise ValueError("score rows must be a nonempty list")
    seen, logical_ids, families, groups = set(), set(), {}, {}
    explanations = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("score row must be an object")
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip() or job_id in seen:
            raise ValueError("missing or duplicate job_id")
        seen.add(job_id)
        if row.get("task_kind") != "explanation":
            continue
        family, model, split, condition = (row.get(k) for k in ("family_id", "model", "split", "condition"))
        if not isinstance(family, str) or not family.strip():
            raise ValueError("explanation family_id is required")
        if (not all(isinstance(v, str) for v in (model, split, condition))
                or model not in model_ids or split not in QUOTAS or condition not in QUOTAS[split]):
            raise ValueError("unknown explanation model, split or condition")
        base_id = f"explanation/{family}/{condition}/{model}"
        expected_id = base_id
        if "prepared_job_id" in row or "prepared_sha256" in row:
            prepared_hash = row.get("prepared_sha256")
            if (row.get("prepared_job_id") != base_id or not isinstance(prepared_hash, str)
                    or re.fullmatch("[0-9a-f]{64}", prepared_hash) is None):
                raise ValueError("prepared job metadata or preparation hash is invalid")
            expected_id = f"release/{prepared_hash}/{base_id}"
        if job_id != expected_id:
            raise ValueError("explanation job_id disagrees with row metadata")
        if base_id in logical_ids:
            raise ValueError("duplicate explanation identity across preparation namespaces")
        logical_ids.add(base_id)
        if family in families and families[family] != split:
            raise ValueError(f"family {family} crosses splits")
        families[family] = split
        if "template_group" in row:
            group = row["template_group"]
            if not isinstance(group, str) or not group:
                raise ValueError("invalid template_group")
            if group in groups and groups[group] != split:
                raise ValueError("template group crosses splits")
            groups[group] = split
        digest = row.get("source_input_sha256")
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("source_input_sha256 must contain a SHA-256 digest")
        if (not isinstance(row.get("run_status"), str)
                or row["run_status"] not in {"ok", "failed", "interrupted", "missing", "pending"}):
            raise ValueError("run_status must be ok, failed, interrupted, missing or pending")
        if "image_path" not in row:
            raise ValueError("image_path must be supplied, including null for OCR-only inputs")
        _image_path(row["image_path"])
        if "reference_image_path" in row:
            _image_path(row["reference_image_path"])
        if "candidate_truncated" in row and type(row["candidate_truncated"]) is not bool:
            raise ValueError("candidate_truncated must be a boolean")
        if row["run_status"] == "ok":
            if not isinstance(row.get("candidate_text"), str):
                raise ValueError("successful candidate_text must be a string, possibly empty")
            score = row.get("score")
            if score is not None and not isinstance(score, dict):
                raise ValueError("score must be an object or null")
            hashes = (score or {}).get("source_sha256", {})
            if not isinstance(hashes, dict):
                raise ValueError("score source hashes must be an object")
            if "candidate" in hashes and hashes["candidate"] != _sha(row["candidate_text"].encode()):
                raise ValueError("candidate text differs from the scored candidate hash")
        explanations.append(row)
    # Reject non-JSON values and NaN before deriving manifests.
    try:
        content_hash(rows)
    except (TypeError, ValueError) as error:
        raise ValueError("score rows must contain finite JSON values") from error
    return sorted(explanations, key=lambda row: row["job_id"])


def _applicable(row):
    counts = {d: None for d in "VIKR"}
    reference = row.get("reference")
    if reference is not None:
        if not isinstance(reference, dict) or not isinstance(reference.get("checklist"), list):
            raise ValueError("reference must contain a checklist list")
        checklist_hash = (row.get("score") or {}).get("source_sha256", {}).get("checklist")
        if checklist_hash is not None and checklist_hash != content_hash(reference["checklist"]):
            raise ValueError("reference checklist differs from the scored checklist hash")
        counts = dict.fromkeys("VIKR", 0)
        seen = set()
        for item in reference["checklist"]:
            if (not isinstance(item, dict) or not isinstance(item.get("dimension"), str)
                    or item["dimension"] not in counts
                    or not isinstance(item.get("item_id"), str) or not item["item_id"]
                    or item["item_id"] in seen or not isinstance(item.get("text"), str)):
                raise ValueError("reference checklist contains an invalid or duplicate item")
            seen.add(item["item_id"])
            counts[item["dimension"]] += 1
    dimensions = (row.get("score") or {}).get("dimensions", {})
    if not isinstance(dimensions, dict):
        raise ValueError("score dimensions must be an object")
    for dimension in "VIKR":
        if dimension not in dimensions:
            continue
        value = dimensions[dimension]
        count = value.get("applicable") if isinstance(value, dict) else None
        if type(count) is not int or count < 0:
            raise ValueError("score applicable item count must be a nonnegative integer")
        if counts[dimension] is not None and counts[dimension] != count:
            raise ValueError("reference checklist and scored applicability disagree")
        counts[dimension] = count
    return counts


def select_audit(rows, *, model_ids, seed=6000, phase='all') -> dict:
    """Select 10 dev + 20 test explanations, or fail with explicit shortfalls.

    Each model contributes dev 2/2/1 and test 4/3/3 in image/hinted/OCR order.
    Each stratum has an independent deterministic seed. Within-phase shuffling
    keeps IDs stable whether dev/test are exported separately or together. Metadata
    bindings are checked locally; upstream input authenticity remains the
    score producer's responsibility because complete jobs are not supplied here.
    """
    if not isinstance(phase,str) or phase not in {'all','dev','test'}:
        raise ValueError('audit phase must be all, dev or test')
    explanations = [r for r in _validate(rows, model_ids, seed) if phase=='all' or r['split']==phase]
    eligible = [row for row in explanations if row["run_status"] == "ok"]
    applicability = {row["job_id"]: _applicable(row) for row in eligible}
    selected, strata, shortages = [], [], []
    for split, quotas in QUOTAS.items():
        if phase!='all' and phase!=split:
            continue
        phase_rows=[]
        for model in sorted(model_ids):
            for condition, required in quotas.items():
                pool = [r for r in eligible if (r["split"], r["model"], r["condition"]) == (split, model, condition)]
                strata.append({"split": split, "model": model, "condition": condition,
                               "required": required, "eligible": len(pool)})
                if len(pool) < required:
                    shortages.append(f"{split}/{model}/{condition}: required {required}, available {len(pool)}")
                else:
                    rng=random.Random(content_hash([seed,split,model,condition]))
                    phase_rows.extend(rng.sample(pool, required))
        random.Random(content_hash([seed,split,'blind-order'])).shuffle(phase_rows)
        selected.extend(phase_rows)
    if shortages:
        raise ValueError("audit quota shortfall; " + "; ".join(shortages))
    sample, private_key = [], []
    next_id={'dev':1,'test':11}
    for row in selected:
        index=next_id[row['split']];next_id[row['split']]+=1
        audit_id = f"A{index:03d}"
        item = {"audit_id": audit_id, **{k: copy.deepcopy(row[k]) for k in _PUBLIC_FIELDS if k in row}}
        item["candidate_truncated"] = row.get("candidate_truncated", False)
        item["applicable_items"] = applicability[row["job_id"]]
        sample.append(item)
        private_key.append({"audit_id": audit_id, **{k: copy.deepcopy(row.get(k)) for k in
                            ("job_id", "prepared_job_id", "prepared_sha256", "family_id", "split", "model",
                             "condition", "source_input_sha256", "score")},
                            "source_row_sha256": content_hash(row)})
    excluded = [{"job_id": r["job_id"], "run_status": r["run_status"],
                 "reason": f"candidate inference {r['run_status']}; no successful candidate to audit"}
                for r in explanations if r["run_status"] != "ok"]
    return {"protocol_id": "reference_reuse_v1", "audit_schema": "small_explanation_audit_v2", "phase":phase,
            "seed": seed, "sample": sample, "private_key": private_key, "excluded": excluded,
            "excluded_by_status": {status: sum(row["run_status"] == status for row in excluded)
                                   for status in ("failed", "interrupted", "missing", "pending")},
            "strata": strata, "source_rows_sha256": content_hash(sorted(rows, key=lambda r: r["job_id"])),
            "sample_sha256": content_hash(sample), "out_of_scope_rows": len(rows) - len(explanations)}


def _blank_csv(sample):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=RATING_FIELDS)
    writer.writeheader()
    writer.writerows({"audit_id": row["audit_id"]} for row in sample)
    return stream.getvalue().encode()


def _candidate_html(sample):
    columns = ("audit_id", "split", "condition", "family_id", "image_path", "reference_image_path",
               "candidate_text", "candidate_truncated", "reference", "ocr_text", "hint_text", "applicable_items")
    def cell(value):
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
        return "<td><pre>" + html.escape(text) + "</pre></td>"
    body = "\n".join("<tr>" + "".join(cell(row.get(k)) for k in columns) + "</tr>" for row in sample)
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Explanation audit</title>"
            "<style>body{font-family:system-ui}table{border-collapse:collapse}th,td{border:1px solid #bbb;"
            "padding:8px;vertical-align:top}pre{white-space:pre-wrap;min-width:120px;max-width:480px;"
            "font:inherit}</style><h1>Explanation audit — blank ratings supplied separately</h1>"
            "<p>Paths refer to existing local materials. No images are downloaded or embedded. "
            "Inspect source references; missing reference material is not a zero score. "
            "Model identities and automatic judge scores are withheld from this page.</p><table><thead><tr>"
            + "".join("<th>" + html.escape(k) + "</th>" for k in columns)
            + "</tr></thead><tbody>" + body + "</tbody></table></html>\n").encode()


_README = """# Small explanation audit

The full study samples 30 explanations: 10 development and 20 test outputs. A
package may export either phase first or both together. Two members independently
complete one blank CSV each (60 initial ratings across the two phases).
Exporting the package records no completed rating, acceptance or adjudication.

Give each rater candidates.html or sample.json, their own reviewer_N.csv, and
the corresponding existing images and source references. Keep private_key.json,
excluded.json and manifest.json with the coordinator: they contain model IDs or
links back to unblinded runs. The public fields withhold model metadata and judge
scores; candidate text is unmodified, so self-identification in text is possible.

Each CSV has audit_id, reviewer, V/I/K/R_coverage, grounded_relation,
invented_claim and notes. Enter your own reviewer name/ID on every row. Leave the
other member's file untouched and do not exchange ratings before completion.
Coverage is 0 (none), 0.5 (partial) or 1 (full). Use NA only when that dimension
has zero applicable source checklist items. Unknown/missing reference materials
must be obtained before rating, not treated as NA or zero. Grounded_relation and
invented_claim use yes, no or uncertain. Record evidence and defensible reference
gaps in notes; source coverage does not settle every possible interpretation.

Image paths are relative to the existing material data root. OCR-only inputs may
have no image_path; reference_image_path, when supplied, identifies the original
image for the audit. Inspect candidate_text alongside the source reference,
checklist and condition; do not confuse omitted visual coverage in OCR-only
outputs with an automatically incorrect interpretation. Optional reference,
OCR and hint fields are copied only when the caller supplied them. Their absence
requires the coordinator to supply source materials separately.

Selection uses seed 6000 by default, sorted input job IDs and model IDs, fixed
strata with independent seeds, and a within-phase shuffle. Development IDs are
A001...A010; test IDs are A011...A030. Phase exports retain those same memberships
and IDs, so calibration can precede test inference. Per model, development
quotas are image_only=2, hinted=2, ocr_only=1; test quotas are 4, 3, 3. Run-status
ok outputs remain eligible even when empty, truncated, or judge-unavailable.
Failed/interrupted inference and missing/pending planned outputs are enumerated
in excluded.json and counted separately by status in the manifest. Missing quotas
abort export. Inputs must be produced by the bound score pipeline: this utility
checks row consistency/hashes but cannot revalidate absent full upstream jobs.

Keep the development ratings separate when calibrating the judge prompt, then
freeze the judge before interpreting the 20 test ratings. Report raw independent
ratings and disagreements without replacing them by automatic adjudication.
This small test audit cannot establish general judge reliability. Poor agreement
keeps coverage results descriptive; it does not justify selective reruns.

The manifest records the actual seed, sampling strata, input/sample hashes and
artifact SHA-256 digests. This is a new-only atomic export; reruns need a new
destination. No model, image download, human check or performance experiment ran.
"""


def export_audit(rows, *, model_ids, output_dir, seed=6000, phase='all') -> dict:
    """Publish the complete new audit package atomically; never overwrite."""
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"audit output already exists: {destination}")
    parent = destination.parent.resolve(strict=True)
    selection = select_audit(rows, model_ids=model_ids, seed=seed, phase=phase)
    sample = selection["sample"]
    readme=(f'This export contains {len(sample)} {phase} audit rows. '
            'Two blank forms are supplied; no human ratings are filled.\n\n'+_README)
    artifacts = {"sample.json": _json_bytes(sample), "private_key.json": _json_bytes(selection["private_key"]),
                 "excluded.json": _json_bytes(selection["excluded"]), "reviewer_1.csv": _blank_csv(sample),
                 "reviewer_2.csv": _blank_csv(sample), "candidates.html": _candidate_html(sample),
                 "README.md": readme.encode()}
    summary = {"sample_count": len(sample), "phase":phase,
               "dev_count":sum(r['split']=='dev' for r in sample),
               "test_count":sum(r['split']=='test' for r in sample),
               "ratings_completed": 0, "excluded_count": len(selection["excluded"]),
               "excluded_by_status": selection["excluded_by_status"],
               "sample_sha256": selection["sample_sha256"], "seed": seed}
    manifest = {k: v for k, v in selection.items() if k not in {"sample", "private_key", "excluded"}}
    manifest.update(summary)
    manifest["artifact_sha256"] = {name: _sha(raw) for name, raw in artifacts.items()}
    artifacts["manifest.json"] = _json_bytes(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        for name, raw in artifacts.items():
            with (staging / name).open("xb") as stream:
                stream.write(raw)
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def _rating_form(rows, samples):
    if not isinstance(rows, list) or len(rows) != len(samples):
        raise ValueError("each completed form must rate every exported audit ID exactly once")
    result, identities = {}, set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(RATING_FIELDS):
            raise ValueError("rating rows must contain exactly the exported CSV fields")
        audit_id = row["audit_id"]
        if not isinstance(audit_id, str) or audit_id not in samples or audit_id in result:
            raise ValueError("unknown or duplicate audit_id in completed form")
        reviewer = row["reviewer"]
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("every completed rating needs an identified reviewer")
        identities.add(unicodedata.normalize("NFKC", reviewer).strip().casefold())
        values = {}
        for dimension in "VIKR":
            field = dimension + "_coverage"
            value = row[field]
            applicable = samples[audit_id]["applicable_items"][dimension]
            if value == "NA":
                if applicable != 0:
                    raise ValueError("NA is allowed only for a known zero-item source dimension")
                values[field] = None
            else:
                if (type(value) not in (str, int, float)
                        or str(value) not in {"0", "0.0", "0.5", "1", "1.0"}):
                    raise ValueError("coverage must be 0, 0.5, 1 or justified NA; blank ratings are incomplete")
                if applicable == 0:
                    raise ValueError("zero-item source dimensions must use NA")
                values[field] = float(value)
        for field in ("grounded_relation", "invented_claim"):
            if not isinstance(row[field], str) or row[field] not in {"yes", "no", "uncertain"}:
                raise ValueError(f"{field} must be yes, no or uncertain")
            values[field] = row[field]
        if not isinstance(row["notes"], str):
            raise ValueError("rating notes must be text")
        result[audit_id] = values
    if len(identities) != 1:
        raise ValueError("each form must belong to one identified reviewer")
    return result, identities.pop()


def summarize_ratings(sample, ratings1, ratings2) -> dict:
    """Summarize raw disagreement without adjudication or reliability claims.

    ``sample`` is select_audit's return value, or a dict containing the exported
    sample list and its manifest sample_sha256. Each ratings argument is the
    complete list read from one CSV. Hash binding detects accidental edits; it
    does not authenticate the identity or independence of the named reviewers.
    """
    if (not isinstance(sample, dict) or not isinstance(sample.get("sample"), list)
            or sample.get("sample_sha256") != content_hash(sample["sample"])):
        raise ValueError("sample hash is missing or differs from the selected sample")
    rows = sample["sample"]
    if any(not isinstance(row,dict) for row in rows):
        raise ValueError('audit samples must be objects')
    phase_counts=(sum(r.get('split')=='dev' for r in rows),sum(r.get('split')=='test' for r in rows))
    if phase_counts not in {(10,0),(0,20),(10,20)}:
        raise ValueError("audit must contain exactly 10 dev, 20 test or both phase samples")
    indexed = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("audit_id"), str)
                or not re.fullmatch(r"A\d{3}", row["audit_id"])
                or row["audit_id"] in indexed or row.get("split") not in {"dev", "test"}):
            raise ValueError("invalid, duplicate or unassigned sample audit ID")
        applicability = row.get("applicable_items")
        if (not isinstance(applicability, dict) or set(applicability) != set("VIKR")
                or any(v is not None and (type(v) is not int or v < 0) for v in applicability.values())):
            raise ValueError("sample requires nonnegative or unknown source item counts")
        indexed[row["audit_id"]] = row
    first, reviewer1 = _rating_form(ratings1, indexed)
    second, reviewer2 = _rating_form(ratings2, indexed)
    if reviewer1 == reviewer2:
        raise ValueError("the two forms must identify different independent reviewers")
    fields = RATING_FIELDS[2:-1]
    splits, disagreements = {}, []
    for split in ("dev", "test"):
        ids = sorted(key for key, row in indexed.items() if row["split"] == split)
        summaries = {}
        for field in fields:
            comparable = [key for key in ids if first[key][field] is not None and second[key][field] is not None]
            differing = [key for key in comparable if first[key][field] != second[key][field]]
            summaries[field] = {"comparable": len(comparable), "disagreements": len(differing),
                                "disagreement_rate": len(differing) / len(comparable) if comparable else None,
                                "na_excluded": len(ids) - len(comparable)}
            disagreements.extend({"audit_id": key, "split": split, "field": field,
                                  "rating1": first[key][field], "rating2": second[key][field]}
                                 for key in differing)
        splits[split] = {"sample_count": len(ids), "fields": summaries}
    return {"sample_sha256": sample["sample_sha256"], "ratings_completed": 2*len(rows),
            "reviewers": [reviewer1, reviewer2], "splits": splits, "disagreements": disagreements,
            "limitations": "Raw independent ratings only. The 20 test samples cannot establish general judge reliability."}
