"""Validate material records and compile inputs separately from gold labels."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import PurePosixPath
import unicodedata


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def family_content_hash(family: dict) -> str:
    return content_hash({k: v for k, v in family.items() if k != "reviews"})


def validate_families(families: list[dict], *, require_reviews: bool = False) -> list[str]:
    errors, identifiers, groups = [], set(), {}
    if not isinstance(families, list) or not families:
        return ["No families supplied"]
    for family in families:
        if not isinstance(family, dict):
            errors.append("Family record must be an object")
            continue
        fid = family.get("family_id", "")
        if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", fid):
            errors.append(f"{fid}: invalid family_id")
            continue
        if fid in identifiers:
            errors.append(f"{fid}: duplicate family_id")
        identifiers.add(fid)
        split, group = family.get("split"), family.get("template_group")
        if split not in ("dev", "test"):
            errors.append(f"{fid}: split must be dev or test")
        if not isinstance(group, str) or not group:
            errors.append(f"{fid}: missing template_group")
        elif group in groups and groups[group] != split:
            errors.append(f"{fid}: template group {group} crosses splits")
        else:
            groups[group] = split
        source = family.get("source", {})
        if not isinstance(source, dict) or not str(source.get("url", "")).startswith(("https://", "http://")):
            errors.append(f"{fid}: source URL required")
            source = {}
        options, order = family.get("options", {}), family.get("option_order", [])
        if not isinstance(options, dict) or set(options) != {"o1", "o2", "o3", "none"}:
            errors.append(f"{fid}: exactly four semantic options required")
            options = {}
        if (not isinstance(order, list) or len(order) != 4 or
                any(not isinstance(x, str) for x in order) or set(order) != set(options)):
            errors.append(f"{fid}: option_order must permute the four option IDs")
        if any(not isinstance(v, str) or not v.strip() for v in options.values()):
            errors.append(f"{fid}: empty option text")
        elif len({unicodedata.normalize('NFKC', v).strip() for v in options.values()}) != len(options):
            errors.append(f"{fid}: duplicate option wording")
        for key in ("caption_zh", "fact_question"):
            value = family.get(key)
            if not isinstance(value, str) or (key == "fact_question" and not value.strip()):
                errors.append(f"{fid}: invalid {key}")
        notes = family.get("backgrounds", {})
        if not isinstance(notes, dict) or set(notes) != {"relevant", "irrelevant", "false"}:
            errors.append(f"{fid}: exactly relevant/irrelevant/false notes required")
        elif any(not isinstance(v, str) or not v.strip() for v in notes.values()):
            errors.append(f"{fid}: empty background note")
        variants = family.get("variants", {})
        if not isinstance(variants, dict) or set(variants) not in ({"original"}, {"original", "edited", "sham"}):
            errors.append(f"{fid}: variants must be original alone or original/edited/sham")
            variants = {}
        for name, variant in variants.items():
            if not isinstance(variant, dict):
                errors.append(f"{fid}/{name}: variant must be an object")
                continue
            if (not isinstance(variant.get("gold_option"), str) or
                    variant.get("gold_option") not in options or type(variant.get("gold_fact")) is not bool):
                errors.append(f"{fid}/{name}: invalid gold labels")
            path = variant.get("image_path", "")
            if (not isinstance(path, str) or not path or "\\" in path or
                    PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts):
                errors.append(f"{fid}/{name}: image path must remain relative to data root")
            if not re.fullmatch(r"[0-9a-f]{64}", str(variant.get("sha256", ""))):
                errors.append(f"{fid}/{name}: image sha256 missing or malformed")
        if all(isinstance(variants.get(n), dict) for n in ("original", "edited", "sham")):
            if variants["sham"].get("gold_option") != variants["original"].get("gold_option"):
                errors.append(f"{fid}: sham must preserve interpretation")
            if variants["edited"].get("gold_option") == variants["original"].get("gold_option"):
                errors.append(f"{fid}: edited image must change interpretation")
        if require_reviews:
            digest = family_content_hash(family)
            reviews = family.get("reviews", [])
            if not isinstance(reviews, list):
                reviews = []
            reviewers, signed = set(), set()
            for review in reviews:
                if not isinstance(review, dict) or review.get("reviewed_content_sha256") != digest:
                    continue
                name = review.get("reviewer")
                if review.get("role") != "human" or not isinstance(name, str) or not name.strip():
                    errors.append(f"{fid}: current review requires an identified human reviewer")
                    continue
                name = unicodedata.normalize("NFKC", name).strip().casefold()
                if name in signed:
                    errors.append(f"{fid}: duplicate or conflicting current review signature")
                signed.add(name)
                try:
                    when = datetime.fromisoformat(review.get("reviewed_at", "").replace("Z", "+00:00"))
                    if when.tzinfo is None or when.utcoffset() is None:
                        raise ValueError("missing timezone")
                except (AttributeError, TypeError, ValueError):
                    errors.append(f"{fid}: current review requires a timezone-aware reviewed_at")
                    continue
                if review.get("decision") == "accept":
                    reviewers.add(name)
                else:
                    errors.append(f"{fid}: current review has unresolved revise/reject or invalid decision")
            if len(reviewers) < 2:
                errors.append(f"{fid}: two distinct current human reviews required")
            if source.get("rights_status") not in ("research_only", "redistributable"):
                errors.append(f"{fid}: image usage rights need review")
    return errors


def compile_cases(families: list[dict], *, permutation_family_ids: list[str]) -> tuple[list[dict], dict]:
    errors = validate_families(families)
    if errors:
        raise ValueError("\n".join(errors))
    test_ids = {f["family_id"] for f in families if f["split"] == "test"}
    selected = set(permutation_family_ids)
    if len(selected) != len(permutation_family_ids) or not selected <= test_ids:
        raise ValueError("Permutation family IDs must be unique test families")
    cases, gold = [], {}

    def emit(family, variant_name, background, control):
        variant = family["variants"][variant_name]
        cid = "/".join((family["family_id"], variant_name, background, control))
        order = list(family["option_order"])
        if control == "permutation":
            order = order[1:] + order[:1]
        text_only = control == "text"
        case = {"case_id": cid, "family_id": family["family_id"],
                "template_group": family["template_group"], "split": family["split"],
                "variant": variant_name, "background": background, "control": control,
                "image_path": None if text_only else variant["image_path"],
                "image_sha256": None if text_only else variant["sha256"],
                "caption_zh": family["caption_zh"],
                "note": "" if background == "none" else family["backgrounds"][background],
                "options": [{"label": label, "semantic_id": oid, "text": family["options"][oid]}
                            for label, oid in zip("ABCD", order)],
                "fact_question": None if text_only else family["fact_question"]}
        cases.append(case)
        gold[cid] = {"option": variant["gold_option"], "fact": None if text_only else variant["gold_fact"]}

    for family in sorted(families, key=lambda f: f["family_id"]):
        for background in ("none", "relevant", "irrelevant", "false"):
            emit(family, "original", background, "core")
        if "edited" in family["variants"]:
            for variant in ("edited", "sham"):
                for background in ("none", "relevant"):
                    emit(family, variant, background, "core")
        if family["split"] == "test":
            for background in ("none", "relevant"):
                emit(family, "original", background, "text")
        if family["family_id"] in selected:
            emit(family, "original", "none", "permutation")
    return cases, gold


def build_jobs(cases: list[dict], models: list[str], workflows: list[str]) -> list[dict]:
    if not models or len(set(models)) != len(models):
        raise ValueError("Model IDs must be nonempty and unique")
    if not workflows or len(set(workflows)) != len(workflows) or not set(workflows) <= {"D", "R", "V"}:
        raise ValueError("Invalid workflow selection")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate cases")
    jobs = []
    for case in cases:
        for model in models:
            for workflow in workflows if case["control"] == "core" else ["D"]:
                job = {"job_id": f"{model}/{workflow}/{case['case_id']}", "model": model,
                       "workflow": workflow, "case": copy.deepcopy(case),
                       "planned_calls": 1 if workflow == "D" else 2}
                job["input_sha256"] = content_hash(job)
                jobs.append(job)
    return jobs


def design_summary(cases: list[dict], jobs: list[dict]) -> dict:
    core = [c for c in cases if c["control"] == "core"]
    calls = sum(j["planned_calls"] for j in jobs)
    return {"core_cases": len(core), "core_dev_cases": sum(c["split"] == "dev" for c in core),
            "core_test_cases": sum(c["split"] == "test" for c in core),
            "control_cases": dict(Counter(c["control"] for c in cases if c["control"] != "core")),
            "final_responses": len(jobs), "planned_calls": calls,
            "intermediate_responses": calls-len(jobs),
            "max_output_tokens": 160*len(jobs)+80*(calls-len(jobs))}
