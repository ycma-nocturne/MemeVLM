"""Import public annotations with attribution for coordinator-side scoring.

Scoring records include answers and checklists. Prompt builders select inputs
with an explicit allowlist.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote

from .data import content_hash
from .intake import REPOSITORY
from .reference_errata import apply_source_errata


_DIMENSIONS = (("visual", "V"), ("identity", "I"), ("knowledge", "K"), ("reasoning", "R"))


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _digest(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value, field):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts):
        raise ValueError(f"Unsafe {field}: expected a relative path")
    return PurePosixPath(value)


def _verify_image(root, value, digest, field):
    relative = _relative_path(value, field)
    expected = _digest(digest, field + " sha256")
    try:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{field} escapes data_root or is not a file")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{field} is not a readable local file") from error
    if actual != expected:
        raise ValueError(f"{field} sha256 mismatch")
    return value


def _reference(row, item):
    prefix = f"source {item}"
    explanation = _text(row.get("checked_gt"), prefix + " checked_gt reference")
    meta = row.get("meta")
    if not isinstance(meta, dict) or meta.get("language") != "zh":
        raise ValueError(f"{prefix} must have language=zh")
    domain = _text(meta.get("domain"), prefix + " meta.domain")
    visual = row.get("visual")
    if not isinstance(visual, dict) or "ocr" not in visual:
        raise ValueError(f"{prefix} visual.ocr is missing")
    ocr = visual["ocr"]
    if ocr is not None and not isinstance(ocr, str):
        raise ValueError(f"{prefix} visual.ocr must be text or null")
    knowledge = row.get("knowledge")
    facts = knowledge.get("facts") if isinstance(knowledge, dict) else None
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"{prefix} knowledge.facts must be a nonempty list")
    for fact in facts:
        _text(fact, prefix + " knowledge fact")
    checks = row.get("eval_checklist")
    if not isinstance(checks, dict) or set(checks) != {name for name, _ in _DIMENSIONS}:
        raise ValueError(f"{prefix} eval_checklist must contain visual/identity/knowledge/reasoning")
    checklist = []
    for name, dimension in _DIMENSIONS:
        values = checks[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{prefix} eval_checklist.{name} must be a nonempty list")
        for number, text in enumerate(values, 1):
            checklist.append({"item_id": f"{dimension}{number:03d}", "dimension": dimension,
                              "text": _text(text, prefix + " checklist item")})
    reasoning = row.get("reasoning")
    if not isinstance(reasoning, dict):
        raise ValueError(f"{prefix} reasoning must be an object")
    for field in ("trigger", "reference", "core_message", "logic"):
        _text(reasoning.get(field), prefix + " reasoning." + field)
    if "logic_detail" not in reasoning:
        raise ValueError(f"{prefix} reasoning.logic_detail is missing")
    if reasoning["logic_detail"] is not None:
        _text(reasoning["logic_detail"], prefix + " reasoning.logic_detail")
    return ocr, copy.deepcopy(facts), {
        "explanation": explanation, "checklist": checklist, "domain": domain,
        "reasoning": copy.deepcopy(reasoning),
    }


def import_references(metadata_path, selection, data_root, *, expected_sha256, revision, errata=None) -> list[dict]:
    """Join selected public IDs to a pinned metadata file and verified local images.

    Paths in selection are relative to data_root. The caller supplies the image
    hashes and source mapping from acquisition receipts. Hash verification does
    not establish image rights, annotation correctness, or semantic equivalence.
    Records retain selection order. This function does not write any files.
    """
    _digest(expected_sha256, "metadata expected_sha256")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must identify an immutable dataset revision")
    try:
        raw = Path(metadata_path).read_bytes()
        root = Path(data_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("metadata_path and data_root must exist and be readable") from error
    if not root.is_dir():
        raise ValueError("data_root must be a directory")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("metadata sha256 mismatch")
    source_rows = {}
    try:
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"metadata line {number} must be an object")
            item = row.get("id")
            if type(item) is not int or item < 1:
                raise ValueError(f"metadata line {number} requires a positive integer source ID")
            if item in source_rows:
                raise ValueError(f"duplicate metadata source ID: {item}")
            source_rows[item] = row
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("metadata must contain valid UTF-8 JSONL records") from error
    if not isinstance(selection, list) or not selection:
        raise ValueError("selection must be a nonempty list")
    source_ids, family_ids, group_splits, result = set(), set(), {}, []
    base = f"https://huggingface.co/datasets/{REPOSITORY}"
    for selected in selection:
        if not isinstance(selected, dict):
            raise ValueError("selection entries must be objects")
        item = selected.get("source_id")
        if type(item) is not int or item < 1:
            raise ValueError("selection requires a positive integer public source_id")
        if item in source_ids:
            raise ValueError(f"duplicate selected source ID: {item}")
        if item not in source_rows:
            raise ValueError(f"missing public source ID: {item}")
        family = _text(selected.get("family_id"), "family_id")
        group = _text(selected.get("template_group"), "template_group")
        if family != family.strip() or group != group.strip():
            raise ValueError("family_id and template_group must not have outer whitespace")
        if family in family_ids:
            raise ValueError(f"duplicate family_id: {family}")
        split = selected.get("split")
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"template leakage across splits: {group}")
        row = source_rows[item]
        published_path = _relative_path(row.get("image_path"), f"source {item} image_path")
        if (len(published_path.parts) != 2 or published_path.parts[0] != "images"
                or row.get("file_name") != row.get("image_path")):
            raise ValueError(f"source {item} has conflicting or unsafe image paths")
        ocr, facts, reference = _reference(row, item)
        image_path = _verify_image(root, selected.get("image_path"), selected.get("image_sha256"),
                                   "image_path")
        source_path = _verify_image(root, selected.get("source_image_path"),
                                    selected.get("source_image_sha256"), "source_image_path")
        if PurePosixPath(source_path).name != published_path.name:
            raise ValueError(f"source {item} image filename differs from the published image")
        source_ids.add(item)
        family_ids.add(family)
        group_splits[group] = split
        result.append({
            "family_id": family, "source_id": item, "split": split, "template_group": group,
            "source": {
                "dataset": REPOSITORY, "source_id": item, "revision": revision,
                "metadata_sha256": actual, "record_sha256": content_hash(row),
                "url": f"{base}/blob/{revision}/metadata.jsonl",
                "image_url": f"{base}/resolve/{revision}/{quote(str(published_path))}",
                "source_image_path": str(published_path),
                "annotation_origin": "inherited_public",
            },
            "image": {"path": image_path, "sha256": selected["image_sha256"],
                      "source_path": source_path, "source_sha256": selected["source_image_sha256"],
                      "adapted": selected["image_sha256"] != selected["source_image_sha256"]},
            "ocr_text": ocr, "knowledge_facts": facts, "reference": reference,
            "provenance": {
                "reference_visibility": "private_scoring_only",
                "annotation_review": "inherited_public_not_new_human_review",
                "image_adaptation_basis": "byte_sha256_difference_only",
                "field_mapping": {"ocr_text": "visual.ocr", "knowledge_facts": "knowledge.facts",
                                  "reference.explanation": "checked_gt",
                                  "reference.checklist": "eval_checklist",
                                  "reference.domain": "meta.domain", "reference.reasoning": "reasoning"},
            },
        })
    return apply_source_errata(result, errata)
