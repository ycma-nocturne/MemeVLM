"""Import existing annotations into an explicitly unreviewed candidate queue."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from urllib.parse import quote

REPOSITORY = "anonymous-neurips-2026/memebench"


def candidate_index(records: list[dict], *, revision: str) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable dataset revision")
    seen, candidates = set(), []
    for row in records:
        item = row.get("id")
        if type(item) is not int or item < 1:
            raise ValueError("Expected positive integer source ID")
        if item in seen:
            raise ValueError(f"duplicate source ID: {item}")
        seen.add(item)
        path = row.get("image_path", "")
        if (not isinstance(path, str) or not path.startswith("images/") or
                ".." in PurePosixPath(path).parts or "\\" in path or ":" in path):
            raise ValueError(f"Unsafe source image path for {item}")
        if row.get("file_name", path) != path:
            raise ValueError(f"Conflicting image paths for {item}")
        if row.get("meta", {}).get("language") != "zh":
            continue
        candidates.append({
            "candidate_id": f"memebench-{item:04d}", "source_id": item,
            "dataset": REPOSITORY, "revision": revision,
            "source_url": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/metadata.jsonl",
            "image_url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{quote(path)}",
            "source_image_path": path, "original_post_url": None,
            "annotation_license": "CC-BY-NC-SA-4.0",
            "image_rights_status": "not_independently_established",
            "status": "unreviewed_candidate", "human_reviews": [],
            "existing_annotation": row,
        })
    return sorted(candidates, key=lambda x: x["source_id"])


def image_receipt(data: bytes, *, source_name: str, expected_sha256: str | None = None) -> dict:
    """Inspect signatures only. A decoder must separately establish readability."""
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Image checksum differs from expected bytes")
    if data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        fmt, extensions = "jpeg", {".jpg", ".jpeg"}
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt, extensions = "png", {".png"}
    elif data.startswith((b"GIF87a", b"GIF89a")):
        fmt, extensions = "gif", {".gif"}
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt, extensions = "webp", {".webp"}
    else:
        raise ValueError("Unrecognized or incomplete supported image signature")
    return {"sha256": digest, "bytes": len(data), "detected_format": fmt,
            "source_name": source_name,
            "extension_mismatch": PurePosixPath(source_name).suffix.lower() not in extensions,
            "decoded": False}
