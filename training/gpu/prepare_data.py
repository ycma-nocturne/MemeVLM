"""Verify training splits and export portable LlamaFactory ShareGPT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import SPLITS, _check_image, _eval_index, _load_split

MANIFEST = "dataset-manifest.json"
REGISTRATION = {
    f"memevlm_{split}": {
        "file_name": f"{split}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }
    for split in SPLITS
}
TABLES = {entry["file_name"] for entry in REGISTRATION.values()} | {"dataset_info.json"}
IMAGE_NAME = re.compile(r"images/([0-9a-f]{64})\.(?:jpg|jpeg|png|webp|gif|bmp|tiff?)")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _file_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in names}


def _protected_english(root: Path, cache: dict) -> tuple[set[str], set[str], set[str], int]:
    ids, groups, pixels = set(), set(), set()
    file = root / "test.jsonl"
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(field), str) or not row[field].strip()
            for field in ("id", "group_id", "image")
        ):
            raise ValueError(f"bilingual/test.jsonl:{number}: invalid held-out metadata")
        if row["id"] in ids:
            raise ValueError(f"bilingual/test.jsonl:{number}: duplicate held-out ID")
        ids.add(row["id"])
        groups.add(row["group_id"])
        pixels.add(_check_image(root, row["image"], row.get("image_sha256"), cache)[2])
    if not ids:
        raise ValueError("bilingual/test.jsonl: expected held-out rows")
    return ids, groups, pixels, len(ids)


def _assemble(project: Path, profile: str) -> tuple[dict, dict]:
    if profile not in ("chinese", "bilingual"):
        raise ValueError("profile must be chinese or bilingual")
    project = project.resolve(strict=True)
    chinese = project / "training"
    bilingual = chinese / "bilingual"
    cache = {}
    split_names = tuple(f"{split}.jsonl" for split in SPLITS)
    sources = {
        "chinese": _file_hashes(chinese, split_names),
        "formal": _file_hashes(project / "data/study", ("config.json", "selection.json", "pairs.json")),
    }
    locked = None
    if profile == "bilingual" or bilingual.exists():
        bilingual = bilingual.resolve(strict=True)
        if not bilingual.is_relative_to(project):
            raise ValueError("bilingual source must be inside the project")
        sources["bilingual"] = _file_hashes(bilingual, (*split_names, "test.jsonl", "pairs.jsonl"))
        locked = _protected_english(bilingual, cache)

    rows = {split: [("zh", row) for row in _load_split(chinese, split, cache)] for split in SPLITS}
    if profile == "bilingual":
        for split in SPLITS:
            english = [("en", row) for row in _load_split(bilingual, split, cache)]
            rows[split] = [item for pair in zip_longest(rows[split], english) for item in pair
                           if item is not None]

    by_split = {split: [row for _, row in pairs] for split, pairs in rows.items()}
    all_rows = by_split["train"] + by_split["validation"]
    if len({row.id for row in all_rows}) != len(all_rows):
        raise ValueError("training and validation must use unique IDs across sources")
    for attr, label in (("group_id", "groups"), ("pixel_sha256", "images")):
        if {getattr(row, attr) for row in by_split["train"]} & {
            getattr(row, attr) for row in by_split["validation"]
        }:
            raise ValueError(f"train and validation share {label}")
    for split, entries in by_split.items():
        if len({(row.pixel_sha256, row.prompt.strip()) for row in entries}) != len(entries):
            raise ValueError(f"{split} contains duplicate image and prompt rows")

    formal_groups, formal_pixels, formal_count = _eval_index(project)
    if any(row.group_id in formal_groups or row.pixel_sha256 in formal_pixels for row in all_rows):
        raise ValueError("a training or validation row overlaps the formal benchmark")
    protections = {"formal": {"groups": len(formal_groups), "images": len(formal_pixels),
                              "image_references": formal_count}}
    if locked is not None:
        ids, groups, pixels, count = locked
        if any(row.id in ids or row.group_id in groups or row.pixel_sha256 in pixels for row in all_rows):
            raise ValueError("a training or validation row overlaps the English held-out split")
        protections["english_test"] = {"rows": count, "groups": len(groups), "images": len(pixels)}

    split_counts = {
        split: {"rows": len(entries), "by_language": dict(Counter(lang for lang, _ in rows[split])),
                "groups": len({row.group_id for row in entries}),
                "images": len({row.pixel_sha256 for row in entries})}
        for split, entries in by_split.items()
    }
    # Confirm the same manifests were read to prevent a mixed-source preflight.
    roots = {"chinese": chinese, "bilingual": bilingual, "formal": project / "data/study"}
    for source, expected in sources.items():
        if _file_hashes(roots[source], tuple(expected)) != expected:
            raise ValueError(f"{source} manifests changed during preflight")
    return by_split, {"manifest_version": 1, "profile": profile, "sources": sources,
                      "splits": split_counts, "protected": protections}


def preflight(project: Path, *, profile: str = "bilingual") -> dict:
    _, manifest = _assemble(project, profile)
    return manifest


def export_dataset(project: Path, output: Path, *, profile: str = "bilingual") -> tuple[dict, str]:
    splits, manifest = _assemble(project, profile)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new export directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memevlm-gpu-", dir=output.parent) as tmp:
        staged = Path(tmp) / "dataset"
        (staged / "images").mkdir(parents=True)
        artifacts = {"dataset_info.json": REGISTRATION}
        hashes, images = {}, {}
        for split, rows in splits.items():
            conversations = []
            for row in rows:
                if row.image_sha256 not in images:
                    relative = f"images/{row.image_sha256}{row.image_path.suffix.lower()}"
                    if not IMAGE_NAME.fullmatch(relative):
                        raise ValueError(f"unsupported source image suffix: {row.image_path.suffix}")
                    raw = row.image_path.read_bytes()
                    if sha256(raw) != row.image_sha256:
                        raise ValueError("a source image changed during export")
                    (staged / relative).write_bytes(raw)
                    images[row.image_sha256] = relative
                    hashes[relative] = row.image_sha256
                example = row.sharegpt()
                example["images"] = [images[row.image_sha256]]
                conversations.append(example)
            artifacts[f"{split}.json"] = conversations
        for name, value in artifacts.items():
            raw = json_bytes(value)
            (staged / name).write_bytes(raw)
            hashes[name] = sha256(raw)
        manifest["files"] = hashes
        (staged / MANIFEST).write_bytes(json_bytes(manifest))
        verified = verify_dataset(staged)
        staged.rename(output)
    return verified


def verify_dataset(directory: Path | str) -> tuple[dict, str]:
    """Return the portable manifest and exact manifest digest after checking its export."""
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / MANIFEST
    if manifest_path.is_symlink() or (directory / "images").is_symlink():
        raise ValueError("dataset manifest and image directory must be local files")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("manifest_version") != 1 or manifest.get("profile") not in ("chinese", "bilingual"):
        raise ValueError("unsupported dataset manifest")
    hashes = manifest["files"]
    if not TABLES.issubset(hashes):
        raise ValueError("dataset manifest lacks required ShareGPT files")
    for name, digest in hashes.items():
        match = IMAGE_NAME.fullmatch(name)
        if name not in TABLES and (match is None or match[1] != digest):
            raise ValueError("unrecognized file in dataset manifest")
        path = directory / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"export file checksum mismatch: {name}")
    if json.loads((directory / "dataset_info.json").read_bytes()) != REGISTRATION:
        raise ValueError("unexpected LlamaFactory dataset registration")
    image_sets = {}
    for split in SPLITS:
        rows = json.loads((directory / f"{split}.json").read_bytes())
        if not isinstance(rows, list) or not rows or len(rows) != manifest["splits"][split]["rows"]:
            raise ValueError(f"unexpected {split} row count")
        image_sets[split] = set()
        for row in rows:
            messages, images = row["messages"], row["images"]
            if (len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant"
                    or not messages[0]["content"].startswith("<image>\n")
                    or len(images) != 1 or images[0] in TABLES or images[0] not in hashes):
                raise ValueError(f"invalid ShareGPT example in {split}")
            image_sets[split].add(images[0])
    if image_sets["train"] & image_sets["validation"] or set.union(*image_sets.values()) != set(hashes) - TABLES:
        raise ValueError("export images must match disjoint train and validation references")
    expected = set(hashes) | {MANIFEST, "images"}
    if {path.relative_to(directory).as_posix() for path in directory.rglob("*")} != expected:
        raise ValueError("unexpected untracked file in export")
    return manifest, sha256(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "export"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, default=PROJECT_ROOT)
        command.add_argument("--profile", choices=("chinese", "bilingual"), default="bilingual")
    commands.choices["export"].add_argument("--output", type=Path, required=True)
    commands.add_parser("verify").add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight(args.project, profile=args.profile)
        else:
            manifest, digest = (verify_dataset(args.dataset) if args.command == "verify"
                                else export_dataset(args.project, args.output, profile=args.profile))
            result = {"dataset_sha256": digest, **manifest}
        print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
