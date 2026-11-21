"""Validate training splits and export single-image SFT data for LlamaFactory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile

from PIL import Image, ImageOps


SPLITS = ("train", "validation")
FIELDS = ("id", "group_id", "image", "prompt", "answer")
MEDIA_TAGS = ("<image>", "<video>", "<audio>")


@dataclass(frozen=True)
class Example:
    id: str
    group_id: str
    image: str
    prompt: str
    answer: str
    image_path: Path
    image_sha256: str
    pixel_sha256: str

    def load_image(self) -> Image.Image:
        """Return an RGB image with its display orientation applied."""
        with Image.open(self.image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    def qwen_messages(self) -> list[dict]:
        """Return messages for a local Qwen3-VL processor."""
        return [
            {"role": "user", "content": [
                {"type": "image", "image": str(self.image_path)},
                {"type": "text", "text": self.prompt},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": self.answer}]},
        ]

    def sharegpt(self) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"<image>\n{self.prompt}"},
                {"role": "assistant", "content": self.answer},
            ],
            "images": [self.image],
        }


def _image_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or any(char in name for char in "\\:\0"):
        raise ValueError("image must use a relative POSIX path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError(f"invalid relative image path: {name!r}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"image must be a file under its data root: {name}")
    return path


def _check_image(root: Path, name: str, expected: str | None, cache: dict) -> tuple[Path, str, str]:
    path = _image_path(root, name)
    if path not in cache:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "is_animated", False):
                raise ValueError(f"expected one still image: {name}")
            rgb = ImageOps.exif_transpose(source).convert("RGB")
        image_hash = hashlib.sha256(raw).hexdigest()
        shape = f"{rgb.width}x{rgb.height}:".encode("ascii")
        pixel_hash = hashlib.sha256(shape + rgb.tobytes()).hexdigest()
        cache[path] = (image_hash, pixel_hash)

    image_hash, pixel_hash = cache[path]
    if expected is not None and expected != image_hash:
        raise ValueError(f"image checksum mismatch: {name}")
    return path, image_hash, pixel_hash


def _load_split(root: Path, split: str, cache: dict) -> list[Example]:
    if split not in SPLITS:
        raise ValueError("split must be train or validation")
    file = root / f"{split}.jsonl"
    examples, ids, questions = [], set(), set()
    for line_no, line in enumerate(file.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        context = f"{file.name}:{line_no}"
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(field), str) or not row[field].strip() for field in FIELDS
            ):
                raise ValueError(f"expected nonempty text fields {', '.join(FIELDS)}")
            if row["id"] in ids:
                raise ValueError(f"duplicate ID: {row['id']}")
            if any(tag in row[field] for field in ("prompt", "answer") for tag in MEDIA_TAGS):
                raise ValueError("prompt and answer must leave media tags to the exporter")
            expected = row.get("image_sha256")
            if "image_sha256" in row and (
                not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
            ):
                raise ValueError("image_sha256 must contain 64 lowercase hex characters")
            path, image_hash, pixel_hash = _check_image(root, row["image"], expected, cache)
            question = pixel_hash, row["prompt"].strip()
            if question in questions:
                raise ValueError("duplicate image and prompt")
            ids.add(row["id"])
            questions.add(question)
            examples.append(Example(**{field: row[field] for field in FIELDS},
                                    image_path=path, image_sha256=image_hash, pixel_sha256=pixel_hash))
        except (OSError, ValueError) as err:
            raise ValueError(f"{context}: {err}") from err

    if not examples:
        raise ValueError(f"{file.name}: expected at least one example")
    return examples


def load_dataset(root: Path | str, split: str) -> list[Example]:
    """Load and verify one split under the training directory."""
    return _load_split(Path(root).resolve(strict=True), split, {})


def _eval_index(eval_root: Path) -> tuple[set[str], set[str], int]:
    study = eval_root / "data/study"
    assets = (eval_root / "data/assets").resolve(strict=True)
    selections = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    pairs = json.loads((study / "pairs.json").read_text(encoding="utf-8"))
    groups = {row["template_group"] for row in selections + pairs if "template_group" in row}
    images = selections + [variant for row in pairs for variant in row["variants"].values()]
    if not images:
        raise ValueError("evaluation manifests contain no images")
    cache = {}
    pixels = {
        _check_image(assets, row["image_path"], row.get("image_sha256"), cache)[2]
        for row in images
    }
    return groups, pixels, len(images)


def _checked_splits(root: Path | str, eval_root: Path | str | None) -> tuple[dict, dict]:
    root = Path(root).resolve(strict=True)
    cache = {}
    splits = {split: _load_split(root, split, cache) for split in SPLITS}
    train, validation = (splits[split] for split in SPLITS)
    for attr, label in (("id", "ID"), ("group_id", "group"), ("pixel_sha256", "image")):
        overlap = {getattr(row, attr) for row in train} & {getattr(row, attr) for row in validation}
        if overlap:
            raise ValueError(f"train and validation share {len(overlap)} {label} value(s)")

    report = {
        split: {
            "rows": len(rows),
            "groups": len({row.group_id for row in rows}),
            "images": len({row.pixel_sha256 for row in rows}),
        }
        for split, rows in splits.items()
    }
    if eval_root is not None:
        groups, pixels, image_references = _eval_index(Path(eval_root).resolve(strict=True))
        if any(row.group_id in groups or row.pixel_sha256 in pixels for row in train + validation):
            raise ValueError("a training/validation group or image overlaps evaluation")
        report["evaluation"] = {
            "image_references": image_references,
            "unique_images": len(pixels),
            "groups": len(groups),
        }
    return splits, report


def validate_dataset(root: Path | str, *, eval_root: Path | str | None = None) -> dict:
    """Check both training splits and, when supplied, their evaluation overlap."""
    return _checked_splits(root, eval_root)[1]


def export_llamafactory(root: Path | str, output: Path | str, *,
                        eval_root: Path | str | None = None) -> dict:
    """Write a ShareGPT export using image paths relative to the training root."""
    splits, report = _checked_splits(root, eval_root)
    output = Path(output).absolute()
    dataset_info = {
        f"memevlm_{split}": {
            "file_name": f"{split}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
        for split in SPLITS
    }
    artifacts = {f"{split}.json": [row.sharegpt() for row in rows] for split, rows in splits.items()}
    artifacts["dataset_info.json"] = dataset_info
    encoded = {name: json.dumps(data, ensure_ascii=False, indent=2) + "\n"
               for name, data in artifacts.items()}
    if output.exists():
        if output.is_dir() and all((output / name).is_file() and
                                   (output / name).read_text(encoding="utf-8") == text
                                   for name, text in encoded.items()):
            return report
        raise FileExistsError(f"export differs from existing path: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memevlm-sft-", dir=output.parent) as tmp:
        staged = Path(tmp) / "data"
        staged.mkdir()
        for name, text in encoded.items():
            (staged / name).write_text(text, encoding="utf-8")
        staged.rename(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (commands.add_parser("validate"), commands.add_parser("export-llamafactory")):
        command.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                             help="directory containing train.jsonl and validation.jsonl")
        command.add_argument("--eval-root", type=Path, help="project root containing data/study and data/assets")
    commands.choices["export-llamafactory"].add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            report = validate_dataset(args.root, eval_root=args.eval_root)
        else:
            report = export_llamafactory(args.root, args.output or args.root / "llamafactory",
                                         eval_root=args.eval_root)
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
