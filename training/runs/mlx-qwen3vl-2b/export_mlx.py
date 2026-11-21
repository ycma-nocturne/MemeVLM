"""Export MemeVLM training examples for the MLX-VLM 0.8.0 SFT loader."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

from training.dataset import _checked_splits


def export(training_root, output, *, eval_root=None):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")

    splits, report = _checked_splits(training_root, eval_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mlx-export-", dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        (staged / "images").mkdir(parents=True)
        for split, examples in splits.items():
            with (staged / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as rows:
                for item in examples:
                    image = Path("images") / f"{item.image_sha256}{item.image_path.suffix.lower()}"
                    target = staged / image
                    if not target.exists():
                        raw = item.image_path.read_bytes()
                        if sha256(raw).hexdigest() != item.image_sha256:
                            raise ValueError(f"image changed during export: {item.image}")
                        target.write_bytes(raw)
                    row = {
                        "images": [image.as_posix()],
                        "messages": [
                            {"role": "user", "content": item.prompt},
                            {"role": "assistant", "content": item.answer},
                        ],
                    }
                    rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.rename(output)

    print(json.dumps({"output": str(output), **report}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path)
    args = parser.parse_args()
    try:
        export(args.training_root, args.output, eval_root=args.eval_root)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
