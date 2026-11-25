"""Build a paired 770-row LoRA comparison and a separate native MemeQA dev set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import file_digest, sha256
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
MEMEQA_REVISION = "e48b332a02f7b25e08459dd7dcf69f19b430c621"
SEMEVAL_REVISION = "796b32fa76b7a5814a48d1870d163058e81e4ce1"
ANNOTATION_DIGESTS = {
    "train": "1a760597a172ecf1fe45ac03b915dd59039071425e2bef8ad4c663449d504737",
    "dev": "954f1ff10230a606cc659c397a44153468025e39e9515b664b73215e814996c8",
    "test": "ca22300a959293a0ff32206855151499ab22e8ba1065800b33cfcef303ef2a2e",
}
ARCHIVE_DIGESTS = {
    "training": "402de437a0952fb1b6bdb578d8eb42b08cb78809de60a494abcd55d093a9dcf9",
    "dev": "83fc1fa124a2132bafadd78e6557bd2bd7bac3628a2d0bf276bed1becfb675e5",
    "test": "6200b07851798e99a06a3ab499dab24f711e06fd4ad7e4ba3f41d3d59b69fe88",
}
FIXED_BASELINE = "8c8973952c8eb68367d56673c97ecaf5c1a76e96e0ac1cf094d96e365d8ad2a3"
TRAIN_TARGET = 256
DEV_TARGET = 96
SEED = 6000
LABELS = "ABCD"
PROMPT = (
    "Answer the multiple-choice question about this meme. Select one option. "
    'Return one JSON object with exactly one key "answer" and a value of "A", "B", "C", or "D".'
)
POSITION_DEPENDENT = re.compile(
    r'\b(?:none|all)\s+of\s+(?:the\s+above|these|the\s+following)\b'
    r'|\b(?:both\s+[A-D]\s+(?:and\s+)?[A-D]|(?:answer|option|choice|letter)s?\s+[A-D])\b'
    r'|\b[A-D]\s*(?:and|/|&)\s*[A-D]\b', re.IGNORECASE,
)
DCT = np.cos(np.pi * np.outer(np.arange(8), np.arange(32) + 0.5) / 32)


@dataclass(frozen=True)
class Visual:
    rgb: str
    fingerprints: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Asset:
    name: str
    source_split: str
    member: str
    data: bytes
    suffix: str
    visual: Visual

    @property
    def digest(self):
        return sha256(self.data).hexdigest()


def rank(*values):
    return sha256(json.dumps([SEED, *values], ensure_ascii=False).encode()).hexdigest()


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def bound_json(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected annotation digest: {path.name}")
    rows = json.loads(raw)
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError(f"duplicate official question ID: {path.name}")
    return rows


def bits(values):
    return int.from_bytes(np.packbits(values).tobytes(), "big")


def perceptual(image):
    pixels = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = (DCT @ pixels @ DCT.T).ravel()
    threshold = np.median(coefficients[1:])
    gradients = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS))
    return bits(coefficients > threshold), bits(gradients[:, 1:] > gradients[:, :-1])


def describe(data):
    with Image.open(BytesIO(data)) as source:
        suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(source.format)
        if suffix is None or getattr(source, "n_frames", 1) != 1:
            raise ValueError("expected one still PNG, JPEG or WebP")
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        content = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
        gray = rgb.convert("L")
        w, h = gray.size
        areas = (gray, ImageOps.mirror(gray), gray.crop((w // 10, h // 10, 9 * w // 10, 9 * h // 10)),
                 gray.crop((0, 0, w, 3 * h // 4)), gray.crop((0, h // 4, w, h)))
        return suffix, Visual(sha256(content).hexdigest(), tuple(map(perceptual, areas)))


def similarity(candidate, protected):
    if candidate.rgb == protected.rgb:
        return "exact_rgb"
    a, b = candidate.fingerprints, protected.fingerprints
    for i, j in ((0, 0), (1, 0), (0, 2), (0, 3), (0, 4), (2, 0), (3, 0), (4, 0), (2, 2)):
        ph = (a[i][0] ^ b[j][0]).bit_count()
        dh = (a[i][1] ^ b[j][1]).bit_count()
        if (i <= 1 and j == 0 and ph <= 7 and dh <= 12) or (ph <= 4 and dh <= 8):
            return "perceptual"
    return ""


def blocked(asset, index):
    for source_id, protected in index:
        match = similarity(asset.visual, protected)
        if match:
            return {"image": asset.name, "matched_image": source_id, "reason": match}
    return None


def load_assets(directory, names):
    with ExitStack() as stack:
        index = {}
        for split, expected in ARCHIVE_DIGESTS.items():
            path = directory / f"semeval-{split}.zip"
            if digest(path) != expected:
                raise ValueError(f"unexpected SemEval archive digest: {path.name}")
            archive = stack.enter_context(ZipFile(path))
            for member in archive.namelist():
                if Path(member).suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                name = Path(member).name
                if name in index:
                    raise ValueError(f"duplicate SemEval image filename: {name}")
                index[name] = (archive, member, split)
        assets = {}
        for name in sorted(names):
            archive, member, split = index[name]
            data = archive.read(member)
            suffix, visual = describe(data)
            assets[name] = Asset(name, split, member, data, suffix, visual)
    return assets


def formal_assets(project):
    study = project / "data/study"
    originals = json.loads((study / "selection.json").read_bytes())
    pairs = json.loads((study / "pairs.json").read_bytes())
    rows = originals + [variant for family in pairs for variant in family["variants"].values()]
    results = []
    for row in rows:
        name = row["image_path"]
        raw = (project / "data/assets" / name).read_bytes()
        if sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError(f"formal image changed: {name}")
        results.append((name, describe(raw)[1]))
    if len(rows) != 105 or len({v.rgb for _, v in results}) != 100:
        raise ValueError("formal data no longer matches the protected census")
    return results


def eligible(rows):
    by_image = defaultdict(list)
    excluded = Counter()
    for row in rows:
        options, key = row["options"], row["answer_key"]
        if row["general_type"] != "single":
            excluded["multiple_answers"] += 1
        elif len(options) != 4:
            excluded["different_option_count"] += 1
        elif any(not isinstance(text, str) or not text.strip() or "<image>" in text
                 for text in [row["question"], *options]):
            excluded["unusable_question_or_option_text"] += 1
        elif len({option.strip().casefold() for option in options}) != 4:
            excluded["duplicate_choice_text"] += 1
        elif any(POSITION_DEPENDENT.search(text) for text in [row["question"], *options]):
            excluded["position_dependent"] += 1
        elif type(key) is not int or key not in range(4):
            raise ValueError(f"unusable official question: {row['id']}")
        else:
            by_image[row["img"]].append(row)
    return by_image, dict(excluded)


def screen(pool, protected, assets):
    kept, removed = {}, []
    for name in sorted(pool, key=lambda n: rank("visual", n)):
        match = blocked(assets[name], protected)
        if match:
            removed.append(match)
        else:
            kept[name] = pool[name]
    return kept, removed


def select_cases(pool, assets, split, count):
    names = sorted(pool, key=lambda name: rank(split, "image", name))
    chosen, removed, approved = [], [], []
    type_counts = Counter()
    for name in names:
        asset = assets[name]
        match = blocked(asset, approved)
        if match:
            removed.append(match)
            continue
        choices = pool[name]
        row = min(choices, key=lambda r: (type_counts[r["specific_type"]], rank(split, r["id"])))
        type_counts[row["specific_type"]] += 1
        approved.append((name, asset.visual))
        chosen.append((row, asset))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"{split}: expected {count} distinct images, found {len(chosen)}")
    return chosen, removed


def rendered_cases(chosen, split):
    rows, gold = [], []
    for i, (source, asset) in enumerate(chosen):
        position = i % 4
        correct = source["answer_key"]
        order = sorted(set(range(4)) - {correct}, key=lambda n: rank(split, source["id"], n))
        order.insert(position, correct)
        case_id = f"mqa-{split}-{i + 1:04d}"
        rows.append({
            "id": case_id, "group_id": f"rgb-{asset.visual.rgb}",
            "image": f"images/{asset.digest}{asset.suffix}", "image_sha256": asset.digest,
            "options": [source["options"][j] for j in order],
            "pilot_case_id": None, "pilot_group": False, "pilot_question": False,
            "question": source["question"], "question_type": source["specific_type"],
            "source_image": source["img"], "source_question_id": source["id"],
        })
        gold.append({"id": case_id, "answer": LABELS[position]})
    return rows, gold


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def replacement_ids(project):
    groups = defaultdict(list)
    path = project / "training/bilingual/pairs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["split"] == "train":
            groups[item["mechanism"]].append(item["group_id"])
    if len(groups) != 8 or {len(ids) for ids in groups.values()} != {21}:
        raise ValueError("unexpected English training family strata")
    return {mechanism: sorted(ids, key=lambda name: rank("replace", name))[:8]
            for mechanism, ids in sorted(groups.items())}


def create_mix(data, project, stage, base_manifest, source_train, candidate_rows, selected):
    base = stage / "baseline"
    mixed = stage / "real-mix"
    (mixed / "images").mkdir(parents=True)
    baserows = json.loads((base / "train.json").read_bytes())
    replaced = replacement_ids(project)
    groups = {name for names in replaced.values() for name in names}
    replacing = Counter(example.group_id for example in source_train if example.group_id in groups)
    if len(replacing) != 64 or set(replacing.values()) != {4}:
        raise ValueError("replacements must contain all four rows per family")
    native_inputs, native_gold = candidate_rows
    native_examples = []
    for case, label, (_, asset) in zip(native_inputs, native_gold, selected, strict=True):
        choices = "\n".join(f"{letter}. {option}" for letter, option in zip(LABELS, case["options"], strict=True))
        native_examples.append({
            "messages": [{"role": "user", "content": f'<image>\n{PROMPT}\n\n{case["question"]}\n{choices}'},
                         {"role": "assistant", "content": json.dumps({"answer": label["answer"]})}],
            "images": [case["image"]],
        })
        target = mixed / case["image"]
        with target.open("xb") as out:
            out.write(asset.data)
    native_iterator = iter(native_examples)
    rows = [next(native_iterator) if src.group_id in groups else row
            for src, row in zip(source_train, baserows, strict=True)]
    if next(native_iterator, None) is not None or len(rows) != 770 or len(set(map(data.json_bytes, rows))) != 770:
        raise ValueError("expected 770 distinct mixed training rows")
    for name in ("dataset_info.json", "validation.json"):
        shutil.copyfile(base / name, mixed / name)
    (mixed / "train.json").write_bytes(data.json_bytes(rows))
    validation = json.loads((base / "validation.json").read_bytes())
    images = {row["images"][0] for row in rows + validation}
    for name in images:
        if not (mixed / name).exists():
            shutil.copyfile(base / name, mixed / name)
    # Keep verification wired to the standard project dataset contract.
    manifest = {**base_manifest, "files": {**{name: digest(mixed / name) for name in images},
                                           **{name: digest(mixed / name) for name in data.TABLES}}}
    manifest["sources"] = {**manifest["sources"], "memeqa_none_minus_train": ANNOTATION_DIGESTS["train"]}
    manifest["splits"] = {**manifest["splits"], "train": {
        "by_language": {"en": 672, "zh": 98}, "groups": 385, "images": 666, "rows": 770,
        "by_source": {"chinese": 98, "english_synthetic": 416, "memeqa_real": 256},
    }}
    if len(images) != 819:
        raise ValueError("unexpected image composition in mixed export")
    (mixed / data.MANIFEST).write_bytes(data.json_bytes(manifest))
    verified, output_hash = data.verify_dataset(mixed)
    if verified["files"]["validation.json"] != base_manifest["files"]["validation.json"]:
        raise ValueError("control and real mix must have byte-identical validation")
    return output_hash, replaced


def inventory(selections):
    return [{"source_question_id": row["id"], "source_image": image.name,
             "source_archive_split": image.source_split, "source_member": image.member,
             "image_sha256": image.digest, "decoded_rgb_sha256": image.visual.rgb,
             "question_type": row["specific_type"]} for row, image in selections]


def assemble(args):
    sys.path.insert(0, str(args.project.resolve()))
    from training.gpu import prepare_data as data

    annotation_paths = {split: args.annotations / f"memeqa-{split}.json" for split in ("train", "dev")}
    annotation_paths["test"] = args.memeqa_test
    rows = {split: bound_json(path, ANNOTATION_DIGESTS[split]) for split, path in annotation_paths.items()}
    images = {split: {row["img"] for row in table} for split, table in rows.items()}
    if tuple(map(len, images.values())) != (570, 190, 190) or any(
        a & b for a, b in combinations(images.values(), 2)
    ):
        raise ValueError("official image splits differ or overlap")
    assets = load_assets(args.archives, set.union(*images.values()))
    protected = [(f"MemeQA-test/{name}", assets[name].visual) for name in sorted(images["test"])]
    protected += [(f"MemeBench/{name}", visual) for name, visual in formal_assets(args.project)]
    official, excluded = {}, {}
    for split in ("train", "dev"):
        official[split], excluded[split] = eligible(rows[split])
    devpool, dev_blocked = screen(official["dev"], protected, assets)
    dev, dev_near = select_cases(devpool, assets, "dev", DEV_TARGET)
    # Exclude the whole upstream dev image split from real training, including unused images.
    train_protected = protected + [(f"MemeQA-dev/{name}", assets[name].visual) for name in sorted(images["dev"])]
    trainpool, train_blocked = screen(official["train"], train_protected, assets)
    train, train_near = select_cases(trainpool, assets, "train", TRAIN_TARGET)
    native_train = rendered_cases(train, "train")
    native_dev = rendered_cases(dev, "dev")
    out = args.output.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memeqa-real-", dir=out.parent) as scratch:
        stage = Path(scratch) / "prepared"
        stage.mkdir()
        control, control_hash = data.export_dataset(args.project, stage / "baseline", profile="bilingual")
        if control_hash != FIXED_BASELINE:
            raise ValueError("synthetic control differs from the original bilingual training receipt")
        source_splits, _ = data._assemble(args.project, "bilingual")
        mix_hash, replaced = create_mix(data, args.project, stage, control, source_splits["train"], native_train, train)
        dev_dir = stage / "native-dev"
        (dev_dir / "images").mkdir(parents=True)
        for case, (_, asset) in zip(native_dev[0], dev, strict=True):
            with (dev_dir / case["image"]).open("xb") as file:
                file.write(asset.data)
        jsonl(dev_dir / "inputs.jsonl", native_dev[0])
        jsonl(dev_dir / "gold.jsonl", native_dev[1])
        manifest = {
            "protocol": "memeqa-none-minus-real-followup-dev-v1", "source_split": "dev",
            "n_questions": DEV_TARGET, "n_images": len(dev),
            "type_counts": dict(Counter(row["question_type"] for row in native_dev[0])),
            "answer_position_counts": dict(Counter(row["answer"] for row in native_dev[1])),
            "inputs_sha256": digest(dev_dir / "inputs.jsonl"), "gold_sha256": digest(dev_dir / "gold.jsonl"),
            "image_files": {row["image"]: row["image_sha256"] for row in native_dev[0]},
            "source_annotations_sha256": ANNOTATION_DIGESTS["dev"],
        }
        (dev_dir / "manifest.json").write_bytes(data.json_bytes(manifest))
        receipt = {
            "protocol": "memeqa-real-followup-v1", "seed": SEED,
            "prepare_sha256": digest(Path(__file__)),
            "baseline_dataset_sha256": control_hash, "real_mix_dataset_sha256": mix_hash,
            "shared_validation_sha256": control["files"]["validation.json"],
            "sources": {"memeqa_revision": MEMEQA_REVISION, "annotation_sha256": ANNOTATION_DIGESTS,
                        "semeval_revision": SEMEVAL_REVISION, "archive_sha256": ARCHIVE_DIGESTS},
            "official_splits": {s: {"questions": len(rows[s]), "images": len(images[s])} for s in rows},
            "pool_before_image_filter": {s: {"images": len(v), "questions": sum(map(len, v.values())),
                                            "type_counts": dict(Counter(row["specific_type"] for rr in v.values() for row in rr))}
                                         for s, v in official.items()},
            "question_exclusions": excluded,
            "protected_memeqa_test_images": len(images["test"]), "protected_memebench_image_references": 105,
            "perceptual_filter": {"full_phash_max": 7, "full_dhash_max": 12, "crop_phash_max": 4, "crop_dhash_max": 8,
                                  "stages": ["EXIF-corrected RGB SHA-256", "64-bit low-frequency DCT + gradient hash",
                                             "full, mirror, 80% center and overlapping vertical crops"]},
            "blocked_images": {"train_protected": train_blocked, "dev_protected": dev_blocked,
                               "train_near_duplicate": train_near, "dev_near_duplicate": dev_near},
            "replaced_english_training_families_by_mechanism": replaced,
            "real_train": inventory(train), "native_dev": inventory(dev),
            "real_train_question_types": dict(Counter(row["specific_type"] for row, _ in train)),
            "native_dev_manifest_sha256": digest(dev_dir / "manifest.json"),
        }
        (stage / "selection-receipt.json").write_bytes(data.json_bytes(receipt))
        jsonl(stage / "real-train-questions.jsonl", native_train[0])
        jsonl(stage / "real-train-gold.jsonl", native_train[1])
        stage.rename(out)
    return {key: receipt[key] for key in ("baseline_dataset_sha256", "real_mix_dataset_sha256",
                                           "shared_validation_sha256", "real_train_question_types",
                                           "native_dev_manifest_sha256")} | {"dev": manifest["type_counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=WORKSPACE / "MemeVLM-team-training")
    parser.add_argument("--annotations", type=Path, default=HERE / "inputs")
    parser.add_argument("--memeqa-test", type=Path, required=True, help="Pinned source test JSON used for image protection")
    parser.add_argument("--archives", type=Path, required=True, help="Directory of pinned SemEval ZIPs")
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    print(json.dumps(assemble(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
