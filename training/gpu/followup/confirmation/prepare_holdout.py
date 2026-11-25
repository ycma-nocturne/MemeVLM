"""Prepare a MemeQA confirmation set from previously unused official development images."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent
sys.path.insert(0, str(FOLLOWUP))
import prepare_data as data


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bound_visual(root, name, expected):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"image outside its asset directory: {name}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected image checksum: {name}")
    return data.describe(raw)[1]


def prior_images(prior):
    receipt_path = prior / "selection-receipt.json"
    receipt = read_json(receipt_path)
    dev = prior / "native-dev"
    manifest_path = dev / "manifest.json"
    manifest = read_json(manifest_path)
    if data.digest(manifest_path) != receipt["native_dev_manifest_sha256"]:
        raise ValueError("prior dev manifest differs from its selection receipt")
    prior_input = dev / "inputs.jsonl"
    if data.digest(prior_input) != manifest["inputs_sha256"]:
        raise ValueError("prior dev inputs differ from the manifest")
    rows = read_rows(prior_input)
    official_names = {row["source_image"] for row in rows}
    inventory = {row["source_image"]: row for row in receipt["native_dev"]}
    if len(official_names) != len(rows) or len(rows) != 96 or official_names != inventory.keys():
        raise ValueError("unexpected previous development image census")
    visuals = []
    for row in rows:
        name, image = row["source_image"], row["image"]
        visual = bound_visual(dev, image, row["image_sha256"])
        if row["group_id"] != f"rgb-{visual.rgb}" or visual.rgb != inventory[name]["decoded_rgb_sha256"]:
            raise ValueError(f"prior dev visual differs for {name}")
        visuals.append((f"prior-dev/{name}", visual))
    return official_names, visuals, {
        "data_selection_sha256": data.digest(receipt_path),
        "prior_dev_manifest_sha256": data.digest(manifest_path),
        "prior_dev_inputs_sha256": manifest["inputs_sha256"],
        "baseline_dataset_sha256": receipt["baseline_dataset_sha256"],
        "real_mix_dataset_sha256": receipt["real_mix_dataset_sha256"],
    }


def training_visuals(prior, binding):
    checked = {}
    split_counts = {}
    for dataset, digest_key in (("baseline", "baseline_dataset_sha256"), ("real-mix", "real_mix_dataset_sha256")):
        root = prior / dataset
        manifest_path = root / "dataset-manifest.json"
        manifest = read_json(manifest_path)
        if data.digest(manifest_path) != binding[digest_key]:
            raise ValueError(f"prior data changed: {dataset}")
        split_counts[dataset] = {}
        for split in ("train", "validation"):
            table = root / f"{split}.json"
            if data.digest(table) != manifest["files"][table.name]:
                raise ValueError(f"prior training table changed: {dataset}/{split}")
            names = {row["images"][0] for row in read_json(table)}
            split_counts[dataset][split] = len(names)
            for name in sorted(names):
                key = manifest["files"][name]
                if key not in checked:
                    checked[key] = (f"{dataset}/{split}/{name}", bound_visual(root, name, key))
    return list(checked.values()), split_counts


def confirmation_cases(dev, testing, prior, archives, project):
    dev_rows = data.bound_json(dev, data.ANNOTATION_DIGESTS["dev"])
    test_rows = data.bound_json(testing, data.ANNOTATION_DIGESTS["test"])
    all_dev = {row["img"] for row in dev_rows}
    all_test = {row["img"] for row in test_rows}
    if (len(all_dev), len(all_test)) != (190, 190) or all_dev & all_test:
        raise ValueError("unexpected official split census or overlap")
    used, prior_visuals, binding = prior_images(prior)
    if not used.issubset(all_dev):
        raise ValueError("previous development includes images outside official dev")
    protected = training_visuals(prior, binding)
    train_visuals, train_counts = protected
    archive_images = data.load_assets(archives, all_dev | all_test)
    test_visuals = [(f"official-test/{name}", archive_images[name].visual) for name in sorted(all_test)]
    formal = data.formal_assets(project)
    formal_visuals = [(f"MemeBench/{name}", visual) for name, visual in formal]
    pool, question_exclusions = data.eligible(dev_rows)
    if not used.issubset(pool):
        raise ValueError("unexpected prior dev change in the eligible question pool")
    unused = {name: rows for name, rows in pool.items() if name not in used}
    external = prior_visuals + test_visuals + formal_visuals + train_visuals
    screened, rejected = data.screen(unused, external, archive_images)
    kept, same_cohort_rejected = [], []
    visual_index = []
    for name in sorted(screened, key=lambda k: data.rank("confirmation-image", k)):
        asset = archive_images[name]
        match = data.blocked(asset, visual_index)
        if match:
            same_cohort_rejected.append(match)
        else:
            kept.extend((row, asset) for row in screened[name])
            visual_index.append((name, asset.visual))
    kept.sort(key=lambda item: data.rank("confirmation-question", item[0]["id"]))
    inputs, gold = data.rendered_cases(kept, "confirm")
    if not inputs or len({x["id"] for x in inputs}) != len(inputs):
        raise ValueError("confirmation needs unique question IDs")
    for case, key, (original, asset) in zip(inputs, gold, kept, strict=True):
        if (case["question"] != original["question"]
                or sorted(case["options"]) != sorted(original["options"])
                or case["options"][data.LABELS.index(key["answer"])] != original["options"][original["answer_key"]]
                or asset.name != original["img"] or asset.name not in all_dev - used):
            raise ValueError(f"confirmation answer or split differs: {case['id']}")
    census = {
        "official_dev": {"questions": len(dev_rows), "images": len(all_dev)},
        "eligible_dev": {"questions": sum(map(len, pool.values())), "images": len(pool)},
        "unused_candidates": {"questions": sum(map(len, unused.values())), "images": len(unused)},
        "question_exclusions": question_exclusions,
        "excluded_prior_dev": 96,
        "checked_prior_training_and_validation": train_counts,
        "checked_prior_training_and_validation_unique_encoded_images": len(train_visuals),
        "checked_official_memeqa_test_images": len(test_visuals),
        "checked_memebench_image_paths": len(formal_visuals),
        "excluded_overlap": rejected,
        "excluded_confirmation_near_duplicates": same_cohort_rejected,
    }
    return inputs, gold, {asset.name: asset for _, asset in kept}, binding, census


def build(args):
    rows, gold, assets, binding, census = confirmation_cases(
        args.dev, args.memeqa_test, args.prior, args.archives, args.project,
    )
    out = args.output.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memeqa-confirmation-", dir=out.parent) as scratch:
        stage = Path(scratch) / "prepared"
        (stage / "images").mkdir(parents=True)
        image_files = {}
        for asset in assets.values():
            name = f"images/{asset.digest}{asset.suffix}"
            (stage / name).write_bytes(asset.data)
            image_files[name] = data.digest(stage / name)
        if len(image_files) != len(assets) or any(
            image_files.get(row["image"]) != row["image_sha256"] for row in rows
        ):
            raise ValueError("confirmation image export differs from source")
        data.jsonl(stage / "inputs.jsonl", rows)
        data.jsonl(stage / "gold.jsonl", gold)
        manifest = {
            "protocol": "memeqa-none-minus-unused-dev-confirmation-v1",
            "source_split": "official-dev-unused-images",
            "n_questions": len(rows), "n_images": len(assets),
            "type_counts": dict(Counter(row["question_type"] for row in rows)),
            "answer_position_counts": dict(Counter(row["answer"] for row in gold)),
            "inputs_sha256": data.digest(stage / "inputs.jsonl"),
            "gold_sha256": data.digest(stage / "gold.jsonl"),
            "image_files": image_files,
            "source_annotations_sha256": data.ANNOTATION_DIGESTS["dev"],
            "source_test_annotations_sha256": data.ANNOTATION_DIGESTS["test"],
            "source_archives_sha256": data.ARCHIVE_DIGESTS,
            "memeqa_revision": data.MEMEQA_REVISION, "semeval_revision": data.SEMEVAL_REVISION,
            "prepare_sha256": data.digest(Path(__file__)),
            "shared_screening_sha256": data.digest(Path(data.__file__)),
            "previous_data": binding, "census": census,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.rename(out)
    keys = ("n_questions", "n_images", "type_counts", "answer_position_counts", "inputs_sha256", "gold_sha256", "census")
    print(json.dumps({**{k: manifest[k] for k in keys},
                      "manifest_sha256": data.digest(out / "manifest.json")}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=FOLLOWUP / "inputs/memeqa-dev.json")
    parser.add_argument("--prior", type=Path, default=FOLLOWUP / "prepared")
    parser.add_argument("--project", type=Path, default=data.WORKSPACE / "MemeVLM-team-training")
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--memeqa-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
