"""Summarize the two returned ratings and the frozen and replayed judge outputs."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


PACKETS = ("explanations", "choices", "materials")
RATERS = ("R1", "R2")
DIMENSIONS = ("V", "I", "K", "R")
COVERAGE = {"covered", "not_covered", "uncertain"}
YES_NO = {"yes", "no", "uncertain"}
FIT = {"supported", "partial", "unsupported", "uncertain"}
FIELDS = {
    "explanation": dict.fromkeys(("grounded_relation", "invented_claim"), YES_NO),
    "source_alignment": {
        "reference_fit": FIT,
        "hint_fit": FIT,
        "hint_leakage": YES_NO,
        "transcript_fit": {"accurate", "incomplete", "incorrect", "missing", "uncertain"},
    },
    "intervention": dict.fromkeys(
        ("target_cue_change", "target_relation_change", "sham_cue_preserved",
         "sham_relation_preserved", "unintended_change"), YES_NO
    ),
}
HEADER = ("reviewer", "phase", "batch", "case_id", "field", "item_id", "value", "evidence")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_forms(handoff, source, rating_suffix):
    manifest = read_json(handoff / "coordinator-only/packet-manifest.json")
    inputs, ratings, cases = {}, {}, {}
    for rating in manifest["packets"]:
        packet, rater = rating["kind"], rating["reviewer"]
        archive = handoff / rating["archive"]
        assert sha256(archive.read_bytes()).hexdigest() == rating["sha256"], archive
        with ZipFile(archive) as bundle:
            payload = bundle.read("packet-data.js")
        assert sha256(payload).hexdigest() == rating["data_sha256"], archive
        form = json.loads(payload.decode().split("=", 1)[1].strip().removesuffix(";"))
        assert (form["kind"], form["reviewer"]) == (packet, rater), archive
        expected = {}
        for phase in form["phases"]:
            for batch in phase["batches"]:
                for case in batch["cases"]:
                    cid = case["id"]
                    case_key = packet, cid
                    assert cases.setdefault(case_key, case) == case, case_key
                    prefix = phase["phase"], batch["id"], cid
                    fields = FIELDS.get(case["type"])
                    if case["type"] == "independent_choice":
                        fields = {
                            "cue": {"yes", "no", "insufficient"},
                            "relation": {option["label"] for option in case["options"]},
                        }
                    expected.update({(*prefix, f, ""): choices for f, choices in fields.items()})
                    for item in case.get("items", []):
                        expected[(*prefix, "check_item", item["id"])] = (
                            COVERAGE if packet == "explanations" else {"supported", "revise", "uncertain"}
                        )
                    expected[(*prefix, "notes", "")] = {""}

        path = source / f"{rater}-{packet}-{rating_suffix}"
        inputs[path.name] = sha256(path.read_bytes()).hexdigest()
        data = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            assert tuple(reader.fieldnames) == HEADER, path
            for row in reader:
                key = tuple(row[column] for column in HEADER[1:6])
                assert None not in row and all(value is not None for value in row.values()), key
                assert row["reviewer"] == rater and key in expected and key not in data, key
                assert row["value"] in expected[key], key
                data[key] = row["value"]
        assert data.keys() == expected.keys(), path
        ratings[packet, rater] = data
    assert set(ratings) == {(packet, rater) for packet in PACKETS for rater in RATERS}
    return inputs, ratings, cases


def paired_forms_agreement(ratings, choice_key):
    metrics = defaultdict(Counter)
    disagreements = Counter()
    for packet in PACKETS:
        raw_a, raw_b = (ratings[packet, rater] for rater in RATERS)
        assert raw_a.keys() == raw_b.keys(), packet
        for key, label_a in raw_a.items():
            phase, _, cid, field, _ = key
            if field == "notes":
                continue
            label_b = raw_b[key]
            if packet == "choices" and field == "relation":
                semantic = choice_key[cid]["option_semantics"]
                label_a, label_b = semantic[label_a], semantic[label_b]
            for split in (phase, "all"):
                counts = metrics[packet, split, field]
                counts["compared"] += 1
                counts["agree"] += label_a == label_b
            if label_a != label_b:
                disagreements[packet] += 1
    return {
        "agreement": [
            {"packet": packet, "phase": phase, "field": field, **counts}
            for (packet, phase, field), counts in sorted(metrics.items())
        ],
        "disagreements": dict(disagreements),
    }


def judge_agreement(ratings, explanation_key):
    selects = {
        "proposal_dev": lambda x: x["phase"] == "dev" and x["legacy_audit_id"],
        "proposal_test": lambda x: x["phase"] == "test" and x["legacy_audit_id"],
        "all_dev": lambda x: x["phase"] == "dev",
        "all_test": lambda x: x["phase"] == "test",
        "all_test_unmasked": lambda x: x["phase"] == "test" and not x["masked_spans"],
    }
    all_results = {}
    for scope, select in selects.items():
        selected = {cid: x for cid, x in explanation_key.items() if select(x)}
        valid = {}
        for label, key in (("frozen", "frozen_score"), ("replay", "replay_score")):
            valid[label] = {}
            for cid, x in selected.items():
                score = x[key]
                if score["status"] != "ok":
                    continue
                mapped = {item["item_id"]: item["verdict"] for item in score["items"]}
                assert len(mapped) == len(score["items"]), cid
                assert set(mapped) == set(x["checklist_item_ids"].values()), cid
                valid[label][cid] = {i: mapped[source] for i, source in x["checklist_item_ids"].items()}
        shared = valid["frozen"].keys() & valid["replay"].keys()
        identical = sum(valid["frozen"][cid] == valid["replay"][cid] for cid in shared)
        all_results[scope] = results = {
            "expected_answers": len(selected),
            "masked_answers": sum(bool(x["masked_spans"]) for x in selected.values()),
            "jointly_valid_answers": len(shared),
            "jointly_valid_identical_answers": identical,
        }
        for label, subset in (("frozen", valid["frozen"]), ("replay", valid["replay"]),
                              ("shared", {cid: valid["frozen"][cid] for cid in shared})):
            results[label] = {
                "valid_answers": len(subset),
                "compared_items": sum(map(len, subset.values())),
                "matching_items": {
                    rater: sum(
                        subset[cid][item] == value
                        for (_, _, cid, field, item), value in ratings["explanations", rater].items()
                        if field == "check_item" and cid in subset
                    )
                    for rater in RATERS
                },
            }
    return all_results


def solutions(reference_records, ratings, cases, explanation_key, material_key, qa_path=None):
    from meme_study.analysis import cluster_bootstrap

    records = {
        row["family_id"]: row
        for row in map(json.loads, reference_records.read_text(encoding="utf-8").splitlines())
    }
    test = {f for f, row in records.items() if row["split"] == "test"}
    ocr = {f for f in test if (records[f]["ocr_text"] or "").strip()}
    flags = {
        rater: {material_key[cid]["family_id"] for (_, _, cid, field, _), value in ratings["materials", rater].items()
                if field == "hint_leakage" and value == "yes"}
        for rater in RATERS
    }
    union, shared = flags["R1"] | flags["R2"], flags["R1"] & flags["R2"]
    assert (len(test), len(ocr), len(test & union)) == (40, 35, 13)
    samples = (
        ("RQ1", "ocr_available", ocr, "image_only", "ocr_only"),
        ("RQ2", "all", test, "hinted", "image_only"),
        ("RQ2", "exclude_either_direct_hint_flag", test - union, "hinted", "image_only"),
    )
    by_answer = {cid: {item["id"]: item["dimension"] for item in c["items"]}
                 for (p, cid), c in cases.items() if p == "explanations"}
    raw = defaultdict(dict)
    for rater in RATERS:
        for (phase, _, cid, field, item), value in ratings["explanations", rater].items():
            if phase == "test" and field == "check_item":
                raw[rater, cid][item] = value

    def contrasts(answers):
        pair = {}
        for (rater, cid), values in answers.items():
            detail = explanation_key[cid]
            model = detail["model"].rsplit("/", 1)[-1]
            assert set(values) == set(by_answer[cid]), cid
            for dim in DIMENSIONS:
                observations = [value for item, value in values.items() if by_answer[cid][item] == dim]
                key = rater, model, detail["family_id"], detail["condition"], dim
                pair[key] = observations.count("covered") / len(observations)
        effects = defaultdict(dict)
        models = sorted({key[1] for key in pair})
        for question, sample, families, after, before in samples:
            for rater in RATERS:
                for model in models:
                    deltas = {
                        dim: {f: pair[rater, model, f, after, dim] - pair[rater, model, f, before, dim]
                              for f in sorted(families)}
                        for dim in DIMENSIONS
                    }
                    if question == "RQ2":
                        deltas["R_minus_K"] = {f: deltas["R"][f] - deltas["K"][f] for f in sorted(families)}
                    effects[f"{question}/{sample}"].setdefault(rater, {})[model] = {
                        dim: cluster_bootstrap([{"template_group": f, "difference": difference}
                                                for f, difference in values.items()])
                        for dim, values in deltas.items()
                    }
        return dict(effects)

    output = {
        "hint_flags": {
            "all_families": len(records),
            "agreed_direct_leakage": len(shared),
            "either_direct_leakage": len(union),
            "test_agreed_direct_leakage": len(test & shared),
            "test_either_direct_leakage": len(test & union),
            "test_total": len(test),
        },
        "raw_human_effects": contrasts(raw),
    }
    if qa_path:
        overlaid = {key: dict(values) for key, values in raw.items()}
        changed = 0
        with qa_path.open(encoding="utf-8-sig", newline="") as stream:
            for entry in csv.DictReader(stream):
                if entry["packet"] != "explanations" or entry["phase"] != "test" or entry["field"] != "check_item":
                    continue
                cid, item = entry["case_id"], entry["item_id"]
                assert entry["proposed_value"] in COVERAGE, (cid, item)
                for rater in RATERS:
                    assert overlaid[rater, cid][item] == entry[f"{rater.lower()}_value"], (rater, cid, item)
                    overlaid[rater, cid][item] = entry["proposed_value"]
                changed += 1
        output["ai_assisted_coverage_sensitivity"] = {
            "test_item_rows_overlay": changed, "effects": contrasts(overlaid)
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True, help="Review packet directory")
    parser.add_argument("--reference-records", type=Path, required=True, help="Frozen reference records JSONL")
    parser.add_argument("--ratings", type=Path, required=True, help="Directory containing the six returned CSV files")
    parser.add_argument("--rating-suffix", default="all.csv", help="Filename suffix after the reviewer and packet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, help="Separate AI-assisted coverage sensitivity overlay")
    args = parser.parse_args()
    if Path(args.rating_suffix).name != args.rating_suffix or not args.rating_suffix.endswith(".csv"):
        parser.error("--rating-suffix must be a filename suffix ending in .csv")
    keys = args.handoff / "coordinator-only"
    explanation_key = read_json(keys / "explanation-key.json")["cases"]
    material_key = read_json(keys / "material-key.json")["case_bindings"]
    legacy = {x["legacy_audit_id"] for x in explanation_key.values() if x["legacy_audit_id"]}
    assert legacy == {f"A{i:03d}" for i in range(1, 31)}
    hashes, raw, cases = load_forms(args.handoff, args.ratings, args.rating_suffix)
    results = {
        "rating_source": "two returned teammate sheets, retrospective",
        "raw_forms_sha256": hashes,
        "answer_counts_per_reviewer": dict(Counter(kind for kind, _ in cases)),
        "raw_form_comparison": paired_forms_agreement(raw, material_key),
        "judge_comparison": judge_agreement(raw, explanation_key),
        "analysis": solutions(args.reference_records, raw, cases, explanation_key, material_key, args.qa),
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
