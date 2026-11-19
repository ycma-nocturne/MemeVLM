"""Real local files exercise compile-only preparation; no inference fixtures."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
from meme_study.data import content_hash
from test_reference_data import FIXTURE_FACT, FIXTURE_REFERENCE, REVISION, source_record


class ReferenceCliTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("meme_study.reference_cli"),
                             "reference prepare CLI is missing")
        from meme_study import reference_cli
        self.api = reference_cli
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "prepared"
        self.metadata = self.root / "metadata.jsonl"
        rows = [source_record(7), source_record(8)]
        self.metadata.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        self.selection, self.hints, self.pairs = [], {}, []
        for item, split, row in zip((7, 8), ("dev", "test"), rows):
            family = f"public-{item}"
            path = f"sources/image_{item:04d}.png"
            digest = self.write_image(path, (20, 30, 40))
            self.selection.append({"family_id": family, "source_id": item, "split": split,
                "template_group": f"template-{item}", "image_path": path, "image_sha256": digest,
                "source_image_path": path, "source_image_sha256": digest})
            self.hints[family] = {"source_record_sha256": content_hash(row), "source_fact_indices": [0],
                "text": FIXTURE_FACT, "preparation_note": "Fixture source fact.",
                "excluded_facts": []}
            variants = {}
            for index, name in enumerate(("original", "edited", "sham")):
                pair_path = f"pairs/{item}-{name}.png"
                pair_digest = self.write_image(pair_path, (index * 40, 80, 90))
                variants[name] = {"image_path": pair_path, "image_sha256": pair_digest,
                    "cue_answers": ["no" if name == "edited" else "yes"],
                    "relation_answers": ["alternative" if name == "edited" else "original"]}
            self.pairs.append({"family_id": family, "split": split, "template_group": f"template-{item}",
                "cue_question": "图中是否有角？", "relation_question": "图文构成哪种关系？",
                "options": [{"label": label, "semantic_id": sid, "text": text} for label, sid, text in
                    [("A", "original", "称呼不匹配"), ("B", "alternative", "称呼匹配"),
                     ("C", "other", "无关角色"), ("D", "insufficient", "证据不足")]],
                "variants": variants, "material_check": None})
        self.config = {"protocol_id": "reference_reuse_v1",
            "metadata": {"revision": REVISION, "sha256": hashlib.sha256(self.metadata.read_bytes()).hexdigest()},
            "model_ids": ["fixture-small", "fixture-large"], "judge_model": "fixture-independent-judge",
            "design": {"families": 2, "dev": 1, "test": 1, "paired_families": 2,
                       "paired_dev": 1, "paired_test": 1, "planned_calls": 38, "attempt_cap": 42}}
        self.paths = {name: self.root / f"{name}.json" for name in ("selection", "hints", "pairs", "config")}
        self.save_inputs()

    def write_image(self, relative, color, *, size=(4, 4), fmt="PNG"):
        path = self.data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, format=fmt)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_inputs(self):
        for name, path in self.paths.items():
            path.write_text(json.dumps(getattr(self, name), ensure_ascii=False, indent=2), encoding="utf-8")

    def prepare(self, **changes):
        values = dict(metadata=self.metadata, selection=self.paths["selection"], hints=self.paths["hints"],
                      pairs=self.paths["pairs"], config=self.paths["config"], data_root=self.data, output=self.out)
        values.update(changes)
        return self.api.prepare(**values)

    def read_rows(self, filename):
        return [json.loads(line) for line in (self.out / filename).read_text().splitlines()]

    def assert_no_package(self):
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob(".prepared-*")), [])

    def test_success_exports_private_sources_and_exact_small_design_without_calls(self):
        summary = self.prepare()
        self.assertEqual(summary["explanation_calls"], 12)  # 2 families * 3 conditions * 2 models
        self.assertEqual(summary["paired_calls"], 12)  # 2 triples * 3 variants * 2 models
        self.assertEqual(summary["permutation_calls"], 2)  # one test family * 2 models
        self.assertEqual(summary["judge_slots"], 12)
        self.assertEqual((summary["planned_calls"], summary["attempt_cap"], summary["actual_model_calls"]), (38, 42, 0))
        expected = {"reference_records.jsonl", "explanation_jobs.jsonl", "pair_jobs.preview.jsonl",
                    "pair_gold.json", "judge_slots.jsonl", "summary.json", "provenance.json", "inputs"}
        self.assertEqual({p.name for p in self.out.iterdir()}, expected)
        refs = self.read_rows("reference_records.jsonl")
        self.assertEqual(refs[0]["reference"]["explanation"], FIXTURE_REFERENCE)
        jobs = self.read_rows("explanation_jobs.jsonl") + self.read_rows("pair_jobs.preview.jsonl")
        for job in jobs:
            text = json.dumps(job, ensure_ascii=False)
            self.assertNotIn(FIXTURE_REFERENCE, text)
            self.assertNotIn("Did the answer", text)
            self.assertNotIn("cue_answers", text)
            self.assertNotIn("relation_answers", text)
        pair_jobs = self.read_rows("pair_jobs.preview.jsonl")
        self.assertTrue(all(j["readiness"] == "pending_material_check" for j in pair_jobs))
        self.assertEqual(len(json.loads((self.out / "pair_gold.json").read_text())), 14)
        self.assertNotEqual(refs[0]["image"]["path"], pair_jobs[0]["image_path"])
        for name, path in self.paths.items():
            self.assertEqual((self.out / "inputs" / f"{name}.json").read_bytes(), path.read_bytes())
        provenance = json.loads((self.out / "provenance.json").read_text())
        self.assertEqual(provenance["model_calls"], 0)
        self.assertEqual(provenance["source_files"]["metadata"]["sha256"], self.config["metadata"]["sha256"])
        self.assertTrue(provenance["private_field_notes"])

    def test_judge_slots_are_dependencies_not_fake_judge_inputs_or_outputs(self):
        self.prepare()
        parents = {j["job_id"]: j for j in self.read_rows("explanation_jobs.jsonl")}
        slots = self.read_rows("judge_slots.jsonl")
        self.assertEqual(len({s["parent_job_id"] for s in slots}), 12)
        for slot in slots:
            self.assertEqual(slot["parent_input_sha256"], parents[slot["parent_job_id"]]["input_sha256"])
            self.assertEqual(slot["readiness"], "pending_candidate_output")
            self.assertNotIn("messages", slot)
            self.assertNotIn("source_output_sha256", slot)
            self.assertNotIn("text", slot)

    def test_wrong_design_count_or_call_budget_is_rejected_without_output(self):
        for key in ("families", "dev", "test", "paired_families", "paired_dev", "paired_test", "planned_calls", "attempt_cap"):
            original = self.config["design"][key]
            self.config["design"][key] = original + 1
            self.save_inputs()
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.prepare()
            self.assert_no_package()
            self.config["design"][key] = original

    def test_unknown_protocol_and_overlapping_judge_are_rejected(self):
        self.config["protocol_id"] = "legacy_or_unknown"
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.config["protocol_id"] = "reference_reuse_v1"
        self.config["judge_model"] = self.config["model_ids"][0]
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_no_package()

    def test_pairs_must_be_selected_families_with_identical_split_and_group(self):
        for key, value in (("family_id", "not-selected"), ("split", "test"), ("template_group", "unbound-group")):
            original = copy.deepcopy(self.pairs)
            self.pairs[0][key] = value
            self.save_inputs()
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.prepare()
            self.assert_no_package()
            self.pairs = original

    def test_incomplete_pair_is_rejected_without_partial_package(self):
        del self.pairs[0]["variants"]["sham"]
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_no_package()

    def test_changed_pair_image_bytes_and_undecodable_bytes_are_rejected(self):
        variant = self.pairs[0]["variants"]["edited"]
        (self.data / variant["image_path"]).write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_no_package()
        variant["image_sha256"] = hashlib.sha256(b"not an image").hexdigest()
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_no_package()

    def test_pair_canvas_and_actual_encoding_must_match(self):
        variant = self.pairs[0]["variants"]["sham"]
        for size, fmt in (((5, 4), "PNG"), ((4, 4), "JPEG")):
            variant["image_sha256"] = self.write_image(variant["image_path"], (10, 20, 30), size=size, fmt=fmt)
            self.save_inputs()
            with self.subTest(size=size, fmt=fmt), self.assertRaises(ValueError):
                self.prepare()
            self.assert_no_package()

    def test_unsafe_pair_paths_or_symlink_escape_are_rejected(self):
        outside = self.root / "outside.png"
        outside.write_bytes((self.data / self.pairs[0]["variants"]["edited"]["image_path"]).read_bytes())
        (self.data / "escape.png").symlink_to(outside)
        variant = self.pairs[0]["variants"]["edited"]
        for path in ("../outside.png", str(outside), "escape.png"):
            variant["image_path"] = path
            self.save_inputs()
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare()
            self.assert_no_package()

    def test_existing_output_even_empty_is_never_overwritten(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_output_parent_must_exist_and_be_writable_directory(self):
        parent_file = self.root / "not-a-directory"
        parent_file.write_text("keep")
        with self.assertRaises((OSError, ValueError)):
            self.prepare(output=parent_file / "output")
        self.assertEqual(parent_file.read_text(), "keep")
        self.assert_no_package()

    @unittest.skipIf(os.geteuid() == 0, "root bypasses POSIX directory write restrictions")
    def test_unwritable_output_parent_fails_without_partial_package(self):
        protected = self.root / "read-only-parent"
        protected.mkdir()
        protected.chmod(0o500)
        try:
            with self.assertRaises(PermissionError):
                self.prepare(output=protected / "prepared")
            self.assertEqual(list(protected.iterdir()), [])
        finally:
            protected.chmod(0o700)

    def test_stale_or_rewritten_hint_is_rejected_before_output(self):
        self.hints["public-7"]["text"] = "Answers copied into an unsupported hint"
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_no_package()

    def test_command_line_exports_package_and_reports_errors_with_nonzero_exit(self):
        args = [sys.executable, "-m", "meme_study.reference_cli", "prepare", "--metadata", str(self.metadata)]
        for key, path in self.paths.items():
            args += ["--" + key, str(path)]
        args += ["--data-root", str(self.data), "--output", str(self.out)]
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["actual_model_calls"], 0)
        second = subprocess.run(args, text=True, capture_output=True, check=False)
        self.assertEqual(second.returncode, 2)
        self.assertTrue(second.stderr)


if __name__ == "__main__":
    unittest.main()
