"""Test source attribution, joins, and reference integrity with synthetic records."""

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REVISION = "d421198c5fd8108b17d336a5d1e8a7c229bd2b1f"
FIXTURE_OCR = "维修站：请按蓝色按钮\n值班机器人"
FIXTURE_FACT = "蓝色按钮启动冷却，红色按钮启动加热。"
FIXTURE_REFERENCE = "机器人把维修按钮认成了咖啡按钮。"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def source_record(item=7):
    """Create a synthetic record with the published field structure."""
    return {
        "file_name": f"images/image_{item:04d}.png", "id": item,
        "image_path": f"images/image_{item:04d}.png",
        "checked_gt": FIXTURE_REFERENCE,
        "meta": {"domain": "Cross_Domain", "meme_type": "Multi_Source",
                 "structure": "Single", "language": "zh"},
        "moderation": {"political": False, "hate_speech": False},
        "eval_checklist": {
            "visual": ["Did the answer read the caption?", "Did it describe the buttons?"],
            "identity": ["Did it identify the robot?"],
            "knowledge": ["Did it explain the button functions?"],
            "reasoning": ["Did it explain the button mix-up?"],
        },
        "visual": {"description": "A robot faces a control panel.", "ocr": FIXTURE_OCR,
                   "entities": [{"id": "e1", "visual_desc": "A robot with a wrench."}]},
        "identity": {"entities": [{"id": "e1", "name": "维修机器人", "source": "合成场景"}]},
        "knowledge": {"facts": [FIXTURE_FACT], "cultural_code": None},
        "reasoning": {"trigger": "Caption and buttons.", "reference": "Control panel labels.",
                      "core_message": "A mistaken button function.", "logic": "IC", "logic_detail": None},
    }


class ReferenceDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("meme_study.reference_data"),
                             "public-reference importer is missing")
        from meme_study.reference_data import import_references
        self.importer = import_references
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "candidates/images").mkdir(parents=True)
        self.metadata = self.root / "metadata.jsonl"
        self.rows = [source_record()]
        self.selection = [self.selection_row()]
        self.write_metadata()

    def selection_row(self, item=7, split="dev", group="fixture-robot-controls"):
        relative = f"candidates/images/image_{item:04d}.png"
        (self.data / relative).write_bytes(PNG)
        digest = hashlib.sha256(PNG).hexdigest()
        return {"family_id": f"fixture-mb-{item:04d}", "source_id": item, "split": split,
                "template_group": group, "image_path": relative, "image_sha256": digest,
                "source_image_path": relative, "source_image_sha256": digest}

    def write_metadata(self):
        self.metadata.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows),
                                 encoding="utf-8")
        self.digest = hashlib.sha256(self.metadata.read_bytes()).hexdigest()

    def run_import(self, **kwargs):
        return self.importer(self.metadata, self.selection, self.data,
                             expected_sha256=kwargs.get("expected_sha256", self.digest),
                             revision=kwargs.get("revision", REVISION))

    def test_import_preserves_source_text_and_keeps_answer_fields_private(self):
        result = self.run_import()
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual((row["family_id"], row["source_id"], row["split"], row["template_group"]),
                         ("fixture-mb-0007", 7, "dev", "fixture-robot-controls"))
        self.assertEqual(row["ocr_text"], FIXTURE_OCR)
        self.assertEqual(row["knowledge_facts"], [FIXTURE_FACT])
        self.assertEqual(row["reference"]["explanation"], FIXTURE_REFERENCE)
        self.assertEqual(row["reference"]["domain"], "Cross_Domain")
        self.assertIsNone(row["reference"]["reasoning"]["logic_detail"])
        self.assertEqual(row["reference"]["checklist"], [
            {"item_id": "V001", "dimension": "V", "text": "Did the answer read the caption?"},
            {"item_id": "V002", "dimension": "V", "text": "Did it describe the buttons?"},
            {"item_id": "I001", "dimension": "I", "text": "Did it identify the robot?"},
            {"item_id": "K001", "dimension": "K", "text": "Did it explain the button functions?"},
            {"item_id": "R001", "dimension": "R", "text": "Did it explain the button mix-up?"},
        ])
        self.assertEqual(row["source"]["annotation_origin"], "inherited_public")
        self.assertEqual(row["source"]["metadata_sha256"], self.digest)
        self.assertEqual(row["source"]["source_id"], 7)
        self.assertIn(f"/{REVISION}/metadata.jsonl", row["source"]["url"])
        self.assertEqual(len(row["source"]["record_sha256"]), 64)
        self.assertFalse(row["image"]["adapted"])
        for private in ("checked_gt", "explanation", "checklist", "reasoning", "reviews", "accepted"):
            self.assertNotIn(private, row)

    def test_null_or_empty_ocr_remains_source_value_without_caption_fallback(self):
        for ocr in (None, ""):
            with self.subTest(ocr=ocr):
                self.rows[0]["visual"]["ocr"] = ocr
                self.selection[0]["caption_zh"] = "Must not be imported as OCR"
                self.write_metadata()
                self.assertEqual(self.run_import()[0]["ocr_text"], ocr)

    def test_byte_adaptation_is_explicit_and_original_hash_is_retained(self):
        (self.data / "adapted.png").write_bytes(PNG + b"adapted-copy")
        self.selection[0]["image_path"] = "adapted.png"
        self.selection[0]["image_sha256"] = hashlib.sha256(PNG + b"adapted-copy").hexdigest()
        image = self.run_import()[0]["image"]
        self.assertTrue(image["adapted"])
        self.assertEqual(image["source_path"], "candidates/images/image_0007.png")
        self.assertEqual(image["source_sha256"], hashlib.sha256(PNG).hexdigest())

    def test_same_bytes_at_a_new_path_are_not_claimed_as_byte_adaptation(self):
        (self.data / "copy.png").write_bytes(PNG)
        self.selection[0]["image_path"] = "copy.png"
        self.assertFalse(self.run_import()[0]["image"]["adapted"])

    def test_metadata_hash_and_revision_must_match_immutable_source(self):
        for change in ({"expected_sha256": "0" * 64}, {"expected_sha256": "bad"}, {"revision": "main"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_import(**change)

    def test_nonpublic_missing_or_duplicate_selected_source_ids_are_rejected(self):
        for item in ("fixture-local-image", "7", True, 0, 999):
            with self.subTest(item=item):
                self.selection[0]["source_id"] = item
                with self.assertRaises(ValueError):
                    self.run_import()
        self.selection = [self.selection_row(), self.selection_row()]
        self.selection[1]["family_id"] = "different-name-same-source"
        with self.assertRaises(ValueError):
            self.run_import()

    def test_duplicate_metadata_ids_are_rejected_even_if_identical(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.write_metadata()
        with self.assertRaises(ValueError):
            self.run_import()

    def test_language_and_metadata_image_identity_must_match(self):
        for section, key, value in (("meta", "language", "en"), (None, "image_path", "../image.png"),
                                    (None, "file_name", "images/image_0008.png")):
            with self.subTest(field=key):
                self.rows = [source_record()]
                target = self.rows[0][section] if section else self.rows[0]
                target[key] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_duplicate_family_or_cross_split_template_is_rejected(self):
        self.rows.append(source_record(8))
        self.selection.append(self.selection_row(8, split="test"))
        self.write_metadata()
        with self.assertRaises(ValueError):
            self.run_import()
        self.selection[1]["template_group"] = "another-template"
        self.selection[1]["family_id"] = self.selection[0]["family_id"]
        with self.assertRaises(ValueError):
            self.run_import()

    def test_same_split_template_members_and_selection_order_are_retained(self):
        self.rows.append(source_record(8))
        self.selection.insert(0, self.selection_row(8))
        self.write_metadata()
        before = copy.deepcopy(self.selection)
        first = self.run_import()
        self.assertEqual([r["source_id"] for r in first], [8, 7])
        self.assertEqual(first, self.run_import())
        self.assertEqual(self.selection, before)

    def test_split_and_group_must_be_explicit(self):
        for key, value in (("split", "train"), ("split", None), ("template_group", " "), ("family_id", "")):
            with self.subTest(field=key):
                self.selection = [self.selection_row()]
                self.selection[0][key] = value
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_missing_401_reference_only_blocks_when_selected(self):
        incomplete = source_record(401)
        del incomplete["checked_gt"]
        self.rows.append(incomplete)
        self.write_metadata()
        self.assertEqual(len(self.run_import()), 1)
        self.selection = [self.selection_row(401)]
        with self.assertRaisesRegex(ValueError, "checked_gt|explanation|reference"):
            self.run_import()

    def test_incomplete_or_malformed_reference_fields_are_rejected(self):
        edits = [("checked_gt", " "), ("eval_checklist", {}), ("knowledge", {"facts": []}),
                 ("visual", {"description": "No OCR key"}), ("reasoning", {}),
                 ("meta", {"language": "zh", "domain": ""})]
        for key, value in edits:
            with self.subTest(field=key):
                self.rows = [source_record()]
                self.rows[0][key] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.run_import()
        for value in ([], [""], "not a list", [True]):
            with self.subTest(checklist=value):
                self.rows = [source_record()]
                self.rows[0]["eval_checklist"]["reasoning"] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_image_escape_missing_file_and_hash_mismatch_are_rejected(self):
        for key in ("image_path", "source_image_path"):
            for value in ("../outside.png", str(self.data / "candidates/images/image_0007.png"),
                          "https://example.com/image.png", "candidates\\image.png", "missing.png"):
                with self.subTest(field=key, path=value):
                    self.selection = [self.selection_row()]
                    self.selection[0][key] = value
                    with self.assertRaises(ValueError):
                        self.run_import()
        for key in ("image_sha256", "source_image_sha256"):
            self.selection = [self.selection_row()]
            self.selection[0][key] = "0" * 64
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.run_import()

    def test_symlink_cannot_escape_data_root(self):
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        (self.data / "link.png").symlink_to(outside)
        self.selection[0]["image_path"] = "link.png"
        with self.assertRaises(ValueError):
            self.run_import()

    def test_source_file_cannot_be_rebound_to_another_published_image(self):
        other = self.selection_row(8)
        self.selection[0]["source_image_path"] = other["source_image_path"]
        self.selection[0]["source_image_sha256"] = other["source_image_sha256"]
        with self.assertRaisesRegex(ValueError, "source.*image|published.*image"):
            self.run_import()

    def test_changed_annotation_changes_record_hash_without_changing_source_id(self):
        first = self.run_import()[0]["source"]
        self.rows[0]["checked_gt"] = "新版公开解释。"
        self.write_metadata()
        second = self.run_import()[0]["source"]
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertNotEqual(first["record_sha256"], second["record_sha256"])
        self.assertNotEqual(first["metadata_sha256"], second["metadata_sha256"])


if __name__ == "__main__":
    unittest.main()
