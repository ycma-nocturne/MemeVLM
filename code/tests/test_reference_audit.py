"""Synthetic audit contracts; these rows are neither model nor human results."""
import copy
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from meme_study.data import content_hash


MODELS = ["fixture-small-model", "fixture-large-model"]
FIELDS = ["audit_id", "reviewer", "V_coverage", "I_coverage", "K_coverage",
          "R_coverage", "grounded_relation", "invented_claim", "notes"]


def scored_rows(*, exact=False):
    result = []
    for split, quotas in (("dev", (2, 2, 1)), ("test", (4, 3, 3))):
        for model in MODELS:
            for condition, quota in zip(("image_only", "hinted", "ocr_only"), quotas):
                for index in range(quota if exact else 8):
                    family = f"{split}-f{index:02d}"
                    job = f"explanation/{family}/{condition}/{model}"
                    text = f"<script>alert('x')</script> 中文解释 {index}"
                    result.append({"job_id": job, "task_kind": "explanation",
                        "family_id": family, "split": split, "model": model,
                        "condition": condition, "image_path": None if condition == "ocr_only" else f"images/{family}.png",
                        "candidate_text": text, "run_status": "ok", "candidate_truncated": False,
                        "source_input_sha256": content_hash(job),
                        "reference": {"explanation": "公开参考解释", "checklist": [
                            {"item_id": d + "1", "dimension": d, "text": d + " 参考条目"}
                            for d in "VIKR"]},
                        "score": {"status": "unavailable", "failure_reason": "judge not available",
                            "source_sha256": {"candidate": hashlib.sha256(text.encode()).hexdigest()},
                            "dimensions": {d: {"applicable": 1, "coverage": None} for d in "VIKR"}}})
    return result


class ReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("meme_study.reference_audit"),
                             "small reference explanation audit is missing")
        from meme_study import reference_audit
        self.api = reference_audit

    def select(self, rows=None, **kwargs):
        return self.api.select_audit(scored_rows() if rows is None else rows,
                                     model_ids=MODELS, **kwargs)

    def test_exact_thirty_unique_rows_obey_every_split_model_condition_quota(self):
        selected = self.select()
        self.assertEqual(len(selected["sample"]), 30)
        self.assertEqual(len({r["audit_id"] for r in selected["sample"]}), 30)
        key = selected["private_key"]
        self.assertEqual(len({r["job_id"] for r in key}), 30)
        counts = Counter((r["split"], r["model"], r["condition"]) for r in key)
        for model in MODELS:
            self.assertEqual([counts[("dev", model, c)] for c in ("image_only", "hinted", "ocr_only")], [2, 2, 1])
            self.assertEqual([counts[("test", model, c)] for c in ("image_only", "hinted", "ocr_only")], [4, 3, 3])

    def test_dev_calibration_export_does_not_require_future_test_outputs(self):
        dev=[row for row in scored_rows() if row['split']=='dev']
        selected=self.select(dev,phase='dev')
        self.assertEqual(len(selected['sample']),10)
        self.assertEqual(selected['phase'],'dev')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'dev'
            summary=self.api.export_audit(dev,model_ids=MODELS,output_dir=output,phase='dev')
            self.assertEqual((summary['dev_count'],summary['test_count']),(10,0))
            with (output/'reviewer_1.csv').open() as stream:
                forms=list(csv.DictReader(stream))
            self.assertEqual(len(forms),10)
            self.assertTrue(all(all(v=='' for k,v in row.items() if k!='audit_id') for row in forms))

    def test_separate_phase_samples_and_ids_match_combined_export(self):
        rows=scored_rows();whole=self.select(rows)
        for phase,count in [('dev',10),('test',20)]:
            separate=self.select([r for r in rows if r['split']==phase],phase=phase)
            self.assertEqual(len(separate['sample']),count)
            self.assertEqual(separate['sample'],[r for r in whole['sample'] if r['split']==phase])
            self.assertEqual(separate['private_key'],[r for r in whole['private_key'] if r['split']==phase])

    def test_dev_ratings_summary_can_precede_test_inference(self):
        selected=self.select([r for r in scored_rows() if r['split']=='dev'],phase='dev')
        first=self.ratings(selected['sample'],'Member one');second=self.ratings(selected['sample'],'Member two')
        summary=self.api.summarize_ratings(selected,first,second)
        self.assertEqual(summary['ratings_completed'],20)
        self.assertEqual(summary['splits']['test']['sample_count'],0)
        self.assertIsNone(summary['splits']['test']['fields']['V_coverage']['disagreement_rate'])

    def test_reordered_input_and_model_list_produce_identical_sample_and_manifest(self):
        rows = scored_rows()
        original = copy.deepcopy(rows)
        first = self.select(rows)
        second = self.api.select_audit(list(reversed(rows)), model_ids=list(reversed(MODELS)))
        self.assertEqual(first, second)
        self.assertEqual(rows, original)
        self.assertNotEqual([r["job_id"] for r in first["private_key"]],
                            [r["job_id"] for r in self.select(rows, seed=6001)["private_key"]])

    def test_quality_and_judge_availability_never_influence_selection(self):
        rows = scored_rows()
        before = self.select(rows)
        for row in rows:
            row["candidate_text"] = ""
            row["candidate_truncated"] = True
            row["score"] = None
        after = self.select(rows)
        self.assertEqual([r["job_id"] for r in before["private_key"]],
                         [r["job_id"] for r in after["private_key"]])
        self.assertTrue(all(r["candidate_text"] == "" and r["candidate_truncated"] for r in after["sample"]))

    def test_failed_and_interrupted_outputs_are_listed_with_reasons(self):
        rows = scored_rows()
        for index, status in enumerate(("failed", "interrupted")):
            rows[index]["run_status"] = status
            rows[index]["candidate_text"] = None
        selected = self.select(rows)
        self.assertEqual({r["run_status"] for r in selected["excluded"]}, {"failed", "interrupted"})
        self.assertTrue(all(r["reason"] for r in selected["excluded"]))
        self.assertFalse({r["job_id"] for r in selected["excluded"]} &
                         {r["job_id"] for r in selected["private_key"]})

    def test_missing_and_pending_planned_outputs_are_explicitly_excluded(self):
        rows = scored_rows()
        for index, status in enumerate(("missing", "pending")):
            rows[index]["run_status"] = status
            rows[index]["candidate_text"] = None
        selected = self.select(rows)
        self.assertEqual(len(selected["sample"]), 30)
        self.assertEqual({row["run_status"] for row in selected["excluded"]}, {"missing", "pending"})
        self.assertEqual(selected["excluded_by_status"], {"failed": 0, "interrupted": 0, "missing": 1, "pending": 1})
        self.assertTrue(all(row["reason"] for row in selected["excluded"]))

    def test_missing_or_insufficient_stratum_rejects_instead_of_undersampling(self):
        rows = [r for r in scored_rows() if not (r["model"] == MODELS[0] and r["split"] == "test" and r["condition"] == "ocr_only")]
        with self.assertRaisesRegex(ValueError, "test.*ocr_only.*required 3.*available 0"):
            self.select(rows)
        rows = scored_rows(exact=True)
        rows[0]["run_status"] = "failed"
        rows[0]["candidate_text"] = None
        with self.assertRaisesRegex(ValueError, "required 2.*available 1"):
            self.select(rows)

    def test_duplicate_tampered_or_cross_split_metadata_rejects(self):
        rows = scored_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.select(rows + [copy.deepcopy(rows[0])])
        for field, value in (("family_id", "other"), ("model", "unknown"),
                             ("condition", "not_a_condition"), ("split", "other"),
                             ("source_input_sha256", "bad-hash"), ("candidate_text", 2)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select(changed)
        changed = copy.deepcopy(rows)
        changed[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.select(changed)

    def test_candidate_hash_mismatch_and_image_path_escape_reject(self):
        for field, value in (("candidate_text", "changed after scoring"), ("image_path", "../outside.png")):
            rows = scored_rows()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select(rows)

    def test_malformed_category_types_and_changed_reference_checklist_reject_cleanly(self):
        for field in ("split", "condition", "model", "run_status"):
            rows = scored_rows()
            rows[0][field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select(rows)
        rows = scored_rows()
        rows[0]["score"]["source_sha256"]["checklist"] = content_hash(rows[0]["reference"]["checklist"])
        rows[0]["reference"]["checklist"][0]["text"] = "changed after scoring"
        with self.assertRaisesRegex(ValueError, "checklist.*hash"):
            self.select(rows)

    def test_release_namespaced_ids_bind_base_metadata_and_preparation_digest(self):
        rows = scored_rows()
        for row in rows:
            row["prepared_job_id"] = row["job_id"]
            row["prepared_sha256"] = "a" * 64
            row["job_id"] = "release/" + "a" * 64 + "/" + row["prepared_job_id"]
        selected = self.select(rows)
        self.assertEqual(len(selected["sample"]), 30)
        self.assertTrue(all(row["prepared_sha256"] == "a" * 64 for row in selected["private_key"]))
        for field, value in (("prepared_sha256", "b" * 64), ("prepared_job_id", "explanation/other/hinted/fixture-small-model"),
                             ("prepared_sha256", "bad"), ("job_id", "arbitrary")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select(changed)
        changed = copy.deepcopy(rows)
        del changed[0]["prepared_job_id"]
        with self.assertRaises(ValueError):
            self.select(changed)

    def test_requires_exactly_two_distinct_models_and_integer_seed(self):
        for models in ([], MODELS[:1], MODELS + ["third"], [MODELS[0], MODELS[0]]):
            with self.subTest(models=models), self.assertRaises(ValueError):
                self.api.select_audit(scored_rows(), model_ids=models)
        with self.assertRaises(ValueError):
            self.select(seed=True)

    def test_export_blinds_metadata_escapes_html_and_leaves_both_forms_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            result = self.api.export_audit(scored_rows(), model_ids=MODELS, output_dir=output)
            self.assertEqual(result["sample_count"], 30)
            self.assertEqual(result["ratings_completed"], 0)
            sample = json.loads((output / "sample.json").read_text())
            self.assertEqual(len(sample), 30)
            anonymous = (output / "sample.json").read_text() + (output / "candidates.html").read_text()
            for model in MODELS:
                self.assertNotIn(model, anonymous)
            self.assertNotIn("source_input_sha256", anonymous)
            self.assertNotIn("<script>", (output / "candidates.html").read_text())
            self.assertIn("&lt;script&gt;", (output / "candidates.html").read_text())
            self.assertIn("公开参考解释", anonymous)
            forms = []
            for name in ("reviewer_1.csv", "reviewer_2.csv"):
                with (output / name).open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, FIELDS)
                    form = list(reader)
                self.assertEqual(len(form), 30)
                self.assertTrue(all(row[field] == "" for row in form for field in FIELDS if field != "audit_id"))
                forms.append(form)
            self.assertEqual(forms[0], forms[1])
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual({r["audit_id"] for r in sample}, {r["audit_id"] for r in forms[0]})
            self.assertFalse(any("adjudicat" in p.name for p in output.iterdir()))

    def test_export_refuses_existing_destination_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one", Path(directory) / "two"
            self.api.export_audit(scored_rows(), model_ids=MODELS, output_dir=first)
            self.api.export_audit(list(reversed(scored_rows())), model_ids=MODELS, output_dir=second)
            self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                             {p.name: p.read_bytes() for p in second.iterdir()})
            with self.assertRaises(FileExistsError):
                self.api.export_audit(scored_rows(), model_ids=MODELS, output_dir=first)
            empty = Path(directory) / "existing-empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                self.api.export_audit(scored_rows(), model_ids=MODELS, output_dir=empty)

    def test_invalid_export_creates_no_partial_package(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            with self.assertRaises(ValueError):
                self.api.export_audit([], model_ids=MODELS, output_dir=output)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def ratings(self, sample, reviewer):
        return [{"audit_id": row["audit_id"], "reviewer": reviewer,
                 **{d + "_coverage": "0.5" for d in "VIKR"},
                 "grounded_relation": "yes", "invented_claim": "no", "notes": ""}
                for row in sample]

    def test_raw_disagreement_uses_independent_ratings_and_keeps_dev_test_separate(self):
        self.assertTrue(hasattr(self.api, "summarize_ratings"), "ratings summary is missing")
        selected = self.select()
        sample = selected["sample"]
        first, second = self.ratings(sample, "Member one"), self.ratings(sample, "Member two")
        dev_index = next(i for i, row in enumerate(sample) if row["split"] == "dev")
        test_index = next(i for i, row in enumerate(sample) if row["split"] == "test")
        second[dev_index]["V_coverage"] = "1"
        second[test_index]["grounded_relation"] = "uncertain"
        before = copy.deepcopy((first, second))
        summary = self.api.summarize_ratings(selected, first, second)
        self.assertEqual(summary["ratings_completed"], 60)
        self.assertEqual(summary["splits"]["dev"]["fields"]["V_coverage"],
                         {"comparable": 10, "disagreements": 1, "disagreement_rate": .1, "na_excluded": 0})
        self.assertEqual(summary["splits"]["test"]["fields"]["grounded_relation"],
                         {"comparable": 20, "disagreements": 1, "disagreement_rate": .05, "na_excluded": 0})
        self.assertEqual(len(summary["disagreements"]), 2)
        self.assertEqual((first, second), before)
        self.assertNotIn("accepted", summary)
        self.assertNotIn("adjudication", summary)
        self.assertNotIn("judge_reliability", summary)

    def test_na_is_only_for_empty_reference_dimensions_and_excluded_from_denominator(self):
        self.assertTrue(hasattr(self.api, "summarize_ratings"), "ratings summary is missing")
        rows = scored_rows()
        for row in rows:
            row["reference"]["checklist"] = [item for item in row["reference"]["checklist"] if item["dimension"] != "I"]
            row["score"]["dimensions"]["I"]["applicable"] = 0
        selected = self.select(rows)
        first, second = self.ratings(selected["sample"], "One"), self.ratings(selected["sample"], "Two")
        for row in first + second:
            row["I_coverage"] = "NA"
        summary = self.api.summarize_ratings(selected, first, second)
        self.assertEqual(summary["splits"]["test"]["fields"]["I_coverage"],
                         {"comparable": 0, "disagreements": 0, "disagreement_rate": None, "na_excluded": 20})
        first[0]["V_coverage"] = "NA"
        with self.assertRaisesRegex(ValueError, "NA"):
            self.api.summarize_ratings(selected, first, second)

    def test_summary_rejects_blank_duplicate_incomplete_or_non_independent_forms(self):
        self.assertTrue(hasattr(self.api, "summarize_ratings"), "ratings summary is missing")
        selected = self.select()
        first, second = self.ratings(selected["sample"], "One"), self.ratings(selected["sample"], "Two")
        for field, value in (("reviewer", ""), ("R_coverage", ""), ("V_coverage", "0.3"),
                             ("grounded_relation", "maybe"), ("invented_claim", 1)):
            changed = copy.deepcopy(first)
            changed[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.api.summarize_ratings(selected, changed, second)
        for bad in (first[:-1], first + [first[0]], self.ratings(selected["sample"], " TWO ")):
            with self.assertRaises(ValueError):
                self.api.summarize_ratings(selected, bad, second)

    def test_summary_rejects_tampered_anonymous_sample(self):
        self.assertTrue(hasattr(self.api, "summarize_ratings"), "ratings summary is missing")
        selected = self.select()
        first, second = self.ratings(selected["sample"], "One"), self.ratings(selected["sample"], "Two")
        selected["sample"][0]["candidate_text"] = "silently modified"
        with self.assertRaisesRegex(ValueError, "sample.*hash"):
            self.api.summarize_ratings(selected, first, second)


if __name__ == "__main__":
    unittest.main()
