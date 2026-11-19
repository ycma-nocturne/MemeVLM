"""Hand-computed statistical counterexamples, not research observations."""
import importlib.util
import unittest


def scored(family, background="none", workflow="R", correct=0,
           variant="original", model="m8", group=None, **changes):
    row = dict(case_id=f"{family}:{variant}:{background}", family_id=family,
               template_group=group or family, split="test", variant=variant,
               background=background, control="core", model=model,
               workflow=workflow, interpretation_correct=correct,
               fact_correct=0, joint_correct=0, valid=True,
               predicted_option="o1" if correct else "o2")
    row.update(changes)
    return row


class PairedAnalysis(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("meme_study.analysis"),
                             "paired analysis is not implemented")
        from meme_study import analysis
        self.analysis = analysis

    def test_repair_uses_own_workflow_baseline_and_empty_harm_is_none(self):
        rows = [scored("a"), scored("b", valid=False),
                scored("a", "relevant", correct=1),
                scored("b", "relevant"),
                scored("a", workflow="D", correct=1),
                scored("b", workflow="D", correct=1)]
        result = self.analysis.paired_background(rows, "m8", "R", "relevant")
        self.assertEqual(result["repair"], dict(num=1, den=2, rate=0.5))
        self.assertEqual(result["harm"], dict(num=0, den=0, rate=None))
        self.assertEqual(result["n_pairs"], 2)

    def test_repair_and_harm_keep_invalid_answers_in_denominators(self):
        rows = [scored("a", correct=1), scored("a", "false", valid=False),
                scored("b"), scored("b", "false", correct=1)]
        result = self.analysis.paired_background(rows, "m8", "R", "false")
        self.assertEqual(result["harm"], dict(num=1, den=1, rate=1.0))
        self.assertEqual(result["repair"], dict(num=1, den=1, rate=1.0))

    def test_background_rejects_missing_duplicate_and_mismatched_pairs(self):
        baseline = scored("a")
        note = scored("a", "relevant")
        examples = [[baseline], [baseline, note, dict(note)],
                    [baseline, dict(note, template_group="wrong")]]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.analysis.paired_background(rows, "m8", "R", "relevant")

    def test_workflow_difference_is_v_minus_r_matched_by_case(self):
        rows = [scored("a", workflow="V", correct=1),
                scored("b", workflow="R", correct=1),
                scored("b", workflow="V", valid=False), scored("a")]
        pairs = self.analysis.paired_workflows(rows, "m8")
        self.assertEqual([p["difference"] for p in pairs], [1, -1])
        self.assertEqual([p["family_id"] for p in pairs], ["a", "b"])
        self.assertEqual([p["template_group"] for p in pairs], ["a", "b"])

    def test_workflows_reject_missing_duplicate_and_metadata_mismatch(self):
        r, v = scored("a"), scored("a", workflow="V")
        for rows in ([r], [r, v, dict(v)], [r, dict(v, background="false")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.analysis.paired_workflows(rows, "m8")

    def test_bootstrap_samples_clusters_with_replacement_and_multiplicity(self):
        pairs = [dict(template_group="a", difference=1),
                 dict(template_group="a", difference=1),
                 dict(template_group="b", difference=0),
                 dict(template_group="c", difference=-1)]
        # seed=1 chooses a,c,a: four +1 rows and one -1 row => 3/5.
        result = self.analysis.cluster_bootstrap(pairs, seed=1, repeats=1)
        self.assertEqual(result["estimate"], 0.25)
        self.assertEqual(result["ci"], [0.6, 0.6])
        self.assertEqual((result["n_clusters"], result["n_pairs"]), (3, 4))

    def test_bootstrap_has_no_interval_for_fewer_than_two_clusters(self):
        result = self.analysis.cluster_bootstrap([
            dict(template_group="same", difference=1),
            dict(template_group="same", difference=-1)])
        self.assertEqual(result["estimate"], 0.0)
        self.assertIsNone(result["ci"])
        empty = self.analysis.cluster_bootstrap([])
        self.assertIsNone(empty["estimate"])
        self.assertIsNone(empty["ci"])

    def test_identical_workflows_have_zero_interval_and_seed_is_reproducible(self):
        pairs = [dict(template_group="a", difference=0),
                 dict(template_group="b", difference=0)]
        first = self.analysis.cluster_bootstrap(pairs, seed=12, repeats=50)
        self.assertEqual(first["ci"], [0.0, 0.0])
        self.assertEqual(first, self.analysis.cluster_bootstrap(pairs, seed=12, repeats=50))

    def test_bootstrap_rejects_nonfinite_values_and_invalid_repeat_count(self):
        for pairs, repeats in [([dict(template_group="a", difference=float("nan"))], 2),
                               ([dict(template_group="a", difference=float("inf"))], 2),
                               ([], 0)]:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.analysis.cluster_bootstrap(pairs, repeats=repeats)

    def test_edit_success_requires_correct_original_and_variants(self):
        rows = [scored("a", correct=1), scored("a", variant="edited"),
                scored("a", variant="sham", correct=1), scored("b"),
                scored("b", variant="edited", correct=1),
                scored("b", variant="sham", correct=1),
                scored("c", correct=1), scored("c", variant="edited", correct=1),
                scored("c", variant="sham", correct=1),
                scored("not-editable", correct=1)]
        result = self.analysis.edit_consistency(rows, "m8", "R")
        self.assertEqual(result["original_edited"], dict(num=1, den=3, rate=1/3))
        self.assertEqual(result["original_sham"], dict(num=2, den=3, rate=2/3))
        self.assertEqual(result["triple"], dict(num=1, den=3, rate=1/3))

    def test_edit_rejects_missing_original_missing_sham_and_duplicate(self):
        original, edited, sham = (scored("a", variant=v) for v in ("original", "edited", "sham"))
        for rows in ([edited, sham], [original, edited],
                     [original, edited, sham, dict(sham)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.analysis.edit_consistency(rows, "m8", "R")


if __name__ == "__main__":
    unittest.main()
