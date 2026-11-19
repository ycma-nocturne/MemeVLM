"""Hand-calculated synthetic statistics; never experimental model results."""
import copy
import importlib.util
import unittest


def explanation(family, condition, counts=None, *, model='m', split='test',
                available=True, ocr=True, run_status='ok'):
    counts = counts or {'V': (1, 1), 'I': (0, 0), 'K': (0, 1), 'R': (0, 1)}
    dimensions = {}
    for dim in 'VIKR':
        covered, applicable = counts.get(dim, (0, 0))
        dimensions[dim] = {'covered': covered if available else None,
                           'applicable': applicable, 'uncertain': 0 if available else None,
                           'coverage': covered / applicable if available and applicable else None,
                           'uncertainty_rate': 0. if available and applicable else None}
    return {'job_id': f'explanation/{family}/{condition}/{model}', 'task_kind': 'explanation',
            'family_id': family, 'split': split, 'template_group': family, 'model': model,
            'condition': condition, 'ocr_available': ocr, 'run_status': run_status,
            'score': {'status': 'ok' if available else 'unavailable', 'dimensions': dimensions,
                      'failure_reason': None if available else 'judge JSON invalid', 'items': []}}


def pair(family, variant, *, cue=1, relation=1, semantic=None, status='ok',
         kind='pair', run_status='ok', model='m', split='test'):
    semantic = semantic or ('changed' if variant == 'edited' else 'original_meaning')
    score = {'status': status, 'cue_correct': cue, 'relation_correct': relation,
             'joint_correct': cue * relation, 'semantic_relation': semantic,
             'cue': 'yes', 'explanation': 'synthetic'}
    if status != 'ok':
        score.update(cue_correct=0, relation_correct=0, joint_correct=0,
                     semantic_relation=None, cue=None)
    return {'job_id': f'{kind}/{family}/{variant}/{model}', 'task_kind': kind,
            'family_id': family, 'split': split, 'template_group': family,
            'model': model, 'condition': variant, 'variant': variant,
            'run_status': run_status, 'score': score}


class ReferenceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_analysis'),
                             'family-paired analysis is not implemented')
        from meme_study.reference_analysis import analyze_reference_scores
        self.analyze = analyze_reference_scores

    def report(self, rows):
        return self.analyze(rows, seed=13, draws=100)

    def explanation_model(self, rows, model='m'):
        return self.report(rows)['splits']['test']['explanation']['models'][model]

    def paired_model(self, rows):
        return self.report(rows)['splits']['test']['paired']['models']['m']

    def test_empty_and_absent_dimensions_have_no_estimate_or_interval(self):
        empty = self.report([])
        self.assertEqual(empty['primary_split'], 'test')
        self.assertEqual(empty['splits']['test']['n_rows'], 0)
        dimension = self.explanation_model([explanation('f1', 'image_only')])[
            'conditions']['image_only']['dimensions']['I']
        self.assertEqual(dimension['denominator'], 0)
        self.assertIsNone(dimension['estimate'])
        self.assertIsNone(dimension['ci95'])
        self.assertEqual(dimension['unavailable_family_ids'], ['f1'])

    def test_unavailable_judges_are_na_and_family_means_do_not_pool_items(self):
        rows = [explanation('f1', 'image_only', {'V': (1, 1)}),
                explanation('f2', 'image_only', {'V': (1, 3)}),
                explanation('f3', 'image_only', {'V': (0, 2)}, available=False)]
        cond = self.explanation_model(rows)['conditions']['image_only']
        self.assertAlmostEqual(cond['dimensions']['V']['estimate'], 2 / 3)
        self.assertEqual(cond['dimensions']['V']['denominator'], 2)
        self.assertEqual(cond['dimensions']['V']['n_expected'], 3)
        self.assertEqual(cond['valid_score_coverage']['numerator'], 2)
        self.assertEqual(cond['valid_score_coverage']['denominator'], 3)
        self.assertEqual(cond['valid_score_coverage']['estimate'], 2 / 3)
        self.assertEqual(cond['counts']['score_status'], {'ok': 2, 'unavailable': 1})
        self.assertEqual(cond['dimensions']['V']['family_ids'], ['f1', 'f2'])

    def test_condition_pairs_use_shared_available_families_and_separate_k_r(self):
        rows = [explanation('f1', 'image_only', {'K': (0, 1), 'R': (1, 2)}),
                explanation('f1', 'hinted', {'K': (1, 1), 'R': (0, 2)}),
                explanation('f2', 'image_only', {'K': (0, 1), 'R': (0, 1)}),
                explanation('f2', 'hinted', available=False),
                explanation('f3', 'hinted', {'K': (1, 1), 'R': (1, 1)})]
        model = self.explanation_model(rows)
        gain = model['condition_differences']['hinted_minus_image_only']
        self.assertEqual(gain['K']['estimate'], 1.)
        self.assertEqual(gain['R']['estimate'], -.5)
        self.assertEqual(gain['K']['denominator'], 1)
        self.assertEqual(gain['K']['n_expected'], 3)
        self.assertIsNone(gain['K']['ci95'])
        self.assertEqual(model['hint_gains']['R_minus_K']['estimate'], -1.5)
        self.assertEqual(model['hint_gains']['directions']['K'],
                         {'improved': 1, 'unchanged': 0, 'worsened': 0, 'n_pairs': 1})
        self.assertEqual(model['conditions']['image_only']['counts']['missing_rows'], 1)

    def test_ocr_null_subgroup_does_not_mix_with_known_transcripts(self):
        rows = [explanation('f1', 'ocr_only', {'R': (1, 1)}, ocr=False),
                explanation('f2', 'ocr_only', {'R': (0, 1)}, ocr=True)]
        model = self.explanation_model(rows)
        self.assertEqual(model['ocr_subgroups']['unavailable']['family_ids'], ['f1'])
        self.assertEqual(model['ocr_subgroups']['available']['conditions']['ocr_only'][
            'dimensions']['R']['estimate'], 0.)
        self.assertEqual(model['ocr_subgroups']['unavailable']['conditions']['ocr_only'][
            'dimensions']['R']['estimate'], 1.)

    def test_model_differences_are_matched_and_test_never_includes_dev(self):
        rows = [explanation('f1', 'image_only', {'R': (1, 1)}, model='a'),
                explanation('f1', 'image_only', {'R': (0, 1)}, model='b'),
                explanation('f2', 'image_only', {'R': (0, 1)}, model='a'),
                explanation('d1', 'image_only', {'R': (0, 1)}, model='a', split='dev')]
        report = self.report(rows)
        diff = report['splits']['test']['explanation']['model_differences'][
            'b_minus_a']['conditions']['image_only']['R']
        self.assertEqual(diff['estimate'], -1.)
        self.assertEqual(diff['denominator'], 1)
        self.assertEqual(diff['n_expected'], 2)
        self.assertEqual(report['splits']['dev']['n_families'], 1)
        self.assertEqual(report['splits']['test']['n_families'], 2)

    def test_ten_triples_require_all_cues_relations_and_actual_sham_semantics(self):
        rows = [pair(f'f{i}', v) for i in range(10) for v in ('original', 'edited', 'sham')]
        # One wrong edited cue, one wrong edited relation, one independently
        # acceptable sham answer which does not preserve the original answer.
        rows[4] = pair('f1', 'edited', cue=0)
        rows[7] = pair('f2', 'edited', relation=0)
        rows[11] = pair('f3', 'sham', semantic='another_acceptable_relation')
        model = self.paired_model(rows)
        triple = model['triples']['all_three_joint']
        self.assertEqual((triple['numerator'], triple['denominator']), (7, 10))
        self.assertEqual(triple['estimate'], .7)
        self.assertEqual(model['triples']['sham_preservation']['estimate'], .9)
        self.assertEqual(model['variants']['edited']['cue_relation_discrepancy']['estimate'], .2)
        self.assertEqual(triple['family_ids'], [f'f{i}' for i in range(10)])

    def test_invalid_and_truncated_count_wrong_but_infrastructure_missing_stays_visible(self):
        rows = [pair('f1', 'original'), pair('f2', 'original', status='invalid'),
                pair('f3', 'original', status='truncated'),
                pair('f4', 'original', run_status='failed'), pair('f5', 'edited')]
        model = self.paired_model(rows)
        rate = model['variants']['original']['joint_correct']
        self.assertEqual((rate['numerator'], rate['denominator']), (1, 3))
        self.assertEqual(rate['n_expected'], 5)
        self.assertEqual(rate['unavailable_family_ids'], ['f4', 'f5'])
        counts = model['variants']['original']['counts']
        self.assertEqual(counts['run_status']['failed'], 1)
        self.assertEqual(counts['missing_rows'], 1)
        self.assertEqual(model['triples']['all_three_joint']['denominator'], 0)
        self.assertEqual(model['triples']['all_three_joint']['n_expected'], 5)

    def test_partial_triples_and_wrong_answers_do_not_turn_answer_flips_into_success(self):
        rows = [pair('complete', v) for v in ('original', 'edited', 'sham')]
        rows += [pair('partial', 'original'), pair('partial', 'edited')]
        rows += [pair('flip', 'original', cue=0, relation=0, semantic='wrong'),
                 pair('flip', 'edited', cue=0, relation=0, semantic='other_wrong'),
                 pair('flip', 'sham', cue=0, relation=0, semantic='wrong')]
        model = self.paired_model(rows)
        triple = model['triples']['all_three_joint']
        self.assertEqual((triple['numerator'], triple['denominator']), (1, 2))
        self.assertEqual(triple['unavailable_family_ids'], ['partial'])
        self.assertEqual(model['triples']['complete_family_ids'], ['complete', 'flip'])

    def test_permutation_compares_semantic_ids_and_requires_two_valid_answers(self):
        rows = [pair('f1', 'original'), pair('f1', 'original', kind='permutation'),
                pair('f2', 'original'),
                pair('f2', 'original', kind='permutation', semantic='changed'),
                pair('f3', 'original', status='invalid'),
                pair('f3', 'original', kind='permutation', status='invalid'),
                pair('f4', 'original', kind='permutation')]
        perm = self.paired_model(rows)['permutations']
        self.assertEqual(perm['semantic_agreement']['estimate'], .5)
        self.assertEqual(perm['semantic_agreement']['denominator'], 2)
        self.assertEqual(perm['semantic_agreement']['n_expected'], 4)
        self.assertEqual(perm['complete_answer_pair_family_ids'], ['f1', 'f2', 'f3'])
        self.assertEqual(perm['valid_answer_pair_family_ids'], ['f1', 'f2'])
        self.assertEqual(perm['semantic_agreement']['contributions'][0]['job_ids'],
                         ['pair/f1/original/m', 'permutation/f1/original/m'])

    def test_bootstrap_uses_family_clusters_seed_and_deterministic_input_order(self):
        rows = [explanation('f1', 'image_only', {'V': (0, 1)}),
                explanation('f2', 'image_only', {'V': (1, 1)})]
        for row in rows:
            row['template_group'] = 'shared'
        a = self.report(rows)
        b = self.report(list(reversed(rows)))
        self.assertEqual(a, b)
        metric = a['splits']['test']['explanation']['models']['m'][
            'conditions']['image_only']['dimensions']['V']
        self.assertEqual(metric['n_clusters'], 2)
        self.assertEqual(metric['ci95'], [0., 1.])
        self.assertEqual(a['seed'], 13)
        self.assertEqual(a['draws'], 100)

    def test_rejects_duplicates_split_leakage_and_mislabeled_conditions(self):
        a = explanation('f1', 'image_only')
        cases = [[a, copy.deepcopy(a)]]
        duplicate = copy.deepcopy(a); duplicate['job_id'] = 'different_job'
        cases.append([a, duplicate])
        mixed = explanation('f1', 'hinted', split='dev'); cases.append([a, mixed])
        group = explanation('f2', 'hinted', split='dev'); group['template_group'] = 'f1'
        cases.append([a, group])
        mixed_group = explanation('f1', 'hinted'); mixed_group['template_group'] = 'other'
        cases.append([a, mixed_group])
        wrong = pair('f1', 'original'); wrong['condition'] = 'image_only'; cases.append([wrong])
        wrong = pair('f1', 'edited', kind='permutation'); cases.append([wrong])
        wrong = explanation('f1', 'original'); cases.append([wrong])
        wrong = copy.deepcopy(a); wrong['score']['dimensions']['V']['coverage'] = float('nan')
        cases.append([wrong])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.report(rows)
        for bad in (0, -1, True, 2.5):
            with self.subTest(draws=bad), self.assertRaises(ValueError):
                self.analyze([], draws=bad)

    def test_item_repair_and_harm_are_paired_by_reference_item_and_family(self):
        before = explanation('f1', 'image_only', {'K': (1, 2)})
        after = explanation('f1', 'hinted', {'K': (1, 2)})
        before['score']['items'] = [
            {'item_id': 'k1', 'dimension': 'K', 'verdict': 'covered'},
            {'item_id': 'k2', 'dimension': 'K', 'verdict': 'uncertain'}]
        after['score']['items'] = [
            {'item_id': 'k2', 'dimension': 'K', 'verdict': 'covered'},
            {'item_id': 'k1', 'dimension': 'K', 'verdict': 'not_covered'}]
        item = self.explanation_model([before, after])['hint_gains']['item_repair_harm']['K']
        self.assertEqual(item['repair']['estimate'], 1.)
        self.assertEqual(item['harm']['estimate'], 1.)
        self.assertEqual(item['repair']['denominator'], 1)
        self.assertEqual(item['source_item_counts'], {'repaired': 1, 'repairable': 1,
                                                      'harmed': 1, 'harmable': 1})


if __name__ == '__main__':
    unittest.main()
