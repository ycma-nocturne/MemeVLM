"""Hand-calculated offline fixtures for scientific reporting, never model results."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from meme_study.reference_scoring import score_checklist, score_pair


def fixture_rows():
    """Two test families and a separate dev family; unequal checklist lengths."""
    rows = []
    for family, split in [('f1', 'test'), ('f2', 'test'), ('d1', 'dev')]:
        for condition in ('image_only', 'hinted', 'ocr_only'):
            counts = {'V': (1, 1 if family != 'f2' else 3), 'I': (0, 0),
                      'K': (int(condition == 'hinted'), 1),
                      'R': (0 if condition == 'hinted' else 1, 2)}
            if split == 'dev':
                counts['V'] = (0, 1)
            checklist, items = [], []
            for dimension, (covered, total) in counts.items():
                for i in range(total):
                    item = f'{dimension}{i}'
                    checklist.append({'item_id': item, 'dimension': dimension, 'text': 'fixture fact'})
                    items.append({'item_id': item, 'verdict': 'covered' if i < covered else 'not_covered',
                                  'evidence': ['fixture'] if i < covered else []})
            text = 'synthetic fixture explanation'
            row = _base(family, split, 'explanation', condition, text)
            row.update(reference={'checklist': checklist}, judge_status='ok', ocr_available=True,
                       score=score_checklist(text, checklist, json.dumps({'items': items})))
            rows.append(row)
        for kind, variant in [('pair', 'original'), ('pair', 'edited'), ('pair', 'sham')] + (
                [('permutation', 'original')] if split == 'test' else []):
            options = [{'label': label, 'semantic_id': semantic, 'text': semantic}
                       for label, semantic in zip('ABCD', ['original', 'changed', 'other', 'insufficient'])]
            gold = {'cue_answers': ['yes'],
                    'relation_answers': ['changed' if variant == 'edited' else 'original']}
            text = ('invalid synthetic response' if family == 'f2' and variant == 'edited' else
                    json.dumps({'cue': 'yes', 'relation': 'B' if variant == 'edited' else 'A',
                                'explanation': 'synthetic fixture'}))
            row = _base(family, split, kind, variant, text)
            row.update(variant=variant, options=options, gold=gold,
                       score=score_pair(text, options, gold))
            rows.append(row)
    return rows


def _base(family, split, kind, condition, text):
    return {'job_id': f'{kind}/{family}/{condition}/m', 'family_id': family, 'split': split,
            'template_group': family, 'model': 'fixture_model%&', 'task_kind': kind,
            'condition': condition, 'run_status': 'ok', 'candidate_text': text,
            'candidate_truncated': False, 'candidate_output_sha256': 'a' * 64,
            'candidate_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'source_input_sha256': 'b' * 64}


class ReferenceReportingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_reporting'),
                             'scientific reference exporter is not implemented')
        from meme_study.reference_reporting import export_reference_results
        self.export = export_reference_results
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'report'
        self.rows = fixture_rows()

    def run_export(self, **kwargs):
        return self.export(kwargs.pop('rows', self.rows), output_dir=kwargs.pop('output_dir', self.out),
                           seed=13, draws=100, synthetic_preview=kwargs.pop('synthetic_preview', True), **kwargs)

    def test_tables_keep_family_weighting_na_denominators_and_separate_dev(self):
        manifest = self.run_export()
        with (self.out / 'test/all_metrics.csv').open() as stream:
            table = list(csv.DictReader(stream))
        chosen = [r for r in table if r['section'] == 'explanation_conditions' and
                  r['condition'] == 'image_only' and not r['subgroup']]
        visual = next(r for r in chosen if r['dimension'] == 'V')
        identity = next(r for r in chosen if r['dimension'] == 'I')
        self.assertIn('SYNTHETIC', visual.get('evidence_label', ''))
        self.assertAlmostEqual(float(visual['estimate']), 2 / 3)
        self.assertEqual((visual['denominator'], visual['n_expected'], visual['n_unavailable']), ('2', '2', '0'))
        self.assertEqual(identity['estimate'], 'NA')
        self.assertEqual((identity['denominator'], identity['n_unavailable']), ('0', '2'))
        self.assertEqual(identity['ci95_low'], 'NA')
        members = json.loads((self.out / 'metric_memberships.json').read_text())['metrics']
        metric = members[visual['membership_key']]
        self.assertEqual(metric['family_ids'], ['f1', 'f2'])
        self.assertEqual([x['value'] for x in metric['contributions']], [1., 1 / 3])
        with (self.out / 'dev/all_metrics.csv').open() as stream:
            dev = list(csv.DictReader(stream))
        self.assertEqual(next(r for r in dev if r['section'] == 'explanation_conditions' and
                             r['condition'] == 'image_only' and r['dimension'] == 'V' and
                             not r['subgroup'])['estimate'], '0.0')
        self.assertEqual(manifest['primary_split'], 'test')
        tex = (self.out / 'test/explanation_conditions.tex').read_text()
        self.assertIn(r'fixture\_model\%\&', tex)
        self.assertIn('NA', tex)

    def test_contrasts_and_joint_metrics_are_bound_to_their_analysis_memberships(self):
        manifest = self.run_export()
        analysis = json.loads((self.out / 'analysis.json').read_text())
        memberships = json.loads((self.out / 'metric_memberships.json').read_text())['metrics']
        with (self.out / 'test/all_metrics.csv').open() as stream:
            table = list(csv.DictReader(stream))
        kr = {r['metric']: r for r in table if r['section'] == 'hint_gains' and not r['subgroup']}
        self.assertEqual(float(kr['K']['estimate']), 1.)
        self.assertEqual(float(kr['R']['estimate']), -.5)
        self.assertEqual(float(kr['R_minus_K']['estimate']), -1.5)
        triples = next(r for r in table if r['metric'] == 'all_three_joint')
        self.assertEqual((float(triples['estimate']), triples['denominator']), (.5, '2'))
        for row in table:
            node = analysis
            for key in json.loads(row['analysis_path']):
                node = node[key]
            self.assertEqual(memberships[row['membership_key']]['contributions'], node['contributions'])
            self.assertEqual(row['estimate'], 'NA' if node['estimate'] is None else str(node['estimate']))
        for figure in manifest['figures']:
            for key in figure['metric_ids']:
                self.assertIn(key, memberships)
            if figure['status'] == 'written':
                for path in figure['paths']:
                    self.assertTrue((self.out / path).is_file())
                self.assertTrue(any(p.endswith('.pdf') for p in figure['paths']))
                self.assertTrue(any(p.endswith('.png') for p in figure['paths']))

    def test_pending_judges_and_missing_slots_require_explicit_partial_export(self):
        self.rows[0].update(judge_status='pending', score=None)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_export()
        self.assertFalse(self.out.exists())
        self.rows = self.rows[:-1]
        manifest = self.run_export(allow_partial=True)
        self.assertFalse(manifest['complete'])
        self.assertTrue(manifest['completeness']['nonterminal_jobs'])
        self.assertTrue(manifest['completeness']['missing_slots'])
        self.assertIn('PARTIAL', manifest['label'])

    def test_zero_inference_cannot_export_results_even_with_allow_partial(self):
        for row in self.rows:
            row.update(run_status='missing', candidate_text=None, candidate_text_sha256=None,
                       candidate_output_sha256=None, score=None)
            if row['task_kind'] == 'explanation':
                row['judge_status'] = 'not_applicable'
        with self.assertRaisesRegex(ValueError, 'no successful candidate'):
            self.run_export(allow_partial=True, synthetic_preview=False)
        self.assertFalse(self.out.exists())

    def test_partial_supplied_universe_is_not_published_as_complete_current_protocol(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_export(synthetic_preview=False)
        self.assertFalse(self.out.exists())

    def test_tampered_scores_and_checklist_item_bindings_are_rejected(self):
        for mutation in ('pair', 'dimension', 'item', 'candidate', 'judge_failed_with_score'):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.rows)
                if mutation == 'pair':
                    rows[3]['score'].update(cue_correct=0, joint_correct=0)
                elif mutation == 'dimension':
                    rows[0]['score']['dimensions']['V'].update(covered=0, coverage=0.)
                elif mutation == 'item':
                    rows[0]['score']['items'][0]['dimension'] = 'R'
                elif mutation == 'candidate':
                    rows[0]['candidate_text'] = 'changed'
                else:
                    rows[0]['judge_status'] = 'failed'
                with self.assertRaises(ValueError):
                    self.run_export(rows=rows)
                self.assertFalse(self.out.exists())

    def test_synthetic_label_and_hash_receipt_are_present_and_existing_output_is_safe(self):
        manifest = self.run_export()
        self.assertIn('SYNTHETIC', manifest['label'])
        self.assertEqual(manifest['model_calls'], 0)
        provenance = json.loads((self.out / 'provenance.json').read_text())
        self.assertEqual(provenance['input_sha256'],
                         hashlib.sha256((self.out / 'scored.jsonl').read_bytes()).hexdigest())
        self.assertIn('reference_reporting.py', provenance['code_sha256'])
        for name, digest in manifest['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            self.run_export()
        link = self.root / 'link'
        link.symlink_to(self.root / 'absent')
        with self.assertRaises(FileExistsError):
            self.run_export(output_dir=link)
        self.assertTrue(link.is_symlink())


if __name__ == '__main__':
    unittest.main()
