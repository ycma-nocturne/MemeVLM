"""Verify AI recommendation adoption with synthetic pair materials."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from meme_study.reference_ai_review import FIELDS, MODEL, import_ai_review
from meme_study.reference_protocol import compile_pair_jobs, pair_content_hash
from test_reference_protocol import pairs, records


class AIReviewTests(unittest.TestCase):
    def setUp(self):
        fixture_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fixture_dir.cleanup)
        root = Path(fixture_dir.name)
        source_records = records()
        source_ids = {record['family_id']: record['source_id'] for record in source_records}
        original = pairs(source_records)
        for pair in original:
            pair['source_id'] = source_ids[pair['family_id']]
            pair['preparation'] = {'origin': 'synthetic fixture'}
        corrected = copy.deepcopy(original)
        self.corrected_source_id = original[0]['source_id']
        option = corrected[0]['options'][0]
        change = {
            'scope': 'B',
            'source_id': self.corrected_source_id,
            'field': f'options[semantic_id={option["semantic_id"]}].text',
            'before': option['text'],
            'after': '入口虽然开放，接近仍有障碍',
            'reason': 'Clarify the relation option.',
        }
        option['text'] = change['after']
        self.original_pairs = root / 'reviewed-pairs.json'
        self.current_pairs = root / 'pairs.json'
        self.review_csv = root / 'recommendations.csv'
        self.candidates = root / 'candidates.json'
        self.notes = root / 'notes.txt'
        self.change_record = root / 'change.json'
        self.original_pairs.write_text(json.dumps(original), encoding='utf-8')
        self.current_pairs.write_text(json.dumps(corrected), encoding='utf-8')
        self.candidates.write_text(json.dumps({'changes': [change]}), encoding='utf-8')
        self.notes.write_text(f'{self.corrected_source_id} {option["semantic_id"]}\n', encoding='utf-8')
        self.change_record.write_text(json.dumps({'pair_change': change}), encoding='utf-8')
        with self.review_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for pair in original:
                revise = pair['source_id'] == self.corrected_source_id
                decision = 'revise' if revise else 'accept'
                writer.writerow({
                    'family_id': pair['family_id'], 'title': 'Synthetic pair', 'split': pair['split'],
                    'content_sha256': pair_content_hash(pair), 'ai_model': MODEL,
                    'initial_ai_recommendation': decision, 'ai_recommendation': decision,
                    'confidence': 'high', 'suggested_action': 'Clarify wording' if revise else '',
                    'review_type': 'AI-assisted recommendation, not human acceptance',
                })

    def adopt(self, tmp, **overrides):
        args = dict(review_csv=self.review_csv, reviewed_pairs=self.original_pairs,
                    current_pairs=self.current_pairs, output_dir=Path(tmp)/'out',
                    correction_candidates=self.candidates, correction_notes=self.notes,
                    change_record=self.change_record)
        args.update(overrides)
        import_ai_review(**args)
        return json.loads((args['output_dir']/'pairs.json').read_text())

    def test_reviews_produce_110_ready_calls_with_ai_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            ps = self.adopt(tmp)
            jobs, gold = compile_pair_jobs(ps, ['m4', 'm8'])
            self.assertEqual(len(jobs), 110)
            self.assertTrue(all(j['readiness'] == 'ready' for j in jobs))
            self.assertTrue(all(p['material_check']['role'] == 'ai' for p in ps))
            fixed = next(p for p in ps if p['source_id'] == self.corrected_source_id)
            self.assertEqual(fixed['material_check']['decision'], 'adopted_correction_applied')
            self.assertEqual(fixed['material_check']['prior_recommendation'], 'revise')
            original = json.loads(self.current_pairs.read_text())
            before, bgold = compile_pair_jobs(original, ['m4','m8'], require_material_checks=False)
            self.assertEqual([j['job_id'] for j in before], [j['job_id'] for j in jobs])
            self.assertEqual([j['messages'] for j in before], [j['messages'] for j in jobs])
            self.assertEqual(bgold, gold)
            human = copy.deepcopy(ps)
            for p in human:
                p['material_check'] = {'decision':'accept', 'role':'human', 'reviewer':'member',
                                       'content_sha256':pair_content_hash(p)}
            human_jobs, _ = compile_pair_jobs(human, ['m4', 'm8'])
            self.assertEqual([j['input_sha256'] for j in jobs], [j['input_sha256'] for j in human_jobs])

    def test_stale_evidence_and_current_content_are_unready(self):
        with tempfile.TemporaryDirectory() as tmp:
            ps = self.adopt(tmp)
            ps[0]['cue_question'] += ' changed'
            with self.assertRaises(ValueError): compile_pair_jobs(ps, ['m'])
            ps = self.adopt_in_other_dir(tmp)
            ps[0]['material_check']['evidence']['review_csv']['sha256'] = '0'*64
            with self.assertRaises(ValueError): compile_pair_jobs(ps, ['m'])

    def adopt_in_other_dir(self, tmp):
        return self.adopt(tmp, output_dir=Path(tmp)/'other')

    def test_unresolved_revise_and_reject_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.adopt(tmp, correction_candidates=None, correction_notes=None, change_record=None)
            data = self.review_csv.read_text(encoding='utf-8').replace(',accept,accept,', ',reject,reject,', 1)
            altered = Path(tmp)/'reject.csv'; altered.write_text(data)
            with self.assertRaises(ValueError): self.adopt(tmp, review_csv=altered)

    def test_human_check_remains_supported(self):
        p = json.loads(self.current_pairs.read_text())[0]
        p['material_check'] = {'decision':'accept','role':'human','reviewer':'member',
                               'content_sha256':pair_content_hash(p)}
        self.assertTrue(compile_pair_jobs([p], ['m'])[0][0]['readiness'] == 'ready')

if __name__ == '__main__': unittest.main()
