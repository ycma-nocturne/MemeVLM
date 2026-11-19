"""Audited source corrections must be explicit, private and reproducible."""
import copy
import hashlib
import json
import unittest

import test_reference_cli as fixtures
from meme_study.data import content_hash
from meme_study.reference_execution import load_prepared
from meme_study.reference_protocol import compile_judge_jobs


class ReferenceErrataTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.ReferenceCliTests('runTest')
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.raw = [json.loads(s) for s in self.fx.metadata.read_text().splitlines()]
        for row in self.raw:
            row['visual']['ocr'] = 'null'
        self.fx.metadata.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in self.raw))
        self.fx.config['metadata']['sha256'] = hashlib.sha256(self.fx.metadata.read_bytes()).hexdigest()
        for row in self.raw:
            self.fx.hints[f'public-{row["id"]}']['source_record_sha256'] = content_hash(row)
        self.errata = {'schema': 'project_reference_errata_v1',
                      'metadata_sha256': self.fx.config['metadata']['sha256'],
                      'source_revision': self.fx.config['metadata']['revision'],
                      'entries': [self.entry('reference.explanation', self.raw[0]['checked_gt'], '仅用于评分的修订参考。'),
                                  self.entry('ocr_text', 'null', None)]}

    def entry(self, field, before, after):
        return {'source_id': 7, 'source_record_sha256': content_hash(self.raw[0]),
                'field': field, 'before': before, 'after': after,
                'reason': '明确核对过的源字段勘误。', 'evidence': 'fixture-review-7'}

    def prepare(self, errata=True, **kwargs):
        if errata:
            self.fx.config['source_errata'] = copy.deepcopy(self.errata)
        else:
            self.fx.config.pop('source_errata', None)
        self.fx.save_inputs()
        return self.fx.prepare(**kwargs)

    def test_apply_preserves_original_values_and_attribution(self):
        raw_bytes = self.fx.metadata.read_bytes()
        self.prepare()
        p = load_prepared(self.fx.out)
        row = p['records'][0]
        self.assertIsNone(row['ocr_text'])
        self.assertEqual(row['reference']['explanation'], '仅用于评分的修订参考。')
        self.assertEqual(row['source']['record_sha256'], content_hash(self.raw[0]))
        self.assertEqual(row['source']['annotation_origin'], 'inherited_public')
        self.assertEqual(row['provenance']['project_errata']['entries'], self.errata['entries'])
        self.assertEqual(row['provenance']['project_errata']['manifest_sha256'], content_hash(self.errata))
        self.assertEqual(self.fx.metadata.read_bytes(), raw_bytes)

    def test_only_explicit_missing_value_is_normalized_without_prompt_leakage(self):
        self.prepare()
        p = load_prepared(self.fx.out)
        self.assertIsNone(p['records'][0]['ocr_text'])
        self.assertEqual(p['records'][1]['ocr_text'], 'null')
        for job in p['explanations']:
            self.assertNotIn('仅用于评分的修订参考', json.dumps(job, ensure_ascii=False))
            self.assertNotIn('明确核对过的源字段勘误', json.dumps(job, ensure_ascii=False))
            if job['family_id'] == 'public-7':
                self.assertFalse(job['ocr_available'])
                if job['condition'] == 'ocr_only':
                    self.assertIn('未提供可用', job['messages'][0]['content'][0]['text'])
            else:
                self.assertTrue(job['ocr_available'])

    def test_reference_only_edit_changes_judge_not_candidate_inputs(self):
        self.prepare(errata=False)
        old = load_prepared(self.fx.out)
        self.errata['entries'] = self.errata['entries'][:1]
        second = self.fx.root / 'prepared2'
        self.prepare(output=second)
        new = load_prepared(second)
        self.assertEqual(old['explanations'], new['explanations'])
        parents = old['explanations']
        outputs = [{'job_id': j['job_id'], 'input_sha256': j['input_sha256'], 'status': 'ok',
                    'text': '单元测试中的合成候选回答。'} for j in parents]
        a = compile_judge_jobs(parents, outputs, old['records'], 'fixture-judge')
        b = compile_judge_jobs(parents, outputs, new['records'], 'fixture-judge')
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 6)
        self.assertIn('仅用于评分的修订参考', json.dumps(b[0], ensure_ascii=False))

    def test_legacy_preparation_still_loads_without_adaptation(self):
        self.prepare(errata=False)
        p = load_prepared(self.fx.out)
        self.assertEqual(p['records'][0]['ocr_text'], 'null')
        self.assertNotIn('project_errata', p['records'][0]['provenance'])

    def test_stale_source_digest_or_before_value_is_rejected_atomically(self):
        variants = []
        for key, value in [('metadata_sha256', '0' * 64), ('source_revision', '0' * 40)]:
            bad = copy.deepcopy(self.errata); bad[key] = value; variants.append(bad)
        for key, value in [('source_record_sha256', '0' * 64), ('before', '不是当前原文'), ('source_id', 999)]:
            bad = copy.deepcopy(self.errata); bad['entries'][0][key] = value; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                self.errata = bad
                with self.assertRaises(ValueError): self.prepare()
                self.fx.assert_no_package()

    def test_unsupported_fields_duplicate_entries_and_invalid_values_rejected(self):
        variants = []
        for key, value in [('field', 'reference.checklist'), ('after', ''), ('reason', ''), ('source_id', True)]:
            bad = copy.deepcopy(self.errata); bad['entries'][0][key] = value; variants.append(bad)
        bad = copy.deepcopy(self.errata); bad['entries'].append(copy.deepcopy(bad['entries'][0])); variants.append(bad)
        bad = copy.deepcopy(self.errata); bad['entries'][1]['after'] = '自动补写的字幕'; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                self.errata = bad
                with self.assertRaises(ValueError): self.prepare()
                self.fx.assert_no_package()

    def test_config_cannot_silently_claim_an_unapplied_correction(self):
        self.prepare(errata=False)
        config_path = self.fx.out / 'inputs/config.json'
        config = json.loads(config_path.read_text()); config['source_errata'] = self.errata
        config_path.write_text(json.dumps(config, ensure_ascii=False))
        self.rehash('inputs/config.json')
        with self.assertRaises(ValueError): load_prepared(self.fx.out)

    def rehash(self, name):
        path = self.fx.out / 'provenance.json'; p = json.loads(path.read_text())
        p['artifact_sha256'][name] = hashlib.sha256((self.fx.out / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(p))

    def test_effective_reference_and_receipt_are_checked_at_runtime(self):
        self.prepare()
        path = self.fx.out / 'reference_records.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['reference']['explanation'] = '未记录的新解释。'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        self.rehash('reference_records.jsonl')
        with self.assertRaises(ValueError): load_prepared(self.fx.out)

    def test_rehashed_stale_judge_dependencies_are_rejected(self):
        self.prepare()
        path = self.fx.out / 'judge_slots.jsonl'
        original = [json.loads(s) for s in path.read_text().splitlines()]
        for key, value in [('reference_sha256', '0' * 64), ('parent_input_sha256', '0' * 64),
                           ('model', 'wrong-fixture-judge')]:
            with self.subTest(key=key):
                rows = copy.deepcopy(original); rows[0][key] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                self.rehash('judge_slots.jsonl')
                with self.assertRaisesRegex(ValueError, 'judge'):
                    load_prepared(self.fx.out)
