"""Verify OCR errata against the shared study inputs."""
import json
import os
from collections import Counter
from pathlib import Path
import unittest

from meme_study.reference_data import import_references
from meme_study.reference_protocol import compile_explanation_jobs


class ReferenceStudyInputTests(unittest.TestCase):
    def test_reviewed_missing_transcripts_are_missing_in_actual_model_inputs(self):
        study_root = os.environ.get('MEMEVLM_STUDY_ROOT')
        if not study_root:
            self.skipTest('Set MEMEVLM_STUDY_ROOT to the shared study root.')
        root = Path(study_root).expanduser()
        study = root / 'data/study'
        config = json.loads((study / 'config.json').read_text())
        records = import_references(
            root / 'data/assets/candidates/raw/memebench-metadata.jsonl',
            json.loads((study / 'selection.json').read_text()), root / 'data/assets',
            expected_sha256=config['metadata']['sha256'],
            revision=config['metadata']['revision'], errata=config['source_errata'])
        reviewed_missing = {
            entry['source_id'] for entry in config['source_errata']['entries']
            if entry['field'] == 'ocr_text' and entry['after'] is None
        }
        missing = [record for record in records if record['ocr_text'] is None]
        self.assertEqual(len(reviewed_missing), 2)
        self.assertTrue(reviewed_missing <= {record['source_id'] for record in missing})
        self.assertEqual(len(missing), 7)
        self.assertEqual(Counter(record['split'] for record in missing), {'dev': 2, 'test': 5})
        families = {record['family_id'] for record in missing}
        jobs = compile_explanation_jobs(
            records, json.loads((study / 'hints.json').read_text()), config['model_ids'])
        for job in jobs:
            with self.subTest(condition=job['condition'], split=job['split']):
                self.assertEqual(job['ocr_available'], job['family_id'] not in families)
                if job['condition'] == 'ocr_only' and job['family_id'] in families:
                    text = job['messages'][0]['content'][0]['text']
                    self.assertNotIn('\nnull', text)
                    self.assertIn('未提供可用', text)
