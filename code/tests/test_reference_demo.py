"""Offline export boundary checks with temporary images and fixture outputs only."""
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from meme_study.data import content_hash
from meme_study.reference_analysis import analyze_reference_scores
from meme_study.reference_scoring import score_checklist, score_pair


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.scripts, self.links = [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.images.append(attrs['src'])
        if tag == 'script':
            self.scripts.append(attrs)
        if tag == 'link':
            self.links.append(attrs)


class ReferenceDemoTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_demo'),
                             'offline reference demo is not implemented')
        from meme_study.reference_demo import export_demo
        self.export = export_demo
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.out = self.root / 'demo'
        self.rows = []
        self.selections = []
        for number in range(3):
            family = f'f{number}'
            original = f'{family}-public.png'
            Image.new('RGB', (4, 3), (number * 60, 20, 90)).save(self.data / original)
            public_sha = hashlib.sha256((self.data / original).read_bytes()).hexdigest()
            source = {'dataset': 'Fixture public reference', 'source_id': number + 1,
                      'revision': 'a' * 40, 'url': 'https://example.org/source',
                      'annotation_origin': 'inherited_public'}
            for condition in ('image_only', 'hinted', 'ocr_only'):
                text = 'Fixture response <script>alert("x")</script> visual clue'
                checklist = [{'item_id': 'V1', 'dimension': 'V', 'text': 'visual clue'},
                             {'item_id': 'R1', 'dimension': 'R', 'text': '<img src=x onerror=bad>'}]
                judge = json.dumps({'items': [
                    {'item_id': 'V1', 'verdict': 'covered', 'evidence': ['visual clue']},
                    {'item_id': 'R1', 'verdict': 'uncertain', 'evidence': []}]})
                row = self.base(family, 'explanation', condition, text, source)
                row.update(image_path=None if condition == 'ocr_only' else original,
                           image_sha256=None if condition == 'ocr_only' else public_sha,
                           reference_image_path=original, reference_image_sha256=public_sha,
                           image_provenance={'path': original, 'sha256': public_sha, 'adapted': False},
                           reference={'explanation': 'Inherited <reference>', 'checklist': checklist},
                           ocr_available=True, ocr_text='Fixture OCR', hint_text='Fixture factual hint',
                           judge_status='ok', score=score_checklist(text, checklist, judge))
                self.rows.append(row)
            for variant in ('original', 'edited', 'sham'):
                filename = f'{family}-{variant}.png'
                Image.new('RGB', (4, 3), (number * 60, 40 + len(variant), 100)).save(self.data / filename)
                digest = hashlib.sha256((self.data / filename).read_bytes()).hexdigest()
                options = [{'label': label, 'semantic_id': f'relation-{label}', 'text': f'Option {label}'}
                           for label in 'ABCD']
                gold = {'cue_answers': ['yes'], 'relation_answers': ['relation-A']}
                text = json.dumps({'cue': 'yes', 'relation': 'B' if variant == 'edited' else 'A',
                                   'explanation': 'Fixture paired response'})
                row = self.base(family, 'pair', variant, text, source)
                row.update(variant=variant, image_path=filename, image_sha256=digest,
                           reference_image_path=original, reference_image_sha256=public_sha,
                           image_provenance={'scope': 'B cleaned project triple', 'variant': variant,
                                             'path': filename, 'sha256': digest, 'preparation': None},
                           cue_question='Visible cue?', relation_question='Which relation?',
                           options=options, gold=gold, score=score_pair(text, options, gold))
                self.rows.append(row)
            for kind, condition in (('explanation', 'hinted'), ('pair', 'edited')):
                self.selections.append({'family_id': family, 'model': 'fixture-model',
                                        'task_kind': kind, 'condition': condition,
                                        'selection_reason': 'Explicit fixture comparison, not ranked by score'})

    @staticmethod
    def base(family, kind, condition, text, source):
        return {'job_id': f'{kind}/{family}/{condition}/fixture-model', 'family_id': family,
                'template_group': family, 'model': 'fixture-model', 'split': 'test',
                'task_kind': kind, 'condition': condition, 'run_status': 'ok',
                'source_input_sha256': 'b' * 64, 'candidate_output_sha256': 'c' * 64,
                'candidate_text': text, 'candidate_truncated': False, 'source': source}

    def run_export(self, **kwargs):
        return self.export(kwargs.pop('rows', self.rows), data_root=self.data,
                           output_dir=kwargs.pop('output_dir', self.out),
                           selections=kwargs.pop('selections', self.selections), **kwargs)

    def test_portable_export_has_local_verified_images_actual_text_and_selection_reasons(self):
        result = self.run_export()
        manifest = json.loads((self.out / 'manifest.json').read_text())
        page = (self.out / 'index.html').read_text()
        parsed = Page()
        parsed.feed(page)
        self.assertEqual(result['selected_cases'], 6)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(manifest['counts']['rows'], 18)
        self.assertEqual(manifest['counts']['candidate_available'], 18)
        self.assertEqual(manifest['selections'], self.selections)
        self.assertTrue(parsed.images)
        self.assertEqual(parsed.scripts, [])
        self.assertEqual(parsed.links, [])
        self.assertNotIn('<script>alert', page)
        self.assertIn('&lt;script&gt;alert', page)
        self.assertIn('Explicit fixture comparison', page)
        self.assertIn('A public original', page)
        self.assertIn('B cleaned project', page)
        self.assertIn('Inherited public reference', page)
        self.assertIn('Cue correctness', page)
        self.assertIn('95% family bootstrap', page)
        for path in parsed.images:
            self.assertTrue(path.startswith('images/'))
            self.assertTrue((self.out / path).is_file())
        for name, digest in manifest['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(len((self.out / 'scored.jsonl').read_text().splitlines()), 18)
        self.assertNotIn(str(self.data), page)

    def test_missing_invalid_truncated_and_empty_outputs_remain_visible_without_zero_imputation(self):
        unavailable = self.rows[1]
        unavailable.update(score=None, judge_status='missing', candidate_truncated=True)
        self.rows[0].update(candidate_text='', score=None)
        self.rows[2].update(run_status='missing', candidate_text=None, candidate_output_sha256=None, score=None)
        bad = self.rows[4]
        bad['candidate_text'] = 'invalid fixture output'
        bad['score'] = score_pair(bad['candidate_text'], bad['options'], bad['gold'])
        self.run_export()
        manifest = json.loads((self.out / 'manifest.json').read_text())
        page = (self.out / 'index.html').read_text()
        self.assertEqual(manifest['counts']['candidate_available'], 17)
        self.assertEqual(manifest['counts']['candidate_truncated'], 1)
        self.assertEqual(manifest['counts']['invalid_pair_outputs'], 1)
        self.assertIn('Candidate was truncated', page)
        self.assertIn('Empty response', page)
        self.assertIn('Unavailable', page)
        self.assertIn('invalid fixture output', page)
        analysis = json.loads((self.out / 'analysis.json').read_text())
        metric = analysis['splits']['test']['explanation']['models']['fixture-model']['conditions']['hinted']['dimensions']['V']
        self.assertEqual(metric['denominator'], 2)
        self.assertEqual(metric['n_unavailable'], 1)

    def test_preview_requires_explicit_flag_and_is_distinctly_labelled(self):
        with self.assertRaisesRegex(ValueError, '6.*8'):
            self.run_export(selections=self.selections[:1])
        self.assertFalse(self.out.exists())
        result = self.run_export(selections=self.selections[:1], synthetic_preview=True)
        self.assertEqual(result['mode'], 'synthetic_preview')
        self.assertIn('SYNTHETIC PREVIEW', (self.out / 'index.html').read_text())

    def test_no_output_export_fails_even_when_preview_is_requested(self):
        for row in self.rows:
            row.update(run_status='missing', candidate_text=None, candidate_output_sha256=None, score=None)
        for preview in (False, True):
            with self.assertRaisesRegex(ValueError, 'no.*response'):
                self.run_export(synthetic_preview=preview)
            self.assertFalse(self.out.exists())

    def test_selection_is_explicit_unique_and_resolves_exact_existing_rows(self):
        for mutation in ('missing_reason', 'duplicate', 'unknown_model', 'unknown_field'):
            with self.subTest(mutation=mutation):
                selections = copy.deepcopy(self.selections)
                if mutation == 'missing_reason':
                    selections[0]['selection_reason'] = ''
                elif mutation == 'duplicate':
                    selections[1] = selections[0]
                elif mutation == 'unknown_model':
                    selections[0]['model'] = 'absent'
                else:
                    selections[0]['best_score'] = True
                with self.assertRaises(ValueError):
                    self.run_export(selections=selections)
                self.assertFalse(self.out.exists())

    def test_refuses_stale_analysis_and_raw_response_hash_mismatch(self):
        analysis = analyze_reference_scores(self.rows, seed=31, draws=50)
        analysis['splits']['test']['n_families'] = 999
        with self.assertRaisesRegex(ValueError, 'analysis'):
            self.run_export(analysis=analysis)
        self.rows[0]['candidate_text'] = 'tampered text'
        with self.assertRaisesRegex(ValueError, 'candidate.*hash|hash.*candidate'):
            self.run_export()
        self.assertFalse(self.out.exists())

    def test_rejects_image_path_traversal_escape_symlink_hash_change_and_nonimage(self):
        original = self.rows[0]['image_path']
        outside = self.root / 'outside.png'
        outside.write_bytes((self.data / original).read_bytes())
        (self.data / 'escape.png').symlink_to(outside)
        for value in ('../outside.png', str(outside), 'escape.png', 'C:\\outside.png'):
            with self.subTest(path=value):
                rows = copy.deepcopy(self.rows)
                rows[0]['image_path'] = value
                with self.assertRaises(ValueError):
                    self.run_export(rows=rows)
                self.assertFalse(self.out.exists())
        (self.data / original).write_bytes(b'not an image')
        with self.assertRaises(ValueError):
            self.run_export()
        for row in self.rows:
            if row.get('image_path') == original:
                row['image_sha256'] = hashlib.sha256(b'not an image').hexdigest()
            if row.get('reference_image_path') == original:
                row['reference_image_sha256'] = hashlib.sha256(b'not an image').hexdigest()
        with self.assertRaises(ValueError):
            self.run_export()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.demo-*')), [])

    def test_existing_directory_and_dangling_symlink_are_never_overwritten(self):
        self.out.mkdir()
        marker = self.out / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(marker.read_text(), 'keep')
        other = self.root / 'linked'
        other.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError):
            self.run_export(output_dir=other)
        self.assertTrue(other.is_symlink())

    def test_unattributed_or_unhashed_selected_image_cannot_be_published(self):
        for field in ('source', 'image_sha256'):
            rows = copy.deepcopy(self.rows)
            del rows[0][field]
            with self.assertRaises(ValueError):
                self.run_export(rows=rows)
            self.assertFalse(self.out.exists())

    def test_pair_correctness_and_truncation_are_bound_to_raw_output(self):
        for mutation in ('truncated', 'incorrect_score'):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.rows)
                if mutation == 'truncated':
                    rows[3]['candidate_truncated'] = True
                else:
                    rows[3]['score'].update(cue_correct=0, joint_correct=0)
                with self.assertRaisesRegex(ValueError, 'pair.*score|score.*pair'):
                    self.run_export(rows=rows, output_dir=self.root / mutation)
                self.assertFalse((self.root / mutation).exists())

    def test_selected_inputs_and_public_reference_materials_cannot_be_silently_omitted(self):
        for index, field in ((0, 'reference'), (1, 'hint_text'), (2, 'ocr_text'),
                             (3, 'cue_question'), (3, 'relation_question'), (3, 'options'), (3, 'gold')):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[index]['score'] = None
                del rows[index][field]
                with self.assertRaises(ValueError):
                    self.run_export(rows=rows, output_dir=self.root / field)
                self.assertFalse((self.root / field).exists())

    def test_provenance_must_match_image_bytes_role_and_original_status(self):
        for number, (index, field, value) in enumerate(((0, 'path', 'other.png'), (0, 'sha256', '0' * 64),
                    (0, 'adapted', True), (2, 'path', 'other.png'), (3, 'variant', 'edited'))):
            with self.subTest(field=field, index=index):
                rows = copy.deepcopy(self.rows)
                rows[index]['image_provenance'][field] = value
                out = self.root / f'provenance-{number}'
                with self.assertRaisesRegex(ValueError, 'provenance|original'):
                    self.run_export(rows=rows, output_dir=out)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
