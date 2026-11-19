"""Material handoff tests use tiny local fixtures, never human decisions or inference."""
import copy
import csv
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from meme_study.data import content_hash
from meme_study.reference_protocol import _validate_pair, pair_content_hash


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.scripts, self.remote_resources, self.text = [], [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.images.append(attrs['src'])
        if tag == 'script':
            self.scripts.append(attrs)
        if tag in {'img', 'script', 'link', 'iframe'}:
            target = attrs.get('src', attrs.get('href', ''))
            if target.startswith(('http:', 'https:', '//')):
                self.remote_resources.append(target)

    def handle_data(self, text):
        self.text.append(text)


class ReferenceMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.reference_materials'),
                             'pair material export/import is not implemented')
        from meme_study.reference_materials import export_pair_review, import_pair_review
        self.export, self.import_review = export_pair_review, import_pair_review
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.out = self.root / 'handoff'
        self.pairs = []
        metadata = []
        for number in range(3):
            variants = {}
            for index, variant in enumerate(('original', 'edited', 'sham')):
                name = f'{number}-{variant}.png'
                Image.new('RGB', (8, 6), (number * 70, index * 70, 40)).save(self.data / name)
                variants[variant] = {
                    'image_path': name,
                    'image_sha256': hashlib.sha256((self.data / name).read_bytes()).hexdigest(),
                    'cue_answers': ['no' if variant == 'edited' else 'yes'],
                    'relation_answers': ['smooth' if variant == 'edited' else 'barrier'],
                }
            self.pairs.append({
                'family_id': f'fixture-{number}', 'source_id': number + 1,
                'split': 'dev' if number == 0 else 'test', 'template_group': f't{number}',
                'cue_question': '是否有尖刺？', 'relation_question': '哪种关系有图像支持？',
                'options': [{'label': label, 'semantic_id': key, 'text': text} for label, key, text in
                            [('A', 'barrier', '没有锁却难接近'), ('B', 'smooth', '容易使用'),
                             ('C', 'other', '另一种关系'), ('D', 'insufficient', '证据不足')]],
                'variants': variants,
                'preparation': {'origin': 'fixture_only', 'source_bank': 'fixture.jsonl',
                                'source_bank_sha256': 'a' * 64,
                                'draft_notes': {'title': '临时测试材料',
                                                'proposed_edit': '移除尖刺',
                                                'proposed_sham': '保留尖刺，改变颜色',
                                                'risk': '<script>alert("not executable")</script>'}},
                'material_check': None,
            })
            metadata.append({'id': number + 1, 'image_path': f'images/image_{number + 1:04d}.png'})
        raw = ''.join(json.dumps(x) + '\n' for x in metadata).encode()
        source = self.data / 'candidates/raw'
        source.mkdir(parents=True)
        (source / 'memebench-metadata.jsonl').write_bytes(raw)
        (source / 'memebench-metadata.receipt.json').write_text(json.dumps({
            'source_url': 'https://huggingface.co/datasets/anonymous-neurips-2026/memebench/resolve/'
                          + 'b' * 40 + '/metadata.jsonl',
            'sha256': hashlib.sha256(raw).hexdigest(),
        }))

    def exported(self):
        return self.export(self.pairs, data_root=self.data, output_dir=self.out)

    def completed_csv(self):
        self.exported()
        with (self.out / 'review.csv').open(newline='', encoding='utf-8-sig') as source:
            rows = list(csv.DictReader(source))
        for row, decision in zip(rows, ('accept', 'revise', 'reject')):
            row.update(decision=decision, reviewer='Fixture tester, not a real reviewer',
                       notes='' if decision == 'accept' else 'Fixture-only reason')
        return rows

    def write_csv(self, rows, *, header=None):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=header or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        path = self.root / 'completed.csv'
        path.write_bytes(stream.getvalue().encode('utf-8-sig'))
        return path

    def test_export_copies_every_image_byte_and_leaves_decisions_and_inputs_unchanged(self):
        before = copy.deepcopy(self.pairs)
        result = self.exported()
        self.assertEqual(self.pairs, before)
        self.assertEqual(result['pair_count'], 3)
        self.assertEqual(result['image_count'], 9)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['human_decisions_created'], 0)
        with (self.out / 'review.csv').open(newline='', encoding='utf-8-sig') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 3)
        for row, pair in zip(rows, self.pairs):
            self.assertEqual(row['content_sha256'], pair_content_hash(pair))
            self.assertEqual([row[k] for k in ('decision', 'reviewer', 'notes')], ['', '', ''])
        for receipt in result['images']:
            source = self.data / receipt['source_path']
            self.assertEqual((self.out / receipt['asset_path']).read_bytes(), source.read_bytes())
        self.assertEqual(json.loads((self.out / 'pairs.json').read_text()), before)

    def test_offline_page_displays_questions_variant_gold_notes_and_attribution_safely(self):
        self.exported()
        parsed = Page()
        parsed.feed((self.out / 'index.html').read_text())
        self.assertEqual(len(parsed.images), 9)
        self.assertEqual(parsed.scripts, [])
        self.assertEqual(parsed.remote_resources, [])
        for name in parsed.images:
            self.assertTrue((self.out / name).is_file())
        visible = ' '.join(parsed.text)
        for value in ('是否有尖刺？', '没有锁却难接近', '容易使用', 'barrier', 'smooth',
                      '移除尖刺', '保留尖刺，改变颜色', '<script>alert', 'MemeBench', 'b' * 40):
            self.assertIn(value, visible)

    def test_tampered_image_is_rejected_before_any_export(self):
        (self.data / '0-edited.png').write_bytes((self.data / '0-original.png').read_bytes())
        with self.assertRaises(ValueError):
            self.exported()
        self.assertFalse(self.out.exists())

    def test_source_metadata_mismatch_is_rejected(self):
        (self.data / 'candidates/raw/memebench-metadata.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            self.exported()
        self.assertFalse(self.out.exists())

    def test_image_symlink_escape_is_rejected(self):
        original = self.data / '0-original.png'
        outside = self.root / 'outside.png'
        original.rename(outside)
        original.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.exported()
        self.assertFalse(self.out.exists())

    def test_existing_export_directory_is_never_overwritten(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.exported()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_import_preserves_all_decisions_binds_csv_and_only_accept_is_ready(self):
        rows = self.completed_csv()
        source = self.write_csv(rows)
        before = copy.deepcopy(self.pairs)
        target = self.root / 'imported'
        result = self.import_review(self.pairs, review_csv=source, output_dir=target)
        self.assertEqual(self.pairs, before)
        imported = json.loads((target / 'pairs.json').read_text())
        self.assertEqual([p['material_check']['decision'] for p in imported], ['accept', 'revise', 'reject'])
        self.assertEqual([_validate_pair(p) for p in imported], [True, False, False])
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(result['source_csv_sha256'], expected)
        self.assertEqual((target / 'source-review.csv').read_bytes(), source.read_bytes())
        for original, pair in zip(before, imported):
            check = pair.pop('material_check')
            self.assertEqual(check['content_sha256'], pair_content_hash(original))
            self.assertEqual(check['source_csv_sha256'], expected)
            self.assertEqual(check['role'], 'human')
            self.assertEqual(check['identity_verification'], 'not_performed')
            self.assertEqual(pair, {k: v for k, v in original.items() if k != 'material_check'})

    def test_import_rejects_blank_duplicate_missing_unknown_stale_or_ambiguous_rows(self):
        good = self.completed_csv()
        cases = []
        for key, value in [('decision', ''), ('decision', 'accept/revise'), ('reviewer', ''),
                           ('content_sha256', 'f' * 64), ('split', 'test')]:
            changed = copy.deepcopy(good)
            changed[0][key] = value
            cases.append(changed)
        cases.extend([good + [good[0]], good[:-1]])
        changed = copy.deepcopy(good)
        changed[0]['family_id'] = 'unknown'
        cases.append(changed)
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.import_review(self.pairs, review_csv=self.write_csv(rows),
                                       output_dir=self.root / 'invalid')
                self.assertFalse((self.root / 'invalid').exists())

    def test_import_rejects_revise_and_reject_without_notes(self):
        rows = self.completed_csv()
        for decision in ('revise', 'reject'):
            rows[0].update(decision=decision, notes='  ')
            with self.assertRaises(ValueError):
                self.import_review(self.pairs, review_csv=self.write_csv(rows),
                                   output_dir=self.root / 'invalid')

    def test_import_rejects_repeated_check_and_modified_question(self):
        rows = self.completed_csv()
        source = self.write_csv(rows)
        checked = copy.deepcopy(self.pairs)
        checked[0]['material_check'] = {'decision': 'revise', 'reviewer': 'earlier member'}
        changed = copy.deepcopy(self.pairs)
        changed[0]['cue_question'] = '另一个问题？'
        for pairs in (checked, changed):
            with self.assertRaises(ValueError):
                self.import_review(pairs, review_csv=source, output_dir=self.root / 'invalid')

    def test_import_rejects_duplicate_headers_and_extra_cells(self):
        rows = self.completed_csv()
        source = self.write_csv(rows)
        raw = source.read_text(encoding='utf-8-sig')
        for corrupted in (raw.replace('notes\n', 'notes,notes\n', 1), raw.replace('\n', ',extra\n', 1)):
            source.write_text(corrupted)
            with self.assertRaises(ValueError):
                self.import_review(self.pairs, review_csv=source, output_dir=self.root / 'invalid')

    def test_import_never_overwrites_existing_directory(self):
        source = self.write_csv(self.completed_csv())
        target = self.root / 'existing'
        target.mkdir()
        with self.assertRaises(FileExistsError):
            self.import_review(self.pairs, review_csv=source, output_dir=target)
        self.assertEqual(list(target.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
