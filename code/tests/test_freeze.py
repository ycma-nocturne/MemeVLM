"""Freeze tests use generated one-pixel PNGs and synthetic reviewer identities only."""
import copy
import hashlib
import importlib.util
import inspect
import json
import io
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from helpers import full_design
from meme_study.data import family_content_hash


def png_fixture(index):
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload))
    pixel = bytes([index % 256, index // 256, 17, 255])
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 1, 1, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0' + pixel)) + chunk(b'IEND', b''))


def approve_fixture(family):
    family['reviews'] = [
        {'reviewer': reviewer, 'role': 'human', 'decision': 'accept',
         'reviewed_at': '2026-10-24T09:00:00+08:00',
         'reviewed_content_sha256': family_content_hash(family)}
        for reviewer in ['SYNTHETIC_TEST_RATER_A', 'SYNTHETIC_TEST_RATER_B']]


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('meme_study.freeze'), 'freeze module is not implemented')
        from meme_study import freeze
        self.freeze = freeze
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'; (self.data / 'images').mkdir(parents=True)
        self.code = self.root / 'code'; self.code.mkdir()
        (self.code / 'fixture.py').write_text('# Synthetic code fixture, not project code.\n')
        self.release = self.root / 'release-v1'
        self.families = full_design()
        index = 0
        for family in self.families:
            for variant in family['variants'].values():
                index += 1
                image = png_fixture(index)
                (self.data / variant['image_path']).write_bytes(image)
                variant['sha256'] = hashlib.sha256(image).hexdigest()
            approve_fixture(family)
        self.permutations = [f'f{i:02d}' for i in range(20, 40)]
        self.config = {'models': {
            '4B': {'checkpoint': 'Qwen/Qwen3-VL-4B-Instruct', 'revision': '1' * 40},
            '8B': {'checkpoint': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '2' * 40}},
            'workflows': ['D', 'R', 'V'], 'attempt_cap': 4180}

    def freeze_now(self, **kwargs):
        args = dict(families=self.families, config=self.config, data_root=self.data,
                    output_dir=self.release, permutation_family_ids=self.permutations, code_root=self.code)
        args.update(kwargs)
        return self.freeze.freeze_release(**args)

    def test_full_release_is_reproducible_gold_separated_and_pixel_decoded(self):
        manifest = self.freeze_now()
        self.assertEqual(manifest['summary']['final_responses'], 2360)
        self.assertEqual(manifest['summary']['planned_calls'], 3800)
        self.assertEqual(manifest['summary']['core_cases'], 360)
        self.assertEqual(manifest['models'], self.config['models'])
        self.assertEqual(manifest['config']['attempt_cap'], 4180)
        self.assertEqual(len(manifest['images']), 120)
        self.assertTrue(all(image['decoded'] is True for image in manifest['images']))
        self.assertTrue(all((image['width'], image['height']) == (1, 1) for image in manifest['images']))
        self.assertEqual(set(manifest['artifact_sha256']),
                         {'families.jsonl', 'cases.jsonl', 'gold.jsonl', 'jobs.jsonl'})
        cases = [json.loads(line) for line in (self.release / 'cases.jsonl').read_text().splitlines()]
        gold = [json.loads(line) for line in (self.release / 'gold.jsonl').read_text().splitlines()]
        self.assertEqual(len(cases), 460)
        self.assertEqual({row['case_id'] for row in cases}, {row['case_id'] for row in gold})
        self.assertTrue(all('gold_option' not in case and 'reviews' not in case for case in cases))
        self.assertEqual(self.freeze.validate_release(self.release, self.data), manifest)

    def test_missing_stale_or_nonhuman_review_refuses_release(self):
        for change in ['missing', 'stale', 'machine']:
            families = copy.deepcopy(self.families)
            if change == 'missing': families[0]['reviews'] = []
            if change == 'stale': families[0]['caption_zh'] = 'changed after review'
            if change == 'machine': families[0]['reviews'][0]['role'] = 'ai'
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, 'review'):
                    self.freeze_now(families=families)
                self.assertFalse(self.release.exists())

    def test_exact_family_edit_and_permutation_design_is_required(self):
        with self.assertRaisesRegex(ValueError, '60|family|families'):
            self.freeze_now(families=self.families[:-1])
        with self.assertRaisesRegex(ValueError, '20|permutation'):
            self.freeze_now(permutation_family_ids=self.permutations[:-1])
        families = copy.deepcopy(self.families)
        families[0]['variants'] = {'original': families[0]['variants']['original']}
        approve_fixture(families[0])
        with self.assertRaisesRegex(ValueError, 'edit'):
            self.freeze_now(families=families)
        families = copy.deepcopy(self.families)
        families[19]['split'] = 'test'; approve_fixture(families[19])
        with self.assertRaisesRegex(ValueError, '20|split|dev'):
            self.freeze_now(families=families)

    def test_unsafe_relative_absolute_and_symlink_paths_are_rejected(self):
        original = self.families[0]['variants']['original']
        outside = self.root / 'outside.png'; outside.write_bytes(png_fixture(1))
        (self.data / 'images' / 'escape.png').symlink_to(outside)
        for path in ['../outside.png', str(outside), 'images/escape.png']:
            families = copy.deepcopy(self.families)
            families[0]['variants']['original']['image_path'] = path
            approve_fixture(families[0])
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, 'path|root|escape'):
                    self.freeze_now(families=families)
        self.assertTrue((self.data / original['image_path']).exists())

    def test_missing_changed_or_unrecognizable_image_is_rejected(self):
        image = self.data / self.families[0]['variants']['original']['image_path']
        saved = image.read_bytes(); image.unlink()
        with self.assertRaises(ValueError): self.freeze_now()
        image.write_bytes(png_fixture(999))
        with self.assertRaisesRegex(ValueError, 'checksum|hash'):
            self.freeze_now()
        image.write_bytes(b'not an image')
        self.families[0]['variants']['original']['sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
        approve_fixture(self.families[0])
        with self.assertRaisesRegex(ValueError, 'signature|image'):
            self.freeze_now()
        image.write_bytes(saved)

    def test_image_bytes_cannot_cross_splits_even_under_different_templates(self):
        dev = self.families[0]['variants']['original']
        test = self.families[20]['variants']['original']
        (self.data / test['image_path']).write_bytes((self.data / dev['image_path']).read_bytes())
        test['sha256'] = dev['sha256']; approve_fixture(self.families[20])
        with self.assertRaisesRegex(ValueError, 'cross.*split'):
            self.freeze_now()

    def test_exact_models_pinned_revisions_workflows_and_attempt_cap_required(self):
        changes = [dict(attempt_cap=4181), dict(workflows=['D', 'R'])]
        bad_model = copy.deepcopy(self.config['models']); bad_model['4B']['revision'] = 'main'
        changes.append(dict(models=bad_model))
        missing_model = copy.deepcopy(self.config['models']); del missing_model['8B']
        changes.append(dict(models=missing_model))
        for change in changes:
            with self.subTest(change=change):
                config = copy.deepcopy(self.config); config.update(change)
                with self.assertRaises(ValueError): self.freeze_now(config=config)
                self.assertFalse(self.release.exists())

    def test_existing_directory_is_never_overwritten_and_concurrent_publish_has_one_winner(self):
        self.release.mkdir(); marker = self.release / 'user-owned'; marker.write_text('preserve')
        with self.assertRaises(FileExistsError): self.freeze_now()
        self.assertEqual(marker.read_text(), 'preserve')
        target = self.root / 'concurrent-release'
        def attempt(_):
            try:
                self.freeze_now(output_dir=target)
                return 'published'
            except FileExistsError:
                return 'exists'
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(attempt, range(2))), ['published', 'exists'])
        self.freeze.validate_release(target, self.data)

    def test_validation_detects_artifact_manifest_image_and_code_changes(self):
        self.freeze_now()
        paths = [self.release / filename for filename in ['families.jsonl', 'cases.jsonl', 'gold.jsonl', 'jobs.jsonl']]
        paths.extend([self.release / 'manifest.json', self.code / 'fixture.py',
                      self.data / self.families[0]['variants']['original']['image_path']])
        for path in paths:
            saved = path.read_bytes()
            try:
                if path.name == 'manifest.json':
                    value = json.loads(saved); value['config']['attempt_cap'] = 9999
                    path.write_text(json.dumps(value))
                else:
                    path.write_bytes(saved + b'\nchanged')
                with self.subTest(path=path.name):
                    with self.assertRaises(ValueError): self.freeze.validate_release(self.release, self.data)
            finally:
                path.write_bytes(saved)
        self.freeze.validate_release(self.release, self.data)

    def test_code_must_exist(self):
        (self.code / 'fixture.py').unlink()
        with self.assertRaisesRegex(ValueError, 'code'):
            self.freeze_now()
        self.assertFalse(self.release.exists())

    def test_signature_only_corrupt_png_cannot_be_frozen(self):
        raw = b'\x89PNG\r\n\x1a\nnot actual image pixels'
        variant = self.families[0]['variants']['original']
        (self.data / variant['image_path']).write_bytes(raw)
        variant['sha256'] = hashlib.sha256(raw).hexdigest()
        approve_fixture(self.families[0])
        with self.assertRaisesRegex(ValueError, 'image|decode'):
            self.freeze_now()
        self.assertFalse(self.release.exists())

    def test_family_encoding_and_dimensions_must_match_actual_pixels(self):
        for image_format, dimensions in [('PNG', (2, 1)), ('JPEG', (1, 1))]:
            families = copy.deepcopy(self.families)
            image = Image.new('RGB', dimensions, (4, 5, 6))
            buffer = io.BytesIO(); image.save(buffer, format=image_format)
            raw = buffer.getvalue()
            variant = families[0]['variants']['edited']
            (self.data / variant['image_path']).write_bytes(raw)
            variant['sha256'] = hashlib.sha256(raw).hexdigest()
            approve_fixture(families[0])
            with self.subTest(image_format=image_format, dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, 'encoding|dimensions|size'):
                    self.freeze_now(families=families, output_dir=self.root / f'mismatch-{image_format}')
            original = self.families[0]['variants']['edited']
            (self.data / original['image_path']).write_bytes(png_fixture(2))

    def test_same_code_can_move_to_gpu_directory_without_changing_manifest(self):
        self.assertIn('code_root', inspect.signature(self.freeze.validate_release).parameters,
                      'release validation cannot select the relocated source directory')
        manifest = self.freeze_now()
        before = (self.release / 'manifest.json').read_bytes()
        relocated = self.root / 'gpu-checkout'
        shutil.copytree(self.code, relocated)
        shutil.rmtree(self.code)
        self.assertEqual(self.freeze.validate_release(self.release, self.data, code_root=relocated), manifest)
        self.assertEqual((self.release / 'manifest.json').read_bytes(), before)
        (relocated / 'fixture.py').write_text('# changed code must fail\n')
        with self.assertRaisesRegex(ValueError, 'code checksum'):
            self.freeze.validate_release(self.release, self.data, code_root=relocated)

    def test_relocated_code_file_set_must_match_even_if_existing_file_is_identical(self):
        self.assertIn('code_root', inspect.signature(self.freeze.validate_release).parameters,
                      'release validation cannot select the relocated source directory')
        self.freeze_now()
        relocated = self.root / 'gpu-checkout'
        shutil.copytree(self.code, relocated)
        (relocated / 'extra.py').write_text('# extra source\n')
        with self.assertRaisesRegex(ValueError, 'code checksum'):
            self.freeze.validate_release(self.release, self.data, code_root=relocated)

    def test_full_release_requires_balanced_original_gold_positions_in_each_split(self):
        families = copy.deepcopy(self.families)
        for family in families:
            family['option_order'] = ['o1', 'o2', 'o3', 'none']
            approve_fixture(family)
        with self.assertRaisesRegex(ValueError, 'balance|position'):
            self.freeze_now(families=families)
        self.assertFalse(self.release.exists())

    def test_overall_position_balance_cannot_hide_split_imbalance(self):
        families = copy.deepcopy(self.families)
        families[0]['option_order'], families[21]['option_order'] = (
            families[21]['option_order'], families[0]['option_order'])
        approve_fixture(families[0]); approve_fixture(families[21])
        with self.assertRaisesRegex(ValueError, 'balance|position'):
            self.freeze_now(families=families)

    def test_edit_gold_cannot_all_be_none(self):
        families = copy.deepcopy(self.families)
        for family in families:
            if 'edited' in family['variants']:
                family['variants']['edited']['gold_option'] = 'none'
                approve_fixture(family)
        with self.assertRaisesRegex(ValueError, 'none|edited'):
            self.freeze_now(families=families)
        self.assertFalse(self.release.exists())

    def test_pilot_allows_only_six_to_ten_reviewed_development_families(self):
        manifest = self.freeze_now(families=self.families[:6], permutation_family_ids=[], scope='pilot')
        self.assertEqual(manifest['scope'], 'pilot')
        self.assertEqual(manifest['summary']['core_test_cases'], 0)
        self.assertEqual(manifest['summary']['final_responses'], 288)
        self.assertEqual(manifest['summary']['planned_calls'], 480)
        self.freeze.validate_release(self.release, self.data)
        for families, permutations in [(self.families[:5], []), (self.families[:11], []),
                                       (self.families[20:26], []), (self.families[:6], self.permutations)]:
            with self.subTest(count=len(families), permutations=permutations):
                with self.assertRaises(ValueError):
                    self.freeze_now(families=families, permutation_family_ids=permutations,
                                    scope='pilot', output_dir=self.root / 'bad-pilot')


if __name__ == '__main__':
    unittest.main()
