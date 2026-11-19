"""Immutable, content-bound research releases; no fabricated review decisions.

Review records are checked for declared human roles and current content hashes.
This is not human identity authentication. Images must decode as static pixels
with matching family encoding/dimensions; this does not establish visual validity.
"""
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

from .data import build_jobs, compile_cases, content_hash, design_summary, validate_families
from .image_io import decode_image_receipt


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _check_config(config):
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    models = config.get('models')
    if not isinstance(models, dict) or set(models) != {'4B', '8B'}:
        raise ValueError('models must contain exactly 4B and 8B')
    for alias, model in models.items():
        if not isinstance(model, dict) or model.get('checkpoint') != f'Qwen/Qwen3-VL-{alias}-Instruct':
            raise ValueError(f'{alias}: incorrect checkpoint')
        if not re.fullmatch(r'[0-9a-f]{40}', str(model.get('revision', ''))):
            raise ValueError(f'{alias}: revision must be a pinned immutable commit SHA')
    if (not isinstance(config.get('workflows'), list)
            or len(config['workflows']) != 3 or set(config['workflows']) != {'D', 'R', 'V'}):
        raise ValueError('workflows must contain exactly D, R, V')
    if type(config.get('attempt_cap')) is not int or config['attempt_cap'] != 4180:
        raise ValueError('attempt_cap must equal 4180')


def _check_design(families, permutation_family_ids, scope):
    if scope not in {'full', 'pilot'}:
        raise ValueError('scope must be full or pilot')
    if not isinstance(families, list):
        raise ValueError('families must be a list')
    if not isinstance(permutation_family_ids, list):
        raise ValueError('permutation family IDs must be a list')
    if scope == 'full' and len(families) != 60:
        raise ValueError('full release requires 60 families')
    if scope == 'pilot' and not 6 <= len(families) <= 10:
        raise ValueError('pilot release requires 6 to 10 development families')
    errors = validate_families(families, require_reviews=True)
    if errors:
        raise ValueError('\n'.join(errors))
    splits = Counter(family['split'] for family in families)
    if scope == 'pilot':
        if set(splits) != {'dev'}:
            raise ValueError('pilot release requires all families in dev split')
        if permutation_family_ids:
            raise ValueError('pilot release has no permutation controls')
    else:
        if splits != {'dev': 20, 'test': 40}:
            raise ValueError('full release requires 20 dev and 40 test families')
        edits = Counter(family['split'] for family in families if 'edited' in family['variants'])
        if edits != {'dev': 10, 'test': 20}:
            raise ValueError('full release requires 10 dev and 20 test editable families')
        if len(permutation_family_ids) != 20:
            raise ValueError('full release requires 20 permutation families')
        for split, expected in [('dev', 5), ('test', 10)]:
            positions = Counter(family['option_order'].index(family['variants']['original']['gold_option'])
                                for family in families if family['split'] == split)
            if positions != {position: expected for position in range(4)}:
                raise ValueError(f'full release requires balanced original gold positions within {split}')
        edited_gold = [family['variants']['edited']['gold_option'] for family in families
                       if 'edited' in family['variants']]
        if edited_gold and all(answer == 'none' for answer in edited_gold):
            raise ValueError('edited gold answers must not all be none')
    test_ids = {family['family_id'] for family in families if family['split'] == 'test'}
    if len(set(permutation_family_ids)) != len(permutation_family_ids) or not set(permutation_family_ids) <= test_ids:
        raise ValueError('permutation IDs must be distinct test families')


def _safe_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts):
        raise ValueError('file path must remain relative to its root')
    try:
        path = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'missing or invalid file path: {relative}') from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'file path escapes root or is not a regular file: {relative}')
    return path


def _images(families, data_root):
    root = Path(data_root).resolve(strict=True)
    receipts, observed = [], {}
    for family in sorted(families, key=lambda row: row['family_id']):
        family_encoding_size = None
        for name, variant in sorted(family['variants'].items()):
            relative = variant['image_path']
            image = _safe_file(root, relative).read_bytes()
            receipt = decode_image_receipt(image, source_name=relative, expected_sha256=variant['sha256'])
            encoding_size = (receipt['detected_format'], receipt['width'], receipt['height'])
            if family_encoding_size is not None and encoding_size != family_encoding_size:
                raise ValueError(f"{family['family_id']}: family image encoding and dimensions must match")
            family_encoding_size = encoding_size
            digest = receipt['sha256']
            if digest in observed:
                if observed[digest] != family['split']:
                    raise ValueError('identical image hash crosses splits')
                raise ValueError('each variant requires a distinct image hash')
            observed[digest] = family['split']
            receipts.append(dict(receipt, family_id=family['family_id'],
                                 variant=name, split=family['split'], image_path=relative))
    return receipts


def _code_snapshot(code_root):
    try:
        root = Path(code_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError('code root does not exist') from exc
    if not root.is_dir():
        raise ValueError('code root must be a directory')
    files = {}
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        name = relative.as_posix()
        files[name] = _sha(_safe_file(root, name).read_bytes())
    if not files:
        raise ValueError('code root contains no Python source files')
    return str(root), files, content_hash(files)


def _payloads(families, config, permutation_family_ids):
    cases, gold = compile_cases(families, permutation_family_ids=permutation_family_ids)
    jobs = build_jobs(cases, sorted(config['models']), config['workflows'])
    values = {
        'families.jsonl': sorted(families, key=lambda row: row['family_id']),
        'cases.jsonl': cases,
        'gold.jsonl': [dict(case_id=case_id, **answer) for case_id, answer in sorted(gold.items())],
        'jobs.jsonl': jobs,
    }
    return {name: ''.join(_json(row) + '\n' for row in rows).encode('utf-8')
            for name, rows in values.items()}, design_summary(cases, jobs)


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(staging, destination):
    """Atomically rename with an exclusive destination, including empty dirs.

    Plain os.rename can replace an existing empty directory. The native
    no-replace flags avoid that race on the supported macOS/Linux platforms.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staging), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError('atomic no-replace directory publication requires macOS or Linux rename support')
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def freeze_release(families, config, data_root, output_dir, *, permutation_family_ids,
                   code_root, scope='full'):
    """Publish a new complete directory or refuse; never overwrite a release.

    ``full`` requires the exact 60-family design. ``pilot`` accepts 6–10
    reviewed development families and no permutation controls. Both bind
    actual image bytes, declared human reviews, model commits and code.
    """
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'release already exists: {destination}')
    # Detach input dictionaries so hashes cannot diverge during generation.
    families, config, permutation_family_ids = json.loads(_json([families, config, permutation_family_ids]))
    _check_config(config)
    _check_design(families, permutation_family_ids, scope)
    images = _images(families, data_root)
    code_root, code_files, code_sha256 = _code_snapshot(code_root)
    payloads, summary = _payloads(families, config, permutation_family_ids)
    if scope == 'full' and (summary['final_responses'], summary['planned_calls']) != (2360, 3800):
        raise ValueError('compiled release differs from the full 2360-output/3800-call design')
    manifest = {
        'schema_version': 1, 'scope': scope, 'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'models': config['models'], 'summary': summary,
        'permutation_family_ids': permutation_family_ids,
        'artifact_sha256': {name: _sha(data) for name, data in payloads.items()},
        'images': images, 'code_root': code_root, 'code_files': code_files,
        'code_sha256': code_sha256,
        'review_validation': 'Declared distinct human roles and current content hashes; identity not authenticated.',
        'image_validation': 'SHA-256, static pixel decoding, and matched family encoding/dimensions; semantic validity is not established.',
    }
    manifest['manifest_sha256'] = content_hash(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    try:
        for name, data in dict(payloads, **{'manifest.json': (_json(manifest) + '\n').encode('utf-8')}).items():
            with (staging / name).open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(staging)
        # Re-read source bytes/code before the publication boundary.
        if _images(families, data_root) != images or _code_snapshot(code_root)[1] != code_files:
            raise ValueError('source data or code changed while preparing the release')
        _publish(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def validate_release(release_dir, data_root, *, code_root=None):
    """Recheck artifacts, family reviews, image bytes and the bound source code.

    A manifest checksum detects accidental/manual edits, not an adversary who
    recomputes every hash. Keep a trusted copy of the manifest for provenance.
    code_root selects a relocated checkout; its complete relative source list
    and content hashes must match. The original manifest is never rewritten.
    """
    try:
        root = Path(release_dir).resolve(strict=True)
        manifest = json.loads(_safe_file(root, 'manifest.json').read_text(encoding='utf-8'))
        digest = manifest.get('manifest_sha256')
        if digest != content_hash({key: value for key, value in manifest.items() if key != 'manifest_sha256'}):
            raise ValueError('manifest checksum mismatch')
        if manifest.get('schema_version') != 1:
            raise ValueError('unsupported release schema')
        artifacts = manifest['artifact_sha256']
        expected_names = {'families.jsonl', 'cases.jsonl', 'gold.jsonl', 'jobs.jsonl'}
        if set(artifacts) != expected_names:
            raise ValueError('release artifact list differs from required files')
        raw = {}
        for name, digest in artifacts.items():
            raw[name] = _safe_file(root, name).read_bytes()
            if _sha(raw[name]) != digest:
                raise ValueError(f'artifact checksum mismatch: {name}')
        families = [json.loads(line) for line in raw['families.jsonl'].decode('utf-8').splitlines()]
        config = manifest['config']
        _check_config(config)
        _check_design(families, manifest['permutation_family_ids'], manifest['scope'])
        if manifest['models'] != config['models']:
            raise ValueError('model config differs from manifest')
        if _images(families, data_root) != manifest['images']:
            raise ValueError('image receipts differ from frozen manifest')
        _, code_files, code_sha256 = _code_snapshot(manifest['code_root'] if code_root is None else code_root)
        if code_files != manifest['code_files'] or code_sha256 != manifest['code_sha256']:
            raise ValueError('code checksum differs from frozen manifest')
        regenerated, summary = _payloads(families, config, manifest['permutation_family_ids'])
        if raw != regenerated or summary != manifest['summary']:
            raise ValueError('compiled artifacts differ from reviewed materials/config')
        return manifest
    except (OSError, KeyError, TypeError, UnicodeError, AttributeError) as exc:
        raise ValueError(f'invalid or incomplete release: {exc}') from exc
