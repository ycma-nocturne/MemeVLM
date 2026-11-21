"""Validate the bilingual release and compile the English paired task."""

from collections import Counter
import hashlib
import io
from pathlib import Path
import re

from PIL import Image, ImageOps

from meme_study.data import content_hash
from meme_study.freeze import _safe_file
from meme_study.reference_scoring import _decode, score_pair


PROTOCOL = 'memevlm_english_pairs_v1'
VARIANTS = ('original', 'edited', 'sham')
SPLITS = ('train', 'validation', 'test')
SEED = 6000
MAX_TOKENS = 512


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'expected nonempty text for {label}')
    return value.strip()


def source_rows(root, split):
    path = root / f'{split}.jsonl'
    raw = path.read_bytes()
    rows = [_decode(line) for line in raw.decode('utf-8').split('\n') if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{path.name} must contain JSON objects')
    return rows, hashlib.sha256(raw).hexdigest()


def check_image(root, item, cache):
    path = _safe_file(root, item.get('image'))
    if path not in cache:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('paired evaluation requires a still image')
            image_format = image.format
            rgb = ImageOps.exif_transpose(image).convert('RGB')
        pixels = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
        cache[path] = hashlib.sha256(raw).hexdigest(), pixels, (image_format, rgb.size)
    digest, pixels, canvas = cache[path]
    if item.get('image_sha256') != digest:
        raise ValueError(f'image checksum mismatch for {item.get("image")}')
    return pixels, canvas


def verify_corpus(root, cache):
    groups, pixels, ids, indexed, checksums = {}, {}, set(), {}, {}
    for split in SPLITS:
        rows, checksums[f'{split}.jsonl'] = source_rows(root, split)
        for row in rows:
            row_id = require_text(row.get('id'), 'id')
            group = require_text(row.get('group_id'), 'group_id')
            if groups.setdefault(group, split) != split:
                raise ValueError(f'group crosses splits: {group}')
            pixel_hash, _ = check_image(root, row, cache)
            if pixels.setdefault(pixel_hash, split) != split:
                raise ValueError('decoded image pixels cross splits')
            if row_id in ids:
                raise ValueError(f'duplicate SFT row: {row_id}')
            ids.add(row_id)
            if row.get('task') == 'paired':
                key = group, row.get('variant')
                if key[1] not in VARIANTS or key in indexed:
                    raise ValueError(f'invalid or duplicate paired SFT variant: {key}')
                indexed[key] = row
    return groups, indexed, checksums


def pair_cases(pair, rank, root, indexed, cache):
    group = require_text(pair.get('group_id'), 'paired group_id')
    if re.fullmatch(r'[A-Za-z0-9_-]+', group) is None:
        raise ValueError('paired group_id must be a portable ID')
    cue = require_text(pair.get('cue_question'), 'cue_question')
    options, variants = pair.get('options'), pair.get('variants')
    if not isinstance(options, dict) or len(options) != 4:
        raise ValueError(f'{group}: expected four relation options')
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError(f'{group}: expected original, edited, and sham')
    first, edited = (require_text(variants[name].get('relation'), f'{name} relation')
                     if isinstance(variants[name], dict) else None for name in ('original', 'edited'))
    if first not in options or edited not in options or first == edited:
        raise ValueError(f'{group}: original and edited require distinct gold relations')
    first_position = rank % 4
    edited_position = (first_position + 1 + (rank // 4) % 3) % 4
    gold_positions = {first_position: first, edited_position: edited}
    distractors = iter(sorted(options.keys() - {first, edited}, key=lambda option: content_hash([SEED, group, option])))
    order = [gold_positions[position] if position in gold_positions else next(distractors)
             for position in range(4)]
    choices = [{'label': label, 'semantic_id': semantic, 'text': require_text(options[semantic], semantic)}
               for label, semantic in zip('ABCD', order)]
    if len({choice['text'].casefold() for choice in choices}) != 4:
        raise ValueError(f'{group}: duplicate option wording')
    choice_lines = '\n'.join(f"{option['label']}. {option['text']}" for option in choices)
    prompt = (f'Inspect the image. Answer the cue question: {cue}\n'
              'Which option best explains the relation between the setup and the punchline?\n'
              f'{choice_lines}\n'
              'Return one JSON object with exactly three keys. Set "cue" to "yes" or "no", '
              '"relation" to "A", "B", "C", or "D", and "explanation" to one short sentence.')
    gold, cases, image_pixels, canvases = [], [], set(), set()
    for variant in VARIANTS:
        item, sft = variants[variant], indexed.get((group, variant))
        if not isinstance(item, dict) or sft is None:
            raise ValueError(f'{group}/{variant}: expected a paired SFT row and reference')
        if any(item.get(key) != sft.get(key) for key in ('image', 'image_sha256')):
            raise ValueError(f'{group}/{variant}: pair and SFT images differ')
        pixels, canvas = check_image(root, item, cache)
        if pixels in image_pixels:
            raise ValueError(f'{group}: each variant needs distinct image pixels')
        image_pixels.add(pixels)
        canvases.add(canvas)
        relation = item.get('relation')
        if item.get('cue') not in ('yes', 'no') or not isinstance(relation, str) or relation not in options:
            raise ValueError(f'{group}/{variant}: invalid cue or relation key')
        answer = {'cue_answers': [item['cue']], 'relation_answers': [item['relation']]}
        gold.append(answer)
        case = {'id': f'{group}/{variant}', 'group_id': group, 'split': pair['split'],
                'variant': variant, 'image': item['image'], 'image_sha256': item['image_sha256'],
                'prompt': prompt, 'options': choices, 'gold': answer}
        cases.append(dict(case, input_sha256=content_hash(case)))
    if gold[0] != gold[2] or gold[0]['relation_answers'] == gold[1]['relation_answers']:
        raise ValueError(f'{group}: expected preserved sham answers and changed edited relation')
    if len(canvases) != 1:
        raise ValueError(f'{group}: variants must share one image encoding and canvas size')
    return cases


def compile_release(root):
    root = Path(root).resolve(strict=True)
    cache = {}
    groups, indexed, checksums = verify_corpus(root, cache)
    pairs, checksums['pairs.jsonl'] = source_rows(root, 'pairs')
    seen, cohorts = set(), {split: [] for split in SPLITS}
    for pair in sorted(pairs, key=lambda row: content_hash([SEED, row.get('split'), row.get('group_id')])):
        group = require_text(pair.get('group_id'), 'paired group_id')
        split = pair.get('split')
        if group in seen or split not in SPLITS or groups.get(group) != split:
            raise ValueError(f'{group}: duplicate group or pair split differs from SFT split')
        seen.add(group)
        cohorts[split].extend(pair_cases(pair, len(cohorts[split]) // 3, root, indexed, cache))
    if seen != {group for group, _ in indexed} or any(not cases for cases in cohorts.values()):
        raise ValueError('pair references must cover the paired SFT rows in each split')
    return cohorts, checksums


def describe(cases):
    balance = {variant: dict.fromkeys('ABCD', 0) for variant in VARIANTS}
    for case in cases:
        gold = case['gold']['relation_answers'][0]
        label = next(choice['label'] for choice in case['options'] if choice['semantic_id'] == gold)
        balance[case['variant']][label] += 1
    return {'families': len(cases) // 3, 'cases': len(cases), 'gold_label_counts': balance}


def score_rows(cases, outcomes):
    rows = []
    for case in cases:
        result = outcomes.get(case['id'])
        score = (score_pair(result['text'], case['options'], case['gold'],
                            truncated=result['finish_reason'] == 'length')
                 if result is not None and result['status'] == 'ok' else None)
        if score is not None and score['cue'] == 'insufficient':
            score = dict(score, status='invalid', cue_correct=0, relation_correct=0, joint_correct=0,
                         reason='cue must be yes or no')
        rows.append({'id': case['id'], 'group_id': case['group_id'], 'variant': case['variant'],
                     'status': result['status'] if result else 'missing', 'score': score})
    return rows


def summarize(scored):
    def count(rows):
        scores = [row['score'] for row in rows if row['score'] is not None]
        return {'planned': len(rows), 'terminal': sum(row['status'] != 'missing' for row in rows),
                'valid': sum(score['status'] == 'ok' for score in scores),
                'cue_correct': sum(score['cue_correct'] for score in scores),
                'relation_correct': sum(score['relation_correct'] for score in scores),
                'joint_correct': sum(score['joint_correct'] for score in scores),
                'execution_status': dict(Counter(row['status'] for row in rows)),
                'format_status': dict(Counter(score['status'] for score in scores))}

    def triple_success(rows):
        groups = {}
        for row in rows:
            groups.setdefault(row['group_id'], []).append(row['score'])
        return {group: len(scores) == 3 and all(score is not None and score['joint_correct'] == 1
                                               for score in scores)
                for group, scores in groups.items()}

    result, triples = {}, {}
    for arm, rows in scored.items():
        triples[arm] = triple_success(rows)
        result[arm] = dict(count(rows), by_variant={variant: count([r for r in rows if r['variant'] == variant])
                                                   for variant in VARIANTS},
                           strict_triple_success=sum(triples[arm].values()),
                           triple_count=len(triples[arm]))
    paired = Counter((triples['base'][group], triples['adapter'][group]) for group in triples['base'])
    result['paired_triples'] = {'both_correct': paired[True, True], 'base_only': paired[True, False],
                                'adapter_only': paired[False, True], 'both_incorrect': paired[False, False],
                                'delta_fraction': (paired[False, True] - paired[True, False]) / len(triples['base'])}
    result['complete'] = all(result[arm]['terminal'] == result[arm]['planned'] for arm in triples)
    return result
