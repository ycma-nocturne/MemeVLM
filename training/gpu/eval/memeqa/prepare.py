"""Freeze 32 official MemeQA test questions and source meme images for research evaluation."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os
import re
import sys
from zipfile import ZipFile

from PIL import Image, ImageOps


SEED = 6000
SIZE = 32
LABELS = 'ABCD'
MEMEQA = 'https://github.com/npnkhoi/memeqa'
SEMEVAL = 'https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus'
POSITION_DEPENDENT = re.compile(
    r'\b(?:none|all)\s+of\s+(?:the\s+above|these|the\s+following)\b'
    r'|\b(?:both\s+[A-D]\s+(?:and\s+)?[A-D]|(?:answer|option|choice|letter)s?\s+[A-D])\b'
    r'|\b[A-D]\s*(?:and|/|&)\s*[A-D]\b', re.IGNORECASE,
)


def digest(data):
    return sha256(data).hexdigest()


def rank(tag, *values):
    return digest(json.dumps([SEED, tag, *values], ensure_ascii=False).encode())


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def pixel_hash(image):
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    return digest(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes())


def validate_bytes(data):
    with Image.open(BytesIO(data)) as image:
        fmt = image.format
        image.verify()
    if fmt == 'PNG' and data.startswith(b'\x89PNG\r\n\x1a\n'):
        suffix = '.png'
    elif fmt == 'JPEG' and data.startswith(b'\xff\xd8\xff'):
        suffix = '.jpg'
    else:
        return None
    with Image.open(BytesIO(data)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            return None
        return suffix, pixel_hash(image), list(image.size), fmt


def parse_sample(project):
    rows = json.loads(Path('memeqa-test.json').read_bytes())
    assert len(rows) == len({row['id'] for row in rows}) == 1887
    groups = defaultdict(list)
    categories = Counter()
    for row in rows:
        if row['general_type'] != 'single' or len(row['options']) != 4:
            continue
        question, choices = row['question'], row['options']
        idx = row['answer_key']
        assert type(idx) is int and 0 <= idx < 4
        if len({s.strip().casefold() for s in choices}) != 4:
            continue
        if any(POSITION_DEPENDENT.search(text) for text in [question, *choices]):
            continue
        groups[row['img']].append(row)
        categories[row['specific_type']] += 1
    with ExitStack() as stack:
        archive_by_name = {}
        sources = ('training', 'dev', 'test')
        for split in sources:
            archive = stack.enter_context(ZipFile(f'semeval-{split}.zip'))
            for member in archive.namelist():
                basename = Path(member).name
                if Path(basename).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                if basename in archive_by_name:
                    raise ValueError(f'Duplicate source filename: {basename}')
                archive_by_name[basename] = (archive, member, split)
        missing = set(groups) - archive_by_name.keys()
        assert not missing, f'Unmatched MemeQA image references: {missing}'
        seen, cases, excluded_encoding, exact_duplicates = set(), [], [], []
        for filename in sorted(groups, key=lambda name: rank('image', name)):
            archive, member, source_split = archive_by_name[filename]
            data = archive.read(member)
            check = validate_bytes(data)
            if check is None:
                excluded_encoding.append(filename)
                continue
            suffix, pixels, dimensions, fmt = check
            if pixels in seen:
                exact_duplicates.append(filename)
                continue
            seen.add(pixels)
            row = min(groups[filename], key=lambda item: rank('question', item['id']))
            cases.append((row, data, source_split, member, fmt, suffix, pixels, dimensions))
            if len(cases) == SIZE:
                break
    assert len(cases) == SIZE

    held_images = {case[6] for case in cases}
    verified_project_images = 0
    for dirname in ('training/images', 'training/bilingual/images', 'data/assets'):
        for path in (project / dirname).rglob('*'):
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp') and path.is_file():
                with Image.open(path) as image:
                    assert pixel_hash(image) not in held_images, f'Overlap: {path.name}'
                    verified_project_images += 1
    return rows, groups, categories, cases, verified_project_images, excluded_encoding, exact_duplicates


def main():
    project = Path(sys.argv[1])
    rows, groups, categories, cases, checked, bad_formats, duplicates = parse_sample(project)
    out = Path('frozen')
    out.mkdir(mode=0o700)
    (out / 'images').mkdir(mode=0o700)
    os.umask(0o077)

    answer_positions = sorted(((label, j) for label in LABELS for j in range(SIZE // 4)),
                              key=lambda x: rank('answer-slot', *x))
    inputs, gold, sources = [], [], []
    for index, ((row, data, split, member, fmt, suffix, pixels, dimensions), (answer_letter, _)) in enumerate(
        zip(cases, answer_positions), start=1
    ):
        case = f'mqa-{index:03}'
        original_choices = row['options']
        answer_index = row['answer_key']
        wrong_indexes = sorted((i for i in range(4) if i != answer_index),
                               key=lambda i: rank('wrong-slot', row['id'], i))
        permutation = wrong_indexes[:]
        permutation.insert(LABELS.index(answer_letter), answer_index)
        choices = [original_choices[i] for i in permutation]
        assert set(choices) == set(original_choices)
        assert choices[LABELS.index(answer_letter)] == original_choices[answer_index]
        path = out / 'images' / f'{case}{suffix}'
        path.write_bytes(data)
        prompt = (f'{row["question"]}\n\n' +
                  '\n'.join(f'{letter}. {choice}' for letter, choice in zip(LABELS, choices)) +
                  '\n\nAnswer with exactly one uppercase choice letter from A, B, C, and D.')
        inputs.append({'id': case, 'source_image': row['img'], 'image': f'images/{path.name}',
                       'image_sha256': digest(data), 'question': row['question'],
                       'options': choices, 'prompt': prompt})
        gold.append({'id': case, 'answer': answer_letter})
        sources.append({'id': case, 'source_question_id': row['id'], 'subtask': row['specific_type'],
                        'source_archive_split': split, 'source_member': member, 'image_format': fmt,
                        'source_image_size': dimensions, 'decoded_rgb_sha256': pixels,
                        'image_sha256': digest(data)})
    for path, values in [(out / 'inputs.jsonl', inputs), (out / 'gold.jsonl', gold)]:
        write_jsonl(path, values)

    manifest = {
        'protocol': 'memeqa-native-none-minus-transfer-v1',
        'purpose': 'Separate English multiple-choice transfer probe. Research evaluation only.',
        'preselection_model_outputs_seen': 0,
        'seed': SEED,
        'n_images': SIZE,
        'selection': [
            'Official MemeQA none_minus/test.json only',
            'Native single-answer questions with four unique options and position-independent wording',
            'MemeQA supplies the experimental test split. SemEval archives only supply source pixels.',
            'Seeded ordering by source image filename, omit repeated decoded RGB, and select first 32',
            'One question per image by seeded source-question-ID rank',
            'Seeded placement of eight gold answers per letter, with independently seeded distractor ordering',
            'Questions and choice texts retain native wording. No moderation exclusions from the upstream research corpus.',
        ],
        'source_annotations': {'url': f'{MEMEQA}/blob/e48b332a02f7b25e08459dd7dcf69f19b430c621/data/none_minus/test.json',
                               'sha256': digest(Path('memeqa-test.json').read_bytes()), 'license': 'MIT',
                               'n_questions': len(rows), 'n_referenced_images': len({r['img'] for r in rows})},
        'source_images': {'project': SEMEVAL, 'revision': '796b32fa76b7a5814a48d1870d163058e81e4ce1',
                          'terms_publication': f'{SEMEVAL}#licensing', 'usage': 'General research use',
                          'archives': {split: digest(Path(f'semeval-{split}.zip').read_bytes())
                                       for split in ('training', 'dev', 'test')}},
        'eligible_questions_by_type': dict(sorted(categories.items())),
        'eligible_unique_filenames_before_image_validation': len(groups),
        'invalid_image_formats_skipped': bad_formats, 'duplicate_rgb_images_skipped': duplicates,
        'selected_question_types': dict(sorted(Counter(r['subtask'] for r in sources).items())),
        'selected_source_splits': dict(sorted(Counter(r['source_archive_split'] for r in sources).items())),
        'selected_image_formats': dict(sorted(Counter(r['image_format'] for r in sources).items())),
        'checked_project_image_files_for_exact_decoded_rgb_overlap': checked,
        'scorer': {'description': 'Exact choice-letter accuracy on the fixed 32-image denominator',
                   'parse': 'strip outside whitespace, then accept exactly one uppercase A/B/C/D',
                   'control': 'Optional question+choices without image. Reuse the identical prompt and denominator.'},
        'sample_provenance': sources,
        'inputs_sha256': digest((out / 'inputs.jsonl').read_bytes()),
        'gold_sha256': digest((out / 'gold.jsonl').read_bytes()),
        'prepare_sha256': digest(Path(__file__).read_bytes()),
    }
    with (out / 'manifest.json').open('x', encoding='utf-8') as index:
        json.dump(manifest, index, ensure_ascii=False, indent=2, sort_keys=True)
        index.write('\n')
    print(json.dumps({k: manifest[k] for k in (
        'n_images', 'selected_image_formats', 'selected_source_splits', 'selected_question_types',
        'eligible_unique_filenames_before_image_validation', 'checked_project_image_files_for_exact_decoded_rgb_overlap',
        'inputs_sha256', 'gold_sha256')}, indent=2))


if __name__ == '__main__':
    main()
