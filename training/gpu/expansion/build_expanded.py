"""Freeze the full eligible native MemeQA None− test for research evaluation."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile
import argparse
import json
import os

from PIL import Image

from pilot_prepare import (
    LABELS, POSITION_DEPENDENT, SEED, digest, pixel_hash, rank, validate_bytes, write_jsonl,
)


PROMPT = (
    "Answer the multiple-choice question about this meme. Select one option. "
    'Return one JSON object with exactly one key "answer" and a value of "A", "B", "C", or "D".'
)
EXPECTED_PILOT = 'e2cdcd0e15cc2bef999f56c39096433bb6f192baf31291dec76818d633a5360c'
SPLITS = ('training', 'dev', 'test')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
PIXEL_CAP = 1_310_720


def verify_sources():
    raw = Path('pilot-manifest.json').read_bytes()
    if digest(raw) != EXPECTED_PILOT:
        raise ValueError('Pilot manifest changed')
    pilot = json.loads(raw)
    annotations = Path('memeqa-test.json').read_bytes()
    if digest(annotations) != pilot['source_annotations']['sha256']:
        raise ValueError('Native test annotations changed')
    for split in SPLITS:
        if digest(Path(f'semeval-{split}.zip').read_bytes()) != pilot['source_images']['archives'][split]:
            raise ValueError(f'SemEval {split} archive changed')
    return pilot, json.loads(annotations)


def select_questions(rows):
    groups = defaultdict(list)
    ids, excluded = {}, Counter()
    for row in rows:
        identifier = row['id']
        if identifier in ids:
            if row != ids[identifier]:
                raise ValueError(f'Conflicting duplicate question ID: {identifier}')
            excluded['duplicate_ids'] += 1
            continue
        ids[identifier] = row
        choices = row['options']
        if row['general_type'] != 'single':
            excluded['multiple_answers'] += 1
        elif len(choices) != 4:
            excluded['different_option_count'] += 1
        elif len({choice.strip().casefold() for choice in choices}) != 4:
            excluded['duplicate_choice_text'] += 1
        elif any(POSITION_DEPENDENT.search(text) for text in [row['question'], *choices]):
            excluded['position_dependent_wording'] += 1
        else:
            if type(row['answer_key']) is not int or row['answer_key'] not in range(4):
                raise ValueError(f'Invalid native answer: {identifier}')
            groups[row['img']].append(row)
    return groups, excluded


def read_images(filenames):
    indexed = {}
    with ExitStack() as stack:
        for split in SPLITS:
            archive = stack.enter_context(ZipFile(f'semeval-{split}.zip'))
            for member in archive.namelist():
                name = Path(member).name
                if name not in filenames:
                    continue
                if name in indexed:
                    raise ValueError(f'Multiple archived sources for {name}')
                payload = archive.read(member)
                checked = validate_bytes(payload)
                if checked is None:
                    raise ValueError(f'Expected a still JPEG or PNG: {name}')
                suffix, pixels, size, image_format = checked
                indexed[name] = {'archive_split': split, 'member': member, 'bytes': payload,
                                 'suffix': suffix, 'pixels': pixels, 'size': size,
                                 'format': image_format, 'sha256': digest(payload)}
    if set(indexed) != filenames:
        raise ValueError(f'Missing images: {sorted(filenames - indexed.keys())}')
    return indexed


def verify_no_project_overlap(project, held_pixels):
    count = 0
    for name in ('training/images', 'training/bilingual/images', 'data/assets'):
        directory = project / name
        if not directory.is_dir():
            raise ValueError(f'Missing project image directory: {name}')
        for path in directory.rglob('*'):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                with Image.open(path) as image:
                    if pixel_hash(image) in held_pixels:
                        raise ValueError(f'Exact project overlap: {path.name}')
                count += 1
    if count != 956:
        raise ValueError(f'Expected audited set of 956 files, found {count}')
    return count


def balanced_slots(n, namespace):
    labels = sorted(LABELS, key=lambda letter: rank(namespace, 'remainder-label', letter))
    counts = {letter: n // len(labels) + (i < n % len(labels)) for i, letter in enumerate(labels)}
    slots = [(letter, i) for letter in LABELS for i in range(counts[letter])]
    return [letter for letter, _ in sorted(slots, key=lambda cell: rank(namespace, 'answer-slot', *cell))]


def collect_cases(questions, source_images, pilot_index):
    # Give every exact decoded RGB family one stored image and one stable group key.
    canonical = {}
    for filename in sorted(source_images):
        image = source_images[filename]
        canonical.setdefault(image['pixels'], filename)
    pilot_cases = {x['source_question_id']: x['id'] for x in pilot_index}
    pilot_pixels = {x['decoded_rgb_sha256'] for x in pilot_index}
    seen = {}

    for row in sorted((row for rows in questions.values() for row in rows),
                      key=lambda x: rank('expanded-case-order', x['id'])):
        image = source_images[row['img']]
        signature = image['pixels'], row['question'], tuple(sorted(row['options']))
        if signature in seen:
            prior = seen[signature]
            if row['options'][row['answer_key']] != prior['options'][prior['answer_key']]:
                raise ValueError('Native duplicate question has a conflicting answer')
            continue
        seen[signature] = row

    ordered = list(seen.values())
    # Balance the primary nonpilot estimate as well as the entire cohort.
    in_pilot = [r for r in ordered if source_images[r['img']]['pixels'] in pilot_pixels]
    out_pilot = [r for r in ordered if source_images[r['img']]['pixels'] not in pilot_pixels]
    out_slots = balanced_slots(len(out_pilot), 'expanded-nonpilot')
    in_slots = balanced_slots(len(in_pilot), 'expanded-pilot')
    targets = dict(zip([r['id'] for r in out_pilot], out_slots))
    targets.update(zip([r['id'] for r in in_pilot], in_slots))
    if max(Counter(targets.values()).values()) - min(Counter(targets.values()).values()) > 1:
        # Remainder labels can collide between strata. The full cohort remains balanced.
        targets = dict(zip([r['id'] for r in ordered], balanced_slots(len(ordered), 'expanded-total')))

    cases, gold, image_index = [], [], {}
    for index, row in enumerate(ordered, start=1):
        pixels = source_images[row['img']]['pixels']
        source_name = canonical[pixels]
        image = source_images[source_name]
        group = f'rgb-{pixels}'
        relative = f"images/{group}{image['suffix']}"
        image_index.setdefault(group, {
            'source_image': source_name, 'archive_split': image['archive_split'], 'source_member': image['member'],
            'path': relative, 'sha256': image['sha256'], 'decoded_rgb_sha256': pixels, 'size': image['size'],
            'format': image['format'], 'encoded_bytes': len(image['bytes']), 'pilot_group': pixels in pilot_pixels,
        })
        letter = targets[row['id']]
        correct = row['answer_key']
        order = sorted((i for i in range(4) if i != correct),
                       key=lambda i: rank('expanded-distractor', row['id'], i))
        order.insert(LABELS.index(letter), correct)
        options = [row['options'][i] for i in order]
        choices = '\n'.join(f'{label}. {text}' for label, text in zip(LABELS, options))

        case_id = f'mqa-{index:04d}'
        cases.append({'id': case_id, 'source_question_id': row['id'], 'source_image': row['img'],
                      'group_id': group, 'question_type': row['specific_type'],
                      'image': relative, 'image_sha256': image['sha256'], 'question': row['question'],
                      'options': options, 'prompt': f"{PROMPT}\n\n{row['question']}\n{choices}",
                      'pilot_group': pixels in pilot_pixels, 'pilot_question': row['id'] in pilot_cases,
                      'pilot_case_id': pilot_cases.get(row['id'])})
        gold.append({'id': case_id, 'answer': letter})
    if sum(row['pilot_question'] for row in cases) != len(pilot_cases):
        raise ValueError('Expanded cohort failed to retain every pilot question')
    if sum(row['pilot_group'] for row in image_index.values()) != len(pilot_pixels):
        raise ValueError('Expanded cohort failed to retain every pilot image group')
    return cases, gold, image_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('frozen'))
    args = parser.parse_args()
    os.umask(0o077)
    pilot, native_rows = verify_sources()
    questions, excluded = select_questions(native_rows)
    source_images = read_images(set(questions))
    checked = verify_no_project_overlap(args.project, {r['pixels'] for r in source_images.values()})
    cases, gold, images = collect_cases(questions, source_images, pilot['sample_provenance'])
    image_dir = args.output / 'images'
    image_dir.mkdir(parents=True, exist_ok=False)
    for record in images.values():
        with (args.output / record['path']).open('xb') as path:
            path.write(source_images[record['source_image']]['bytes'])
    write_jsonl(args.output / 'inputs.jsonl', cases)
    write_jsonl(args.output / 'gold.jsonl', gold)

    nonpilot = [row for row in cases if not row['pilot_group']]
    nonpilot_ids = {row['id'] for row in nonpilot}
    raw_sizes = [record['size'] for record in images.values()]
    selected = len(cases)
    manifest = {
        'protocol': 'memeqa-native-none-minus-full-eligible-transfer-v1',
        'scope': 'All eligible author-test observations, independently reported by image group.',
        'seed': SEED, 'selection_uses_model_outputs': False, 'contains_previously_evaluated_pilot': True,
        'sample_selection': (
            'No sampling. Retain each native single-answer four-option position-independent question, '
            'with unique option texts and unique question ID. Merge exact decoded RGB groups.'
        ),
        'n_questions': selected, 'n_images': len(images),
        'source_annotations': pilot['source_annotations'], 'source_images': pilot['source_images'],
        'source_pilot_manifest_sha256': EXPECTED_PILOT,
        'pilot_question_ids': sorted(row['source_question_id'] for row in cases if row['pilot_question']),
        'pilot_group_ids': sorted(key for key, value in images.items() if value['pilot_group']),
        'n_pilot_questions': sum(row['pilot_question'] for row in cases),
        'n_pilot_group_questions': sum(row['pilot_group'] for row in cases),
        'nonpilot_exclusion': 'Exclude every question from any prior pilot image, not only the 32 pilot questions.',
        'n_nonpilot_questions': len(nonpilot), 'n_nonpilot_groups': len({row['group_id'] for row in nonpilot}),
        'excluded_question_counts': dict(sorted(excluded.items())),
        'duplicate_question_image_content_skipped': sum(map(len, questions.values())) - selected,
        'n_native_mapped_image_names': len(source_images),
        'asset_check': {'project_image_files_compared': checked, 'exact_rgb_overlap': 0,
                        'verified_actual_formats': dict(sorted(Counter(x['format'] for x in images.values()).items()))},
        'shape_check': {'max_raw_width': max(w for w, _ in raw_sizes),
                        'max_raw_height': max(h for _, h in raw_sizes),
                        'max_raw_pixels': max(w * h for w, h in raw_sizes),
                        'max_raw_aspect_ratio': max(max(w, h) / min(w, h) for w, h in raw_sizes),
                        'max_prompt_utf8_bytes': max(len(row['prompt'].encode()) for row in cases),
                        'raw_images_above_backend_processed_pixel_cap': sum(w*h > PIXEL_CAP for w, h in raw_sizes),
                        'existing_h800_backend_processed_pixel_cap': PIXEL_CAP, 'existing_h800_backend_batch_size': 1},
        'type_counts': dict(sorted(Counter(row['question_type'] for row in cases).items())),
        'answer_position_counts': dict(sorted(Counter(row['answer'] for row in gold).items())),
        'nonpilot_answer_position_counts': dict(sorted(Counter(row['answer'] for row in gold
                                                               if row['id'] in nonpilot_ids).items())),
        'cases_sha256': digest((args.output / 'inputs.jsonl').read_bytes()),
        'inputs_sha256': digest((args.output / 'inputs.jsonl').read_bytes()),
        'gold_sha256': digest((args.output / 'gold.jsonl').read_bytes()),
        'builder_sha256': digest(Path(__file__).read_bytes()),
        'pilot_helper_sha256': digest(Path(__file__).with_name('pilot_prepare.py').read_bytes()),
        'image_groups': images,
    }
    with (args.output / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({key: manifest[key] for key in (
        'n_questions', 'n_images', 'n_nonpilot_questions', 'n_nonpilot_groups', 'n_pilot_questions',
        'n_pilot_group_questions', 'excluded_question_counts', 'shape_check', 'answer_position_counts',
        'nonpilot_answer_position_counts', 'inputs_sha256', 'gold_sha256') if key in manifest}, indent=2))


if __name__ == '__main__':
    main()
