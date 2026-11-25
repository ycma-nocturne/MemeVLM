"""Score one dev-selected LoRA profile against three matched baseline seeds."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import random
from statistics import mean, stdev
from tempfile import TemporaryDirectory


SEEDS = ('42', '43', '44')
PROFILES = ('real', 'real-low')
LABELS = ('A', 'B', 'C', 'D')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows_by_id(path):
    raw = path.read_bytes()
    rows = {}
    for lineno, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        case_id = row.get('id') if isinstance(row, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in rows:
            raise ValueError(f'{path}:{lineno}: expected a unique case ID')
        rows[case_id] = row
    return rows, sha256(raw).hexdigest()


def selected_profile(selection):
    raw = selection.read_bytes()
    value = json.loads(raw)
    by_seed = value.get('selected_by_seed') if isinstance(value, dict) else None
    if (not isinstance(by_seed, dict) or set(by_seed) != set(SEEDS)
            or by_seed['42'] not in PROFILES or any(value != by_seed['42'] for value in by_seed.values())):
        raise ValueError('declare one common dev-selected profile for seeds 42, 43, and 44')
    return by_seed['42'], sha256(raw).hexdigest()


def load_frozen(frozen):
    protocol_path = frozen / 'protocol.json'
    raw_protocol = protocol_path.read_bytes()
    protocol = json.loads(raw_protocol)
    native = protocol['native']
    cases, cases_sha = rows_by_id(frozen / 'native' / 'inputs.jsonl')
    gold, gold_sha = rows_by_id(frozen / 'native' / 'gold.jsonl')
    checksums = {'protocol.json': sha256(raw_protocol).hexdigest(), 'inputs.jsonl': cases_sha,
                 'gold.jsonl': gold_sha, 'manifest.json': digest(frozen / 'native' / 'manifest.json')}
    for name, field in (('inputs.jsonl', 'inputs_sha256'), ('gold.jsonl', 'gold_sha256'),
                        ('manifest.json', 'manifest_sha256')):
        if checksums[name] != native[field]:
            raise ValueError(f'MemeQA {name} differs from the frozen protocol')

    labels = {key: row['answer'] for key, row in gold.items()}
    if labels.keys() != cases.keys() or any(answer not in LABELS for answer in labels.values()):
        raise ValueError('frozen questions and gold answers must have the same IDs and valid labels')
    by_image, pilot = {}, {}
    for case_id, case in cases.items():
        image = case['group_id']
        flag = case['pilot_group']
        if not isinstance(image, str) or not image or type(flag) is not bool or pilot.setdefault(image, flag) != flag:
            raise ValueError('frozen image groups must have a consistent pilot flag')
        by_image.setdefault(image, []).append(case_id)
    clusters = [tuple(ids) for image, ids in by_image.items() if not pilot[image]]
    primary_labels = {case_id: labels[case_id] for cluster in clusters for case_id in cluster}
    if (len(cases), len(by_image), len(primary_labels), len(clusters)) != (
        native['questions'], native['images'], native['new_questions'], native['new_images']
    ):
        raise ValueError('native cohort differs from the frozen question/image census')
    return cases.keys(), primary_labels, clusters, protocol['bootstrap'], checksums


def score_ledger(path, all_ids, labels, expected_sha=None):
    rows, ledger_sha = rows_by_id(path)
    if expected_sha is not None and ledger_sha != expected_sha:
        raise ValueError(f'{path}: ledger differs from archived receipt')
    if rows.keys() != labels.keys() and rows.keys() != all_ids:
        raise ValueError(f'{path}: expected the exact primary or all-eligible question IDs')
    if any('choice' not in row or row['choice'] not in (*LABELS, None) for row in rows.values()):
        raise ValueError(f'{path}: choice must be A–D or null')
    correct = {key for key, answer in labels.items() if rows[key]['choice'] == answer}
    count = len(labels)
    summary = {'correct': len(correct), 'denominator': count, 'accuracy': len(correct) / count,
               'valid': sum(rows[key]['choice'] is not None for key in labels),
               'ledger_cases': len(rows), 'ledger_sha256': ledger_sha}
    return correct, summary


def paired_contrast(clusters, candidate, control, draws, seed):
    """Resample whole images, pooling their questions and retaining all seeds."""
    counts = [(len(cluster), *(sum(case in arm for case in cluster) for arm in candidate + control))
              for cluster in clusters]
    cut = len(candidate) + 1

    def delta(totals):
        return (mean(totals[1:cut]) - mean(totals[cut:])) / totals[0]

    point = delta(tuple(map(sum, zip(*counts))))
    rng = random.Random(seed)
    samples = sorted(delta(tuple(map(sum, zip(*rng.choices(counts, k=len(counts)))))) for _ in range(draws))
    return {'delta': point, 'paired_image_bootstrap_ci95': [samples[int(q * (draws - 1))] for q in (.025, .975)]}


def summarize_profile(summaries):
    rates = [row['accuracy'] for row in summaries]
    return {'mean': mean(rates), 'seed_sample_std': stdev(rates)}


def archived_comparators(frozen, all_ids, labels, clusters, draws, seed, protocol_sha, candidate):
    scores, summaries = {}, {}
    for arm in ('base', *(f'bilingual-{value}' for value in SEEDS)):
        directory = frozen / 'native-runs' / arm
        receipt = read_json(directory / 'completion.json')
        binding_path = directory / 'binding.json'
        binding = read_json(binding_path)
        if (receipt['arm'] != arm or receipt['cases'] != len(all_ids)
                or receipt['binding_sha256'] != digest(binding_path)
                or binding['protocol_sha256'] != protocol_sha or binding['arm'] != arm):
            raise ValueError(f'{arm}: archived response binding differs from the frozen protocol')
        scores[arm], summaries[arm] = score_ledger(directory / 'responses.jsonl', all_ids, labels,
                                                  receipt['ledger_sha256'])
    base = [scores['base']]
    bilingual = [scores[f'bilingual-{value}'] for value in SEEDS]
    return {'per_run': summaries, 'bilingual': summarize_profile([summaries[f'bilingual-{value}'] for value in SEEDS]),
            'bilingual_minus_base': paired_contrast(clusters, bilingual, base, draws, seed),
            'selected_minus_base': paired_contrast(clusters, candidate, base, draws, seed),
            'selected_minus_bilingual': paired_contrast(clusters, candidate, bilingual, draws, seed)}


def score(runs, frozen, selection, include_archived=False):
    profile, selection_sha = selected_profile(selection)
    all_ids, labels, clusters, bootstrap, checksums = load_frozen(frozen)
    draws, seed = bootstrap['draws'], bootstrap['seed']
    scores, summaries = {}, {}
    for arm in ('baseline', profile):
        for training_seed in SEEDS:
            name = f'{arm}-{training_seed}'
            scores[name], summaries[name] = score_ledger(runs / name / 'responses.jsonl', all_ids, labels)
    baseline = [scores[f'baseline-{training_seed}'] for training_seed in SEEDS]
    candidate = [scores[f'{profile}-{training_seed}'] for training_seed in SEEDS]
    primary = paired_contrast(clusters, candidate, baseline, draws, seed)
    per_seed = {training_seed: {'delta': (len(a) - len(b)) / len(labels),
                                'selected_only_correct': len(a - b), 'baseline_only_correct': len(b - a)}
                for training_seed, a, b in zip(SEEDS, candidate, baseline)}
    all_seed_positive = all(row['delta'] > 0 for row in per_seed.values())
    interval_positive = primary['paired_image_bootstrap_ci95'][0] > 0
    result = {'cohort': {'questions': len(labels), 'images_outside_pilot': len(clusters), 'sha256': checksums},
              'selection': {'profile': profile, 'sha256': selection_sha},
              'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'paired image cluster',
                            'interval_scope': 'conditional on the fitted adapters'},
              'per_run': summaries,
              'by_profile': {name: summarize_profile([summaries[f'{name}-{s}'] for s in SEEDS])
                             for name in ('baseline', profile)},
              'selected_minus_baseline': {**primary, 'per_seed': per_seed},
              'stability': {'all_seed_deltas_positive': all_seed_positive,
                            'paired_interval_above_zero': interval_positive,
                            'consistent_positive_on_this_cohort': all_seed_positive and interval_positive}}
    if include_archived:
        result['archived_context'] = archived_comparators(frozen, all_ids, labels, clusters, draws, seed,
                                                          checksums['protocol.json'], candidate)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='Root with {arm}-{seed}/responses.jsonl')
    parser.add_argument('--frozen', type=Path, required=True, help='Original final-expansion directory')
    parser.add_argument('--selection', type=Path, required=True,
                        help='Dev decision JSON. selected_by_seed must map 42, 43, and 44 to the same profile.')
    parser.add_argument('--include-archived', action='store_true', help='Add verified earlier base and bilingual context')
    parser.add_argument('--output', type=Path, help='Write a new score JSON instead of printing it')
    args = parser.parse_args()
    try:
        output = json.dumps(score(args.runs, args.frozen, args.selection, args.include_archived),
                            indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.output is None:
            print(output, end='')
        else:
            with TemporaryDirectory(dir=args.output.parent) as directory:
                pending = Path(directory) / 'score.json'
                pending.write_text(output, encoding='utf-8')
                os.link(pending, args.output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
