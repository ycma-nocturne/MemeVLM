"""Replay the independent MemeQA holdout comparison for the combined setting."""

import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT / 'ablations'))
from collect import canonical_sha, checkpoint_binding, load_dev, read_bundle, score_cell
from score import LABELS, SEEDS, digest, paired_contrast, rows_by_id, summarize_profile


ARMS = ('real', 'batch4', 'combined')
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6000


def read_holdout(directory):
    cases, inputs_sha = rows_by_id(directory / 'inputs.jsonl')
    gold, gold_sha = rows_by_id(directory / 'gold.jsonl')
    labels = {key: item['answer'] for key, item in gold.items()}
    if not cases or cases.keys() != labels.keys() or any(label not in LABELS for label in labels.values()):
        raise ValueError('holdout questions and answers must have identical IDs and valid A–D labels')
    by_image = {}
    for key, row in cases.items():
        image = row['group_id']
        if not isinstance(image, str) or not image:
            raise ValueError(f'{key}: holdout needs a valid image group')
        by_image.setdefault(image, []).append(key)
    hashes = {'inputs_sha256': inputs_sha, 'gold_sha256': gold_sha, 'manifest_sha256': digest(directory / 'manifest.json'),
              'questions': len(cases), 'images': len(by_image)}
    return labels, list(by_image.values()), hashes


def check_ids(ids, labels, split):
    if len(ids) != len(labels) or set(ids) != labels.keys():
        raise ValueError(f'{split} IDs differ from the frozen question list')


def select_combined(bundle, checkpoints, ids, labels, hashes, shared):
    seen = set()
    for seed in SEEDS:
        run = f'combined-{seed}'
        receipt = bundle['training'][run]
        steps = receipt['checkpoint_steps']
        if (tuple(steps) != (50, 100, 150, 193) or set(map(str, steps)) != receipt['checkpoints'].keys()
                or (receipt['arm'], str(receipt['seed']), str(receipt['data_seed'])) != ('combined', seed, seed)
                or (receipt['optimizer_steps'], receipt['stop_step'], receipt['planned_optimizer_steps']) != (193, 193, 579)
                or (receipt['effective_batch_size'], receipt['lora_rank'], receipt['lora_alpha'], receipt['lora_scale']) != (4, 16, 32, 2)):
            raise ValueError(f'{run}: training differs from the frozen combined setting')
        receipt_sha = canonical_sha(receipt)
        candidates = []
        for step in steps:
            key = f'{run}/{step}'
            seen.add(key)
            adapter = checkpoint_binding(receipt, step)
            _, stats = score_cell(bundle['dev'][key], ids, labels, run, step, 'dev',
                                  {**shared, 'inputs_sha256': hashes['inputs_sha256'], 'manifest_sha256': hashes['manifest_sha256'],
                                   'adapter': adapter, 'selection_sha256': None, 'training_receipt_sha256': receipt_sha})
            candidates.append({'step': step, 'adapter': adapter, **{name: stats[name] for name in
                              ('correct', 'ledger_sha256', 'binding_sha256')}})
        best = max(candidates, key=lambda row: (row['correct'], -row['step']))
        if any(checkpoints[seed].get(key) != value for key, value in best.items()):
            raise ValueError(f'{run}: checkpoint does not match the dev accuracy winner')
    if set(bundle['dev']) != seen:
        raise ValueError('expected exactly twelve combined development checkpoints')


def compare(candidate, control, correct, stats, clusters):
    pairs = [[f'{arm}-{seed}' for seed in SEEDS] for arm in (candidate, control)]
    a, b = [[correct[run] for run in names] for names in pairs]
    result = paired_contrast(clusters, a, b, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    result['per_seed'] = {}
    for seed, left, right in zip(SEEDS, *pairs):
        result['per_seed'][seed] = {'delta_pp': 100 * (stats[left]['accuracy'] - stats[right]['accuracy']),
                                    'candidate_only_correct': len(correct[left] - correct[right]),
                                    'control_only_correct': len(correct[right] - correct[left])}
    interval = [100 * value for value in result['paired_image_bootstrap_ci95']]
    return {'candidate': candidate, 'control': control, 'delta_pp': 100 * result['delta'],
            'paired_image_bootstrap_ci95_pp': interval, 'per_seed': result['per_seed']}


def collect(args):
    bundle, bundle_sha = read_bundle(args.bundle)
    ablations, ablation_sha = read_bundle(args.ablations)
    previous, previous_sha = read_bundle(args.previous)
    holdout_gold, clusters, holdout = read_holdout(args.holdout)
    dev_gold, dev_hashes = load_dev(args.dev)
    check_ids(bundle['test_ids'], holdout_gold, 'holdout')
    check_ids(bundle['dev_ids'], dev_gold, 'development')
    if holdout_gold.keys() & dev_gold.keys():
        raise ValueError('confirmation and checkpoint selection share question IDs')
    selection = bundle['selection']
    selection_sha = canonical_sha(selection)
    selected_dev = selection['dev']
    if (bundle.get('selection_sha256', selection_sha) != selection_sha
            or tuple(sorted(selection['test_arms'])) != tuple(sorted(ARMS))
            or any(selection['holdout'].get(key) != value for key, value in holdout.items())
            or any(selected_dev.get(key) != value for key, value in dev_hashes.items())
            or (selected_dev.get('questions'), selected_dev.get('images')) != (len(dev_gold), len(dev_gold))):
        raise ValueError('test selection differs from the frozen arms or cohort manifests')
    runs = {f'{arm}-{seed}' for arm in ARMS for seed in SEEDS}
    checkpoints = selection['checkpoints']
    if (runs != bundle['training'].keys() or runs != bundle['test'].keys() or checkpoints.keys() != set(ARMS)
            or any(entries.keys() != set(SEEDS) for entries in checkpoints.values())):
        raise ValueError('expected training, selected checkpoints, and test records for all nine arms and seeds')

    # These references contribute frozen checkpoint and decoding identities, not scores from the older test.
    prior_binding = previous['test']['real-42']['binding']
    shared = {key: prior_binding[key] for key in ('policy', 'prompt', 'reference_protocol_sha256')}
    for arm, prior in (('real', previous), ('batch4', ablations)):
        for seed in SEEDS:
            run = f'{arm}-{seed}'
            if (bundle['training'][run] != prior['training'][run]
                    or checkpoints[arm][seed] != prior['selection']['checkpoints'][arm][seed]
                    or any(prior['test'][run]['binding'][key] != value for key, value in shared.items())):
                raise ValueError(f'{run}: inherited checkpoint, training receipt or decoding policy changed')

    fixed = ('dataset_sha256', 'data_selection_sha256', 'train_sha256', 'validation_sha256', 'train_rows',
             'validation_rows', 'model_id', 'model_revision', 'model_files', 'learning_rate')
    for seed in SEEDS:
        anchor = bundle['training'][f'batch4-{seed}']
        combined = bundle['training'][f'combined-{seed}']
        if any(anchor[key] != combined[key] for key in fixed):
            raise ValueError(f'combined-{seed}: data, model, or peak learning rate differs from batch4')
    select_combined(bundle, checkpoints['combined'], bundle['dev_ids'], dev_gold, dev_hashes, shared)

    correct, stats, table = {}, {}, []
    for arm in ARMS:
        for seed in SEEDS:
            run = f'{arm}-{seed}'
            training = bundle['training'][run]
            selected = checkpoints[arm][seed]
            step = selected['step']
            correct[run], stats[run] = score_cell(bundle['test'][run], bundle['test_ids'], holdout_gold, run, step, 'test',
                                                 {**shared, 'inputs_sha256': holdout['inputs_sha256'],
                                                  'manifest_sha256': holdout['manifest_sha256'],
                                                  'adapter': checkpoint_binding(training, step), 'selection_sha256': selection_sha,
                                                  'training_receipt_sha256': canonical_sha(training)})
            table.append({'arm': arm, 'seed': int(seed), 'selected_step': step, 'dev_correct': selected['correct'],
                          'dev_questions': len(dev_gold), 'correct': stats[run]['correct'], 'questions': len(holdout_gold),
                          'valid': stats[run]['valid'], 'accuracy_pct': 100 * stats[run]['accuracy']})

    primary = compare('combined', 'batch4', correct, stats, clusters)
    primary['stable_positive'] = primary['paired_image_bootstrap_ci95_pp'][0] > 0 and all(
        row['delta_pp'] > 0 for row in primary['per_seed'].values())
    secondary = compare('batch4', 'real', correct, stats, clusters)
    profiles = {}
    for arm in ARMS:
        profile = summarize_profile([stats[f'{arm}-{seed}'] for seed in SEEDS])
        profiles[arm] = {'accuracy_pct': 100 * profile['mean'], 'seed_sample_sd_pp': 100 * profile['seed_sample_std']}
    summary = {'holdout': holdout, 'development': dev_hashes, 'profiles': profiles, 'test_by_seed': table,
               'primary': primary, 'prespecified_secondary': secondary,
               'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'unit': 'image cluster',
                             'interval_scope': 'conditional on the three fitted seed pairs'},
               'sha256': {'confirmation_bundle': bundle_sha, 'selection': selection_sha, 'prior_ablations': ablation_sha,
                          'prior_data_lr': previous_sha,
                          'test_ledgers': {run: row['ledger_sha256'] for run, row in stats.items()}}}
    contrasts = []
    for scope, comparison in (('primary', primary), ('prespecified_secondary', secondary)):
        lower, upper = comparison['paired_image_bootstrap_ci95_pp']
        contrasts.append({'scope': scope, 'candidate': comparison['candidate'], 'control': comparison['control'],
                          'delta_pp': comparison['delta_pp'], 'ci95_lower_pp': lower, 'ci95_upper_pp': upper,
                          **{f'delta_seed_{seed}_pp': row['delta_pp'] for seed, row in comparison['per_seed'].items()},
                          'stable_positive': comparison.get('stable_positive')})
    return summary, {'test_by_seed.csv': table, 'contrasts.csv': contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Completed confirmation compact-results.json.gz')
    parser.add_argument('--holdout', type=Path, required=True, help='Frozen confirmation inputs, gold and manifest directory')
    parser.add_argument('--ablations', type=Path, default=PARENT / 'ablations/compact-results.json.gz')
    parser.add_argument('--previous', type=Path, default=PARENT / 'results/compact-results.json.gz')
    parser.add_argument('--dev', type=Path, default=PARENT / 'prepared/native-dev')
    parser.add_argument('--output', type=Path, required=True, help='Create a new score directory')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('choose a new score directory')
        summary, tables = collect(args)
        with TemporaryDirectory(dir=args.output.parent) as scratch:
            pending = Path(scratch) / 'scores'
            pending.mkdir()
            (pending / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                                  encoding='utf-8')
            for name, rows in tables.items():
                with (pending / name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            pending.rename(args.output)
    except (OSError, TypeError, KeyError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
