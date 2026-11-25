"""Replay targeted MemeQA ablations and export report-ready curves and contrasts."""

import argparse
import csv
import gzip
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import random
from statistics import mean
import sys
from tempfile import TemporaryDirectory


PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))
from score import LABELS, SEEDS, digest, load_frozen, paired_contrast, rows_by_id, summarize_profile


ARMS = ('batch4', 'rank4', 'rank16', 'alpha8')
PREVIOUS_ARMS = ('baseline', 'real', 'real-low')
ALPHA = 0.05
HISTORY_METRICS = ('loss', 'eval_loss', 'learning_rate', 'grad_norm')
CONFIG = {'baseline': (8, 8, 16), 'real': (8, 8, 16), 'real-low': (8, 8, 16), 'batch4': (4, 8, 16),
          'rank4': (8, 4, 8), 'rank16': (8, 16, 32), 'alpha8': (8, 8, 8)}


def canonical_sha(record):
    raw = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    return sha256(raw.encode()).hexdigest()


def read_bundle(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw)), sha256(raw).hexdigest()


def load_dev(directory):
    inputs, inputs_sha = rows_by_id(directory / 'inputs.jsonl')
    gold, gold_sha = rows_by_id(directory / 'gold.jsonl')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    labels = {case: row['answer'] for case, row in gold.items()}
    if (inputs.keys() != labels.keys() or len(labels) != 96 or any(x not in LABELS for x in labels.values())
            or manifest['inputs_sha256'] != inputs_sha or manifest['gold_sha256'] != gold_sha):
        raise ValueError('development cases and labels differ from the frozen 96-question manifest')
    return labels, {'inputs_sha256': inputs_sha, 'gold_sha256': gold_sha, 'manifest_sha256': digest(manifest_path)}


def score_cell(cell, ids, gold, run, step, split, expected):
    arm, seed = run.rsplit('-', 1)
    binding, receipt, choices = (cell[key] for key in ('binding', 'completion', 'choices'))
    if (binding['arm'], str(binding['seed']), binding['step'], binding['split']) != (arm, seed, step, split):
        raise ValueError(f'{run}/{step}: prediction identity differs from its {split} row')
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{run}/{step}: {split} binding differs from the frozen inputs, policy, or checkpoint')
    if (not isinstance(choices, str) or len(choices) != len(ids) or any(c not in 'ABCD-' for c in choices)
            or receipt['cases'] != len(ids) or receipt['valid'] != len(ids) - choices.count('-')
            or receipt['binding_sha256'] != canonical_sha(binding)):
        raise ValueError(f'{run}/{step}: compact choices differ from the {split} completion receipt')
    correct = {key for key, choice in zip(ids, choices, strict=True) if choice == gold[key]}
    return correct, {'correct': len(correct), 'denominator': len(ids), 'valid': receipt['valid'],
                     'accuracy': len(correct) / len(ids), 'ledger_sha256': receipt['ledger_sha256'],
                     'binding_sha256': receipt['binding_sha256']}


def checkpoint_binding(training, step):
    checkpoint = training['checkpoints'][str(step)]
    return {'adapter_model.safetensors': checkpoint['adapter_sha256'],
            'adapter_config.json': checkpoint['adapter_config_sha256']}


def configuration(arm, training):
    expected = CONFIG[arm]
    keys = ('effective_batch_size', 'lora_rank', 'lora_alpha')
    if arm in ARMS or any(key in training for key in keys):
        values = tuple(training[key] for key in keys)
        if values != expected or training['lora_scale'] != values[2] / values[1]:
            raise ValueError(f'{arm}: training configuration differs from the planned ablation')
    return {**dict(zip(keys, expected)), 'lora_scale': expected[2] / expected[1]}


def inspect_history(state, run, phase, training, chosen):
    records = state if isinstance(state, list) else state.get('log_history', [])
    if isinstance(state, dict) and state and (state.get('global_step', training['optimizer_steps']),
            state.get('max_steps', training['planned_optimizer_steps'])) != (
            training['optimizer_steps'], training['planned_optimizer_steps']):
        raise ValueError(f'{run}: trainer state does not match the completed run receipt')
    arm, seed = run.rsplit('-', 1)
    batch = CONFIG[arm][0]
    epochs = {row['step']: row.get('epoch') for row in records if 'eval_loss' in row}
    if records:
        losses = {row['step']: row['eval_loss'] for row in records if 'eval_loss' in row}
        expected = {int(step): entry['sft_validation_loss'] for step, entry in training['checkpoints'].items()}
        if losses != expected:
            raise ValueError(f'{run}: trainer history validation losses differ from the checkpoint receipts')
    long = [{'phase': phase, 'arm': arm, 'seed': int(seed), 'step': row['step'], 'epoch': row.get('epoch'),
             'nominal_train_rows_seen': min(row['step'] * batch, training['train_rows']),
             'metric': metric, 'value': row[metric], 'selected_checkpoint': row['step'] == chosen}
            for row in records for metric in HISTORY_METRICS if metric in row]
    return epochs, long


def replay(bundle, arms, tested, phase, dev_truth, test_truth, shared, states):
    dev_gold, dev_hashes = dev_truth
    test_gold, test_hashes = test_truth
    dev_ids, test_ids = bundle['dev_ids'], bundle['test_ids']
    for name, ids, gold in (('development', dev_ids, dev_gold), ('test', test_ids, test_gold)):
        if len(ids) != len(gold) or set(ids) != gold.keys():
            raise ValueError(f'{phase}: {name} IDs differ from the frozen cohort')
    selection = bundle['selection']
    selection_sha = canonical_sha(selection)
    if bundle.get('selection_sha256', selection_sha) != selection_sha:
        raise ValueError(f'{phase}: selection digest differs from its compact receipt')
    if selection['dev'] != dev_hashes or selection['questions_per_seed'] != len(dev_ids):
        raise ValueError(f'{phase}: checkpoint selection used different development files')
    names = {f'{arm}-{seed}' for arm in arms for seed in SEEDS}
    if set(bundle['training']) != names or set(bundle['test']) != {f'{arm}-{seed}' for arm in tested for seed in SEEDS}:
        raise ValueError(f'{phase}: wrong set of training or test arms')
    if set(states) - names:
        raise ValueError(f'{phase}: trainer history refers to an undeclared arm')

    selected, results, correct_sets = {}, {}, {}
    curves, log, test_rows = [], [], []
    expected_dev = set()
    for arm in arms:
        for seed in SEEDS:
            run = f'{arm}-{seed}'
            training = bundle['training'][run]
            batch = configuration(arm, training)['effective_batch_size']
            steps = training['checkpoint_steps']
            if (str(training['seed']) != seed or str(training['data_seed']) != seed or training['arm'] != arm
                    or not steps or steps != sorted(set(steps)) or set(map(str, steps)) != set(training['checkpoints'])
                    or steps[-1] != training['stop_step'] or training['optimizer_steps'] != training['stop_step']):
                raise ValueError(f'{run}: invalid completed checkpoint schedule or run identity')
            receipt_sha = canonical_sha(training)
            candidates = []
            for step in steps:
                key = f'{run}/{step}'
                expected_dev.add(key)
                cell = bundle['dev'][key]
                adapter = checkpoint_binding(training, step)
                _, stats = score_cell(cell, dev_ids, dev_gold, run, step, 'dev',
                                      {**shared, 'inputs_sha256': dev_hashes['inputs_sha256'],
                                       'manifest_sha256': dev_hashes['manifest_sha256'], 'adapter': adapter,
                                       'selection_sha256': None, 'training_receipt_sha256': receipt_sha})
                frozen_record, = (row for row in selection['scores'][arm][seed] if row['step'] == step)
                if (frozen_record['adapter'] != adapter
                        or any(frozen_record[key] != stats[key] for key in ('correct', 'ledger_sha256', 'binding_sha256'))):
                    raise ValueError(f'{run}/{step}: development score differs from frozen selection')
                candidates.append((step, stats, frozen_record))
            chosen, chosen_stats, chosen_record = max(candidates, key=lambda item: (item[1]['correct'], -item[0]))
            if chosen_record != selection['checkpoints'][arm][seed]:
                raise ValueError(f'{run}: checkpoint differs from development-only selection')
            selected[run] = chosen_stats['accuracy']
            epochs, long = inspect_history(states.get(run, {}), run, phase, training, chosen)
            log.extend(long)
            for step, stats, _ in candidates:
                loss = training['checkpoints'][str(step)]['sft_validation_loss']
                if not isfinite(loss):
                    raise ValueError(f'{run}/{step}: checkpoint has no finite SFT validation loss')
                curves.append({'phase': phase, 'arm': arm, 'seed': int(seed), 'step': step, 'epoch': epochs.get(step),
                               'nominal_train_rows_seen': min(step * batch, training['train_rows']),
                               'peak_learning_rate': training['learning_rate'], 'stop_step': training['stop_step'],
                               'planned_optimizer_steps': training['planned_optimizer_steps'],
                               'sft_validation_loss': loss, 'dev_accuracy_pct': 100 * stats['accuracy'],
                               'dev_valid': stats['valid'], 'dev_correct': stats['correct'], 'dev_questions': len(dev_ids),
                               'selected_on_dev': step == chosen})
            if arm in tested:
                correct, stats = score_cell(bundle['test'][run], test_ids, test_gold, run, chosen, 'test',
                                            {**shared, **test_hashes, 'adapter': checkpoint_binding(training, chosen),
                                             'selection_sha256': selection_sha, 'training_receipt_sha256': receipt_sha})
                correct_sets[run] = correct
                results[run] = {**stats, 'step': chosen}
                test_rows.append({'phase': phase, 'arm': arm, 'seed': int(seed), 'selected_step': chosen,
                                  'dev_accuracy_pct': 100 * chosen_stats['accuracy'], 'test_accuracy_pct': 100 * stats['accuracy'],
                                  'correct': stats['correct'], 'valid': stats['valid'], 'questions': len(test_ids)})
    if set(bundle['dev']) != expected_dev:
        raise ValueError(f'{phase}: development runs differ from checkpoint schedule')
    for arm in arms:
        if not isclose(selection['mean_accuracy'][arm], mean(selected[f'{arm}-{seed}'] for seed in SEEDS), abs_tol=1e-12):
            raise ValueError(f'{arm}: development mean differs from the frozen selection')
    return results, correct_sets, curves, log, test_rows


def configuration_rows(bundle, phase, states, counts):
    if set(counts) - bundle['training'].keys() or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError(f'{phase}: invalid measured adapter parameter counts')
    rows = []
    for run, training in bundle['training'].items():
        arm, seed = run.rsplit('-', 1)
        state = states.get(run, {})
        if isinstance(state, dict):
            runtime = next((row['train_runtime'] for row in reversed(state.get('log_history', []))
                            if 'train_runtime' in row), None)
            flops = state.get('total_flos')
        else:
            runtime, flops = None, None
        rows.append({'phase': phase, 'arm': arm, 'seed': int(seed), **configuration(arm, training),
                     'measured_adapter_parameters': counts.get(run), 'peak_learning_rate': training['learning_rate'],
                     'train_rows': training['train_rows'], 'validation_rows': training['validation_rows'],
                     'optimizer_steps': training['optimizer_steps'],
                     'planned_optimizer_steps': training['planned_optimizer_steps'],
                     'trainer_runtime_seconds': runtime, 'trainer_estimated_flos': flops})
    return rows


def question_type_rows(frozen, expected_inputs_sha, labels, old_correct, current_correct):
    cases, actual_sha = rows_by_id(frozen / 'native/inputs.jsonl')
    if actual_sha != expected_inputs_sha:
        raise ValueError('question-type metadata differs from the frozen input')
    grouped = {}
    for case in labels:
        row = cases[case]
        grouped.setdefault(row['question_type'], []).append(case)
    rows = []
    for phase, outcomes in (('prior_data_lr', old_correct), ('targeted_ablation', current_correct)):
        for run, correct_set in outcomes.items():
            arm, seed = run.rsplit('-', 1)
            for question_type, ids in sorted(grouped.items()):
                correct = sum(case in correct_set for case in ids)
                rows.append({'phase': phase, 'arm': arm, 'seed': int(seed), 'question_type': question_type,
                             'images': len({cases[case]['group_id'] for case in ids}), 'questions': len(ids),
                             'correct': correct, 'accuracy_pct': 100 * correct / len(ids)})
    return rows


def simultaneous_interval(clusters, candidate, control, draws, seed):
    """Use the paired image bootstrap with Bonferroni tails for four comparisons."""
    pairs = [(len(group), sum(sum(case in run for case in group) for run in candidate)
              - sum(sum(case in run for case in group) for run in control)) for group in clusters]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        denominator, difference = map(sum, zip(*rng.choices(pairs, k=len(pairs))))
        samples.append(difference / (len(SEEDS) * denominator))
    samples.sort()
    tail = ALPHA / (2 * len(ARMS))
    return [samples[int(q * (draws - 1))] for q in (tail, 1 - tail)]


def contrast(candidate_name, control_name, phase, result, correct, clusters, bootstrap, adjusted=False):
    names = [[f'{arm}-{seed}' for seed in SEEDS] for arm in (candidate_name, control_name)]
    candidate, control = [[correct[key] for key in keys] for keys in names]
    cand, ref = [summarize_profile([result[key] for key in keys]) for keys in names]
    draws, seed = bootstrap['draws'], bootstrap['seed']
    paired = paired_contrast(clusters, candidate, control, draws, seed)
    lower, upper = paired['paired_image_bootstrap_ci95']
    low_adj, high_adj = simultaneous_interval(clusters, candidate, control, draws, seed) if adjusted else (None, None)
    return {'phase': phase, 'split': 'test', 'candidate': candidate_name, 'control': control_name,
            'candidate_mean_pct': 100 * cand['mean'], 'candidate_seed_sd_pp': 100 * cand['seed_sample_std'],
            'control_mean_pct': 100 * ref['mean'], 'control_seed_sd_pp': 100 * ref['seed_sample_std'],
            'delta_pp': 100 * paired['delta'], 'nominal_ci95_lower_pp': 100 * lower, 'nominal_ci95_upper_pp': 100 * upper,
            'simultaneous_ci98_75_lower_pp': None if low_adj is None else 100 * low_adj,
            'simultaneous_ci98_75_upper_pp': None if high_adj is None else 100 * high_adj,
            **{f'delta_seed_{value}_pp': 100 * (result[a]['accuracy'] - result[b]['accuracy'])
               for value, a, b in zip(SEEDS, *names)},
            'interpretation': ('higher' if low_adj > 0 else 'lower' if high_adj < 0 else 'uncertain') if adjusted
            else ('higher' if lower > 0 else 'lower' if upper < 0 else 'uncertain')}


def collect(args):
    new, new_sha = read_bundle(args.bundle)
    previous, previous_sha = read_bundle(args.previous)
    dev_truth = load_dev(args.dev)
    _, labels, clusters, bootstrap, frozen = load_frozen(args.frozen)
    control = previous['training']['real-42']
    same = ('dataset_sha256', 'data_selection_sha256', 'train_sha256', 'validation_sha256', 'train_rows',
            'validation_rows', 'model_id', 'model_revision', 'model_files', 'learning_rate')
    if control['learning_rate'] != 1e-4 or any(any(run[key] != control[key] for key in same)
                                               for run in new['training'].values()):
        raise ValueError('ablation changes the fixed data, model or base learning rate')
    if (set(new['selection']['test_arms']) != set(ARMS) or set(previous['selection']['selected_by_seed']) != set(SEEDS)
            or set(previous['selection']['selected_by_seed'].values()) != {'real'}):
        raise ValueError('expected all four test ablations and the previously selected real 1e-4 control')
    old_binding = previous['test']['real-42']['binding']
    shared = {'policy': old_binding['policy'], 'prompt': old_binding['prompt'],
              'reference_protocol_sha256': frozen['protocol.json']}
    test_truth = labels, {'inputs_sha256': frozen['inputs.jsonl'], 'manifest_sha256': frozen['manifest.json']}
    previous_states = new.get('previous_trainer_states', {})
    new_states = new.get('trainer_states', {})
    old, old_correct, old_curves, old_log, old_test = replay(previous, PREVIOUS_ARMS, ('baseline', 'real'),
                                                           'prior_data_lr', dev_truth, test_truth, shared, previous_states)
    current, current_correct, current_curves, current_log, current_test = replay(new, ARMS, ARMS,
                                                                                 'targeted_ablation', dev_truth, test_truth, shared, new_states)
    prior_means = previous['selection']['mean_accuracy']
    previous_test = contrast('real', 'baseline', 'prior_data_lr', old, old_correct, clusters, bootstrap)
    comparisons = [previous_test]
    for name, ref in (('real', 'baseline'), ('real', 'real-low')):
        comparisons.append({'phase': 'prior_data_lr', 'split': 'dev', 'candidate': name, 'control': ref,
                            'candidate_mean_pct': 100 * prior_means[name], 'control_mean_pct': 100 * prior_means[ref],
                            'delta_pp': 100 * (prior_means[name] - prior_means[ref]),
                            'interpretation': 'checkpoint or learning-rate selection evidence'})
    comparisons += [contrast(arm, 'real', 'targeted_ablation', {**old, **current}, {**old_correct, **current_correct},
                             clusters, bootstrap, adjusted=True) for arm in ARMS]
    summary = {'sha256': {'ablation_bundle': new_sha, 'previous_bundle': previous_sha,
                          'frozen_test': frozen, 'frozen_dev': dev_truth[1]},
               'cohort': {'test_questions': len(labels), 'test_images': len(clusters), 'dev_questions': len(dev_truth[0])},
               'bootstrap': {**bootstrap, 'multiplicity': 'Bonferroni over the four targeted test contrasts',
                             'simultaneous_level': 1 - ALPHA / len(ARMS)},
               'question_type_scope': 'Descriptive question counts and accuracy. An image may appear in multiple types.',
               'history_coverage': {'prior_runs': sorted(previous_states), 'ablation_runs': sorted(new_states)},
               'results': {**old, **current}, 'contrasts': comparisons}
    return summary, {'checkpoint_curves.csv': old_curves + current_curves, 'training_history.csv': old_log + current_log,
                     'test_by_seed.csv': old_test + current_test, 'contrasts.csv': comparisons,
                     'configuration.csv': configuration_rows(previous, 'prior_data_lr', previous_states,
                                                             new.get('previous_adapter_parameters', {}))
                     + configuration_rows(new, 'targeted_ablation', new_states, new.get('adapter_parameters', {})),
                     'test_by_question_type.csv': question_type_rows(args.frozen, frozen['inputs.jsonl'], labels,
                                                                    old_correct, current_correct)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Completed compact ablation bundle (.json.gz)')
    parser.add_argument('--previous', type=Path, default=PARENT / 'results/compact-results.json.gz')
    parser.add_argument('--dev', type=Path, default=PARENT / 'prepared/native-dev')
    parser.add_argument('--frozen', type=Path, required=True, help='Original final-expansion directory')
    parser.add_argument('--output', type=Path, required=True, help='Create a new analysis directory')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('choose a new output directory')
        summary, tables = collect(args)
        with TemporaryDirectory(dir=args.output.parent) as scratch:
            staged = Path(scratch) / 'analysis'
            staged.mkdir()
            (staged / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
            for name, rows in tables.items():
                fields = list(dict.fromkeys(field for row in rows for field in row))
                if not fields:
                    fields = ['phase', 'arm', 'seed', 'step', 'epoch', 'nominal_train_rows_seen',
                              'metric', 'value', 'selected_checkpoint']
                with (staged / name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            staged.rename(args.output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
