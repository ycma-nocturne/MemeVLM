"""Freeze, run, and score the synthetic English paired evaluation."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(PROJECT / 'code'), str(PROJECT)]

from meme_study import reference_scoring, runner
from meme_study.data import content_hash
from meme_study.freeze import _safe_file

import backend
import materials


ARMS = ('base', 'adapter')
SPLITS = {'dev': 'validation', 'test': 'test'}


def read_json(path):
    value = reference_scoring._decode(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def write_new(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            runner._sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def release_identity(cohorts, checksums):
    implementation = [Path(__file__), Path(materials.__file__), Path(backend.__file__),
                      Path(reference_scoring.__file__), Path(runner.__file__),
                      PROJECT / 'code/meme_study/backends.py', PROJECT / 'code/meme_study/reference_backends.py',
                      PROJECT / 'code/meme_study/data.py', PROJECT / 'code/meme_study/freeze.py']
    return {
        'protocol': materials.PROTOCOL, 'version': 1, 'sources': checksums,
        'cases': {split: {**materials.describe(cohorts[cohort]), 'sha256': content_hash(cohorts[cohort])}
                  for split, cohort in SPLITS.items()},
        'policy': backend.POLICY, 'code_sha256': {path.name: materials.sha(path) for path in implementation},
    }


def trained_identity(args, checksums):
    from training.gpu import launch, prepare_data

    manifest, dataset_digest = prepare_data.verify_dataset(args.dataset)
    if manifest.get('sources', {}).get('bilingual') != checksums:
        raise ValueError('evaluation sources differ from the bilingual release bound to training')
    selected_path = args.adapter_run / 'selected-adapter.json'
    selected = read_json(selected_path)
    if (selected.get('receipt_version') != 1 or selected.get('mode') != 'train'
            or type(selected.get('optimizer_steps')) is not int or selected['optimizer_steps'] < 1):
        raise ValueError('evaluation requires a selected adapter from a completed training run')
    model_files = launch.verify_model(args.model)
    expected = {'dataset_sha256': dataset_digest, 'model_id': backend.MODEL_ID,
                'model_revision': backend.MODEL_REVISION, 'model_files': model_files, 'profile': manifest['profile']}
    adapter = args.adapter_run / 'trainer'
    for name, field in (('adapter_model.safetensors', 'adapter_sha256'),
                        ('adapter_config.json', 'adapter_config_sha256')):
        expected[field] = materials.sha(adapter / name)
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError('selected adapter receipt differs from verified training inputs or artifacts')
    if read_json(adapter / 'adapter_config.json').get('peft_type') != 'LORA':
        raise ValueError('evaluation requires the selected trained PEFT LoRA adapter')
    return {**expected, 'selected_adapter_receipt_sha256': materials.sha(selected_path),
            'training_validator_sha256': {module.__name__: materials.sha(module.__file__)
                                          for module in (launch, prepare_data)}}


def event_binding(case):
    return {'case_id': case['id'], 'input_sha256': case['input_sha256']}


def checked_response(row):
    response = runner._validate_response(row, materials.MAX_TOKENS)
    tokens = row.get('output_token_ids')
    if (response['input_tokens'] < 1 or not isinstance(tokens, list) or len(tokens) != response['output_tokens']
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('token counts and output IDs must match the generation receipt')
    eos_ids = backend.POLICY['generation']['eos_token_id']
    reason = ('stop' if tokens and tokens[-1] in eos_ids
              else 'length' if len(tokens) == materials.MAX_TOKENS else None)
    if reason is None or response['finish_reason'] != reason or any(token in eos_ids for token in tokens[:-1]):
        raise ValueError('generation must reach EOS or the token cap')
    return dict(response, output_token_ids=tokens)


def read_events(path, cases):
    by_id = {case['id']: case for case in cases}
    dispatched, outcomes = {}, {}
    for row in runner._read_lines(path):
        case_id = row.get('case_id')
        if case_id not in by_id or row.get('input_sha256') != by_id[case_id]['input_sha256']:
            raise ValueError('ledger event differs from the frozen case')
        if row.get('event') == 'dispatch':
            if case_id in dispatched:
                raise ValueError('duplicate evaluation dispatch')
            dispatched[case_id] = row
        elif row.get('event') == 'outcome':
            if case_id not in dispatched or case_id in outcomes:
                raise ValueError('outcome needs one unreconciled dispatch')
            if row.get('status') == 'ok':
                checked_response(row)
            elif row.get('status') not in {'failed', 'interrupted'} or not row.get('error_type'):
                raise ValueError('invalid terminal evaluation status')
            outcomes[case_id] = row
        else:
            raise ValueError('unknown evaluation ledger event')
    return dispatched, outcomes


def execute_cases(directory, cases, config, data_root, predictor):
    data_root = Path(data_root).resolve(strict=True)
    ledger = directory / 'events.jsonl'
    receipt = directory / 'run.json'
    directory.mkdir(parents=True, exist_ok=True)
    with runner._run_lock(directory):
        existing = read_json(receipt) if receipt.is_file() else None
        if existing is not None and existing.get('identity') != config:
            raise ValueError('run identity differs from the frozen split and arm')
        if existing is None and ledger.exists():
            raise ValueError('ledger requires an existing run receipt')
        dispatched, outcomes = read_events(ledger, cases)
        for case in cases:
            case_id = case['id']
            if case_id in dispatched and case_id not in outcomes:
                result = {**event_binding(case), 'event': 'outcome', 'timestamp': runner._now(),
                          'status': 'interrupted', 'error_type': 'InterruptedDispatch'}
                runner._append(ledger, result)
                outcomes[case_id] = result
        unseen = [case for case in cases if case['id'] not in dispatched]
        if unseen:
            hardware = predictor.prepare()
            if existing is None:
                write_new(receipt, {'identity': config, 'hardware': hardware})
            elif existing.get('hardware') != hardware:
                raise ValueError('CUDA runtime differs from the recorded run')
        for case in unseen:
            image = _safe_file(data_root, case['image'])
            if materials.sha(image) != case['image_sha256']:
                raise ValueError('image bytes changed since release validation')
            binding = event_binding(case)
            runner._append(ledger, {**binding, 'event': 'dispatch', 'timestamp': runner._now()})
            try:
                raw = predictor.generate({key: value for key, value in case.items() if key != 'gold'})
                result = dict(checked_response(raw), status='ok')
            except Exception as error:
                result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error) or repr(error)}
            result.update(binding, event='outcome', timestamp=runner._now())
            runner._append(ledger, result)
            outcomes[case['id']] = result
        return {'selected': len(cases), 'new_attempts': len(unseen),
                'statuses': dict(Counter(outcomes[case['id']]['status'] for case in cases)),
                'events_sha256': materials.sha(ledger) if ledger.is_file() else None}


def arm_identity(freeze_sha, split, arm, cases):
    return {'protocol': materials.PROTOCOL, 'freeze_sha256': freeze_sha,
            'split': split, 'arm': arm, 'cases_sha256': content_hash(cases)}


def score_run(args, cases, freeze_digest):
    reports, outcomes = {}, {}
    for arm in ARMS:
        directory = args.runs / args.split / arm
        if not directory.is_dir() or not (directory / 'run.json').is_file():
            reports[arm], outcomes[arm] = None, {}
            continue
        with runner._run_lock(directory):
            run = read_json(directory / 'run.json')
            if run.get('identity') != arm_identity(freeze_digest, args.split, arm, cases):
                raise ValueError(f'{arm} run differs from the frozen comparison')
            path = directory / 'events.jsonl'
            _, outcomes[arm] = read_events(path, cases)
            reports[arm] = {'events_sha256': materials.sha(path) if path.is_file() else None,
                            'hardware': run['hardware']}
    scored = {arm: materials.score_rows(cases, rows) for arm, rows in outcomes.items()}
    summary = materials.summarize(scored)
    if not summary['complete'] and not args.allow_partial:
        raise ValueError('both arms need terminal outcomes for every planned case')
    value = {'protocol': materials.PROTOCOL, 'split': args.split, 'freeze_sha256': freeze_digest,
             'runs': reports, 'summary': summary, 'scores': scored}
    write_new(args.output, value)
    return value


def run(args):
    cohorts, checksums = materials.compile_release(args.data_root)
    release = release_identity(cohorts, checksums)
    if args.command == 'preflight':
        return {'protocol': materials.PROTOCOL, 'sources': checksums,
                'cohorts': {split: materials.describe(rows) for split, rows in cohorts.items()}}
    if args.command == 'freeze':
        release['training'] = trained_identity(args, checksums)
        write_new(args.freeze, release)
        return {'freeze_sha256': materials.sha(args.freeze), 'cases': release['cases']}

    freeze_raw = args.freeze.read_bytes()
    frozen = reference_scoring._decode(freeze_raw.decode('utf-8'))
    if not isinstance(frozen, dict) or {key: value for key, value in frozen.items() if key != 'training'} != release:
        raise ValueError('source, prompt, evaluator, or runtime changed since freeze')
    freeze_digest = hashlib.sha256(freeze_raw).hexdigest()
    cases = cohorts[SPLITS[args.split]]
    if args.command == 'score':
        return score_run(args, cases, freeze_digest)
    if frozen.get('training') != trained_identity(args, checksums):
        raise ValueError('model or training artifacts changed since freeze')
    predictor = backend.CudaBackend(args.model, args.adapter_run / 'trainer' if args.arm == 'adapter' else None,
                                    args.data_root, args.device)
    identity = arm_identity(freeze_digest, args.split, args.arm, cases)
    return execute_cases(args.runs / args.split / args.arm, cases, identity, args.data_root, predictor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=lambda value: Path(value).resolve(),
                        default=PROJECT / 'training' / 'bilingual')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('preflight')
    freeze = commands.add_parser('freeze')
    execute = commands.add_parser('run')
    score = commands.add_parser('score')
    for command in (freeze, execute, score):
        command.add_argument('--freeze', type=Path, required=True)
    for command in (freeze, execute):
        for name in ('model', 'adapter-run', 'dataset'):
            command.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    for command in (execute, score):
        command.add_argument('--split', choices=SPLITS, required=True)
        command.add_argument('--runs', type=Path, required=True)
    execute.add_argument('--arm', choices=ARMS, required=True)
    execute.add_argument('--device', default='cuda:0')
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
