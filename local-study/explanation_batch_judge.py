"""Freeze and execute the local Task A judge with bounded native MLX batches."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import internvl_batch


def commit_events(path, identity, state, events):
    """Add a validated event group while preserving the exact ledger prefix."""
    from meme_study.reference_runner import _append_event, _load_ledger
    from meme_study.runner import _sync_directory

    chain = {key: state[key] for key in ('events', 'last_sha256')}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.judge-ledger-', delete=False) as stream:
        pending = Path(stream.name)
        try:
            if path.exists():
                with path.open('rb') as source:
                    shutil.copyfileobj(source, stream)
                os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
            stream.flush()
            stream.close()
            for kind, payload in events:
                _append_event(pending, chain, kind, payload)
            committed = _load_ledger(pending, identity)
            os.replace(pending, path)
            _sync_directory(path.parent)
        finally:
            pending.unlink(missing_ok=True)
    return committed


def outcome_records(dispatches, generate):
    from meme_study.reference_runner import _error_record, _retain_raw
    from meme_study.runner import _now, _validate_response

    try:
        raw_rows = generate()
        if not isinstance(raw_rows, list) or len(raw_rows) != len(dispatches):
            raise ValueError('judge backend must return one result for each dispatched job')
    except Exception as error:
        return [dict(_error_record(common, 'failed', type(error).__name__, str(error) or repr(error)),
                     raw_receipt=None, raw_receipt_format='json') for common in dispatches]

    outcomes = []
    for common, raw in zip(dispatches, raw_rows):
        try:
            receipt = _validate_response(raw, common['max_new_tokens'])
            record = {**common, 'status': 'ok', 'timestamp': _now(), **receipt,
                      'truncated': receipt['finish_reason'] == 'length'}
        except (ValueError, TypeError) as error:
            record = _error_record(common, 'failed', type(error).__name__, str(error))
            if isinstance(raw, dict) and isinstance(raw.get('text'), str):
                record['text'] = raw['text']
        record['raw_receipt'], record['raw_receipt_format'] = _retain_raw(raw)
        outcomes.append(record)
    return outcomes


def run_groups(path, identity, jobs, config, prepare, generate):
    from meme_study.data import content_hash
    from meme_study.reference_runner import _common, _error_record, _load_ledger, _validate_batch

    state = _load_ledger(path, identity)
    initial_attempts = state['attempts']
    if not state['events']:
        raise ValueError('judge execution needs an existing ledger with candidate results')
    if not jobs:
        return state, 0
    _validate_batch(jobs, config)
    if any(job['task_kind'] != 'judge' for job in jobs) or len({job['split'] for job in jobs}) != 1:
        raise ValueError('one judge invocation requires one split of judge jobs')
    unseen, pending = [], []
    selected = {job['job_id'] for job in jobs}
    for job in jobs:
        job_id = job['job_id']
        binding = state['jobs'].get(job_id)
        if binding is None:
            unseen.append(job)
            continue
        batch_id, bound_job = binding
        if bound_job != job:
            raise ValueError('compiled judge input differs from the ledger binding')
        if job_id in state['terminal']:
            continue
        batch = state['batches'][batch_id]
        if batch['run_config'] != config or job_id not in state['dispatches']:
            raise ValueError('finish the previously bound judge jobs with their original runtime')
        if any(member['job_id'] not in selected for member in batch['jobs']):
            raise ValueError('select a prefix large enough to resume the existing judge batch')
        pending.append(_error_record(state['dispatches'][job_id], 'interrupted', 'InterruptedDispatch',
                                     'Pending dispatch has no outcome; no automatic retry.'))
    if pending:
        state = commit_events(path, identity, state, [('outcome', record) for record in pending])

    limit = identity['attempt_cap']
    rows = config['native_batching']['batch_size']
    if unseen and state['attempts'] < limit:
        prepare()
    for start in range(0, len(unseen), rows):
        group = unseen[start:start + rows]
        batch_id = 'local-a-batch/' + content_hash([job['job_id'] for job in group])
        batch = {'batch_id': batch_id, 'jobs': group, 'run_config': config}
        events, dispatched = [('batch', batch)], []
        attempt = state['attempts']
        for job in group:
            if attempt == limit:
                common = _common(job, identity['study_id'], batch_id, None)
                blocked = _error_record(common, 'failed', 'BudgetExhausted',
                                        'Shared study attempt cap reached before dispatch.')
                events.append(('blocked', blocked))
                continue
            attempt += 1
            common = _common(job, identity['study_id'], batch_id, attempt)
            events.append(('dispatch', common))
            dispatched.append(common)
        state = commit_events(path, identity, state, events)
        if dispatched:
            outcomes = outcome_records(dispatched, lambda: generate(group[:len(dispatched)]))
            state = commit_events(path, identity, state, [('outcome', row) for row in outcomes])
    return state, state['attempts'] - initial_attempts


def judge_config(code, args, receipt=None):
    config = code.runtime('judge', args)
    for name, expected in args.pin['files'].items():
        if config['model_files_sha256'].get(name) != expected['sha256']:
            raise ValueError(f'pinned model digest mismatch: {name}')
    receipt_digest = (receipt or {}).get('runtime', {}).get('judge', {}).get(
        'native_batching', {}).get('technical_probe_sha256')
    if args.calibration_receipt is not None:
        receipt_digest = code.sha(args.calibration_receipt)
    config['native_batching'] = {
        'batch_size': args.batch_size,
        'prefill_token_budget': internvl_batch.PREFILL_TOKEN_BUDGET,
        'group_order': 'contiguous in official seed-6000 judge order, with a smaller final group',
        'embeddings': 'native single-request multimodal preprocessing',
        'decoder': 'native MLX BatchGenerator greedy',
        'decoder_code_sha256': code.sha(internvl_batch.__file__),
        'judge_executor_sha256': code.sha(__file__),
        'technical_probe_sha256': receipt_digest,
    }
    return config


def execute(code, args):
    from meme_study.runner import _run_lock

    note = args.test_judge_note.read_text(encoding='utf-8') if args.test_judge_note else None
    study = code.ExplanationStudy(code.load_prepared(args.prepared), note)
    identity = {'schema_version': 1, 'protocol_id': code.reference_protocol.PROTOCOL_ID,
                'study_id': study.study_id, 'attempt_cap': code.ATTEMPT_CAP,
                'runner_sha256': code.sha(code.reference_runner.__file__)}
    run_lock = _run_lock(args.runs) if args.command == 'run' and args.runs.is_dir() else nullcontext()
    with code._execution_lock(args.runs), run_lock:
        snapshot = code.read_ledger(study, args.runs)
        if args.command == 'freeze':
            if any(job['split'] == 'test' for _, job in snapshot['jobs'].values()):
                raise ValueError('freeze must precede all test dispatches')
            roles = {stage: code.runtime(stage, args) for stage in code.ARMS}
            roles['judge'] = judge_config(code, args)
            tasks = [job for job in study.jobs if job['split'] == 'test']
            code._check_images(tasks, args.data_root)
            value = {'scope': code.SCOPE, 'protocol': study.frozen_protocol, 'runtime': roles,
                     'prepared_sha256': study.prepared, 'judge_note_sha256': code.content_hash(study.notes),
                     'test_explanations': len(tasks), 'development': code.summarize(study, snapshot, 'dev')}
            code.write_new(args.freeze_receipt, value)
            return {'freeze_sha256': code.sha(args.freeze_receipt), 'test_explanations': len(tasks)}

        frozen, freeze_sha = (code.locked_test(study, snapshot, args.freeze_receipt)
                              if args.split == 'test' else (None, None))
        config = judge_config(code, args, frozen)
        if frozen is not None:
            code.freeze_guard(frozen, study.frozen_protocol, 'judge', config)
            config['test_freeze_sha256'] = freeze_sha
        jobs = study.judge_jobs(args.split, snapshot)[:args.max_jobs]
        code._check_images(jobs, args.data_root)
        backend = code.MLXBackend('judge', args, config)

        def generate(group):
            code._check_images(group, args.data_root)
            requests = internvl_batch.build_requests(backend, group)
            receipts, stats = internvl_batch.generate_batch(backend, requests, config['generation'])
            return [dict(row, batch_stats=stats) for row in receipts]

        path = args.runs / code._LEDGER
        end, new_attempts = run_groups(path, identity, jobs, config, backend.prepare, generate)
        return {'scope': code.SCOPE, 'split': args.split, 'selected': len(jobs), 'attempts': end['attempts'],
                'new_attempts': new_attempts, 'ledger_sha256': code.sha(path),
                'status': dict(Counter(end['terminal'][job['job_id']]['status'] for job in jobs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    for name in ('prepared', 'data-root', 'runs', 'base-model', 'adapter', 'freeze-receipt',
                 'calibration-receipt', 'dev-calibration-note'):
        parser.add_argument('--' + name, type=lambda path: Path(path).resolve())
    parser.add_argument('--batch-size', type=int, choices=range(2, 7), default=2)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    run = commands.add_parser('run')
    run.add_argument('--split', choices=('dev', 'test'), required=True)
    run.add_argument('--max-jobs', type=int)
    args = parser.parse_args()
    if args.command == 'freeze' and (args.base_model is None or args.adapter is None):
        parser.error('freeze requires --base-model and --adapter')
    if args.command == 'run' and args.max_jobs is not None and (args.split != 'dev' or args.max_jobs < 1):
        parser.error('a positive --max-jobs prefix is available only on dev')

    project = args.project.resolve()
    local = project / 'local-study'
    sys.path[:0] = [str(local), str(project / 'code')]
    import explanation_study as code

    args.runs = args.runs or local / 'explanation-runs'
    args.prepared = args.prepared or local / 'prepared'
    args.data_root = args.data_root or project / 'data' / 'assets'
    args.freeze_receipt = args.freeze_receipt or args.runs.with_name(args.runs.name + '.freeze.json')
    args.test_judge_note = args.dev_calibration_note
    args.pin = json.loads((local / 'models' / 'snapshots.json').read_bytes())['judge']
    args.judge_model = project / args.pin['directory']
    args.judge_revision = args.pin['revision']
    try:
        print(json.dumps(execute(code, args), sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
