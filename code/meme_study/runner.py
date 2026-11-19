"""CPU orchestration with durable attempt accounting and conservative resume.

Only backend.generate performs inference. No gold, scoring, output repair,
automatic retry or model-loading code belongs here. The same output directory
is an immutable run identity and the scope of its global attempt budget.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from . import prompts
from .prompts import build_messages


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append(path, value):
    """Flush a complete line before returning; never rewrite historical logs."""
    line = (_json(value) + '\n').encode('utf-8')
    with path.open('ab') as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _read_lines(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError(f'{path.name}: torn log; manual reconciliation required')
    try:
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('non-object log record')
        return rows
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'{path.name}: corrupt log; manual reconciliation required') from exc


@contextmanager
def _run_lock(directory):
    # Keep this inode permanently. Unlinking a lock can allow two owners.
    with (directory / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process owns this run and attempt budget') from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _stages(workflow):
    if workflow == 'D':
        return [('final', 160)]
    if workflow == 'R':
        return [('draft', 80), ('final', 160)]
    if workflow == 'V':
        return [('observation', 80), ('final', 160)]
    raise ValueError('unknown workflow')


def _validate_jobs(jobs, attempt_cap):
    if type(attempt_cap) is not int or attempt_cap < 0:
        raise ValueError('attempt_cap must be a nonnegative integer')
    if not isinstance(jobs, list):
        raise ValueError('jobs must be a list')
    seen = set()
    for job in jobs:
        for key in ('job_id', 'model', 'input_sha256'):
            if not isinstance(job.get(key), str) or not job[key]:
                raise ValueError(f'job requires {key}')
        if job['job_id'] in seen:
            raise ValueError('duplicate job_id')
        seen.add(job['job_id'])
        stages = _stages(job['workflow'])
        if type(job.get('planned_calls')) is not int or job['planned_calls'] != len(stages):
            raise ValueError('planned_calls does not match workflow')
        # Preflight only the first stage; no artificial second-stage response.
        build_messages(job['case'], job['workflow'], stages[0][0])


def _bind_run(directory, jobs, backend, attempt_cap):
    config = getattr(backend, 'run_config', {})
    if not isinstance(config, dict):
        raise ValueError('backend.run_config must be a JSON object')
    manifest = {
        'schema_version': 1, 'jobs': jobs, 'jobs_sha256': _digest(jobs),
        'attempt_cap': attempt_cap, 'run_config': config,
        'prompts_sha256': hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Round-trip to detach caller-owned dictionaries before any backend call.
    manifest = json.loads(_json(manifest))
    path = directory / 'run.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('resume refused: jobs, input hashes, code or run parameters changed')
    else:
        if any((directory / name).exists() for name in ('events.jsonl', 'records.jsonl')):
            raise ValueError('logs without run.json require manual reconciliation')
        temporary = directory / '.run.json.pending'
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(_json(manifest) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(directory)
    return manifest


def _load_events(directory, jobs, run_config=None):
    run_manifest = json.loads((directory / 'run.json').read_text())
    if run_config is None:
        run_config = run_manifest['run_config']
    by_id = {job['job_id']: job for job in jobs}
    stages = {job_id: {} for job_id in by_id}
    attempts = {}
    for event in _read_lines(directory / 'events.jsonl'):
        job_id, stage, attempt = event.get('job_id'), event.get('stage'), event.get('attempt')
        if job_id not in by_id or stage not in dict(_stages(by_id[job_id]['workflow'])):
            raise ValueError('event does not belong to this run; manual reconciliation required')
        if event.get('input_sha256') != by_id[job_id]['input_sha256']:
            raise ValueError('event input hash mismatch; manual reconciliation required')
        if event.get('job_sha256') != _digest(by_id[job_id]):
            raise ValueError('event job hash mismatch; manual reconciliation required')
        job = by_id[job_id]
        cap = dict(_stages(job['workflow']))[stage]
        expected = {'schema_version': 1, 'stage_attempt': 1,
                    'model': job['model'], 'run_config': run_config, 'max_new_tokens': cap,
                    'checkpoint': run_config.get('checkpoint', job['model']),
                    'device': run_config.get('device', 'unspecified'),
                    'runner_sha256': run_manifest['runner_sha256'],
                    'prompts_sha256': run_manifest['prompts_sha256']}
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError('event configuration differs from run manifest; manual reconciliation required')
        if event.get('event') == 'dispatch':
            if type(attempt) is not int or attempt != len(attempts) + 1 or stage in stages[job_id]:
                raise ValueError('duplicate/out-of-order attempt; manual reconciliation required')
            sequence = [name for name, _ in _stages(by_id[job_id]['workflow'])]
            for earlier in sequence[:sequence.index(stage)]:
                prior = stages[job_id].get(earlier, {}).get('outcome', {})
                if prior.get('event') != 'result':
                    raise ValueError('missing completed first stage; manual reconciliation required')
            previous = (stages[job_id][sequence[0]]['outcome']['text']
                        if stage == 'final' and len(sequence) == 2 else None)
            messages = build_messages(job['case'], job['workflow'], stage, previous)
            if event.get('messages_sha256') != _digest(messages):
                raise ValueError('stage message hash differs from reconstructed input; manual reconciliation required')
            attempts[attempt] = event
            stages[job_id][stage] = {'dispatch': event}
        elif event.get('event') in {'result', 'error'}:
            pending = stages[job_id].get(stage, {})
            if (attempt not in attempts or pending.get('dispatch', {}).get('attempt') != attempt
                    or 'outcome' in pending):
                raise ValueError('outcome without unique dispatch; manual reconciliation required')
            if event.get('messages_sha256') != pending['dispatch']['messages_sha256']:
                raise ValueError('outcome differs from dispatch message hash; manual reconciliation required')
            if event['event'] == 'result':
                _validate_response(event, cap)
            pending['outcome'] = event
        else:
            raise ValueError('unknown event; manual reconciliation required')
    return stages, len(attempts)


def _validate_response(result, max_new_tokens):
    if not isinstance(result, dict):
        raise ValueError('backend receipt must be an object')
    if not isinstance(result.get('text'), str):
        raise ValueError('backend receipt requires raw text')
    for field in ('input_tokens', 'output_tokens'):
        if type(result.get(field)) is not int or result[field] < 0:
            raise ValueError(f'backend receipt requires nonnegative {field}')
    if result['output_tokens'] > max_new_tokens:
        raise ValueError('backend exceeded max_new_tokens')
    if not isinstance(result.get('finish_reason'), str) or not result['finish_reason']:
        raise ValueError('backend receipt requires finish_reason')
    latency = result.get('latency_s')
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
        raise ValueError('backend receipt requires finite nonnegative latency_s')
    return {key: result[key] for key in
            ('text', 'input_tokens', 'output_tokens', 'finish_reason', 'latency_s')}


def _record(job, outcome=None, status=None, error=None):
    row = {'schema_version': 1, 'job_id': job['job_id'], 'case_id': job['case']['case_id'],
           'model': job['model'], 'workflow': job['workflow'],
           'input_sha256': job['input_sha256'], 'job_sha256': _digest(job)}
    if outcome is not None:
        for key in ('stage', 'attempt', 'timestamp', 'text', 'input_tokens', 'output_tokens',
                    'finish_reason', 'latency_s', 'error', 'error_type', 'run_config',
                    'checkpoint', 'device', 'messages_sha256', 'raw_receipt'):
            if key in outcome:
                row[key] = outcome[key]
        status = ('error' if outcome['event'] == 'error' else
                  'truncated' if outcome['finish_reason'] == 'length' else 'completed')
    row['status'] = status
    row['truncated'] = status == 'truncated'
    if error is not None:
        row['error'] = error
    return row


def run_jobs(jobs, backend, output_dir, *, attempt_cap, before_dispatch=None):
    """Run/resume an immutable manifest and return counts plus job records.

    Files: run.json binds inputs/config/code; events.jsonl records durable
    dispatch intents and outcomes; records.jsonl contains one terminal record
    per answered or infrastructure-failed job. Budget/pending-intent statuses
    appear in summary records but are not final model outputs.

    Every intent consumes budget, even after a crash. An unresolved intent is
    never repeated automatically. Completed stages are reusable only inside
    their own job. Infrastructure errors terminate that job; other jobs run.
    An optional before_dispatch(context) guard may return False before any
    local attempt is recorded. The CLI uses this for a study-wide budget.
    """
    _validate_jobs(jobs, attempt_cap)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _run_lock(directory):
        manifest = _bind_run(directory, jobs, backend, attempt_cap)
        jobs, config = manifest['jobs'], manifest['run_config']
        states, attempts = _load_events(directory, jobs, config)
        if attempts > attempt_cap:
            raise ValueError('persisted attempts exceed the bound budget')
        saved = {}
        for row in _read_lines(directory / 'records.jsonl'):
            if row.get('job_id') not in states or row['job_id'] in saved:
                raise ValueError('unknown/duplicate final record; manual reconciliation required')
            saved[row['job_id']] = row
        # A missing log must never turn a saved response into a fresh request.
        # Check all saved rows before dispatching anything in this invocation.
        by_id = {job['job_id']: job for job in jobs}
        for job_id, existing in saved.items():
            terminal = [entry['outcome'] for stage, entry in states[job_id].items()
                        if 'outcome' in entry
                        and (stage == 'final' or entry['outcome']['event'] == 'error')]
            if len(terminal) != 1 or _record(by_id[job_id], terminal[0]) != existing:
                raise ValueError('final record has no matching terminal event; manual reconciliation required')
        records = []
        for job in jobs:
            state = states[job['job_id']]
            previous = None
            row = None
            for stage, cap in _stages(job['workflow']):
                entry = state.get(stage)
                if entry is not None and 'outcome' not in entry:
                    row = _record(job, status='reconciliation_required', error=
                                  'Unfinished dispatch intent: manual reconciliation required; no automatic retry.')
                    break
                outcome = entry.get('outcome') if entry else None
                if outcome is None:
                    if attempts >= attempt_cap:
                        row = _record(job, status='budget_exhausted', error='Attempt cap reached before dispatch.')
                        break
                    messages = build_messages(job['case'], job['workflow'], stage, previous)
                    context = {'job_id': job['job_id'], 'stage': stage, 'attempt': attempts + 1,
                               'max_new_tokens': cap, 'input_sha256': job['input_sha256']}
                    if before_dispatch is not None and before_dispatch(context) is False:
                        row = _record(job, status='budget_exhausted', error='External dispatch guard refused a new attempt.')
                        break
                    attempts += 1
                    common = {
                        'schema_version': 1, 'job_id': job['job_id'], 'stage': stage,
                        'attempt': attempts, 'stage_attempt': 1, 'input_sha256': job['input_sha256'],
                        'job_sha256': _digest(job), 'model': job['model'],
                        'checkpoint': config.get('checkpoint', job['model']),
                        'device': config.get('device', 'unspecified'), 'run_config': config,
                        'max_new_tokens': cap, 'messages_sha256': _digest(messages),
                        'runner_sha256': manifest['runner_sha256'],
                        'prompts_sha256': manifest['prompts_sha256'],
                    }
                    dispatch = dict(common, event='dispatch', timestamp=_now())
                    _append(directory / 'events.jsonl', dispatch)
                    start = time.perf_counter()
                    raw_receipt = None
                    try:
                        raw_receipt = backend.generate(messages, max_new_tokens=cap)
                        receipt = _validate_response(raw_receipt, cap)
                    except Exception as exc:
                        outcome = dict(common, event='error', timestamp=_now(),
                                       error=str(exc), error_type=type(exc).__name__,
                                       latency_s=time.perf_counter() - start)
                        if raw_receipt is not None:
                            try:
                                outcome['raw_receipt'] = json.loads(_json(raw_receipt))
                            except (TypeError, ValueError):
                                outcome['raw_receipt'] = repr(raw_receipt)
                            if isinstance(raw_receipt, dict) and isinstance(raw_receipt.get('text'), str):
                                outcome['text'] = raw_receipt['text']
                    else:
                        outcome = dict(common, event='result', timestamp=_now(), **receipt)
                    _append(directory / 'events.jsonl', outcome)
                    state[stage] = {'dispatch': dispatch, 'outcome': outcome}
                if outcome['event'] == 'error' or stage == 'final':
                    row = _record(job, outcome)
                    break
                previous = outcome['text']
            if row['status'] in {'completed', 'truncated', 'error'}:
                existing = saved.get(job['job_id'])
                if existing is not None and existing != row:
                    raise ValueError('final record differs from event log; manual reconciliation required')
                if existing is None:
                    _append(directory / 'records.jsonl', row)
            elif job['job_id'] in saved:
                raise ValueError('final record has no terminal event; manual reconciliation required')
            records.append(row)
        return {
            'attempts': attempts, 'attempt_cap': attempt_cap, 'jobs': len(jobs),
            'completed': sum(row['status'] in {'completed', 'truncated'} for row in records),
            'truncated': sum(row['status'] == 'truncated' for row in records),
            'error': sum(row['status'] == 'error' for row in records),
            'budget_exhausted': sum(row['status'] == 'budget_exhausted' for row in records),
            'reconciliation_required': sum(row['status'] == 'reconciliation_required' for row in records),
            'records': records,
        }
