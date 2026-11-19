"""Orchestrate model calls with a shared append-only attempt ledger.

Flush each dispatch before calling the backend and count it against the global
budget. On resume, mark pending dispatches as interrupted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .data import content_hash
from .runner import _append, _json, _now, _read_lines, _run_lock, _validate_response


_PROTOCOL = 'reference_reuse_v1'
_CAPS = {'explanation': 512, 'pair': 512, 'permutation': 512, 'judge': 768}
_LEDGER = 'reference-ledger.jsonl'


def _detach(value):
    try:
        return json.loads(_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('inputs and run configuration must be finite JSON values') from exc


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')


def _messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('job messages must be a nonempty conversation')
    for message in messages:
        if (not isinstance(message, dict) or not isinstance(message.get('role'), str)
                or message['role'] not in {'user', 'assistant', 'system'}
                or not isinstance(message.get('content'), list) or not message['content']):
            raise ValueError('job message has an invalid role or content')
        for block in message['content']:
            if not isinstance(block, dict):
                raise ValueError('message blocks must be objects')
            if block.get('type') == 'text':
                if not isinstance(block.get('text'), str):
                    raise ValueError('text block requires text')
            elif block.get('type') == 'image':
                if ('image' in block) == ('url' in block):
                    raise ValueError('image block requires exactly one image or url path')
                _identifier(block.get('image', block.get('url')), 'image path')
            else:
                raise ValueError('unsupported message block')


def _validate_batch(jobs, config):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('jobs must be a nonempty list')
    if not isinstance(config, dict):
        raise ValueError('backend.run_config must be an object')
    _identifier(config.get('checkpoint'), 'backend checkpoint')
    identifiers, models = set(), set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('job must be an object')
        for name in ('job_id', 'model', 'input_sha256'):
            _identifier(job.get(name), name)
        if job['job_id'] in identifiers:
            raise ValueError('duplicate job ID in batch')
        identifiers.add(job['job_id'])
        models.add(job['model'])
        if job.get('protocol_id') != _PROTOCOL or job.get('readiness') != 'ready':
            raise ValueError('only ready reference_reuse_v1 jobs may execute')
        kind = job.get('task_kind')
        if not isinstance(kind, str) or kind not in _CAPS:
            raise ValueError('unsupported reference task kind')
        if type(job.get('max_new_tokens')) is not int or job['max_new_tokens'] != _CAPS[kind]:
            raise ValueError('job token cap differs from its reference task')
        if job['input_sha256'] != content_hash({k: v for k, v in job.items() if k != 'input_sha256'}):
            raise ValueError('job input hash does not match the complete job')
        _messages(job.get('messages'))
    if models != {config['checkpoint']}:
        raise ValueError('one batch requires one model matching backend checkpoint exactly')


def _common(job, study_id, batch_id, attempt):
    return {'protocol_id': _PROTOCOL, 'study_id': study_id, 'batch_id': batch_id,
            'job_id': job['job_id'], 'input_sha256': job['input_sha256'],
            'model': job['model'], 'task_kind': job['task_kind'],
            'max_new_tokens': job['max_new_tokens'], 'attempt': attempt}


def _error_record(common, status, error_type, error):
    return {**common, 'status': status, 'timestamp': _now(), 'text': None,
            'truncated': False, 'error_type': error_type, 'error': error}


def _check_record(record, common):
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in common.items()):
        raise ValueError('terminal output differs from its bound dispatch')
    status = record.get('status')
    if status == 'ok':
        _validate_response(record, common['max_new_tokens'])
        if type(record.get('truncated')) is not bool or record['truncated'] != (record['finish_reason'] == 'length'):
            raise ValueError('terminal truncation flag differs from its finish reason')
    elif status in {'failed', 'interrupted'}:
        _identifier(record.get('error'), 'terminal error')
        _identifier(record.get('error_type'), 'terminal error type')
    else:
        raise ValueError('unknown terminal status')


def _load_ledger(path, study):
    events = _read_lines(path)
    if path.exists() and not events:
        raise ValueError('empty existing ledger requires manual reconciliation')
    state = {'events': len(events), 'last_sha256': None, 'batches': {}, 'jobs': {},
             'dispatches': {}, 'terminal': {}, 'attempts': 0}
    for index, event in enumerate(events, 1):
        if (event.get('sequence') != index or event.get('previous_sha256') != state['last_sha256']
                or event.get('event_sha256') != content_hash({k: v for k, v in event.items() if k != 'event_sha256'})):
            raise ValueError('ledger sequence or checksum changed; manual reconciliation required')
        kind, payload = event.get('event'), event.get('payload')
        if not isinstance(payload, dict):
            raise ValueError('ledger event requires an object payload')
        state['last_sha256'] = event['event_sha256']
        if index == 1:
            if kind != 'study' or payload != study:
                raise ValueError('study identity, attempt cap or runner code changed')
            continue
        if kind == 'batch':
            _identifier(payload.get('batch_id'), 'batch ID')
            batch_id = payload['batch_id']
            if batch_id in state['batches']:
                raise ValueError('duplicate batch binding in ledger')
            _validate_batch(payload.get('jobs'), payload.get('run_config'))
            for job in payload['jobs']:
                if job['job_id'] in state['jobs']:
                    raise ValueError('same job ID is bound to more than one batch')
                state['jobs'][job['job_id']] = (batch_id, job)
            state['batches'][batch_id] = payload
            continue
        job_id = payload.get('job_id')
        if not isinstance(job_id, str) or job_id not in state['jobs']:
            raise ValueError('ledger event has no bound job')
        batch_id, job = state['jobs'][job_id]
        if kind == 'dispatch':
            attempt = state['attempts'] + 1
            common = _common(job, study['study_id'], batch_id, attempt)
            if (payload != common or job_id in state['dispatches'] or job_id in state['terminal']
                    or attempt > study['attempt_cap']):
                raise ValueError('duplicate, mismatched or over-budget dispatch')
            state['dispatches'][job_id] = payload
            state['attempts'] = attempt
        elif kind == 'outcome':
            if job_id not in state['dispatches'] or job_id in state['terminal']:
                raise ValueError('terminal result has no unique pending dispatch')
            _check_record(payload, state['dispatches'][job_id])
            state['terminal'][job_id] = payload
        elif kind == 'blocked':
            common = _common(job, study['study_id'], batch_id, None)
            if (job_id in state['dispatches'] or job_id in state['terminal']
                    or state['attempts'] < study['attempt_cap']
                    or payload.get('status') != 'failed' or payload.get('error_type') != 'BudgetExhausted'):
                raise ValueError('invalid blocked-job record')
            _check_record(payload, common)
            state['terminal'][job_id] = payload
        else:
            raise ValueError('unknown ledger event')
    return state


def _append_event(path, state, kind, payload):
    event = {'event': kind, 'sequence': state['events'] + 1, 'timestamp': _now(),
             'previous_sha256': state['last_sha256'], 'payload': payload}
    event['event_sha256'] = content_hash(event)
    _append(path, event)
    state['events'] += 1
    state['last_sha256'] = event['event_sha256']


def _retain_raw(receipt):
    try:
        return _detach(receipt), 'json'
    except ValueError:
        return repr(receipt), 'repr'


def run_reference_jobs(jobs, backend, output_dir, *, batch_id, study_id, attempt_cap=913) -> list[dict]:
    """Run or resume one immutable, single-model batch within one shared study.

    The output directory is exclusively locked for the whole invocation. Its
    reference-ledger.jsonl binds the study, every complete batch and every
    attempted call. Exceptions consume attempts and become failed outputs;
    process interruptions leave a durable pending dispatch. Answer contents are
    never parsed or repaired here. Completed outputs, including truncated raw
    text, are returned unchanged on resume. Budget-blocked jobs consume no call.
    """
    _identifier(batch_id, 'batch_id')
    _identifier(study_id, 'study_id')
    if type(attempt_cap) is not int or not 1 <= attempt_cap <= 913:
        raise ValueError('attempt_cap must be an integer from 1 through 913')
    jobs = _detach(jobs)
    config = _detach(getattr(backend, 'run_config', None))
    _validate_batch(jobs, config)
    if not callable(getattr(backend, 'generate', None)):
        raise ValueError('backend must provide generate')
    study = {'schema_version': 1, 'protocol_id': _PROTOCOL, 'study_id': study_id,
             'attempt_cap': attempt_cap,
             'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    batch = {'batch_id': batch_id, 'jobs': jobs, 'run_config': config}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _LEDGER
    with _run_lock(directory):
        if not path.exists() and any(p.name != '.run.lock' for p in directory.iterdir()):
            raise ValueError('new study output must be empty; existing files require reconciliation')
        state = _load_ledger(path, study)
        if not state['events']:
            _append_event(path, state, 'study', study)
        if batch_id in state['batches']:
            if state['batches'][batch_id] != batch:
                raise ValueError('resume refused: complete job list or backend configuration changed')
        else:
            if any(job['job_id'] in state['jobs'] for job in jobs):
                raise ValueError('same job ID cannot execute in another batch')
            _append_event(path, state, 'batch', batch)
        # Replay bound evidence before generating any response in this call.
        state = _load_ledger(path, study)
        records = []
        for job in jobs:
            job_id = job['job_id']
            if job_id in state['terminal']:
                records.append(copy.deepcopy(state['terminal'][job_id]))
                continue
            if job_id in state['dispatches']:
                record = _error_record(state['dispatches'][job_id], 'interrupted',
                                       'InterruptedDispatch', 'Pending dispatch has no outcome; no automatic retry.')
                _append_event(path, state, 'outcome', record)
            elif state['attempts'] >= attempt_cap:
                common = _common(job, study_id, batch_id, None)
                record = _error_record(common, 'failed', 'BudgetExhausted',
                                       'Shared study attempt cap reached before dispatch.')
                _append_event(path, state, 'blocked', record)
            else:
                common = _common(job, study_id, batch_id, state['attempts'] + 1)
                # A durable intent consumes the attempt even if generation is interrupted.
                _append_event(path, state, 'dispatch', common)
                state['attempts'] += 1
                state['dispatches'][job_id] = common
                raw = None
                try:
                    raw = backend.generate(copy.deepcopy(job['messages']), max_new_tokens=job['max_new_tokens'])
                    receipt = _validate_response(raw, job['max_new_tokens'])
                    record = {**common, 'status': 'ok', 'timestamp': _now(), **receipt,
                              'truncated': receipt['finish_reason'] == 'length'}
                except Exception as exc:
                    record = _error_record(common, 'failed', type(exc).__name__, str(exc) or repr(exc))
                    if isinstance(raw, dict) and isinstance(raw.get('text'), str):
                        record['text'] = raw['text']
                record['raw_receipt'], record['raw_receipt_format'] = _retain_raw(raw)
                _append_event(path, state, 'outcome', record)
            state['terminal'][job_id] = record
            records.append(copy.deepcopy(record))
        return records
