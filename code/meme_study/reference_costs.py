"""Summarize calls, wall time, and observed resources from study records.

Wall time includes setup. The single-card price estimate uses process wall time.
Missing observations remain null.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .freeze import _publish
from .reference_execution import read_reference_run
from .reference_protocol import PROTOCOL_ID
from .reference_runner import _LEDGER

_MEMORY_SCOPE = 'current CUDA process / torch allocator, not whole-device utilization'
_TIMING_SCOPE = ('serial process wall time for backend construction, model setup, execution and '
                 'bookkeeping; excludes preflight and lock wait; not allocated GPU time')
_METRICS = ('input_tokens', 'output_tokens', 'latency_s')
_MEMORY = ('torch_memory_allocated_bytes', 'torch_memory_reserved_bytes',
           'torch_peak_allocated_bytes', 'torch_peak_reserved_bytes')
_CALL_FIELDS = ('attempt', 'attempted', 'job_id', 'batch_id', 'prepared_sha256', 'input_sha256',
                'model', 'task_kind', 'split', 'condition', 'family_id', 'parent_job_id',
                'candidate_model', 'status', 'truncated', 'finish_reason', 'input_tokens',
                'output_tokens', 'latency_s', 'receipt_source', 'raw_receipt_present',
                'max_allocated_bytes', 'max_reserved_bytes', 'resource_snapshot_count',
                'per_call_peak_reset', 'memory_scope', 'output_sha256', 'raw_receipt_sha256')
_INVOCATION_FIELDS = ('invocation_id', 'model', 'task', 'split', 'prepared_sha256',
                      'started_at', 'finished_at', 'status', 'profile_complete',
                      'attempts_before', 'attempts_after', 'new_attempts', 'selected_jobs',
                      'wall_seconds', 'reported_wall_seconds', 'setup_seconds',
                      'setup_attempted', 'setup_completed', 'max_allocated_bytes',
                      'max_reserved_bytes', 'resource_snapshot_count', 'memory_scope',
                      'started_sha256', 'finished_sha256')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _number(value, name, *, integer=False):
    if ((type(value) is not int if integer else type(value) not in (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative ' + ('integer' if integer else 'number'))
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a SHA-256 digest')
    return value


def _timestamp(value, name):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'{name} must be a UTC ISO timestamp') from exc
    return parsed


def _memory(snapshots):
    available = []
    for snapshot in snapshots:
        if snapshot is None:
            continue
        if not isinstance(snapshot, dict) or type(snapshot.get('available')) is not bool:
            raise ValueError('resource snapshot must declare available as boolean')
        if not snapshot['available']:
            if any(key in snapshot and snapshot[key] is not None for key in _MEMORY):
                raise ValueError('unavailable resource snapshot cannot contain memory readings')
            continue
        for key in _MEMORY:
            _number(snapshot.get(key), key, integer=True)
        total = _number(snapshot.get('total_memory_bytes'), 'total_memory_bytes', integer=True)
        allocated, reserved, peak_allocated, peak_reserved = [snapshot[key] for key in _MEMORY]
        if (allocated > reserved or peak_allocated < allocated or peak_reserved < reserved
                or peak_allocated > peak_reserved or max(allocated, reserved, peak_allocated, peak_reserved) > total):
            raise ValueError('resource memory readings are inconsistent')
        available.append(snapshot)
    return {'max_allocated_bytes': max((max(s[_MEMORY[0]], s[_MEMORY[2]]) for s in available), default=None),
            'max_reserved_bytes': max((max(s[_MEMORY[1]], s[_MEMORY[3]]) for s in available), default=None),
            'resource_snapshot_count': len(available), 'memory_scope': _MEMORY_SCOPE}


def _call_rows(snapshot):
    outputs = {output['job_id']: output for output in snapshot['outputs']}
    rows = []
    # Bound but undispatched jobs have no incurred call; blocked jobs are still visible.
    ids = list(snapshot['dispatches']) + [key for key, output in outputs.items() if output['attempt'] is None]
    for job_id in ids:
        job, output = snapshot['jobs'][job_id], outputs.get(job_id)
        dispatch = snapshot['dispatches'].get(job_id)
        row = dict.fromkeys(_CALL_FIELDS)
        row.update({field: job.get(field) for field in
                    ('job_id', 'prepared_sha256', 'input_sha256', 'model', 'task_kind',
                     'split', 'condition', 'family_id', 'parent_job_id')})
        row.update(attempt=dispatch['attempt'] if dispatch else None, attempted=dispatch is not None,
                   batch_id=dispatch['batch_id'] if dispatch else output['batch_id'],
                   status=output['status'] if output else 'unresolved',
                   truncated=output['truncated'] if output and output['status'] == 'ok' else None,
                   raw_receipt_present=False, receipt_source=None, per_call_peak_reset=None)
        parent = job.get('parent_job_id')
        if job['task_kind'] == 'judge':
            parent_job = snapshot['jobs'].get(parent)
            if (parent_job is None or parent_job['task_kind'] != 'explanation'
                    or parent_job['input_sha256'] != job.get('parent_input_sha256')):
                raise ValueError('judge parent candidate is missing or differs from its bound input')
            row['candidate_model'] = parent_job['model']
        else:
            row['candidate_model'] = job['model']
        raw = output.get('raw_receipt') if output else None
        row['output_sha256'] = _sha(_json(output)) if output else None
        row['raw_receipt_sha256'] = _sha(_json(raw)) if raw is not None else None
        if output and output['status'] == 'ok':
            for key in (*_METRICS, 'finish_reason'):
                row[key] = output[key]
            row['receipt_source'] = 'validated_terminal'
        if isinstance(raw, dict):
            row['raw_receipt_present'] = True
            if output['status'] == 'ok':
                for key in (*_METRICS, 'text', 'finish_reason'):
                    if raw.get(key) != output[key] or type(raw.get(key)) is not type(output[key]):
                        raise ValueError(f'raw receipt {key} disagrees with validated terminal output')
                row['receipt_source'] = 'validated_terminal_and_raw'
            else:
                # Failed validation may still leave independently valid partial measurements.
                if isinstance(raw.get('finish_reason'), str) and raw['finish_reason']:
                    row['finish_reason'] = raw['finish_reason']
                    row['truncated'] = raw['finish_reason'] == 'length'
                for key in _METRICS:
                    value = raw.get(key)
                    try:
                        _number(value, key, integer=key != 'latency_s')
                    except ValueError:
                        continue
                    if key == 'output_tokens' and value > job['max_new_tokens']:
                        continue
                    row[key] = value
                if any(row[key] is not None for key in _METRICS):
                    row['receipt_source'] = 'partial_failed_raw_receipt'
            resource = raw.get('resource_usage')
            if resource is not None:
                if (not isinstance(resource, dict) or type(resource.get('available')) is not bool
                        or type(resource.get('peak_stats_reset')) is not bool):
                    raise ValueError('raw resource_usage requires availability and peak reset flags')
                if resource['available'] and not resource['peak_stats_reset']:
                    raise ValueError('available per-call resource usage requires reset peak statistics')
                row['per_call_peak_reset'] = resource['available'] and resource['peak_stats_reset']
                row.update(_memory([resource.get('before'), resource.get('after')]))
                if resource['available'] and row['resource_snapshot_count'] != 2:
                    raise ValueError('available per-call resource usage requires before and after snapshots')
            else:
                row.update(_memory([]))
        else:
            row.update(_memory([]))
        rows.append(row)
    return rows


def _prefixes(raw):
    result = {0: (None, 0)}
    length, attempts = 0, 0
    for line in raw.splitlines(keepends=True):
        length += len(line)
        if line.strip() and json.loads(line)['event'] == 'dispatch':
            attempts += 1
        result[length] = (_sha(raw[:length]), attempts)
    return result


def _check_prefix(record, suffix, prefixes):
    size, digest, attempts = (record.get('ledger_bytes_' + suffix),
                             record.get('ledger_' + suffix + '_sha256'), record.get('attempts_' + suffix))
    _number(size, 'ledger prefix bytes', integer=True)
    _number(attempts, 'ledger prefix attempts', integer=True)
    if size not in prefixes or prefixes[size] != (digest, attempts):
        raise ValueError('profile ledger prefix hash, byte count or attempts disagree with current ledger')
    return size, attempts


def _read_object(path, source_hashes):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'profile requires a regular non-symlink file: {path}')
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('profile JSON is invalid') from exc
    if not isinstance(value, dict):
        raise ValueError('profile must be a JSON object')
    source_hashes[str(path)] = _sha(raw)
    return value, _sha(raw)


def _profile_rows(profiles, snapshot, ledger_raw):
    rows, sources, covered = [], {}, set()
    path = Path(profiles)
    if not path.exists():
        return rows, sources, covered
    if path.is_symlink() or not path.is_dir():
        raise ValueError('profiles must be a non-symlink directory')
    prefixes = _prefixes(ledger_raw)
    dispatches = sorted(snapshot['dispatches'].values(), key=lambda d: d['attempt'])
    for directory in sorted(path.iterdir()):
        if directory.is_symlink() or not directory.is_dir() or re.fullmatch('[0-9a-f]{32}', directory.name) is None:
            raise ValueError('profile directory must contain only invocation UUID directories')
        if {child.name for child in directory.iterdir()} - {'started.json', 'finished.json'}:
            raise ValueError('unexpected file in invocation profile')
        started, start_sha = _read_object(directory / 'started.json', sources)
        if (started.get('schema_version') != 1 or started.get('protocol_id') != PROTOCOL_ID
                or started.get('invocation_id') != directory.name):
            raise ValueError('profile start identity or protocol is invalid')
        start_time = _timestamp(started.get('started_at'), 'started_at')
        _digest(started.get('prepared_sha256'), 'prepared_sha256')
        _text(started.get('model'), 'model')
        _text(started.get('timing_scope'), 'timing_scope')
        if started.get('task') not in {'explanation', 'pair', 'judge'} or started.get('split') not in {'all', 'dev', 'test'}:
            raise ValueError('profile task or split is invalid')
        selected = started.get('selected_job_ids')
        if (not isinstance(selected, list) or any(not isinstance(x, str) or not x for x in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('selected job IDs must be unique text values')
        before_size, before_attempts = _check_prefix(started, 'before', prefixes)
        row = {key: started.get(key) for key in _INVOCATION_FIELDS}
        row.update(status='incomplete', profile_complete=False, selected_jobs=len(selected),
                   started_sha256=start_sha, **_memory([]))
        finished_path = directory / 'finished.json'
        if finished_path.exists():
            finished, finish_sha = _read_object(finished_path, sources)
            if (finished.get('schema_version') != 1 or finished.get('protocol_id') != PROTOCOL_ID
                    or finished.get('invocation_id') != directory.name
                    or finished.get('started_sha256') != start_sha):
                raise ValueError('profile finish is not bound to its start')
            if finished.get('status') not in {'completed', 'error'}:
                raise ValueError('unknown profile finish status')
            if _timestamp(finished.get('finished_at'), 'finished_at') < start_time:
                raise ValueError('profile finish precedes its start')
            wall = _number(finished.get('wall_seconds'), 'wall_seconds')
            setup = finished.get('setup_seconds')
            if setup is not None and _number(setup, 'setup_seconds') > wall:
                raise ValueError('setup_seconds cannot exceed wall_seconds')
            if any(type(finished.get(key)) is not bool for key in ('setup_attempted', 'setup_completed')):
                raise ValueError('profile setup flags must be boolean')
            if finished['setup_completed'] and not finished['setup_attempted']:
                raise ValueError('setup cannot complete without an attempt')
            _digest(finished.get('profile_code_sha256'), 'profile_code_sha256')
            if finished['status'] == 'error':
                error = finished.get('error')
                if not isinstance(error, dict):
                    raise ValueError('error invocation requires error details')
                _text(error.get('type'), 'error type')
                if not isinstance(error.get('message'),str):
                    raise ValueError('error message must be text, possibly empty for an interrupt')
            elif finished.get('error') is not None:
                raise ValueError('completed invocation cannot declare an error')
            row.update({key: finished.get(key) for key in ('status', 'finished_at', 'setup_seconds',
                                                          'setup_attempted', 'setup_completed')})
            row.update(reported_wall_seconds=wall, finished_sha256=finish_sha,
                       **_memory([finished.get('resource_after_setup')]))
            after_values = [finished.get(key) for key in
                            ('attempts_after', 'ledger_after_sha256', 'ledger_bytes_after')]
            if finished.get('attempts_after') is None:
                if any(value is not None for value in after_values):
                    raise ValueError('unknown profile ledger end must omit all snapshot fields')
            else:
                after_size, after_attempts = _check_prefix(finished, 'after', prefixes)
                if after_size < before_size or after_attempts < before_attempts:
                    raise ValueError('profile ledger prefix moved backwards')
                added = [d for d in dispatches if before_attempts < d['attempt'] <= after_attempts]
                for dispatch in added:
                    job = snapshot['jobs'][dispatch['job_id']]
                    tasks = {'pair', 'permutation'} if started['task'] == 'pair' else {started['task']}
                    if (job['job_id'] not in selected or job['model'] != started['model']
                            or job['task_kind'] not in tasks
                            or (started['split'] != 'all' and job.get('split') != started['split'])):
                        raise ValueError('profile selected job membership disagrees with newly dispatched calls')
                    if dispatch['attempt'] in covered:
                        raise ValueError('overlapping profiles duplicate an attempted call')
                    covered.add(dispatch['attempt'])
                output = finished.get('output_summary')
                if output is not None:
                    if not isinstance(output, dict):
                        raise ValueError('profile output_summary must be an object or null')
                    expected = {'model': started['model'], 'task': started['task'], 'split': started['split'],
                                'selected_jobs': len(selected), 'attempts_total': after_attempts,
                                'new_attempts': after_attempts - before_attempts}
                    if any(key in output and output[key] != value for key, value in expected.items()):
                        raise ValueError('profile output summary disagrees with ledger or selected jobs')
                row.update(profile_complete=True, wall_seconds=wall, attempts_after=after_attempts,
                           new_attempts=after_attempts - before_attempts)
        rows.append(row)
    return rows, sources, covered


def _receipts(rows):
    result = {}
    for field in _METRICS:
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = {'known_count': len(values), 'missing_count': len(rows) - len(values),
                         'known_sum': sum(values) if values else None}
    return result


def _csv(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def export_reference_costs(*, runs, profiles, output_dir, hkd_per_card_hour=12):
    """Atomically publish new accounting artifacts from verified existing evidence."""
    _number(hkd_per_card_hour, 'hkd_per_card_hour')
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'cost output already exists: {destination}')
    parent = destination.parent.resolve(strict=True)
    snapshot = read_reference_run(runs)
    ledger_path = Path(runs) / _LEDGER
    ledger_raw = ledger_path.read_bytes() if ledger_path.exists() else b''
    if (_sha(ledger_raw) if ledger_raw else None) != snapshot['ledger_sha256']:
        raise ValueError('run ledger changed while reading costs')
    calls = _call_rows(snapshot)
    invocations, sources, covered = _profile_rows(profiles, snapshot, ledger_raw)
    attempted = [row for row in calls if row['attempted']]
    complete_profiles = bool(invocations) and all(row['profile_complete'] for row in invocations) and len(covered) == snapshot['attempts']
    known_wall = [row['wall_seconds'] for row in invocations if row['wall_seconds'] is not None]
    wall_sum = sum(known_wall) if known_wall else None
    wall_total = wall_sum if complete_profiles else None
    groups = defaultdict(list)
    for row in attempted:
        groups[tuple(row.get(key) for key in ('model', 'task_kind', 'split', 'condition', 'candidate_model'))].append(row)
    summary = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'attempts_total': snapshot['attempts'],
               'accounting_scope': 'all unique dispatched attempts in the shared ledger, across every preparation release',
               'blocked_without_attempt': len(calls) - len(attempted),
               'attempt_status_counts': dict(Counter(row['status'] for row in attempted)),
               'truncated_attempts': sum(row['truncated'] is True for row in attempted),
               'receipts': _receipts(attempted), 'profiles_complete': complete_profiles,
               'profile_coverage': {'invocations': len(invocations), 'attempts_covered': len(covered),
                                    'attempts_without_profile': snapshot['attempts'] - len(covered),
                                    'incomplete_invocations': sum(not row['profile_complete'] for row in invocations),
                                    'known_wall_invocations': len(known_wall),
                                    'missing_wall_invocations': len(invocations) - len(known_wall)},
               'serial_process_wall_seconds': wall_total, 'known_profile_wall_seconds': wall_sum,
               'timing_scope': _TIMING_SCOPE, 'actual_allocated_gpu_hours': None,
               'actual_allocated_gpu_hours_status': 'unknown: no allocation or billing record supplied',
               'hypothetical_cost_proxy': {'hkd_per_card_hour': hkd_per_card_hour, 'assumed_cards': 1,
                                           'basis': 'serial process wall time at an assumed single-card hourly rate',
                                           'cost_hkd': wall_total / 3600 * hkd_per_card_hour if wall_total is not None else None,
                                           'known_profiles_only_cost_hkd': wall_sum / 3600 * hkd_per_card_hour if wall_sum is not None else None,
                                           'is_actual_bill': False, 'is_h100_benchmark': False},
               'memory': {'scope': _MEMORY_SCOPE,
                          'max_allocated_bytes': max((row['max_allocated_bytes'] for row in calls + invocations if row['max_allocated_bytes'] is not None), default=None),
                          'max_reserved_bytes': max((row['max_reserved_bytes'] for row in calls + invocations if row['max_reserved_bytes'] is not None), default=None),
                          'attempts_with_snapshots': sum(row['resource_snapshot_count'] > 0 for row in attempted),
                          'attempts_without_snapshots': sum(row['resource_snapshot_count'] == 0 for row in attempted),
                          'per_call_peak_reset_attempts': sum(row['per_call_peak_reset'] is True for row in attempted)},
               'groups': [{**dict(zip(('model', 'task_kind', 'split', 'condition', 'candidate_model'), key)),
                           'attempts': len(rows), 'receipts': _receipts(rows)}
                          for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))],
               'limitations': ['Profile coverage verifies dispatched attempts, but the ledger cannot reveal missing zero-attempt invocations.',
                               'Known receipt sums omit missing observations and are not whole-study totals when missing_count is nonzero.',
                               'Allocator peaks may include model memory and earlier process activity unless a successful per-call reset is recorded.',
                               'Neither allocator memory nor process wall time measures whole-device utilization or billed allocation.']}
    artifacts = {'per-call.csv': _csv(calls, _CALL_FIELDS),
                 'per-invocation.csv': _csv(invocations, _INVOCATION_FIELDS),
                 'summary.json': _json(summary)}
    provenance = {'schema_version': 1, 'protocol_id': PROTOCOL_ID,
                  'ledger_sha256': snapshot['ledger_sha256'], 'ledger_size_bytes': len(ledger_raw),
                  'study': snapshot['study'], 'profile_sha256': sources,
                  'code_sha256': {name: _sha((Path(__file__).parent / name).read_bytes()) for name in
                                  ('reference_costs.py', 'reference_execution.py', 'reference_runner.py')},
                  'artifact_sha256': {name: _sha(raw) for name, raw in artifacts.items()}}
    artifacts['provenance.json'] = _json(provenance)
    if (ledger_path.read_bytes() if ledger_path.exists() else b'') != ledger_raw:
        raise ValueError('run ledger changed during cost export')
    if any(_sha(Path(path).read_bytes()) != digest for path, digest in sources.items()):
        raise ValueError('invocation profile changed during cost export')
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=parent))
    try:
        for name, raw in artifacts.items():
            with (staging / name).open('xb') as stream:
                stream.write(raw)
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
