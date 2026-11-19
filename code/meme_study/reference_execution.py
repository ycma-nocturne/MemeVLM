"""Bind prepared inputs to attempts and score recorded outputs.

Reuse bound results for unchanged jobs across immutable releases, including
failures. A shared ledger tracks one attempt budget across releases.
"""
from __future__ import annotations

import copy
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

from .data import content_hash
from .freeze import _publish, _safe_file
from .reference_cli import _check_config, _json_bytes, _jsonl_bytes, _read_json, _sha
from .reference_protocol import (PROTOCOL_ID, compile_explanation_jobs,
                                 compile_judge_jobs, compile_judge_slots, compile_pair_jobs, summarize_plan)
from .reference_runner import _LEDGER, _load_ledger, run_reference_jobs
from .reference_scoring import score_checklist, score_pair
from .reference_profiling import ResourceInvocation
from .reference_errata import verify_applied_errata

STUDY_ID = 'reference-reuse-v1'
_ARTIFACTS = {'reference_records.jsonl', 'explanation_jobs.jsonl', 'pair_jobs.preview.jsonl',
              'pair_gold.json', 'judge_slots.jsonl', 'summary.json', 'inputs/selection.json',
              'inputs/hints.json', 'inputs/pairs.json', 'inputs/config.json'}


def _rows(raw, name):
    return [_read_json(line, name) for line in raw.splitlines() if line.strip()]


def load_prepared(directory):
    """Verify the saved artifacts and recompile jobs; do not access GPU libraries."""
    root = Path(directory).resolve(strict=True)
    provenance_raw = _safe_file(root, 'provenance.json').read_bytes()
    provenance = _read_json(provenance_raw, 'provenance.json')
    if (provenance.get('protocol_id') != PROTOCOL_ID
            or set(provenance.get('artifact_sha256', {})) != _ARTIFACTS):
        raise ValueError('unknown or incomplete prepared artifact manifest')
    artifacts = {}
    for name, digest in provenance['artifact_sha256'].items():
        raw = _safe_file(root, name).read_bytes()
        if _sha(raw) != digest:
            raise ValueError(f'prepared artifact changed: {name}')
        artifacts[name] = _rows(raw, name) if name.endswith('.jsonl') else _read_json(raw, name)
    config = artifacts['inputs/config.json']
    _check_config(config)
    records, pairs = artifacts['reference_records.jsonl'], artifacts['inputs/pairs.json']
    verify_applied_errata(records, config.get('source_errata'))
    explanations = compile_explanation_jobs(records, artifacts['inputs/hints.json'], config['model_ids'])
    if compile_judge_slots(explanations, records, config['judge_model']) != artifacts['judge_slots.jsonl']:
        raise ValueError('prepared judge dependencies do not match their parent inputs and references')
    pair_jobs, gold = compile_pair_jobs(pairs, config['model_ids'], require_material_checks=False)
    if (explanations != artifacts['explanation_jobs.jsonl']
            or pair_jobs != artifacts['pair_jobs.preview.jsonl'] or gold != artifacts['pair_gold.json']):
        raise ValueError('prepared jobs/gold do not match their saved compiler inputs')
    plan = summarize_plan(explanations, pair_jobs)
    if any(artifacts['summary.json'].get(k) != v for k, v in plan.items()):
        raise ValueError('prepared summary differs from its jobs')
    if not 1 <= config['design']['attempt_cap'] <= 913 or config['design']['attempt_cap'] != plan['attempt_cap']:
        raise ValueError('prepared attempt cap differs from the reference design')
    # Hash the portable artifact manifest, not the machine's absolute source paths.
    return {'path': root, 'prepared_sha256': content_hash(provenance['artifact_sha256']),
            'provenance_sha256': _sha(provenance_raw), 'provenance': provenance,
            'artifacts': artifacts, 'config': config, 'records': records,
            'explanations': explanations, 'pairs': pair_jobs, 'gold': gold, 'plan': plan}


def read_reference_run(directory):
    """Read a verified append-only snapshot without creating or repairing files."""
    path = Path(directory) / _LEDGER
    if not path.exists():
        return {'study': None, 'attempts': 0, 'jobs': {}, 'outputs': [], 'batches': {},
                'dispatches': {}, 'ledger_sha256': None, 'ledger_size_bytes':0}
    raw = path.read_bytes()
    events = _rows(raw, _LEDGER)
    if not events or events[0].get('event') != 'study':
        raise ValueError('run ledger has no study header')
    study = events[0]['payload']
    if (study.get('protocol_id') != PROTOCOL_ID or study.get('study_id') != STUDY_ID
            or type(study.get('attempt_cap')) is not int or not 1 <= study['attempt_cap'] <= 913):
        raise ValueError('run ledger belongs to another study')
    state = _load_ledger(path, study)
    # A concurrent writer must not produce a mixed snapshot.
    if path.read_bytes() != raw:
        raise ValueError('run ledger changed while reading; retry after execution finishes')
    return {'study': study, 'attempts': state['attempts'],
            'jobs': {key: value[1] for key, value in state['jobs'].items()},
            'outputs': list(state['terminal'].values()), 'batches': state['batches'],
            'dispatches': state['dispatches'], 'ledger_sha256': _sha(raw), 'ledger_size_bytes':len(raw)}


def _base_job(job):
    result = copy.deepcopy(job)
    if 'prepared_job_id' in result:
        result['job_id'] = result.pop('prepared_job_id')
        result.pop('prepared_sha256')
        result['input_sha256'] = content_hash({k: v for k, v in result.items() if k != 'input_sha256'})
    return result


def _bind(base, prepared_sha256, snapshot):
    matches = [old for old in snapshot['jobs'].values() if _base_job(old) == base]
    if len(matches) > 1:
        raise ValueError('multiple attempts bound to identical inputs; reconcile explicitly')
    if matches:
        return copy.deepcopy(matches[0])
    job = copy.deepcopy(base)
    job['prepared_job_id'] = job['job_id']
    job['prepared_sha256'] = prepared_sha256
    job['job_id'] = f'release/{prepared_sha256}/{job["prepared_job_id"]}'
    job['input_sha256'] = content_hash({k: v for k, v in job.items() if k != 'input_sha256'})
    return job


def _bound(package, snapshot):
    if snapshot['study'] and snapshot['study']['attempt_cap'] != package['config']['design']['attempt_cap']:
        raise ValueError('prepared budget differs from the shared study budget')
    return [_bind(job, package['prepared_sha256'], snapshot)
            for job in package['explanations'] + package['pairs']]


def _check_output(job, snapshot):
    bound = snapshot['jobs'].get(job['job_id'])
    if bound is not None and bound != job:
        raise ValueError('runtime job differs from current prepared input')
    return next((output for output in snapshot['outputs'] if output['job_id'] == job['job_id']), None)


def _check_images(jobs, root):
    checked = set()
    for job in jobs:
        image = job.get('image_path')
        if image is None:
            continue
        pair = (image, job['image_sha256'])
        if pair not in checked:
            if _sha(_safe_file(root, image).read_bytes()) != pair[1]:
                raise ValueError(f'image bytes changed: {image}')
            checked.add(pair)


class _ImageCheckedBackend:
    def __init__(self, backend, jobs, root):
        self.backend, self.root = backend, root
        self.run_config = {**backend.run_config, 'image_validation': 'sha256_before_each_call',
                           'execution_code_sha256': _sha(Path(__file__).read_bytes())}
        self.expected = {}
        for job in jobs:
            key = content_hash(job['messages'])
            self.expected.setdefault(key, []).append(job)

    def generate(self, messages, max_new_tokens):
        _check_images(self.expected[content_hash(messages)], self.root)
        return self.backend.generate(messages, max_new_tokens=max_new_tokens)


@contextmanager
def _execution_lock(runs):
    # Keep the inode. A sibling lock covers preflight and setup before the
    # lower-level runner creates its own run directory and dispatch lock.
    directory = Path(runs).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f'.{directory.name}.reference-stage.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another process owns this reference study preparation and execution') from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def execute_reference(*, prepared, data_root, runs, model, task='explanation', split='all',
                      backend_factory=None, device='cuda:0', max_jobs=None, **backend_options):
    with _execution_lock(runs):
        return _execute_reference(prepared=prepared, data_root=data_root, runs=runs, model=model,
                                  task=task, split=split, backend_factory=backend_factory,
                                  device=device, max_jobs=max_jobs, **backend_options)


def _execute_reference(*, prepared, data_root, runs, model, task='explanation', split='all',
                       backend_factory=None, device='cuda:0', max_jobs=None, **backend_options):
    """Execute a model/stage/split; cached exact inputs never consume a new attempt."""
    if task not in {'explanation', 'pair', 'judge'} or split not in {'all', 'dev', 'test'}:
        raise ValueError('unknown task or split')
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1 or split != 'dev'):
        raise ValueError('max_jobs must be a positive integer and is only allowed for dev')
    package = load_prepared(prepared)
    snapshot = read_reference_run(runs)
    config = package['config']
    if model not in ([config['judge_model']] if task == 'judge' else config['model_ids']):
        raise ValueError('model does not match its configured role')
    jobs = _bound(package, snapshot)
    chosen_split = lambda j: split == 'all' or j['split'] == split
    if task == 'judge':
        parents = [j for j in jobs if j['task_kind'] == 'explanation' and chosen_split(j)]
        parent_outputs = [_check_output(j, snapshot) for j in parents]
        if any(output is None for output in parent_outputs):
            raise ValueError('judge requires all selected parent explanations to have terminal records')
        jobs = [_bind(j, package['prepared_sha256'], snapshot) for j in
                compile_judge_jobs(parents, parent_outputs, package['records'], model,
                                   judge_notes_by_split=config.get('judge_notes_by_split'))]
    else:
        kinds = {'explanation'} if task == 'explanation' else {'pair', 'permutation'}
        jobs = [j for j in jobs if j['task_kind'] in kinds and j['model'] == model and chosen_split(j)]
    if any(j['readiness'] != 'ready' for j in jobs):
        raise ValueError('selected pair materials still need a recorded current-content material review')
    # A deterministic prefix, including cached records: repeating a pilot never
    # advances its sample or repairs failures. Full continuation reuses these IDs.
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    root = Path(data_root).resolve(strict=True)
    _check_images(jobs, root)
    if not jobs:
        return {'protocol_id': PROTOCOL_ID, 'task': task, 'selected_jobs': 0, 'new_attempts': 0,
                'skipped_failed_parents': len(parent_outputs) if task == 'judge' else 0}
    if backend_factory is None:
        from .reference_backends import create_backend
        backend_factory = create_backend
    metadata={'prepared_sha256':package['prepared_sha256'],'model':model,'task':task,'split':split,
              'selected_job_ids':[j['job_id'] for j in jobs], 'max_jobs':max_jobs}
    with ResourceInvocation(runs=runs,metadata=metadata,snapshot=snapshot,read_snapshot=read_reference_run) as profile:
        result = _dispatch_reference(jobs=jobs,snapshot=snapshot,config=config,root=root,runs=runs,
                                     model=model,task=task,split=split,device=device,
                                     backend_factory=backend_factory,backend_options=backend_options,profile=profile)
        result['prepared_sha256']=package['prepared_sha256']
        result['profiling_receipt']=str(profile.directory/'finished.json')
        profile.result=result
        return result


def _dispatch_reference(*,jobs,snapshot,config,root,runs,model,task,split,device,
                        backend_factory,backend_options,profile):
    raw_backend = backend_factory(model, root, device=device, **backend_options)
    backend = _ImageCheckedBackend(raw_backend, jobs, root)
    selected_ids = {j['job_id'] for j in jobs}
    # Resume an interrupted original batch intact, without dispatching unselected inputs.
    completed = {r['job_id'] for r in snapshot['outputs']}
    for batch in snapshot['batches'].values():
        if any(j['job_id'] in selected_ids for j in batch['jobs']) and batch['run_config'] != backend.run_config:
            raise ValueError('selected cached inputs use another runtime configuration; no silent rerun/reuse')
    needs_setup=bool(selected_ids-completed) and callable(getattr(raw_backend,'prepare',None))
    with profile.setup(needs_setup):
        if needs_setup:
            raw_backend.prepare()
    profile.resources(raw_backend)
    for batch in snapshot['batches'].values():
        batch_ids = {j['job_id'] for j in batch['jobs']}
        if (batch_ids - completed) & selected_ids:
            if not batch_ids <= selected_ids:
                raise ValueError('resume the original split/batch before selecting a smaller subset')
            run_reference_jobs(batch['jobs'], backend, runs, batch_id=batch['batch_id'],
                               study_id=STUDY_ID, attempt_cap=config['design']['attempt_cap'])
    new_jobs = [j for j in jobs if j['job_id'] not in snapshot['jobs']]
    if new_jobs:
        batch_id = f'{task}/{model}/{split}/' + content_hash([j['job_id'] for j in new_jobs])
        run_reference_jobs(new_jobs, backend, runs, batch_id=batch_id, study_id=STUDY_ID,
                           attempt_cap=config['design']['attempt_cap'])
    after = read_reference_run(runs)
    outputs = [_check_output(j, after) for j in jobs]
    counts = Counter(r['status'] for r in outputs if r is not None)
    return {'protocol_id': PROTOCOL_ID,
            'model': model, 'task': task, 'split': split, 'selected_jobs': len(jobs),
            'attempts_total': after['attempts'], 'new_attempts': after['attempts'] - snapshot['attempts'],
            'backend_setup_seconds': profile.setup_seconds, 'reused_outputs': len(selected_ids & completed),
            **{status: counts[status] for status in ('ok', 'failed', 'interrupted')}}


def score_reference(*, prepared, runs, output, allow_partial=False):
    """Score verified raw outputs; unavailable judgments remain null, never zero."""
    package, snapshot = load_prepared(prepared), read_reference_run(runs)
    jobs = _bound(package, snapshot)
    explanations = [j for j in jobs if j['task_kind'] == 'explanation']
    candidate_outputs = [_check_output(j, snapshot) for j in explanations]
    judge_jobs = compile_judge_jobs(explanations, [o for o in candidate_outputs if o is not None],
                                    package['records'], package['config']['judge_model'],
                                    judge_notes_by_split=package['config'].get('judge_notes_by_split'))
    judges = {j['parent_job_id']: _bind(j, package['prepared_sha256'], snapshot) for j in judge_jobs}
    references = {r['family_id']: r for r in package['records']}
    pairs = {r['family_id']: r for r in package['artifacts']['inputs/pairs.json']}
    rows = []
    for job in jobs:
        candidate = _check_output(job, snapshot)
        status = candidate['status'] if candidate else ('pending' if job['readiness'] != 'ready' else 'missing')
        row = {key: job[key] for key in ('job_id', 'prepared_job_id', 'prepared_sha256',
               'task_kind', 'family_id', 'split', 'template_group', 'model', 'condition')}
        row.update(source_input_sha256=job['input_sha256'], run_status=status, score=None,
                   image_path=job.get('image_path'), image_sha256=job.get('image_sha256'),
                   candidate_text=candidate.get('text') if candidate else None,
                   candidate_text_sha256=_sha(candidate['text'].encode())
                       if candidate and isinstance(candidate.get('text'), str) else None,
                   candidate_truncated=candidate.get('truncated', False) if candidate else False,
                   candidate_output_sha256=content_hash(candidate) if candidate else None)
        ref = references[job['family_id']]
        row.update(source=ref['source'], reference_image_path=ref['image']['path'],
                   reference_image_sha256=ref['image']['sha256'],
                   public_reference_image_provenance=ref['image'])
        if job['task_kind'] == 'explanation':
            row.update(ocr_available=job['ocr_available'], reference_image_path=ref['image']['path'],
                       reference=ref['reference'], ocr_text=ref['ocr_text'],
                       image_provenance=ref['image'],
                       hint_text=package['artifacts']['inputs/hints.json'][job['family_id']]['text'])
            judge_job = judges.get(job['job_id'])
            judge = _check_output(judge_job, snapshot) if judge_job else None
            row.update(judge_job_id=judge_job['job_id'] if judge_job else None,
                       judge_profile_sha256=judge_job['judge_profile_sha256'] if judge_job else None,
                       judge_status=judge['status'] if judge else ('missing' if status == 'ok' else 'not_applicable'),
                       judge_output_sha256=content_hash(judge) if judge else None)
            if status == 'ok' and judge and judge['status'] == 'ok':
                row['score'] = score_checklist(candidate['text'], ref['reference']['checklist'],
                                                judge['text'], truncated=judge['truncated'])
        else:
            row['variant'] = job['variant']
            pair = pairs[job['family_id']]
            row.update(options=job['options'], cue_question=pair['cue_question'],
                       relation_question=pair['relation_question'], gold=package['gold'][job['prepared_job_id']],
                       image_provenance={'scope':'B cleaned project triple, separate from public A image',
                                         'variant':job['variant'], 'path':job['image_path'],
                                         'sha256':job['image_sha256'], 'preparation':pair.get('preparation')},
                       material_check=pair.get('material_check'))
            if status == 'ok':
                row['score'] = score_pair(candidate['text'], job['options'],
                                          package['gold'][job['prepared_job_id']], truncated=candidate['truncated'])
        rows.append(row)
    complete = all(r['run_status'] in {'ok', 'failed', 'interrupted'}
                   and r.get('judge_status') != 'missing' for r in rows)
    if not complete and not allow_partial:
        raise ValueError('study is incomplete; use allow_partial for an explicitly incomplete export')
    summary = {'protocol_id': PROTOCOL_ID, 'prepared_sha256': package['prepared_sha256'],
               'complete': complete, 'allow_partial': allow_partial, 'rows': len(rows),
               'run_status': dict(Counter(r['run_status'] for r in rows)),
               'explanations_with_valid_judge': sum(r['task_kind'] == 'explanation'
                    and (r['score'] or {}).get('status') == 'ok' for r in rows),
               'attempts_all_releases': snapshot['attempts'], 'model_calls_in_this_operation': 0}
    artifacts = {'scored.jsonl': _jsonl_bytes(rows), 'summary.json': _json_bytes(summary)}
    provenance = {'protocol_id': PROTOCOL_ID, 'prepared_sha256': package['prepared_sha256'],
                  'prepared_provenance_sha256': package['provenance_sha256'],
                  'ledger_sha256': snapshot['ledger_sha256'],
                  'artifact_sha256': {name: _sha(raw) for name, raw in artifacts.items()},
                  'scoring_code_sha256': _sha(Path(__file__).with_name('reference_scoring.py').read_bytes()),
                  'execution_code_sha256': _sha(Path(__file__).read_bytes()), 'model_calls': 0}
    artifacts['provenance.json'] = _json_bytes(provenance)
    _write_bundle(output, artifacts)
    return summary


def _write_bundle(output, artifacts):
    destination = Path(output).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'output already exists: {destination}')
    parent = destination.parent.resolve(strict=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=parent))
    try:
        for name, raw in artifacts.items():
            (staging / name).write_bytes(raw)
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_scores(directory):
    root = Path(directory).resolve(strict=True)
    provenance = _read_json(_safe_file(root, 'provenance.json').read_bytes(), 'score provenance')
    if provenance.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('unknown score protocol')
    for name in ('scored.jsonl', 'summary.json'):
        if _sha(_safe_file(root, name).read_bytes()) != provenance['artifact_sha256'][name]:
            raise ValueError(f'score artifact changed: {name}')
    return _rows(_safe_file(root, 'scored.jsonl').read_bytes(), 'scores')
