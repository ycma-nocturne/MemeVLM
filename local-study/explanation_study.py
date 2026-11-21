"""Execute and score the private MLX 2B explanation comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
import time

from meme_study import data, reference_analysis, reference_audit, reference_execution, reference_protocol
from meme_study import reference_runner, reference_scoring
from meme_study.data import content_hash
from meme_study.reference_execution import (_ImageCheckedBackend, _check_images, _execution_lock,
                                             _write_bundle, load_prepared)
from meme_study.reference_runner import _LEDGER, _load_ledger, run_reference_jobs
from meme_study.reference_scoring import score_checklist
from meme_study import runner

ARMS = {'base': 'local-qwen3vl-2b-base', 'lora': 'local-qwen3vl-2b-lora'}
JUDGE = 'local-internvl3-9b-mlx-judge'
SCOPE = 'local_mlx_2b_task_a'
BASE_REVISION = '9c4f5209e57b31f4b9dfba735de3fb983739c9cc'
BASE_WEIGHT_SHA256 = '4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb'
QWEN_PIXELS = {'min_pixels': 262144, 'max_pixels': 1310720}
JUDGE_MAX_TILES = 4
GENERATION = {'temperature': 0.0, 'skip_special_tokens': True}
RESERVED = ('<image>', '<img>', '</img>', '<IMG_CONTEXT>', '<|im_start|>', '<|im_end|>')
ATTEMPT_CAP = 720


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


class ExplanationStudy:
    def __init__(self, package, test_note=None):
        self.prepared = package['prepared_sha256']
        self.records = package['records']
        self.refs = {row['family_id']: row for row in self.records}
        self.hints = package['artifacts']['inputs/hints.json']
        self.notes = dict(package['config'].get('judge_notes_by_split', {}))
        if test_note is not None:
            if len(test_note) > 2000:
                raise ValueError('test judge note exceeds 2000 characters')
            self.notes['test'] = test_note
        self.jobs = reference_protocol.compile_explanation_jobs(self.records, self.hints, list(ARMS.values()))
        self.study_id = f'{SCOPE}/{self.prepared}'
        self.frozen_protocol = content_hash({'scope': SCOPE, 'prepared_sha256': self.prepared,
                                             'jobs': self.jobs, 'judge': JUDGE, 'notes': self.notes})

    def judge_jobs(self, split, snapshot, *, require_candidates=True):
        parents = [job for job in self.jobs if job['split'] == split]
        outputs = [bound_output(job, snapshot) for job in parents]
        if require_candidates and any(output is None for output in outputs):
            raise ValueError('judge requires terminal candidate records for both arms in the selected split')
        jobs = reference_protocol.compile_judge_jobs(
            parents, [output for output in outputs if output is not None], self.records,
            JUDGE, judge_notes_by_split=self.notes)
        return sorted(jobs, key=lambda job: content_hash([6000, job['job_id']]))

    def select(self, split, stage, snapshot=None):
        if stage == 'judge':
            return self.judge_jobs(split, snapshot)
        return [job for job in self.jobs if job['split'] == split and job['model'] == ARMS[stage]]


def read_ledger(study, runs):
    path = Path(runs) / _LEDGER
    original = sha(path) if path.is_file() else None
    identity = {'schema_version': 1, 'protocol_id': reference_protocol.PROTOCOL_ID,
                'study_id': study.study_id, 'attempt_cap': ATTEMPT_CAP,
                'runner_sha256': sha(reference_runner.__file__)}
    state = _load_ledger(path, identity)
    if path.is_file() and sha(path) != original:
        raise ValueError('ledger changed while reading; repeat after execution finishes')
    state['file_sha256'] = original
    return state


def bound_output(job, state):
    entry = state['jobs'].get(job['job_id'])
    if entry is not None and entry[1] != job:
        raise ValueError(f'ledger job differs from current compiled input: {job["job_id"]}')
    return state['terminal'].get(job['job_id'])


def run_batches(study, runs, jobs, backend):
    state = read_ledger(study, runs)
    selected = {job['job_id'] for job in jobs}
    for job in jobs:
        bound_output(job, state)
    batches = []
    for batch in state['batches'].values():
        ids = {job['job_id'] for job in batch['jobs']}
        if not selected.intersection(ids):
            continue
        if batch['run_config'] != backend.run_config:
            raise ValueError('selected existing jobs use different runtime settings')
        if ids - state['terminal'].keys():
            if not ids <= selected:
                raise ValueError('select a prefix large enough to resume the existing batch')
            batches.append((batch['batch_id'], batch['jobs']))
    unseen = [job for job in jobs if job['job_id'] not in state['jobs']]
    if unseen:
        batches.append((f"local-a/{content_hash([j['job_id'] for j in unseen])}", unseen))
    if batches and hasattr(backend, 'prepare'):
        backend.prepare()
    for batch_id, members in batches:
        run_reference_jobs(members, backend, runs, batch_id=batch_id,
                           study_id=study.study_id, attempt_cap=ATTEMPT_CAP)
    result = read_ledger(study, runs)
    return {'scope': SCOPE, 'selected': len(selected), 'attempts': result['attempts'],
            'new_attempts': result['attempts'] - state['attempts'],
            'status': dict(Counter(result['terminal'][key]['status'] for key in selected)),
            'ledger_sha256': result['file_sha256']}


def runtime(stage, args):
    is_judge = stage == 'judge'
    role = JUDGE if is_judge else ARMS[stage]
    folder = args.judge_model if is_judge else args.base_model
    revision = args.judge_revision if is_judge else BASE_REVISION
    if folder is None or revision is None or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError(f'{stage} needs a local model directory and a 40-character revision')
    config = json.loads((folder / 'config.json').read_bytes())
    if is_judge and config.get('model_type') not in {'internvl', 'internvl_chat'}:
        raise ValueError('judge snapshot must be an MLX InternVL model')
    if not is_judge and (config.get('model_type') != 'qwen3_vl'
                         or config.get('quantization', {}).get('bits') != 4
                         or config.get('text_config', {}).get('hidden_size') != 2048):
        raise ValueError('candidate snapshot must be 4-bit Qwen3-VL-2B')
    model_files = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and (
        p.suffix in {'.json', '.safetensors', '.model', '.txt', '.jinja'} or p.name == 'vocab')}
    if not any(name.endswith('.safetensors') for name in model_files):
        raise ValueError('model snapshot needs local safetensors weights')
    if not is_judge and model_files.get('model.safetensors') != BASE_WEIGHT_SHA256:
        raise ValueError('Qwen weights differ from the published 2B 4-bit checkpoint')
    libraries = {}
    for name in ('mlx', 'mlx-vlm', 'transformers', 'Pillow'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    processor = {'processor': 'native MLX Qwen3-VL', **QWEN_PIXELS}
    if is_judge:
        processor_config = folder / 'processor_config.json'
        extra = json.loads(processor_config.read_bytes()) if processor_config.is_file() else {}
        processor = {'processor': 'native MLX InternVL', 'dynamic_max_num': JUDGE_MAX_TILES,
                     'dynamic_min_num': config.get('min_dynamic_patch', 1),
                     'dynamic_use_thumbnail': config.get('use_thumbnail', True),
                     'tile_pixels': extra.get('image_size', config.get('force_image_size',
                         config.get('vision_config', {}).get('image_size', 448)))}
    result = {
        'checkpoint': role, 'scope': SCOPE, 'backend': 'mlx-vlm-local',
        'declared_snapshot_revision': revision, 'model_files_sha256': model_files,
        'generation': dict(GENERATION, max_new_tokens=768 if is_judge else 512),
        'preprocessing': processor,
        'image_validation': 'sha256_before_each_call',
        'execution_code_sha256': sha(reference_execution.__file__),
        'trust_remote_code': False, 'libraries': libraries,
        'code_sha256': {'local': sha(__file__), **{module.__name__: sha(module.__file__) for module in
                        (reference_protocol, reference_runner, reference_execution, reference_scoring,
                         reference_analysis, reference_audit, data, runner)}},
    }
    if stage == 'lora':
        if args.adapter is None or args.adapter.suffix != '.safetensors':
            raise ValueError('LoRA role needs the selected adapter safetensors file')
        config_path = args.adapter.with_name('adapter_config.json')
        if json.loads(config_path.read_bytes()).get('fine_tune_type') != 'lora':
            raise ValueError('adapter_config.json must describe LoRA')
        result.update(adapter_sha256=sha(args.adapter), adapter_config_sha256=sha(config_path))
    return result


class MLXBackend:
    def __init__(self, stage, args, config):
        self.run_config = config
        self.folder = args.judge_model if stage == 'judge' else args.base_model
        self.adapter = args.adapter if stage == 'lora' else None
        self.assets = args.data_root
        self.judge = stage == 'judge'

    def prepare(self):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.trainer import prepare_model_for_training

        model, processor = load(str(self.folder), trust_remote_code=False)
        if self.adapter is not None:
            model = prepare_model_for_training(model, train_type='lora', checkpoint_path=self.adapter)
        model.eval()
        image_processor = processor.image_processor
        if self.judge:
            if not hasattr(image_processor, 'dynamic_max_num'):
                raise ValueError('native judge processor cannot enforce the tile cap')
            image_processor.dynamic_max_num = JUDGE_MAX_TILES
            settings = self.run_config['preprocessing']
            for attribute, key in (('dynamic_min_num', 'dynamic_min_num'),
                                   ('dynamic_use_thumbnail', 'dynamic_use_thumbnail'),
                                   ('size', 'tile_pixels')):
                if getattr(image_processor, attribute, None) != settings[key]:
                    raise ValueError(f'native judge processor differs from recorded {key}')
        else:
            for name, value in QWEN_PIXELS.items():
                setattr(image_processor, name, value)
        self.model, self.processor = model, processor
        self.generate_mlx, self.template = generate, apply_chat_template

    def generate(self, messages, max_new_tokens):
        from meme_study.freeze import _safe_file
        blocks = messages[0]['content']
        images = [str(_safe_file(self.assets, block['image'])) for block in blocks if block['type'] == 'image']
        if len(messages) != 1 or messages[0]['role'] != 'user' or len(images) > 1 or self.judge and not images:
            raise ValueError('invalid local explanation or judge conversation')
        raw = '\n'.join(block['text'] for block in blocks if block['type'] == 'text')
        if any(marker in raw for marker in RESERVED):
            raise ValueError('study input contains reserved model-template tokens')
        prompt = self.template(self.processor, self.model.config, messages, num_images=len(images))
        started = time.perf_counter()
        output = self.generate_mlx(self.model, self.processor, prompt, image=images or None,
                                   max_tokens=max_new_tokens, **GENERATION)
        return {'text': output.text, 'input_tokens': output.prompt_tokens,
                'output_tokens': output.generation_tokens, 'finish_reason': output.finish_reason,
                'latency_s': time.perf_counter() - started}


def freeze_guard(receipt, protocol, stage, config):
    if not isinstance(receipt, dict) or receipt.get('protocol') != protocol or (
            receipt.get('runtime', {}).get(stage) != config):
        raise ValueError('test freeze does not match the selected inputs and runtime')


def locked_test(study, snapshot, path):
    if not path.is_file():
        raise ValueError('test needs a freeze receipt written before dispatch')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(raw)
    if receipt.get('scope') != SCOPE or receipt.get('protocol') != study.frozen_protocol:
        raise ValueError('test freeze does not match the compiled inputs and judge note')
    stages = {model: stage for stage, model in {**ARMS, 'judge': JUDGE}.items()}
    for batch in snapshot['batches'].values():
        if not any(job['split'] == 'test' for job in batch['jobs']):
            continue
        config = dict(batch['run_config'])
        if config.pop('test_freeze_sha256', None) != digest:
            raise ValueError('recorded test batch was bound to another freeze receipt')
        freeze_guard(receipt, study.frozen_protocol, stages.get(config['checkpoint']), config)
    return receipt, digest


def score_rows(study, snapshot, split):
    judge_jobs = {job['parent_job_id']: job for job in study.judge_jobs(split, snapshot, require_candidates=False)}
    rows = []
    for job in study.jobs:
        if job['split'] != split:
            continue
        parent = bound_output(job, snapshot)
        ref = study.refs[job['family_id']]
        judge_job = judge_jobs.get(job['job_id'])
        judge = bound_output(judge_job, snapshot) if judge_job else None
        score = (score_checklist(parent['text'], ref['reference']['checklist'], judge['text'],
                                 truncated=judge['truncated']) if judge and judge['status'] == 'ok' else None)
        row = {key: job[key] for key in ('job_id', 'task_kind', 'family_id', 'split',
                                        'template_group', 'model', 'condition', 'ocr_available',
                                        'image_path', 'image_sha256')}
        row.update(source_input_sha256=job['input_sha256'], run_status=parent['status'] if parent else 'missing',
                   candidate_text=parent.get('text') if parent else None,
                   candidate_truncated=parent['truncated'] if parent else False,
                   candidate_output_sha256=content_hash(parent) if parent else None,
                   source=ref['source'], reference_image_path=ref['image']['path'],
                   reference=ref['reference'], ocr_text=ref['ocr_text'],
                   hint_text=study.hints[job['family_id']]['text'],
                   judge_job_id=judge_job['job_id'] if judge_job else None,
                   judge_profile_sha256=judge_job['judge_profile_sha256'] if judge_job else None,
                   judge_status=judge['status'] if judge else ('missing' if judge_job else 'not_applicable'),
                   judge_output_sha256=content_hash(judge) if judge else None, score=score)
        rows.append(row)
    return rows


def summarize(study, snapshot, split):
    rows = score_rows(study, snapshot, split)
    return {'scope': SCOPE, 'split': split, 'prepared_sha256': study.prepared,
            'ledger_sha256': snapshot['file_sha256'], 'attempts': snapshot['attempts'],
            'planned_explanations': len(rows),
            'candidate_status': dict(Counter(row['run_status'] for row in rows)),
            'judge_status': dict(Counter(row['judge_status'] for row in rows)),
            'score_status': dict(Counter((row['score'] or {}).get('status', 'missing') for row in rows)),
            'terminal_coverage': all(row['run_status'] != 'missing' and row['judge_status'] != 'missing'
                                     for row in rows)}


def run(args):
    note = args.test_judge_note.read_text(encoding='utf-8') if args.test_judge_note else None
    study = ExplanationStudy(load_prepared(args.prepared), note)
    receipt_path = args.freeze_receipt or args.runs.parent / f'{args.runs.name}.freeze.json'
    with _execution_lock(args.runs):
        snapshot = read_ledger(study, args.runs)
        test = locked_test(study, snapshot, receipt_path) if getattr(args, 'split', None) == 'test' else None
        if args.command == 'status':
            return {split: summarize(study, snapshot, split) for split in ('dev', 'test')}
        if args.command == 'freeze':
            if any(job[1]['split'] == 'test' for job in snapshot['jobs'].values()):
                raise ValueError('freeze must precede all test dispatches')
            roles = {stage: runtime(stage, args) for stage in (*ARMS, 'judge')}
            tasks = [job for job in study.jobs if job['split'] == 'test']
            _check_images(tasks, args.data_root)
            value = {'scope': SCOPE, 'protocol': study.frozen_protocol, 'runtime': roles,
                     'prepared_sha256': study.prepared, 'judge_note_sha256': content_hash(study.notes),
                     'test_explanations': len(tasks), 'development': summarize(study, snapshot, 'dev')}
            write_new(receipt_path, value)
            return {'freeze_sha256': sha(receipt_path), 'test_explanations': len(tasks)}
        if args.command == 'run':
            if args.max_jobs is not None and (args.max_jobs < 1 or args.split != 'dev'):
                raise ValueError('a positive max-jobs prefix is available only on dev')
            jobs = study.select(args.split, args.stage, snapshot)[:args.max_jobs]
            config = runtime(args.stage, args)
            if test is not None:
                freeze_guard(test[0], study.frozen_protocol, args.stage, config)
                config['test_freeze_sha256'] = test[1]
            _check_images(jobs, args.data_root)
            backend = MLXBackend(args.stage, args, config)
            checked = _ImageCheckedBackend(backend, jobs, args.data_root)
            checked.prepare = backend.prepare
            return run_batches(study, args.runs, jobs, checked)
        if args.command == 'judge-jobs':
            jobs = study.judge_jobs(args.split, snapshot)
            write_new(args.output, {'scope': SCOPE, 'ledger_sha256': snapshot['file_sha256'], 'jobs': jobs})
            return {'judge_jobs': len(jobs), 'output_sha256': sha(args.output)}
        rows = score_rows(study, snapshot, args.split)
        if args.command == 'audit':
            result = reference_audit.export_audit(rows, model_ids=list(ARMS.values()),
                                                  output_dir=args.output, phase=args.split)
            result['scope'] = SCOPE
            return result
        if args.command == 'score':
            summary = summarize(study, snapshot, args.split)
            if not summary['terminal_coverage'] and not args.allow_partial:
                raise ValueError('selected split is incomplete; --allow-partial produces an explicit partial export')
            summary['allow_partial'] = args.allow_partial
            report = reference_analysis.analyze_reference_scores(rows)
            report.update(scope=SCOPE, primary_split=args.split)
            artifacts = {
                'summary.json': json_bytes(summary), 'analysis.json': json_bytes(report),
                'scored.jsonl': b''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
                                        + b'\n' for row in rows),
            }
            artifacts['provenance.json'] = json_bytes({
                'scope': SCOPE, 'prepared_sha256': study.prepared, 'ledger_sha256': snapshot['file_sha256'],
                'test_freeze_sha256': test[1] if test is not None else None,
                'artifact_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()},
            })
            _write_bundle(args.output, artifacts)
            return summary
        raise ValueError('unknown operation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'data-root', 'runs'):
        parser.add_argument('--' + name, required=True, type=lambda path: Path(path).resolve())
    for name in ('base-model', 'adapter', 'judge-model', 'freeze-receipt'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--dev-calibration-note', '--test-judge-note', dest='test_judge_note', type=Path,
                        help='Apply a note derived from dev to the frozen test judge prompt.')
    parser.add_argument('--judge-revision')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    sub.add_parser('freeze')
    command = sub.add_parser('run')
    command.add_argument('--stage', required=True, choices=(*ARMS, 'judge'))
    command.add_argument('--split', required=True, choices=('dev', 'test'))
    command.add_argument('--max-jobs', type=int)
    for name in ('judge-jobs', 'score', 'audit'):
        command = sub.add_parser(name)
        command.add_argument('--split', required=True, choices=('dev', 'test'))
        command.add_argument('--output', required=True, type=Path)
        if name == 'score':
            command.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
