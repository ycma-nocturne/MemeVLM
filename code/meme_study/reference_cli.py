"""Reference-reuse preparation, execution, scoring, analysis and small audits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .data import content_hash
from .freeze import _publish, _safe_file
from .image_io import decode_image_receipt
from .reference_data import import_references
from .reference_protocol import (PROTOCOL_ID, compile_explanation_jobs, compile_pair_jobs,
                                 compile_judge_slots, summarize_plan)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n" for row in rows).encode()


def _read_json(raw, name):
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name}: duplicate JSON key {key}")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
        content_hash(value)  # Reject NaN/Infinity before compiling or writing.
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error


def _check_config(config):
    if not isinstance(config, dict) or config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unknown protocol; expected reference_reuse_v1")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict) or not {"revision", "sha256"} <= set(metadata):
        raise ValueError("config requires pinned metadata revision and sha256")
    models, judge = config.get("model_ids"), config.get("judge_model")
    if (not isinstance(models, list) or len(models) != 2
            or any(not isinstance(model, str) or not model.strip() for model in models)
            or len(set(models)) != 2):
        raise ValueError("config requires two distinct tested model IDs")
    if not isinstance(judge, str) or not judge.strip() or judge in models:
        raise ValueError("config requires an independent judge model")
    notes=config.get('judge_notes_by_split',{})
    if (not isinstance(notes,dict) or not set(notes)<={'dev','test'}
            or any(not isinstance(note,str) or len(note)>2000 for note in notes.values())):
        raise ValueError('judge_notes_by_split must map dev/test to at most 2000 characters each')
    design = config.get("design")
    fields = ("families", "dev", "test", "paired_families", "paired_dev", "paired_test",
              "planned_calls", "attempt_cap")
    if not isinstance(design, dict) or any(type(design.get(k)) is not int or design[k] < 0 for k in fields):
        raise ValueError("config design requires nonnegative integer family and call counts")


def _verify_images(records, pairs, root):
    """Verify the encoded bytes and decode every pixel."""
    receipts, cache = [], {}

    def check(path, digest, family, variant, scope):
        actual_path = _safe_file(root, path)
        key = (str(actual_path), digest)
        if key not in cache:
            cache[key] = decode_image_receipt(actual_path.read_bytes(), source_name=path,
                                              expected_sha256=digest)
        receipt = cache[key]
        receipts.append({"scope": scope, "family_id": family, "variant": variant,
                         "path": path, **receipt})
        return receipt

    for row in records:
        check(row["image"]["path"], row["image"]["sha256"], row["family_id"], "original", "A")
    for pair in pairs:
        dimensions = set()
        for name in ("original", "edited", "sham"):
            image = pair["variants"][name]
            receipt = check(image["image_path"], image["image_sha256"], pair["family_id"], name, "B")
            dimensions.add((receipt["width"], receipt["height"], receipt["detected_format"]))
        if len(dimensions) != 1:
            raise ValueError(f'{pair["family_id"]}: pair images must have matching dimensions and encoding')
    return receipts


def prepare(*, metadata, selection, hints, pairs, config, data_root, output):
    """Publish a complete preparation directory atomically.

    Keep reference answers in scoring artifacts. Pair readiness follows
    current material checks, and judge slots bind to candidate outputs later.
    """
    destination = Path(output).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    source_paths = {name: Path(path).resolve(strict=True) for name, path in
                    (("metadata", metadata), ("selection", selection), ("hints", hints),
                     ("pairs", pairs), ("config", config))}
    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    values = {name: _read_json(raw, name) for name, raw in source_bytes.items() if name != "metadata"}
    config, selection, hints, pairs = (values[name] for name in ("config", "selection", "hints", "pairs"))
    _check_config(config)
    root = Path(data_root).resolve(strict=True)
    records = import_references(source_paths["metadata"], selection, root,
                                expected_sha256=config["metadata"]["sha256"],
                                revision=config["metadata"]["revision"],
                                errata=config.get('source_errata'))
    indexed = {row["family_id"]: row for row in records}
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs must be a nonempty list")
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get("family_id") not in indexed:
            raise ValueError("each pair must be a selected public family")
        original = indexed[pair["family_id"]]
        if pair.get("split") != original["split"] or pair.get("template_group") != original["template_group"]:
            raise ValueError("pair split/template_group differs from its selected original")
    if not isinstance(hints, dict):
        raise ValueError("hints must be an object keyed by family_id")
    explanation_jobs = compile_explanation_jobs(records, hints, config["model_ids"])
    pair_jobs, gold = compile_pair_jobs(pairs, config["model_ids"], require_material_checks=False)
    summary = summarize_plan(explanation_jobs, pair_jobs)
    splits, pair_splits = Counter(row["split"] for row in records), Counter(row["split"] for row in pairs)
    actual_design = {"families": len(records), "dev": splits["dev"], "test": splits["test"],
                     "paired_families": len(pairs), "paired_dev": pair_splits["dev"],
                     "paired_test": pair_splits["test"], "planned_calls": summary["planned_calls"],
                     "attempt_cap": summary["attempt_cap"]}
    for key, actual in actual_design.items():
        if config["design"][key] != actual:
            raise ValueError(f"configured {key}={config['design'][key]} differs from actual {actual}")
    summary.update(actual_design)
    image_receipts = _verify_images(records, pairs, root)
    judge_slots = compile_judge_slots(explanation_jobs, records, config['judge_model'])
    artifacts = {"reference_records.jsonl": _jsonl_bytes(records),
                 "explanation_jobs.jsonl": _jsonl_bytes(explanation_jobs),
                 "pair_jobs.preview.jsonl": _jsonl_bytes(pair_jobs), "pair_gold.json": _json_bytes(gold),
                 "judge_slots.jsonl": _jsonl_bytes(judge_slots), "summary.json": _json_bytes(summary)}
    for name in ("selection", "hints", "pairs", "config"):
        artifacts[f"inputs/{name}.json"] = source_bytes[name]
    provenance = {
        "protocol_id": PROTOCOL_ID, "operation": "prepare_only_no_inference", "model_calls": 0,
        "source_files": {name: {"path": str(source_paths[name]), "sha256": _sha(raw), "bytes": len(raw)}
                         for name, raw in source_bytes.items()},
        "data_root": str(root), "image_receipts": image_receipts,
        "artifact_sha256": {name: _sha(raw) for name, raw in artifacts.items()},
        "private_field_notes": {
            "reference_records.jsonl": "Private reference.explanation/checklist/domain/reasoning; never pass whole records to tested models.",
            "pair_gold.json": "Private variant-specific answer sets; not original-image public checklists.",
            "inputs/pairs.json": "Private pair references and any supplied material-check records.",
            "judge_slots.jsonl": "Dependencies only; no candidate outputs or executable judge prompts yet.",
        },
        "material_checks_generated": 0,
        "image_scope": "A public originals and B cleaned triples are independent inputs; image decoding is not semantic acceptance.",
    }
    artifacts["provenance.json"] = _json_bytes(provenance)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        for name, raw in artifacts.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        # Refuse a package if any supplied inputs changed during preparation.
        for name, path in source_paths.items():
            if path.read_bytes() != source_bytes[name]:
                raise ValueError(f"{name} changed during preparation")
        for receipt in image_receipts:
            if _sha(_safe_file(root, receipt["path"]).read_bytes()) != receipt["sha256"]:
                raise ValueError("image changed during preparation")
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Validate and export inputs; perform no model calls")
    for name in ("metadata", "selection", "hints", "pairs", "config", "data-root", "output"):
        command.add_argument("--" + name, required=True, type=Path)
    command = commands.add_parser('inspect', help='Check saved inputs/images without model loading')
    command.add_argument('--prepared', required=True, type=Path)
    command.add_argument('--data-root', type=Path)
    command = commands.add_parser('status', help='Read shared attempt ledger without changing it')
    command.add_argument('--runs', required=True, type=Path)
    command = commands.add_parser('run', help='Execute one configured model; requires its GPU environment')
    for name in ('prepared', 'data-root', 'runs'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--model', required=True)
    command.add_argument('--task', choices=('explanation','pair','judge'), default='explanation')
    command.add_argument('--split', choices=('dev','test','all'), default='dev')
    command.add_argument('--max-jobs', type=int, help='Dev-only deterministic prefix including cached jobs; omit to continue the full stage')
    command.add_argument('--device', default='cuda:0')
    command.add_argument('--max-tiles', type=int, choices=(4,6))
    command = commands.add_parser('score', help='Join verified outputs and score; no model calls')
    for name in ('prepared','runs','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--allow-partial', action='store_true')
    command.add_argument('--allow-judge-json-fence', action='store_true',
                         help='Accept a single fenced JSON block around the judge response')
    command = commands.add_parser('analyze', help='Family-paired statistics from a verified score bundle')
    for name in ('scores','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--seed', type=int, default=6000)
    command.add_argument('--draws', type=int, default=10000)
    command = commands.add_parser('audit', help='Export the fixed 30-row blinded double-review forms')
    for name in ('scores','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--model-ids', nargs=2, required=True)
    command.add_argument('--seed', type=int, default=6000)
    command.add_argument('--phase', choices=('all','dev','test'), default='all')
    command = commands.add_parser('demo', help='Export 6–8 explicitly selected actual-response examples')
    for name in ('scores','data-root','selection','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--synthetic-preview', action='store_true',
                         help='Label the entire package as synthetic fixture-only, never measured results')
    command = commands.add_parser('material-review', help='Export the targeted B triple checks with blank decisions')
    for name in ('pairs','data-root','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command = commands.add_parser('material-import', help='Import supplied member decisions into a new pairs file')
    for name in ('pairs','review','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command = commands.add_parser('report-results', help='Export source-bound scientific tables and figures')
    for name in ('scores','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--seed', type=int, default=6000)
    command.add_argument('--draws', type=int, default=10000)
    command.add_argument('--allow-partial', action='store_true')
    command.add_argument('--synthetic-preview', action='store_true')
    command = commands.add_parser('costs', help='Export observed call/profile resources, not provider billing')
    for name in ('runs','output'):
        command.add_argument('--'+name, required=True, type=Path)
    command.add_argument('--profiles', type=Path)
    command.add_argument('--hkd-per-card-hour', type=float, default=12)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(metadata=args.metadata, selection=args.selection, hints=args.hints,
                             pairs=args.pairs, config=args.config, data_root=args.data_root, output=args.output)
        else:
            from .reference_execution import (load_prepared, read_reference_run, execute_reference,
                                               score_reference, load_scores, _write_bundle)
            if args.command == 'inspect':
                package = load_prepared(args.prepared)
                result = {**package['plan'], 'prepared_sha256': package['prepared_sha256'], 'images_checked':0}
                if args.data_root:
                    images = _verify_images(package['records'], package['artifacts']['inputs/pairs.json'],
                                            args.data_root.resolve(strict=True))
                    result['images_checked'] = len(images)
            elif args.command == 'status':
                snapshot = read_reference_run(args.runs)
                result = {'study': snapshot['study'], 'attempts': snapshot['attempts'],
                          'registered_jobs': len(snapshot['jobs']), 'terminal_outputs': len(snapshot['outputs']),
                          'statuses': dict(Counter(o['status'] for o in snapshot['outputs'])),
                          'ledger_sha256': snapshot['ledger_sha256']}
            elif args.command == 'run':
                options = {'max_tiles':args.max_tiles} if args.max_tiles is not None else {}
                if options and args.task != 'judge':
                    raise ValueError('--max-tiles is only valid for the judge')
                result = execute_reference(prepared=args.prepared, data_root=args.data_root,
                    runs=args.runs, model=args.model, task=args.task, split=args.split, device=args.device, max_jobs=args.max_jobs, **options)
            elif args.command == 'score':
                result = score_reference(prepared=args.prepared, runs=args.runs, output=args.output,
                                         allow_partial=args.allow_partial,
                                         allow_judge_json_fence=args.allow_judge_json_fence)
            elif args.command == 'analyze':
                from .reference_analysis import analyze_reference_scores
                report = analyze_reference_scores(load_scores(args.scores), seed=args.seed, draws=args.draws)
                raw = _json_bytes(report)
                provenance = {'protocol_id': PROTOCOL_ID, 'model_calls':0,
                    'scores_sha256': _sha((args.scores/'scored.jsonl').read_bytes()),
                    'analysis_sha256': _sha(raw), 'seed':args.seed, 'draws':args.draws,
                    'analysis_code_sha256': _sha(Path(__file__).with_name('reference_analysis.py').read_bytes())}
                _write_bundle(args.output, {'analysis.json': raw, 'provenance.json': _json_bytes(provenance)})
                result = {'output': str(args.output), **provenance}
            elif args.command == 'audit':
                from .reference_audit import export_audit
                result = export_audit(load_scores(args.scores), model_ids=args.model_ids,
                                      output_dir=args.output, seed=args.seed, phase=args.phase)
            elif args.command == 'demo':
                from .reference_demo import export_demo
                selections = _read_json(args.selection.read_bytes(), 'demo selection')
                result = export_demo(load_scores(args.scores), data_root=args.data_root,
                                     output_dir=args.output, selections=selections,
                                     synthetic_preview=args.synthetic_preview)
            elif args.command == 'material-review':
                from .reference_materials import export_pair_review
                result = export_pair_review(_read_json(args.pairs.read_bytes(),'pairs'),
                                            data_root=args.data_root,output_dir=args.output)
            elif args.command == 'material-import':
                from .reference_materials import import_pair_review
                result = import_pair_review(_read_json(args.pairs.read_bytes(),'pairs'),
                                            review_csv=args.review,output_dir=args.output)
            elif args.command == 'report-results':
                from .reference_reporting import export_reference_results
                result = export_reference_results(load_scores(args.scores),output_dir=args.output,
                    seed=args.seed,draws=args.draws,allow_partial=args.allow_partial,
                    synthetic_preview=args.synthetic_preview)
            elif args.command == 'costs':
                from .reference_costs import export_reference_costs
                resolved_runs=args.runs.resolve()
                profiles=args.profiles or resolved_runs.parent/(resolved_runs.name+'.profiles')
                result = export_reference_costs(runs=args.runs,profiles=profiles,output_dir=args.output,
                                                hkd_per_card_hour=args.hkd_per_card_hour)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
