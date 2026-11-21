"""Compare 2B base and LoRA with JSON-constrained Task B decoding."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from meme_study import reference_analysis, reference_protocol, reference_scoring
from meme_study.data import content_hash
from meme_study.freeze import _safe_file
from meme_study.runner import _validate_response


ARMS = ("base", "lora")
SCOPE = "local_qwen3vl_2b_task_b_constrained"
FREEZE_PURPOSE = "authorize_fixed_held_out_constrained_predictions"
DECODING_SCHEMA = {
    "type": "object",
    "properties": {
        "cue": {"type": "string", "enum": ["yes", "no", "insufficient"]},
        "relation": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "explanation": {"type": "string", "minLength": 1, "pattern": r"\S"},
    },
    "required": ["cue", "relation", "explanation"],
    "additionalProperties": False,
}
GENERATION = dict(temperature=0.0, min_pixels=262144, max_pixels=1310720,
                  skip_special_tokens=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as staging:
        pending = Path(staging.name)
        try:
            json.dump(value, staging, ensure_ascii=False, indent=2, allow_nan=False)
            staging.write("\n")
            staging.flush()
            os.fsync(staging.fileno())
            staging.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def compile_matrix(project, split):
    project = Path(project).resolve(strict=True)
    pairs = [row for row in read_json(project / "data/study/pairs.json")
             if row["split"] == split]
    jobs, gold = reference_protocol.compile_pair_jobs(pairs, list(ARMS))
    assets = (project / "data/assets").resolve(strict=True)
    for name, expected in {(job["image_path"], job["image_sha256"]) for job in jobs}:
        if digest(_safe_file(assets, name)) != expected:
            raise ValueError(f"image checksum mismatch: {name}")
    return jobs, gold, assets


def installed_versions():
    result = {}
    for name in ("mlx", "mlx-vlm", "transformers", "Pillow", "llguidance"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def manifest(project, model, adapter, split, jobs, gold):
    config = read_json(model / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("model directory must contain a Qwen3-VL checkpoint")
    names = {path.name for pattern in ("*.json", "*.safetensors", "*.model", "*.jinja",
                                        "merges.txt", "*.py")
             for path in model.glob(pattern) if path.is_file()}
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("model directory contains no safetensors weights")
    adapter_config = adapter.with_name("adapter_config.json")
    if read_json(adapter_config).get("fine_tune_type") != "lora":
        raise ValueError("adapter_config.json must describe a LoRA adapter")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "split": split,
        "model_files": {name: digest(model / name) for name in sorted(names)},
        "adapter_sha256": digest(adapter),
        "adapter_config_sha256": digest(adapter_config),
        "source_pairs_sha256": digest(Path(project) / "data/study/pairs.json"),
        "compiled_inputs_sha256": content_hash({"jobs": jobs, "gold": gold}),
        "generation": GENERATION,
        "constrained_decoding": {
            "engine": "mlx-vlm llguidance JSON Schema",
            "schema": DECODING_SCHEMA,
            "schema_sha256": content_hash(DECODING_SCHEMA),
        },
        "max_new_tokens": sorted({job["max_new_tokens"] for job in jobs}),
        "installed_versions": installed_versions(),
        "code_sha256": {name: digest(Path(path)) for name, path in {
            "runner": __file__, "compiler": reference_protocol.__file__,
            "scores": reference_scoring.__file__, "analysis": reference_analysis.__file__,
        }.items()},
    }


def freeze(project, model, adapter, receipt):
    jobs, gold, _ = compile_matrix(project, "test")
    settings = manifest(project, Path(model), Path(adapter), "test", jobs, gold)
    receipt = Path(receipt)
    with receipt.open("x", encoding="utf-8") as output:
        json.dump({"purpose": FREEZE_PURPOSE, "manifest": settings},
                  output, indent=2)
        output.write("\n")
    return {"freeze_sha256": digest(receipt), "planned_jobs": len(jobs)}


def load_predictor(model_dir, adapter):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.structured import build_json_schema_logits_processor
    from mlx_vlm.trainer import prepare_model_for_training

    model, processor = load(str(model_dir), trust_remote_code=False)
    if adapter is not None:
        model = prepare_model_for_training(model, train_type="lora", checkpoint_path=adapter)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    grammar = build_json_schema_logits_processor(tokenizer, DECODING_SCHEMA)

    def predict(job, image):
        prompt = apply_chat_template(processor, model.config, job["messages"], num_images=1)
        started = time.perf_counter()
        result = generate(model, processor, prompt, image=[str(image)],
                          max_tokens=job["max_new_tokens"],
                          logits_processors=[grammar.clone()], **GENERATION)
        return {"text": result.text, "input_tokens": result.prompt_tokens,
                "output_tokens": result.generation_tokens, "finish_reason": result.finish_reason,
                "latency_s": time.perf_counter() - started}

    return predict


def bound_record(path, job, run_sha):
    row = read_json(path)
    if (row.get("job_id"), row.get("input_sha256"), row.get("run_sha256")) != (
            job["job_id"], job["input_sha256"], run_sha):
        raise ValueError(f"cached prediction binding mismatch: {path.name}")
    response = _validate_response(row["response"], job["max_new_tokens"])
    if content_hash(response) != row.get("response_sha256"):
        raise ValueError(f"cached prediction response hash mismatch: {path.name}")
    return response


def prediction_index(records, jobs):
    return {job["job_id"]: records / f"{content_hash(job['job_id'])}.json" for job in jobs}


def summarize(jobs, gold, records, run_sha, split):
    score_rows, predictions = [], []
    index = prediction_index(records, jobs)
    measure = split != "test" or all(path.is_file() for path in index.values())
    for job in jobs:
        row = {key: job[key] for key in ("job_id", "family_id", "template_group", "model", "split",
                "task_kind", "condition", "variant", "input_sha256", "image_sha256")}
        path = index[job["job_id"]]
        response = bound_record(path, job, run_sha) if path.is_file() else None
        score = reference_scoring.score_pair(
            response["text"], job["options"], gold[job["job_id"]],
            truncated=response["finish_reason"] == "length",
        ) if measure and response is not None else None
        score_rows.append({**row, "run_status": "ok" if response is not None else "pending",
                           "score": score})
        if response is not None:
            predictions.append({**row, **response, "score": score})

    if not measure:
        return {
            "scope": SCOPE, "split": split, "run_sha256": run_sha, "planned_jobs": len(jobs),
            "completed_jobs": len(predictions), "valid_predictions": None, "arms": {},
            "matched_triple_differences": None,
        }, predictions

    analysis = reference_analysis.analyze_reference_scores(score_rows)
    arms = analysis["splits"][split]["paired"]["models"]
    values = {arm: {row["family_id"]: row["value"] for row in
                    arms[arm]["triples"]["all_three_joint"]["contributions"]} for arm in ARMS}
    common = sorted(values["base"].keys() & values["lora"].keys())
    gains = sum(values["lora"][key] > values["base"][key] for key in common)
    losses = sum(values["lora"][key] < values["base"][key] for key in common)
    return {
        "scope": SCOPE, "split": split, "run_sha256": run_sha, "planned_jobs": len(jobs),
        "completed_jobs": len(predictions),
        "valid_predictions": sum(row["score"]["status"] == "ok" for row in predictions),
        "arms": arms,
        "matched_triple_differences": {"n_common": len(common), "improved": gains,
                                       "regressed": losses, "tied": len(common) - gains - losses},
    }, predictions


def run(project, model, adapter, output, *, split, receipt=None, max_per_arm=None,
        loader=load_predictor):
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if max_per_arm is not None and (max_per_arm < 1 or split != "dev"):
        raise ValueError("max-per-arm must be positive and is available for development")
    if split == "test" and receipt is None:
        raise ValueError("held-out execution requires an explicit matching freeze receipt")
    if split == "dev" and receipt is not None:
        raise ValueError("a freeze receipt applies to held-out execution")

    model, adapter, output = (Path(p).resolve() for p in (model, adapter, output))
    jobs, gold, assets = compile_matrix(project, split)
    settings = manifest(project, model, adapter, split, jobs, gold)
    if receipt is not None:
        receipt = Path(receipt)
        raw = receipt.read_bytes()
        if json.loads(raw) != {"purpose": FREEZE_PURPOSE, "manifest": settings}:
            raise ValueError("freeze receipt does not match the proposed held-out run")
        settings["freeze_sha256"] = hashlib.sha256(raw).hexdigest()
    run_sha = content_hash(settings)

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "run.json"
        if manifest_path.exists():
            if read_json(manifest_path) != settings:
                raise ValueError("run manifest changed. Choose a fresh output directory.")
        else:
            if set(path.name for path in output.iterdir()) != {".lock"}:
                raise ValueError("choose a fresh output directory")
            write_json(manifest_path, settings)
        records = output / "records"
        records.mkdir(exist_ok=True)
        paths = prediction_index(records, jobs)
        if set(records.glob("*.json")) - set(paths.values()):
            raise ValueError("unexpected prediction records in the run")
        for job in jobs:
            if paths[job["job_id"]].exists():
                bound_record(paths[job["job_id"]], job, run_sha)
        try:
            for arm in ARMS:
                chosen = [job for job in jobs if job["model"] == arm][:max_per_arm]
                pending = [job for job in chosen if not paths[job["job_id"]].exists()]
                if not pending:
                    continue
                predict = loader(model, adapter if arm == "lora" else None)
                try:
                    for job in pending:
                        image = _safe_file(assets, job["image_path"])
                        if digest(image) != job["image_sha256"]:
                            raise ValueError(f"image changed before prediction: {job['image_path']}")
                        response = _validate_response(predict(job, image), job["max_new_tokens"])
                        write_json(paths[job["job_id"]], {"job_id": job["job_id"],
                                   "input_sha256": job["input_sha256"], "run_sha256": run_sha,
                                   "response_sha256": content_hash(response), "response": response})
                finally:
                    del predict
                    gc.collect()
        finally:
            summary, predictions = summarize(jobs, gold, records, run_sha, split)
            write_json(output / "summary.json", summary)
            write_json(output / "predictions.json", predictions)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Check development input hashes and show counts")
    freezing = commands.add_parser("freeze", help="Freeze exact held-out settings without generation")
    running = commands.add_parser("run", help="Generate predictions sequentially and retain completed records")
    for command in (planning, freezing, running):
        command.add_argument("--project", type=Path, required=True)
    for command in (freezing, running):
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--adapter", type=Path, required=True)
    freezing.add_argument("--receipt", type=Path, required=True)
    running.add_argument("--split", choices=("dev", "test"), required=True)
    running.add_argument("--freeze", type=Path)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--max-per-arm", type=int)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            jobs, _, _ = compile_matrix(args.project, "dev")
            report = {"split": "dev", "planned_jobs": len(jobs),
                      "families": len({job["family_id"] for job in jobs})}
        elif args.command == "freeze":
            report = freeze(args.project, args.model_dir, args.adapter, args.receipt)
        else:
            summary = run(args.project, args.model_dir, args.adapter, args.output, split=args.split,
                          receipt=args.freeze, max_per_arm=args.max_per_arm)
            report = {key: summary[key] for key in ("split", "planned_jobs", "completed_jobs",
                                                    "valid_predictions", "matched_triple_differences")}
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
