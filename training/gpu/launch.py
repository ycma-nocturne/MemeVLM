"""Plan, launch, resume, and verify the fixed Qwen3-VL 4B CUDA LoRA recipe."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[1]))

from training.gpu.prepare_data import file_sha256, json_bytes, sha256, verify_dataset

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_FILES = {
    "model-00001-of-00002.safetensors": "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors": "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
    "chat_template.json": "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    "config.json": "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
    "generation_config.json": "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors.index.json": "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
PYTORCH = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "torchaudio": "2.8.0+cu128"}
RECIPE = HERE / "qwen3vl_4b_lora.json"
ADAPTER = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
BINDING = "run-manifest.json"
SELECTED = "selected-adapter.json"
CHECKPOINT = re.compile(r"checkpoint-([1-9][0-9]*)")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(json_bytes(value))
        scratch = Path(tmp.name)
    scratch.replace(path)


def verify_model(model: Path | str) -> dict[str, str]:
    """Check the exact weights, tokenizer, processor and model config at the pinned revision."""
    model = Path(model).resolve(strict=True)
    if not model.is_dir():
        raise ValueError("choose a local Qwen3-VL model directory")
    for alternate in ("model.safetensors", "pytorch_model.bin", "adapter_config.json", "processor_config.json",
                      "chat_template.jinja", "chat_templates"):
        if (model / alternate).exists():
            raise ValueError(f"unexpected overriding model file: {alternate}")
    for name, expected in MODEL_FILES.items():
        if file_sha256(model / name) != expected:
            raise ValueError(f"file differs from the pinned Qwen revision: {name}")
    return MODEL_FILES.copy()


def make_config(dataset: Path, model: Path, trainer: Path) -> dict:
    config = _read_json(RECIPE)
    if config["model_revision"] != MODEL_REVISION:
        raise ValueError("recipe model revision mismatch")
    config.update(model_name_or_path=str(model), dataset_dir=str(dataset), media_dir=str(dataset),
                  output_dir=str(trainer))
    return config


def _runtime() -> dict:
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("CUDA training requires Linux and Python 3.12")
    requirements = dict(line.split("==", 1) for line in (HERE / "requirements.txt").read_text().splitlines()
                        if line and not line.startswith("#"))
    versions = {}
    for package, expected in (requirements | PYTORCH).items():
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise ValueError(f"{package} {installed} found, expected {expected}")
        versions[package] = installed
    import torch

    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise ValueError("CUDA 12.8 PyTorch and an accessible NVIDIA GPU are required")
    if torch.cuda.device_count() != 1:
        raise ValueError("select one visible NVIDIA GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("the selected GPU must support native BF16")
    return versions


def latest_checkpoint(trainer: Path) -> Path:
    candidates = [(int(match[1]), path) for path in trainer.glob("checkpoint-*")
                  if (match := CHECKPOINT.fullmatch(path.name))]
    if not candidates:
        raise ValueError("no training checkpoint available for resume")
    step, checkpoint = max(candidates)
    required = (ADAPTER, ADAPTER_CONFIG, "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    if checkpoint.is_symlink() or any(
        not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0 for name in required
    ):
        raise ValueError(f"incomplete checkpoint: {checkpoint.name}")
    state = _read_json(checkpoint / "trainer_state.json")
    if state["global_step"] != step:
        raise ValueError("checkpoint step does not match trainer state")
    best = state.get("best_model_checkpoint")
    if best and (Path(best).parent != trainer or not (Path(best) / ADAPTER).is_file()):
        raise ValueError("restore the original run path and previous best checkpoint before resuming")
    return checkpoint


def _valid_metric(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def selected_adapter(output: Path, binding: dict) -> dict:
    trainer = output / "trainer"
    state = _read_json(trainer / "trainer_state.json")
    best, loss = state["best_model_checkpoint"], state["best_metric"]
    match = CHECKPOINT.fullmatch(Path(best).name) if isinstance(best, str) else None
    if match is None or not _valid_metric(loss) or state["global_step"] != state["max_steps"]:
        raise ValueError("training is incomplete or the best validation checkpoint is missing")
    checkpoint = trainer / Path(best).name
    hashes = {name: file_sha256(trainer / name) for name in (ADAPTER, ADAPTER_CONFIG)}
    if any(file_sha256(checkpoint / name) != digest for name, digest in hashes.items()):
        raise ValueError("saved adapter differs from the best checkpoint")
    final_loss = _read_json(trainer / "eval_results.json")["eval_loss"]
    if not _valid_metric(final_loss):
        raise ValueError("final validation loss must be finite")
    return {**binding, "receipt_version": 1, "best_checkpoint": str(checkpoint.relative_to(output)),
            "best_eval_loss": loss, "final_eval_loss": final_loss, "optimizer_steps": state["global_step"],
            "adapter_sha256": hashes[ADAPTER], "adapter_config_sha256": hashes[ADAPTER_CONFIG]}


def _verify_run(output: Path, manifest: dict, dataset_digest: str) -> dict:
    binding = _read_json(output / BINDING)
    receipt = _read_json(output / SELECTED)
    if binding["dataset_sha256"] != dataset_digest or binding["profile"] != manifest["profile"]:
        raise ValueError("run and dataset do not match")
    if binding["model_id"] != MODEL_ID or binding["model_revision"] != MODEL_REVISION:
        raise ValueError("unexpected model in the run receipt")
    if receipt != selected_adapter(output, binding):
        raise ValueError("selected adapter receipt differs from the current run files")
    return receipt


def _reload(output: Path, model: Path, receipt: dict) -> dict:
    _runtime()
    if verify_model(model) != receipt["model_files"]:
        raise ValueError("model files differ from training")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    base = AutoModelForImageTextToText.from_pretrained(
        model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"},
        local_files_only=True, trust_remote_code=False,
    )
    adapted = PeftModel.from_pretrained(base, output / "trainer", is_trainable=False).eval()
    count = sum(parameter.numel() for name, parameter in adapted.named_parameters() if "lora_" in name)
    if not count:
        raise ValueError("PEFT reload did not attach LoRA parameters")
    return {"reloaded": True, "lora_parameters": count, "adapter_sha256": receipt["adapter_sha256"]}


def _run(args: argparse.Namespace, manifest: dict, dataset_digest: str) -> dict:
    dataset = args.dataset.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(dataset) or output.is_relative_to(model):
        raise ValueError("choose an output directory outside the dataset and model")
    runtime = _runtime()
    model_files = verify_model(model)
    resume = args.command == "resume"
    config = make_config(dataset, model, output / "trainer")
    portable_config = {key: value for key, value in config.items()
                       if key not in {"model_name_or_path", "dataset_dir", "media_dir", "output_dir"}}
    validation_rows = manifest["splits"]["validation"]["rows"]
    binding = {"mode": "train", "profile": manifest["profile"], "dataset_sha256": dataset_digest,
               "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_files": model_files,
               "config_sha256": sha256(json_bytes(portable_config)), "validation_rows": validation_rows,
               "dependencies": runtime}
    if not resume:
        output.mkdir(parents=True)
    with (output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if (output / SELECTED).exists():
                raise ValueError("the full run already has a selected adapter")
            if _read_json(output / BINDING) != binding:
                raise ValueError("resume requires the original dataset, model, recipe and dependencies")
            checkpoint = latest_checkpoint(output / "trainer")
            config["resume_from_checkpoint"] = str(checkpoint)
        else:
            _write_json(output / BINDING, binding)
        # JSON is also YAML. LlamaFactory 0.9.5 reads the .yaml config through OmegaConf.
        resolved = output / ("resume.yaml" if resume else "run.yaml")
        _write_json(resolved, config)
        env = os.environ | {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "FORCE_TORCHRUN": "0",
                            "PYTHONHASHSEED": str(config["seed"]),
                            "USE_MCA": "0", "USE_V1": "0", "USE_RAY": "0", "WANDB_MODE": "disabled"}
        subprocess.run([sys.executable, "-m", "llamafactory.cli", "train", str(resolved)], env=env, check=True)
        receipt = selected_adapter(output, binding)
        _write_json(output / SELECTED, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "train", "resume", "verify", "reload"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name != "verify":
            command.add_argument("--model", type=Path, required=True)
    commands.add_parser("verify-model").add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify-model":
            result = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_files": verify_model(args.model)}
        else:
            manifest, dataset_digest = verify_dataset(args.dataset)
            if args.command == "plan":
                result = {"dataset_sha256": dataset_digest, "profile": manifest["profile"],
                          "splits": manifest["splits"], "model_id": MODEL_ID, "mode": "train",
                          "config": make_config(args.dataset.resolve(), args.model.resolve(),
                                                args.output.resolve() / "trainer")}
            elif args.command in ("verify", "reload"):
                result = _verify_run(args.output.resolve(strict=True), manifest, dataset_digest)
                if args.command == "reload":
                    result = _reload(args.output, args.model, result)
            else:
                result = _run(args, manifest, dataset_digest)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError,
            importlib.metadata.PackageNotFoundError, subprocess.CalledProcessError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
