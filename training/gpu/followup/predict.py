"""Run the frozen 4B MemeQA generation policy on development or test images."""
import argparse
import gc
import json
from pathlib import Path
import sys


STEPS = (25, 50, 75, 97)
SEEDS = (42, 43, 44)
ARMS = ("baseline", "real", "real-low")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "work"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run_prediction(parser.parse_args())


def run_prediction(args, *, subdirectory=None, independent_holdout=False, source_subdirectory=None):
    root, work = args.root.resolve(), args.work.resolve()
    if not work.is_relative_to(root):
        raise ValueError("Follow-up outputs must stay within the experiment root")
    output = (work / subdirectory).resolve() if subdirectory else work
    if not output.is_relative_to(work):
        raise ValueError("Prediction outputs must stay within the follow-up directory")
    project = root / "project"
    sys.path[:0] = [str(project / name) for name in ("training/gpu/eval", "code", ".")]
    from backend import CudaBackend, POLICY
    from meme_study import runner
    from meme_study.freeze import _safe_file
    from run_eval import checked_response, write_new
    from training.gpu.eval.memeqa.run_memeqa import LABELS, PROMPT, answer, digest, lines
    from training.gpu.launch import verify_model

    prior = root / "evaluation/freeze.json"
    historical = json.loads(prior.read_text())
    protocol_path = root / "final-expansion/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    model = root / "model"
    if digest(prior) != protocol["prior_freeze_sha256"]:
        raise ValueError("Reference freeze differs from the archived protocol")
    if POLICY != historical["policy"] or verify_model(model) != historical["training"]["model_files"]:
        raise ValueError("Inference model or policy differs from the archived comparator")

    source = output if source_subdirectory is None else (work / source_subdirectory).resolve()
    run = (source / "runs" / f"{args.arm}-{args.seed}").resolve()
    if not run.is_relative_to(work):
        raise ValueError("Adapter must stay within the follow-up directory")
    receipt_path = run / "run-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    data_receipt = work / "prepared/selection-receipt.json"
    if digest(data_receipt) != receipt["data_selection_sha256"]:
        raise ValueError("The training data selection receipt has changed")
    selection_path = output / "selection.json"
    if args.split == "test":
        selection = json.loads(selection_path.read_text())
        seed_key = str(args.seed)
        arms = selection["test_arms"] if subdirectory else ("baseline", selection["selected_by_seed"][seed_key])
        if args.arm not in arms:
            raise ValueError("This arm was not selected on development data")
        selected = selection["checkpoints"][args.arm][seed_key]
        step = selected["step"]
        data = output / "prepared" if independent_holdout else root / "final-expansion/native"
        held = selection["holdout"] if independent_holdout else protocol["native"]
        for filename, key in (("inputs.jsonl", "inputs_sha256"), ("manifest.json", "manifest_sha256")):
            if digest(data / filename) != held[key]:
                raise ValueError(f"Frozen test changed: {filename}")
        cases = lines(data / "inputs.jsonl")
        if independent_holdout:
            expected = held["questions"], held["images"]
        else:
            pilot_groups = set(json.loads((data / "manifest.json").read_text())["pilot_group_ids"])
            cases = [row for row in cases if row["group_id"] not in pilot_groups]
            expected = held["new_questions"], held["new_images"]
        steps = (step,)
    else:
        data = work / "prepared/native-dev"
        data_binding = json.loads(data_receipt.read_text())
        manifest_path = data / "manifest.json"
        if digest(manifest_path) != data_binding["native_dev_manifest_sha256"]:
            raise ValueError("The development manifest differs from its frozen receipt")
        if digest(data / "inputs.jsonl") != json.loads(manifest_path.read_text())["inputs_sha256"]:
            raise ValueError("The development inputs differ from their manifest")
        cases = lines(data / "inputs.jsonl")
        expected = (96, 96)
        steps = tuple(receipt["checkpoint_steps"]) if subdirectory else STEPS
    ids = {row["id"] for row in cases}
    groups = {row["group_id"] for row in cases}
    if len(cases) != len(ids) or (len(ids), len(groups)) != expected:
        raise ValueError("MemeQA inputs differ from the expected image/question counts")
    images = {row["image"]: row["image_sha256"] for row in cases}
    if len(images) != len(groups) or any(digest(_safe_file(data, path)) != sha for path, sha in images.items()):
        raise ValueError("MemeQA evaluation images differ from the prepared inputs")

    for step in steps:
        entry = receipt["checkpoints"][str(step)]
        adapter = (run / entry["adapter_dir"]).resolve()
        if not adapter.is_relative_to(run):
            raise ValueError("Adapter path escapes its run directory")
        fingerprints = {name: digest(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")}
        if fingerprints != {"adapter_config.json": entry["adapter_config_sha256"],
                            "adapter_model.safetensors": entry["adapter_sha256"]}:
            raise ValueError("Adapter bytes differ from the training receipt")
        if args.split == "test" and fingerprints != selected["adapter"]:
            raise ValueError("Adapter differs from the model selected on development data")
        arm_name = f"{args.arm}-{args.seed}"
        out = output / ("dev-runs" if args.split == "dev" else "test-runs") / arm_name
        if args.split == "dev":
            out /= str(step)
        out.mkdir(parents=True, exist_ok=True)
        binding = {"arm": args.arm, "seed": args.seed, "step": step, "split": args.split,
                   "inputs_sha256": digest(data / "inputs.jsonl"), "manifest_sha256": digest(data / "manifest.json"),
                   "adapter": fingerprints, "training_receipt_sha256": digest(receipt_path),
                   "selection_sha256": digest(selection_path) if args.split == "test" else None,
                   "reference_protocol_sha256": digest(protocol_path), "policy": POLICY, "prompt": PROMPT}
        with runner._run_lock(out):
            binding_path = out / "binding.json"
            if not binding_path.exists():
                write_new(binding_path, binding)
            elif json.loads(binding_path.read_text()) != binding:
                raise ValueError("Existing responses belong to another evaluation")
            ledger = out / "responses.jsonl"
            rows = runner._read_lines(ledger)
            previous = {row["id"] for row in rows}
            if len(rows) != len(previous) or not previous.issubset(ids):
                raise ValueError("Prediction ledger has incompatible question IDs")
            hardware_path = out / "hardware.json"
            if previous != ids:
                backend = CudaBackend(model, adapter, data, "cuda:0")
                hardware = backend.prepare()
                if not hardware_path.exists():
                    write_new(hardware_path, hardware)
                elif json.loads(hardware_path.read_text()) != hardware:
                    raise ValueError("Resumed predictions use a different runtime")
                for index, case in enumerate((row for row in cases if row["id"] not in previous), len(previous) + 1):
                    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LABELS, case["options"], strict=True))
                    prompt = f'{PROMPT}\n\n{case["question"]}\n{choices}'
                    response = checked_response(backend.generate({"image": case["image"], "prompt": prompt}))
                    runner._append(ledger, {"id": case["id"], "choice": answer(response), **response})
                    if index % 50 == 0 or index == len(cases):
                        print(f"{args.split} {arm_name} step {step}: {index}/{len(cases)}", flush=True)
                torch = backend.torch
                del backend
                gc.collect()
                torch.cuda.empty_cache()
            rows = runner._read_lines(ledger)
            if len(rows) != len(ids) or {row["id"] for row in rows} != ids:
                raise ValueError("Prediction ledger is incomplete")
            completion = {"cases": len(rows), "valid": sum(row["choice"] is not None for row in rows),
                          "ledger_sha256": digest(ledger), "binding_sha256": digest(binding_path)}
            path = out / "completion.json"
            if not path.exists():
                write_new(path, completion)
            elif json.loads(path.read_text()) != completion:
                raise ValueError("Completion receipt differs from its predictions")


if __name__ == "__main__":
    main()
