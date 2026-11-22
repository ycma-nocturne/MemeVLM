"""Evaluate one fixed 4B arm on the expanded native MemeQA question set."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    expand = root / "final-expansion"
    data = expand / "native"
    sys.path.insert(0, str(root / "project/training/gpu/eval"))
    sys.path.insert(0, str(expand))
    import run_eval
    from backend import CudaBackend, POLICY
    from meme_study import runner
    from meme_study.freeze import _safe_file
    from run_memeqa import PROMPT, LABELS, answer, digest, lines
    from training.gpu.launch import verify_model

    protocol = json.loads((expand / "protocol.json").read_text())
    arm = next(item for item in protocol["arms"] if item["name"] == args.arm)
    prior_path = root / "evaluation/freeze.json"
    prior = json.loads(prior_path.read_text())
    if digest(prior_path) != protocol["prior_freeze_sha256"] or POLICY != prior["policy"]:
        raise ValueError("Backend policy differs from the frozen original test")
    if digest(data / "inputs.jsonl") != protocol["native"]["inputs_sha256"]:
        raise ValueError("Native questions differ from the fixed expanded cohort")
    if digest(data / "manifest.json") != protocol["native"]["manifest_sha256"]:
        raise ValueError("Native manifest differs from the fixed expanded cohort")
    if digest(Path(__file__)) != protocol["code"]["run_native.py"]:
        raise ValueError("Native runner differs from the fixed analysis plan")
    cases = lines(data / "inputs.jsonl")
    ids = {case["id"] for case in cases}
    groups = {case["group_id"] for case in cases}
    if len(ids) != len(cases) or len(cases) != protocol["native"]["questions"] or len(groups) != protocol["native"]["images"]:
        raise ValueError("Native cohort has inconsistent census")
    images = {case["image"]: case["image_sha256"] for case in cases}
    if len(images) != len(groups) or any(digest(_safe_file(data, name)) != sha for name, sha in images.items()):
        raise ValueError("Native input images differ from frozen images")
    if verify_model(root / "model") != prior["training"]["model_files"]:
        raise ValueError("Base model differs from the original paired test")
    adapter = None
    bound = None
    if arm["adapter_run"]:
        run = root / arm["adapter_run"]
        adapter = run / "trainer"
        receipt_path = run / "selected-adapter.json"
        receipt = json.loads(receipt_path.read_text())
        for name, key in (("adapter_model.safetensors", "adapter_sha256"),
                          ("adapter_config.json", "adapter_config_sha256")):
            if digest(adapter / name) != receipt[key]:
                raise ValueError(f"Selected adapter {name} differs from its receipt")
        if receipt["optimizer_steps"] != 291 or receipt["model_files"] != prior["training"]["model_files"]:
            raise ValueError("Incompatible selected adapter training budget or base")
        bound = {"selected_receipt_sha256": digest(receipt_path), "adapter_sha256": receipt["adapter_sha256"]}
    path = expand / "native-runs" / args.arm
    path.mkdir(parents=True, exist_ok=True)
    with runner._run_lock(path):
        binding_path = path / "binding.json"
        binding = {"arm": args.arm, "protocol_sha256": digest(expand / "protocol.json"),
                   "label_prompt": PROMPT, "generation_policy": POLICY, "adaptation": bound}
        if binding_path.is_file():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Resume inputs differ from frozen run")
        else:
            run_eval.write_new(binding_path, binding)

        ledger = path / "responses.jsonl"
        recorded = runner._read_lines(ledger)
        seen = {row["id"] for row in recorded}
        if len(seen) != len(recorded) or not seen.issubset(ids):
            raise ValueError("Invalid ledger IDs")
        unseen = [case for case in cases if case["id"] not in seen]
        hardware_path = path / "hardware.json"
        if unseen:
            backend = CudaBackend(root / "model", adapter, data, "cuda:0")
            hardware = backend.prepare()
            if hardware_path.is_file():
                if json.loads(hardware_path.read_text()) != hardware:
                    raise ValueError("Resume runtime differs from previous generations")
            else:
                run_eval.write_new(hardware_path, hardware)
            for n, case in enumerate(unseen, len(seen) + 1):
                choices = "\n".join(f"{letter}. {option}" for letter, option in zip(LABELS, case["options"]))
                prompt = f'{PROMPT}\n\n{case["question"]}\n{choices}'
                response = run_eval.checked_response(backend.generate({"image": case["image"], "prompt": prompt}))
                runner._append(ledger, {"id": case["id"], "choice": answer(response), **response})
                if n % 50 == 0 or n == len(cases):
                    print(f"{args.arm}: {n}/{len(cases)}", flush=True)

        final = runner._read_lines(ledger)
        if len(final) != len(cases) or {row["id"] for row in final} != ids:
            raise ValueError("Expanded inference incomplete")
        result = {"arm": args.arm, "cases": len(final), "valid": sum(row["choice"] is not None for row in final),
                  "ledger_sha256": digest(ledger), "binding_sha256": digest(binding_path),
                  "hardware": json.loads(hardware_path.read_text())}
        completed = path / "completion.json"
        if completed.is_file() and json.loads(completed.read_text()) != result:
            raise ValueError("Completed receipt differs from current records")
        if not completed.is_file():
            run_eval.write_new(completed, result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

