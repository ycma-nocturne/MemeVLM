"""Run an exact-choice comparison on a fixed sample of native MemeQA test questions."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import sys

PROMPT = (
    "Answer the multiple-choice question about this meme. Select one option. "
    'Return one JSON object with exactly one key "answer" and a value of "A", "B", "C", or "D".'
)
LABELS = "ABCD"


def digest(path):
    with path.open("rb") as handle:
        from hashlib import file_digest
        return file_digest(handle, "sha256").hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def answer(row):
    def distinct_keys(items):
        if len({key for key, _ in items}) != len(items):
            raise ValueError("Duplicate answer keys")
        return dict(items)

    if row["finish_reason"] != "stop":
        return None
    try:
        value = json.loads(row["text"], object_pairs_hook=distinct_keys)
    except ValueError:
        return None
    if isinstance(value, dict) and set(value) == {"answer"} and value["answer"] in list(LABELS):
        return value["answer"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "output", "project", "model", "adapter-run", "prior-freeze"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "training/gpu/eval"))
    import run_eval
    from backend import CudaBackend, POLICY
    from meme_study import runner
    from meme_study.freeze import _safe_file
    from training.gpu.launch import verify_model

    cases = lines(args.data / "inputs.jsonl")
    gold = {row["id"]: row["answer"] for row in lines(args.data / "gold.jsonl")}
    assert len(cases) == len(gold) == 32 and {row["id"] for row in cases} == gold.keys()
    assert len({row["source_image"] for row in cases}) == len({row["image_sha256"] for row in cases}) == 32
    assert Counter(gold.values()) == Counter(dict.fromkeys(LABELS, 8))
    for case in cases:
        assert isinstance(case["question"], str) and case["question"].strip()
        assert len(case["options"]) == 4 and len(set(case["options"])) == 4
        assert digest(_safe_file(args.data, case["image"])) == case["image_sha256"]
    prior = json.loads(args.prior_freeze.read_text())
    assert prior["policy"] == POLICY and POLICY["use_model_defaults"] is False
    assert verify_model(args.model) == prior["training"]["model_files"]
    adapter = args.adapter_run / "trainer"
    for path, key in [("adapter_config.json", "adapter_config_sha256"),
                      ("adapter_model.safetensors", "adapter_sha256")]:
        assert digest(adapter / path) == prior["training"][key]
    args.output.mkdir(exist_ok=True)
    freeze_path = args.output / "freeze.json"
    frozen = {
        "protocol": "memeqa-native-test-sample-v1",
        "planned_per_arm": len(cases),
        "scoring": "Exact one-key JSON. Invalid outputs count as incorrect.",
        "prompt": PROMPT,
        "inputs_sha256": digest(args.data / "inputs.jsonl"),
        "gold_sha256": digest(args.data / "gold.jsonl"),
        "selection_manifest_sha256": digest(args.data / "manifest.json"),
        "same_4b_as_english_freeze_sha256": digest(args.prior_freeze),
        "runner_sha256": digest(Path(__file__)),
        "policy": POLICY,
    }
    run_eval.write_new(freeze_path, frozen)
    results, hardware, hashes = {}, {}, {}
    for arm in ("base", "adapter"):
        print(f"{arm}: loading model", flush=True)
        backend = CudaBackend(args.model, adapter if arm == "adapter" else None, args.data, "cuda:0")
        hardware[arm] = backend.prepare()
        ledger = args.output / f"{arm}.jsonl"
        assert not ledger.exists()
        results[arm] = []
        for case in cases:
            choices = "\n".join(f"{letter}. {option}" for letter, option in zip(LABELS, case["options"]))
            prompt = f'{PROMPT}\n\n{case["question"]}\n{choices}'
            response = run_eval.checked_response(backend.generate({"image": case["image"], "prompt": prompt}))
            choice = answer(response)
            runner._append(ledger, {"id": case["id"], **response, "choice": choice})
            results[arm].append({"id": case["id"], "valid": choice is not None, "correct": choice == gold[case["id"]]})
        print(f"{arm}: generated {len(results[arm])} answers", flush=True)
        hashes[arm] = digest(ledger)
        del backend

    paired = Counter((a["correct"], b["correct"]) for a, b in zip(results["base"], results["adapter"]))
    gains, losses = paired[False, True], paired[True, False]
    discordant = gains + losses
    exact_p = min(1.0, 2 * sum(comb(discordant, i) for i in range(min(gains, losses) + 1)) / (2**discordant))
    assert hardware["base"] == hardware["adapter"]
    report = {
        "protocol": frozen["protocol"],
        "freeze_sha256": digest(freeze_path),
        "per_arm": {arm: {"generated": len(rows), "valid": sum(row["valid"] for row in rows),
                          "correct": sum(row["correct"] for row in rows)} for arm, rows in results.items()},
        "paired": {"both_correct": paired[True, True], "base_only": losses, "adapter_only": gains,
                   "both_incorrect": paired[False, False], "two_sided_exact_p": exact_p},
        "ledger_sha256": hashes,
        "hardware": hardware["base"],
    }
    run_eval.write_new(args.output / "scores.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

