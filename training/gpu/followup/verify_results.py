import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

work = Path(sys.argv[1])
history = Path(sys.argv[2])
sys.path.insert(0, str(work))
from score import score

bundle = json.loads(gzip.decompress((work / "results/compact-results.json.gz").read_bytes()))
selection = work / "results/selection.json"
official_scores = bundle["scores"]
assert sha256(selection.read_bytes()).hexdigest() == official_scores["selection"]["sha256"]
dev = work / "prepared/native-dev"
gold = {row["id"]: row["answer"] for row in map(json.loads, (dev / "gold.jsonl").read_text().splitlines())}
assert len(bundle["dev_ids"]) == len(gold) == 96
assert len(bundle["test_ids"]) == 816
for run, value in bundle["dev"].items():
    arm_seed, step = run.split("/")
    arm, seed = arm_seed.rsplit("-", 1)
    accuracy_rows = bundle["selection"]["scores"][arm][seed]
    expected, = (row for row in accuracy_rows if row["step"] == int(step))
    correct = sum(gold[key] == label for key, label in zip(bundle["dev_ids"], value["choices"], strict=True))
    assert correct == expected["correct"], run

with TemporaryDirectory() as scratch:
    predictions = Path(scratch)
    for name, value in bundle["test"].items():
        destination = predictions / name
        destination.mkdir()
        rows = (json.dumps({"id": key, "choice": label if label != "-" else None})
                for key, label in zip(bundle["test_ids"], value["choices"], strict=True))
        (destination / "responses.jsonl").write_text("\n".join(rows) + "\n")
    actual = score(predictions, history, selection, include_archived=True)
for key in ("by_profile", "selected_minus_baseline", "stability", "cohort"):
    assert actual[key] == official_scores[key], key
for name, parsed in actual["per_run"].items():
    for key in ("denominator", "accuracy", "correct", "valid"):
        assert parsed[key] == official_scores["per_run"][name][key], (name, key)
print("Recomputed all 36 dev cells, 6 final runs and the paired confidence interval.")
print("Selection", bundle["selection"]["mean_accuracy"], bundle["selection"]["selected_by_seed"])
print("Per run", {name: (row["correct"], row["valid"]) for name, row in actual["per_run"].items()})
print("Result", actual["by_profile"], actual["selected_minus_baseline"], actual["stability"])
print("Archived contrasts", actual["archived_context"]["selected_minus_bilingual"], actual["archived_context"]["selected_minus_base"])
