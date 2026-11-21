# Reproduce the Apple Silicon study

Complete [setup](../README.md#set-up). Run from the extracted project root. The handoff includes the selected epoch 1 adapter, its training receipts, the held-out run records, and the freeze files.

## Prepare the benchmark

```bash
export PYTHONPATH=.:code:local-study
work=$(mktemp -d)
assets=data/assets
study=data/study
prepared="$work/prepared"
python -m meme_study.reference_cli prepare \
  --metadata "$assets/candidates/raw/memebench-metadata.jsonl" \
  --selection "$study/selection.json" --hints "$study/hints.json" \
  --pairs "$study/pairs.json" --config "$study/config.json" \
  --data-root "$assets" --output "$prepared"
python -m meme_study.reference_cli inspect --prepared "$prepared" --data-root "$assets"
```

The expected prepared SHA-256 is `e50d1e56349da4021dee6e88b82d38e7bb5d7194ac397c4b5be2f9d44846a4f5`. The original plan uses 105 image references.

## Replay the recorded scoring

```bash
python local-study/explanation_score_fences.py \
  --prepared "$prepared" --runs results/explanations/runs \
  --freeze-receipt local-study/explanation-runs.freeze.json \
  --scoring-freeze local-study/explanation-scoring.freeze.json \
  score --split test --output "$work/a-replayed"
python local-study/verify_results.py --results results
```

Task A scores the saved judge responses strictly and, in a supplementary analysis, removes a sole enclosing JSON fence. Task B checks exact cue and relation answers before combining the three variants. See the [protocol](protocol.md) for denominators and selection.

## Rerun training and generation

Create a separate Python 3.12 environment on Apple Silicon. Restore the pinned checkpoints by SHA-256 and keep the selected adapter in the handoff for evaluation replay.

```bash
run=training/runs/mlx-qwen3vl-2b
python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install -r "$run/requirements-resolved.txt"
python local-study/models/restore.py base --download
# Restore the independent judge for Task A.
python local-study/models/restore.py judge --download
export HF_HUB_OFFLINE=1
```

To retrain, move `$run/full/` and `local-study/training/epoch-02/` out of a fresh extracted copy and keep their original receipts. Run `python "$run/local_train.py" full`, then `python local-study/training/continue_lora.py 2`. The two epochs use 98 training examples and 33 validation examples. Rebuild the prepared model inputs on a separate copy with `export_mlx.py --training-root training --output "$work/mlx-data" --eval-root .`, followed once by `cap_image_pixels.py "$work/mlx-data"`. The handoff includes the checked input copy at `$run/data`.

To regenerate candidate explanations, use `local-study/explanation_study.py` on the development split for `base` and `lora`. Then run `local-study/explanation_batch_judge.py` with `--batch-size 6`. Write fresh judge and scoring freezes before held-out generation. Run `pair_study.py` and `pair_constrained_study.py` from the same directory for the free and constrained Task B policies. Use their `freeze` commands for fresh held-out runs. CLI help lists the model, adapter, dataset, freeze, and output flags. Keep the new outputs next to their own receipts.
