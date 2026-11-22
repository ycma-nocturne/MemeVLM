# English paired evaluation

Compare Qwen3-VL-4B-Instruct with the selected GPU LoRA on the project-authored English families in `training/bilingual/`. The synthetic English test contains 32 held-out families and 96 image variants.

One family has original, edited, and sham images. The edited image changes the target relation. The sham preserves the original answer. Both arms receive the same English prompt and the same option order across a family's variants. Seed 6000 orders families, and the gold labels rotate across A–D within each split.

## Run

Use the CUDA environment in [GPU training](../README.md). The recorded comparison used BF16 and greedy decoding with the model's sampling defaults disabled. Run from the extracted project root with the private English corpus in `training/bilingual/`. Set `MODEL` to the verified local Qwen snapshot, `DATASET` to the verified trainer export, and `ADAPTER_RUN` to the completed GPU training output containing `selected-adapter.json` and `trainer/`.

```sh
eval_script=training/gpu/eval/run_eval.py
work=/path/to/evaluation-output
mkdir -p "$work"

python "$eval_script" freeze --model "$MODEL" --dataset "$DATASET" \
  --adapter-run "$ADAPTER_RUN" --freeze "$work/freeze.json"

for arm in base adapter; do
  python "$eval_script" run --split test --arm "$arm" \
    --model "$MODEL" --dataset "$DATASET" --adapter-run "$ADAPTER_RUN" \
    --freeze "$work/freeze.json" --runs "$work/runs"
done

python "$eval_script" score --split test --freeze "$work/freeze.json" \
  --runs "$work/runs" --output "$work/test-scores.json"
```

Use the same run command after an interruption. The ledger preserves terminal results, marks an unresolved dispatch as interrupted, and continues with untouched cases. Keep the freeze, score file, run receipts, and raw event ledgers together. CUDA results are tied to the recorded software and GPU environment.

## Scoring

Each response supplies a yes/no cue, an A–D relation, and a short explanation in one JSON object. Cue and relation must both match. The explanation must be present. Strict family success requires correct answers on all three variants. The score file records planned denominators, format validity, per-variant counts, paired family outcomes, and the adapter-minus-base difference. Failed and invalid cases remain in the planned denominator.

[Qwen snapshot](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [Transformers 4.57.1](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/qwen3_vl) · [PEFT 0.18.1](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/peft_model.py)

## Replay saved scores

Use the original freeze and response ledgers from the team handoff with the same corpus. Run the `score` command above with a new output path. It runs on CPU with the dependencies in `code/requirements-cpu.txt`.

The supplementary scorer removes a single enclosing JSON code fence before applying the same scoring. To replay it, stage a copy of the original `evaluation/` records in a fresh working directory. Set `REPO` to the checkout, `EVIDENCE` to the handoff's English evaluation directory, and `CORPUS` to the handoff's `implementation/training/bilingual` directory.

~~~sh
work=$(mktemp -d)
mkdir "$work/evaluation"
ln -s "$REPO" "$work/project"
ln -s "$CORPUS" "$work/evaluation/english-corpus"
ln -s "$EVIDENCE/runs" "$work/evaluation/runs"
cp "$EVIDENCE/analysis-plan.json" "$EVIDENCE/freeze.json" "$work/evaluation/"
cp "$REPO/training/gpu/eval/score_supplement.py" "$work/evaluation/"
PYTHONPATH="$REPO/code" python -B "$work/evaluation/score_supplement.py"
cmp "$work/evaluation/supplementary-scores.json" "$EVIDENCE/supplementary-scores.json"
~~~
