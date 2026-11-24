# Expanded 4B experiments

The experiment compares the pinned Qwen3-VL-4B base with two LoRA profiles across seeds 42, 43, and 44. The Chinese profile repeats 98 unique rows to match the bilingual profile's 770 rows per epoch. Both profiles use 291 optimizer updates and the same 193 validation rows. Checkpoint selection uses validation loss at steps 97, 194, and 291.

## Stage the workspace

Use the [4B CUDA environment](../README.md), pinned model snapshot, and both private team handoffs. Set `REPO` to this checkout and `WORK` to an absolute experiment directory outside the checkout. Put the repository code and private project inputs under `WORK/project`. The image overlap check uses `training/images/`, `training/bilingual/images/`, and `data/assets/` within that directory. Put the pinned Qwen snapshot under `WORK/model`. Use the original bilingual export at WORK/inputs/sft-bilingual. Copy the original English paired freeze, receipts, and ledgers to WORK/evaluation. Put the original English image release at WORK/evaluation/english-corpus and the original seed-42 bilingual adapter run at WORK/output/bilingual.

~~~sh
exp="$WORK/final-expansion"
mkdir -p "$exp"
cp "$REPO/training/gpu/expansion/"*.py "$exp/"
cp "$REPO/training/gpu/eval/memeqa/run_memeqa.py" "$exp/"
cp "$REPO/training/gpu/eval/memeqa/prepare.py" "$exp/pilot_prepare.py"
~~~

Use the expanded handoff's protocol.json in final-expansion/. It binds the input manifests, prompt policy, and scoring code. Rebuild the native inputs from the pinned [MemeQA and SemEval files](../eval/memeqa/README.md). Set `NATIVE_INPUTS` to the absolute directory holding the downloaded archives and `memeqa-test.json`. Save the original pilot selection manifest there as `pilot-manifest.json`.

~~~sh
(cd "$NATIVE_INPUTS" && python "$exp/build_expanded.py" --project "$WORK/project" --output "$exp/native")
python "$WORK/project/training/gpu/prepare_data.py" export --project "$WORK/project" \
  --profile chinese --output "$exp/inputs/chinese-source"
python "$exp/matched_controls.py" --project "$WORK/project" prepare \
  --bilingual "$WORK/inputs/sft-bilingual" --chinese "$exp/inputs/chinese-source" \
  --output "$exp/inputs/chinese-control"
~~~

Use Pillow 13.0.0 to rebuild the native inputs. Use a separate pinned CUDA environment for training and inference.

## Train and evaluate

Run each new training job on one allocated GPU. The bilingual seeds are 43 and 44. The Chinese seeds are 42, 43, and 44. For bilingual, use PROFILE=bilingual, ARM=bilingual and DATASET="$WORK/inputs/sft-bilingual". For Chinese, use PROFILE=chinese, ARM=chinese-replay and DATASET="$exp/inputs/chinese-control".

~~~sh
run="$exp/train-runs/$PROFILE-$SEED"
python "$exp/matched_controls.py" --project "$WORK/project" train \
  --arm "$ARM" --seed "$SEED" --dataset "$DATASET" --model "$WORK/model" --output "$run"
~~~

For each new adapter, use the [paired English evaluator](../eval/README.md) to freeze and run the adapter arm. Write freeze.json and runs/ under final-expansion/synthetic-runs/PROFILE-SEED. The aggregate verifies the original base and bilingual seed-42 ledgers before reusing them.

Use the expanded runner for each of base, bilingual-42, bilingual-43, bilingual-44, chinese-42, chinese-43, and chinese-44.

~~~sh
python -B "$exp/run_native.py" --root "$WORK" --arm "$RUN_NAME"
python -B "$exp/aggregate.py" --root "$WORK"
~~~

The final aggregation runs once after all arms complete. It reports all eligible native questions and the primary subset outside the original pilot images. The team handoffs supply the frozen receipts, raw responses, and scoring keys for CPU replay.

