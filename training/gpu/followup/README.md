# MemeQA follow-up

The training comparison uses 770 records per epoch. The real-meme recipe replaces 64 complete authored families with 256 four-option questions on distinct images from the official MemeQA None− training split. Both recipes keep the same 98 Chinese examples and 193 training-validation records. The 96-question development set selects checkpoints and the peak learning rate.

## Results

| Comparison | Cohort | Three-seed mean accuracy | Paired difference |
| --- | --- | --- | --- |
| Real-meme recipe vs original recipe, both at `1e-4` | Official test, 816 questions on 134 images | 72.47% vs 55.15% | +17.32 pp [14.66, 20.07] |
| Effective batch 4 vs 8 at rank 8 | Confirmation, 433 questions on 77 images | 80.68% vs 78.29% | +2.39 pp [0.61, 4.26] |
| Rank 16 vs 8 at effective batch 4 | Same confirmation cohort | 82.06% vs 80.68% | +1.39 pp [−0.23, 3.15] |

The confirmation images come from previously unused images in the official MemeQA development split. We screened them against training, checkpoint-development and prior test images. Intervals are 95% paired image-bootstrap intervals from 10,000 draws, conditional on the three fitted seeds. The data comparison measures the complete training recipe, including the change in image source and supervision format.

At fixed `1e-4`, mean development accuracy was 56.25% for the original recipe and 77.43% for the real-meme recipe. The latter scored 71.18% at `5e-5`. On the original test cohort, the four one-factor ablations gave +3.72 pp for batch 4, −2.49 pp for rank 4, +2.33 pp for rank 16 and −3.02 pp for scale 1, each relative to batch 8, rank 8, scale 2. Their aggregate includes Bonferroni-adjusted intervals.

We use the real-meme recipe with peak LR `1e-4`, effective batch 4, and LoRA rank 8, alpha 16. Batch 4 improved all three seeds on the confirmation cohort. The rank-16 combination doubled the adapter size to 33,030,144 parameters and its paired interval included zero.

[Development scores](results/development.json) · [Paired data comparison](results/data-test.json) · [One-factor ablations](results/ablation.json) · [Configuration confirmation](results/confirmation.json)

## Prepare inputs

Use Python 3.12, Pillow 13.0.0 and NumPy 2.5.4 for data preparation. Put a copy of this repository under `$ROOT/project` and these scripts under `$WORK`, a new directory inside `$ROOT`. Supply the authorized original training files and `data/assets/` under `$ROOT/project`, the original paired freeze at `$ROOT/evaluation/freeze.json` and the frozen expanded protocol and native corpus under `$ROOT/final-expansion/`. The [expanded evaluation guide](../expansion/README.md) specifies these inputs.

Populate `$ANNOTATIONS` with `memeqa-train.json`, `memeqa-dev.json` and `memeqa-test.json` from the pinned `data/none_minus` revision below. Populate `$ARCHIVES` with `semeval-training.zip`, `semeval-dev.zip` and `semeval-test.zip` using the [pinned archive instructions](../eval/memeqa/README.md#prepare-the-inputs). The preparation scripts verify both sources and preserve the original question and option wording.

```sh
python -B "$WORK/prepare_data.py" --project "$ROOT/project" \
  --annotations "$ANNOTATIONS" --memeqa-test "$ANNOTATIONS/memeqa-test.json" \
  --archives "$ARCHIVES" --output "$WORK/prepared"
python -B "$WORK/confirmation/prepare_holdout.py" --project "$ROOT/project" \
  --dev "$ANNOTATIONS/memeqa-dev.json" --prior "$WORK/prepared" \
  --memeqa-test "$ANNOTATIONS/memeqa-test.json" --archives "$ARCHIVES" \
  --output "$WORK/confirmation/prepared"
```

Keep the generated images, answer keys and per-question response ledgers in your research workspace. [MemeQA source revision](https://github.com/npnkhoi/memeqa/tree/e48b332a02f7b25e08459dd7dcf69f19b430c621/data/none_minus) · [SemEval usage terms](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus#licensing)

## Train and evaluate

Use `$ROOT/model` for the pinned Qwen3-VL-4B-Instruct snapshot and `$ROOT/.venv-cuda` for the [CUDA environment](../README.md). The Slurm `bootstrap.sbatch` script can provision both inside `$ROOT`. Create the log directories before launch. Set the account and partition for your cluster.

```sh
mkdir -p "$WORK/logs" "$WORK/ablations/logs" "$WORK/confirmation/logs"
sbatch --output="$WORK/logs/bootstrap-%j.log" "$WORK/bootstrap.sbatch" "$ROOT" "$WORK"
```

Run each of the following jobs after the preceding stage succeeds. These resource flags provide three single-GPU lanes on a node. Adjust the resource request to the cluster.

```sh
sbatch --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=120G --gres=gpu:3 --time=01:30:00 \
  --output="$WORK/logs/data-%j.log" "$WORK/packed.sbatch" "$ROOT" "$WORK" 3
sbatch --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=120G --gres=gpu:3 --time=01:30:00 \
  --output="$WORK/logs/ablation-%j.log" "$WORK/ablations/run.sbatch" "$ROOT" "$WORK" 3
sbatch --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=120G --gres=gpu:3 --time=01:30:00 \
  --output="$WORK/logs/confirmation-%j.log" "$WORK/confirmation/run.sbatch" "$ROOT" "$WORK"
```

All stages pair seeds 42, 43 and 44 and execute one epoch. Batch 8 checkpoints are steps 25, 50, 75 and 97 under the 291-step schedule. Batch 4 checkpoints are steps 50, 100, 150 and 193 under the 579-step schedule. The smaller batch uses more updates at 770 presentations. Select by development accuracy and break ties toward earlier checkpoints. For the learning rate, ties retain `1e-4`. The confirmation stage reuses the selected rank-8 checkpoints.

## Score saved predictions

Use the authorized predictions and labels in the paths above. The first command scores the native test. The other commands replay the saved compact analysis inputs with fresh output directories.

```sh
python -B "$WORK/score.py" --runs "$WORK/test-runs" --frozen "$ROOT/final-expansion" \
  --selection "$WORK/selection.json" --output "$WORK/recomputed-data.json"
python -B "$WORK/ablations/collect.py" --bundle "$WORK/ablations/compact-results.json.gz" \
  --previous "$WORK/results/compact-results.json.gz" --dev "$WORK/prepared/native-dev" \
  --frozen "$ROOT/final-expansion" --output "$WORK/recomputed-ablation"
python -B "$WORK/confirmation/score_confirmation.py" \
  --bundle "$WORK/confirmation/compact-results.json.gz" --holdout "$WORK/confirmation/prepared" \
  --ablations "$WORK/ablations/compact-results.json.gz" --previous "$WORK/results/compact-results.json.gz" \
  --dev "$WORK/prepared/native-dev" --output "$WORK/recomputed-confirmation"
```
