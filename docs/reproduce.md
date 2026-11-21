# Reproduce the study

Run the commands from `code/` after completing [setup](../README.md#set-up).

## 1. Verify the inputs

Use the private input snapshot with the layout below. Set `MEMEVLM_STUDY_ROOT` to its project directory when working from a separate code checkout.

| Path under the study root | Contents |
| --- | --- |
| `data/study/{config,selection,hints,pairs}.json` | Exact study configuration, split, background hints, and triple definitions with scoring references |
| `data/assets/candidates/raw/memebench-metadata.jsonl` | Pinned upstream metadata |
| `data/assets/<image_path>` | 60 originals listed in `selection.json` and 45 triple images listed under `variants` in `pairs.json` |

The MemeBench dataset uses CC BY-NC-SA 4.0. To recover an upstream original, use `source_image_url` and verify `image_sha256` in `selection.json`. The pinned metadata file has SHA-256 `9abb2951ea6f2e681207e4b2917514f69e4c5b45426ef9fbc5326bbdfa837010`. [Dataset card](https://huggingface.co/datasets/anonymous-neurips-2026/memebench) · [Pinned metadata](https://huggingface.co/datasets/anonymous-neurips-2026/memebench/resolve/d421198c5fd8108b17d336a5d1e8a7c229bd2b1f/metadata.jsonl)

```bash
study_root="${MEMEVLM_STUDY_ROOT:-$(cd .. && pwd)}"
study="$study_root/data/study"
assets="$study_root/data/assets"
work="$(cd .. && pwd)/local"
prepared="$work/prepared"
runs="$work/runs"
mkdir -p "$work"

python -m meme_study.reference_cli prepare \
  --metadata "$assets/candidates/raw/memebench-metadata.jsonl" \
  --selection "$study/selection.json" --hints "$study/hints.json" \
  --pairs "$study/pairs.json" --config "$study/config.json" \
  --data-root "$assets" --output "$prepared"

python -m meme_study.reference_cli inspect --prepared "$prepared" --data-root "$assets"
```

`prepare` requires a new output directory. Use `inspect` on subsequent checks. The current input snapshot yields 105 image checks, 830 planned calls, an attempt cap of 913, and `prepared_sha256` equal to `e50d1e56349da4021dee6e88b82d38e7bb5d7194ac397c4b5be2f9d44846a4f5`.

## 2. Run development jobs on CUDA

Create separate Python environments for Qwen and InternVL on the GPU host. Install a compatible CUDA-enabled PyTorch and torchvision pair in both, then install the shared dependencies with Transformers `4.57.1` for Qwen and `4.48.3` for InternVL. Set `QWEN_PY` and `INTERNVL_PY` to those Python executables.

```bash
"$QWEN_PY" -m pip install -r requirements-reference-gpu.txt 'transformers==4.57.1'
"$INTERNVL_PY" -m pip install -r requirements-reference-gpu.txt 'transformers==4.48.3'
```

Run a development pilot and record the code commit, input hash, checkpoint revisions, CUDA and driver versions, GPU, and resolved dependencies. The adapters pin checkpoint revisions. Keep one `runs` directory and its adjacent `runs.profiles` directory for all stages.

```bash
q4=Qwen/Qwen3-VL-4B-Instruct
q8=Qwen/Qwen3-VL-8B-Instruct
judge=OpenGVLab/InternVL3-9B

for model in "$q4" "$q8"; do
  "$QWEN_PY" -m meme_study.reference_cli run \
    --prepared "$prepared" --data-root "$assets" --runs "$runs" \
    --model "$model" --task explanation --split dev --max-jobs 3
done
```

Continue the full development split after the pilot. Run the judge after both models have terminal explanation records. Resuming a stage uses the same ledger.

```bash
split=dev
(
  set -e
  for model in "$q4" "$q8"; do
    for task in explanation pair; do
      bash scripts/reference_job.sh "$QWEN_PY" "$prepared" "$assets" "$runs" \
        "$model" "$task" "$split"
    done
  done
  bash scripts/reference_job.sh "$INTERNVL_PY" "$prepared" "$assets" "$runs" \
    "$judge" judge "$split"
)
```

## 3. Score, audit, and export

```bash
python -m meme_study.reference_cli score --prepared "$prepared" --runs "$runs" \
  --output "$work/scores-dev" --allow-partial
python -m meme_study.reference_cli audit --scores "$work/scores-dev" \
  --output "$work/audit-dev" --model-ids "$q4" "$q8" --phase dev
```

Have two reviewers score the 10 blinded development explanations independently. After calibration, record the frozen inputs and judge instructions. Run the full-stage commands with `split=test`, using the same shared ledger. If calibration changes the configuration, use the new prepared directory and record its hash.

Once the test run is complete, use fresh output directories. Export the 20 blinded test explanations for independent review and retain the two sets of ratings with the run records.

```bash
scores="$work/scores"
python -m meme_study.reference_cli score --prepared "$prepared" --runs "$runs" --output "$scores"
python -m meme_study.reference_cli audit --scores "$scores" --output "$work/audit-test" \
  --model-ids "$q4" "$q8" --phase test
python -m meme_study.reference_cli analyze --scores "$scores" --output "$work/analysis"
python -m meme_study.reference_cli report-results --scores "$scores" --output "$work/report"
python -m meme_study.reference_cli costs --runs "$runs" --output "$work/costs"
```
