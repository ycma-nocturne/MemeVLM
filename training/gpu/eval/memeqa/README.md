# MemeQA transfer evaluation

The probe samples 32 unique images from the authors' MemeQA **None− test split** with seed 6000. It keeps one native four-option question per image, preserves question and option wording, and balances gold positions across A–D. Both 4B arms receive the image and question. The runner builds a one-key JSON prompt and scores the exact choice.

## Prepare the inputs

From the repository root, use Python 3.12 and Pillow 13.0.0 in a CPU environment. The SemEval archives supply the source images for research use.

~~~sh
repo="$PWD"
work=$(mktemp -d)
python3.12 -m venv "$work/prepare-env"
"$work/prepare-env/bin/python" -m pip install Pillow==13.0.0
cd "$work"

mqa=e48b332a02f7b25e08459dd7dcf69f19b430c621
sem=796b32fa76b7a5814a48d1870d163058e81e4ce1
curl -fLsS --retry 2 \
  "https://raw.githubusercontent.com/npnkhoi/memeqa/$mqa/data/none_minus/test.json" \
  -o memeqa-test.json
for split in training dev test; do
  curl -fLsS --retry 2 \
    "https://raw.githubusercontent.com/di-dimitrov/SEMEVAL-2021-task6-corpus/$sem/data/${split}_set_task3.zip" \
    -o "semeval-$split.zip"
done

shasum -a 256 -c <<'SHA256'
ca22300a959293a0ff32206855151499ab22e8ba1065800b33cfcef303ef2a2e  memeqa-test.json
402de437a0952fb1b6bdb578d8eb42b08cb78809de60a494abcd55d093a9dcf9  semeval-training.zip
83fc1fa124a2132bafadd78e6557bd2bd7bac3628a2d0bf276bed1becfb675e5  semeval-dev.zip
6200b07851798e99a06a3ab499dab24f711e06fd4ad7e4ba3f41d3d59b69fe88  semeval-test.zip
SHA256

"$work/prepare-env/bin/python" "$repo/training/gpu/eval/memeqa/prepare.py" "${PROJECT:-$repo}"
~~~

The script writes the sample to `frozen/`. Set `PROJECT` to the absolute path of the team handoff's implementation directory to repeat the recorded overlap check against 956 project images and recreate the exact manifest. Keep generated images and answer keys in the working directory.

## Run the comparison

Use the [pinned 4B CUDA environment](../../README.md). Set `REPO` to the checkout, `INPUT` to the prepared `frozen/` directory, `MODEL` to the pinned local model snapshot, and `ADAPTER_RUN` to the selected adapter run. Set `PRIOR_FREEZE` to the [synthetic English evaluation](../README.md) freeze. Choose a new `OUTPUT` directory whose parent exists.

~~~sh
python -B "$REPO/training/gpu/eval/memeqa/run_memeqa.py" \
  --data "$INPUT" --output "$OUTPUT" --project "$REPO" \
  --model "$MODEL" --adapter-run "$ADAPTER_RUN" --prior-freeze "$PRIOR_FREEZE"
~~~

The runner verifies the model and adapter against the prior freeze. Use the response ledgers and associated scoring files from the team handoff to replay the recorded 13/32 and 17/32 results.

~~~sh
python -B "$REPO/training/gpu/eval/memeqa/score_saved.py" "$EVIDENCE"
~~~

`EVIDENCE` is the MemeQA evidence directory in the team handoff. To replay a new run, copy `gold.jsonl` and `manifest.json` from `INPUT` into `OUTPUT`, name the copied manifest `selection-manifest.json`, and copy the exact `run_memeqa.py` into `OUTPUT`. Then use `OUTPUT` as `EVIDENCE`.

[MemeQA paper](https://aclanthology.org/2025.acl-long.927/) · [Author repository](https://github.com/npnkhoi/memeqa) · [SemEval research terms](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus#licensing)
