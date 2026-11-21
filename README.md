# MemeVLM

MemeVLM evaluates how vision-language models use visual evidence and background knowledge to interpret memes. The completed Apple Silicon study compares Qwen3-VL-2B-Instruct with and without a LoRA adapter on Chinese memes. It varies the input between image, image plus a factual hint, and OCR text, and checks model responses to controlled image edits.

The repository contains the implementation and experiment settings. Use the private course handoff for the exact images, split manifests, adapters, run records, and submission documents.

## Set up

Use Python 3.12. Check out this repository and apply its code to a fresh copy of the private handoff. Run these commands from the checkout, with `BUNDLE` set to the path of `MemeVLM-gpu-private.zip`.

```bash
work=$(mktemp -d)
unzip -q "$BUNDLE" -d "$work"
(cd "$work/MemeVLM" && shasum -a 256 -c SHA256SUMS.txt) &&
  git archive HEAD | tar -x -C "$work/MemeVLM"
cd "$work/MemeVLM"

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-analysis.txt
PYTHONPATH=code python -m meme_study.reference_cli --help
```

## Reproduce

| Workflow | Guide |
| --- | --- |
| Completed Chinese 2B LoRA training and held-out evaluation | [Apple Silicon study](local-study/reproduce.md) |
| Chinese and bilingual 4B LoRA training and English held-out evaluation | [CUDA training](training/gpu/README.md) |
| Original fixed-weight 4B/8B evaluation design | [Reference study](docs/reproduce.md) |

The CUDA workflows are follow-up experiments. See the [study protocol](docs/protocol.md) for the evaluation design and the [English data guide](training/bilingual/README.md) for the bilingual split.
