# MemeVLM

MemeVLM tests whether a vision-language model updates its interpretation when an edit changes a meme's punchline. We fine-tuned Qwen3-VL-4B-Instruct with bilingual LoRA on 770 training examples and selected the checkpoint with the lowest loss on 193 validation examples. The BF16 run used one NVIDIA H800.

## Results

| Evaluation | 4B base | 4B + LoRA |
| --- | ---: | ---: |
| Synthetic English, cue and relation both correct | 63/96 | 92/96 |
| Synthetic English, target edit correct | 20/32 | 32/32 |
| Synthetic English, all three variants correct | 12/32 | 30/32 |
| Natural English, MemeQA choice correct | 13/32 | 17/32 |

An English family contains an original, a target edit, and a sham edit that preserves the intended relation. On this synthetic test, both arms had 96/96 valid JSON responses and 96/96 correct cues. The gain comes from relation selection within the shared synthetic design. MemeQA is an exploratory sample of 32 original images with native four-option questions.

## Set up

Use Python 3.12. Set up CPU scoring and analysis from the project root.

~~~sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/requirements-analysis.txt
~~~

The team handoff supplies the exact training data, test inputs, selected adapter, and response ledgers. For the recorded runs, extract the handoff and use its implementation directory as the project root.

## Reproduce

- [4B CUDA training](training/gpu/README.md) and [synthetic English evaluation](training/gpu/eval/README.md)
- [MemeQA inputs, inference, and score replay](training/gpu/eval/memeqa/README.md)
- [Earlier Chinese 2B study on Apple Silicon](local-study/reproduce.md) and [fixed-weight reference protocol](docs/protocol.md)
