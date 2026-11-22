# MemeVLM

MemeVLM tests whether a vision-language model updates its interpretation when an edit changes a meme's punchline. We compare one frozen Qwen3-VL-4B-Instruct base with two LoRA profiles trained across seeds 42, 43, and 44 on NVIDIA H800 GPUs.

## Results

| Evaluation | 4B base | Chinese LoRA | Bilingual LoRA |
| --- | ---: | ---: | ---: |
| Synthetic English, all three variants correct (32 families) | 37.5% | 57.3% ± 3.6 pp | 92.7% ± 1.8 pp |
| Natural English, exact MemeQA choice (816 questions, 134 images) | 53.55% | 53.92% ± 0.25 pp | 54.82% ± 0.51 pp |

LoRA results show mean ± sample standard deviation across three seeds. Each run uses 291 optimizer updates and the same 193 validation rows. Bilingual training uses 770 distinct rows. The Chinese control repeats 98 unique rows to match the update budget.

The synthetic score requires all three variants of a family to be correct. The expanded MemeQA evaluation covers 1,027 eligible questions on 166 images. Its primary comparison excludes all images from the earlier 32-image pilot. The bilingual gain over the base is +1.27 pp on images outside the pilot, with a paired 95% interval of -1.14 to +3.58 pp. The interval includes zero.

## Set up

Use Python 3.12. Set up CPU scoring and analysis from the project root.

~~~sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/requirements-analysis.txt
~~~

The private team handoffs supply the exact training data, test inputs, adapters, and response ledgers.

## Reproduce

- [Six-run training and expanded evaluation](training/gpu/expansion/README.md)
- [Pinned CUDA environment](training/gpu/README.md) and [synthetic English scoring](training/gpu/eval/README.md)
- [Earlier MemeQA pilot](training/gpu/eval/memeqa/README.md) and [Chinese 2B study](local-study/reproduce.md)
