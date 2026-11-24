# MemeVLM

MemeVLM evaluates whether a vision-language model updates its interpretation when an edit changes a meme's meaning. It compares mixed-corpus and replay LoRA recipes under the same update budget and tests transfer to MemeQA.

| Evaluation | Frozen 4B | Replay LoRA | Mixed-corpus LoRA |
| --- | ---: | ---: | ---: |
| Strict accuracy on 32 controlled edit families | 37.5% | 57.3% | 92.7% |
| Exact choice on 816 MemeQA questions across 134 images | 53.55% | 53.92% | 54.82% |

LoRA scores are means across three paired seeds. Each run completed 291 updates. All six selected step 97 on the shared validation set. Mixed-corpus LoRA improves strict accuracy over replay by 35.42 percentage points (95% paired interval +21.88 to +50.00). Its MemeQA advantage over replay is 0.90 points (−1.18 to +2.99). The advantage over the frozen base is 1.27 points (−1.14 to +3.58).

In a separate MemeBench study, frozen 4B and 8B each pass 5 of 10 edit triples. Two reviewers find higher reasoning coverage with full images than with OCR alone. Hint gains narrow when families flagged for direct leakage are removed. The relative gains in knowledge and reasoning remain uncertain.

## Report and presentation

- [Final report](docs/final/Final_Report.pdf) and [LaTeX](docs/final/source/report/)
- [Presentation](docs/final/Final_Presentation.pptx), [speaker guide](docs/final/Speaker_Guide.md) and [editable source](docs/final/source/presentation/)
- [Proposal](docs/final/Proposal.pdf) and [LaTeX](docs/final/source/proposal/)
- [Build instructions and aggregate results](docs/final/README.md)

## Set up

Use Python 3.12. Install the CPU dependencies from the repository root.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/requirements-analysis.txt
```

## Reproduce

- [Matched 4B training and evaluation](training/gpu/expansion/README.md)
- [Frozen MemeBench comparisons](docs/reproduce.md) and [judge replay and review analysis](repro/chinese/README.md)
- [CPU analysis setup](code/requirements-analysis.txt) and [GPU environment](training/gpu/README.md)

The report and presentation build from the included figures and aggregate results. Exact experiment reruns require the corresponding authorized data, checkpoints and response ledgers.
