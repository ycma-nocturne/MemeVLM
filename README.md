# MemeVLM

MemeVLM evaluates whether a vision-language model updates its interpretation when an edit changes a meme's meaning. It compares mixed-corpus and replay LoRA recipes under the same update budget and tests transfer to MemeQA.

## MemeQA follow-up

On the fixed official test cohort of 816 questions across 134 images, replacing 256 authored training records with real-meme questions from the official training split raised mean accuracy from 55.15% to 72.47%. The paired gain was **17.32 percentage points** (95% image-bootstrap CI [14.66, 20.07]). The comparison fixes the training row count and evaluates adaptation within MemeQA.

The final recipe uses real-meme training data, peak LR `1e-4`, effective batch 4 and LoRA rank 8, alpha 16. On a separate confirmation cohort of 433 questions across 77 previously unused official development images, batch 4 at rank 8 improved accuracy from 78.29% to 80.68% (+2.39 pp, 95% CI [0.61, 4.26]). The batch-4 rank-16 combination reached 82.06% (+1.39 pp over batch 4 at rank 8, 95% CI [−0.23, 3.15]) with twice the adapter parameters.

[Follow-up protocol, ablations and reproduction](training/gpu/followup/README.md)

## Initial transfer and controlled edits

| Evaluation | Frozen 4B | Replay LoRA | Mixed-corpus LoRA |
| --- | ---: | ---: | ---: |
| Strict accuracy on 32 controlled edit families | 37.5% | 57.3% | 92.7% |
| Exact choice on 816 MemeQA questions across 134 images | 53.55% | 53.92% | 54.82% |

LoRA scores are means across three paired seeds. Each run completed 291 updates. All six selected step 97 on the shared validation set. Mixed-corpus LoRA improves strict accuracy over replay by 35.42 percentage points (95% paired interval +21.88 to +50.00). Its MemeQA advantage over replay is 0.90 points (−1.18 to +2.99). The advantage over the frozen base is 1.27 points (−1.14 to +3.58).

In a separate MemeBench study, frozen 4B and 8B each pass 5 of 10 edit triples. Two reviewers find higher reasoning coverage with full images than with OCR alone. Hint gains narrow when families flagged for direct leakage are removed. The relative gains in knowledge and reasoning remain uncertain.

## Initial report and presentation

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

- [MemeQA follow-up](training/gpu/followup/README.md)
- [Matched 4B training and evaluation](training/gpu/expansion/README.md)
- [Frozen MemeBench comparisons](docs/reproduce.md) and [judge replay and review analysis](repro/chinese/README.md)
- [CPU analysis setup](code/requirements-analysis.txt) and [GPU environment](training/gpu/README.md)

The report and presentation build from the included figures and aggregate results. Exact experiment reruns require the corresponding authorized data, checkpoints and response ledgers.
