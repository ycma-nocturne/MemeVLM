# MemeVLM

MemeVLM studies how vision-language models use visual evidence and background knowledge to interpret Chinese memes. It compares image, image with a background hint, and OCR-only inputs, then uses controlled image edits to test visual sensitivity.

## Quick start

Use Python 3.12 on macOS or Linux. The test suite uses synthetic fixtures and runs on CPU.

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests
python -m meme_study.reference_cli --help
```

- [Study protocol](docs/protocol.md) describes the experimental conditions and scoring.
- [Reproduction guide](docs/reproduce.md) covers team inputs, input verification, GPU runs, and result export.

Use the team’s private handoff for the exact study inputs. Keep run records, grading references, and reviewer forms in team storage. In pull requests, describe the change and list the checks you ran.
