# Rebuild the research figures

From this directory, run `sh build.sh` with pdfLaTeX, PGF/TikZ, Latin Modern, and Poppler to render the architecture and matched-training diagrams. Run the plot scripts with Python 3 and Matplotlib.

```sh
python3 plot_training_diagnostics.py
python3 plot_memeqa_transitions.py
python3 plot_reference_rq3.py --summary ../../evidence/reference-study/rq3.json --out .
python3 plot_reference_coverage.py --analysis ../../evidence/reference-study/coverage-test.json --split test --out .

replay=$(mktemp -d)
python3 plot_reference_coverage.py --analysis ../../evidence/reference-study/coverage-span-replay.json --split test --out "$replay"
for format in pdf png svg; do
  mv "$replay/reference-coverage-test.$format" "reference-coverage-replay.$format"
done
rmdir "$replay"
```

The training plot uses trailing ten-step means for 1,746 training losses, plus means and sample standard deviations at the three validation checkpoints. The MemeQA plot compares corrected and regressed answers across the 816-question primary cohort. The reference-coverage plots use family-paired differences and intervals.

[Training curves](../../evidence/training-loss/curves.csv) · [MemeQA transitions](memeqa-transitions.json) · [MemeBench aggregates](../../evidence/reference-study/)
