# Build the paper

Run with Python 3 and TeX Live or MacTeX from this directory.

```sh
python3 generate_report_tables.py ../../evidence/full-results.json
python3 write_coverage_results.py --aggregate ../../evidence/reference-study/coverage-test.json > coverage-results.tex
python3 write_coverage_results.py --replay --aggregate ../../evidence/reference-study/coverage-span-replay.json > coverage-replay-results.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
```

The compiled PDF is `build/report.pdf`. Keep [figures](../figures/) and [illustrations](../assets/) at their existing relative paths. The [aggregate study protocol](protocol.json) specifies the training budget, cohorts, and bootstrap plan.
