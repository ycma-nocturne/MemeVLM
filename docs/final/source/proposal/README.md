# Build the proposal

Run with TeX Live or MacTeX from this directory.

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build proposal.tex
```

The compiled PDF is `build/proposal.pdf`.
