# MemeVLM report, proposal and talk

The [MemeQA follow-up](../../training/gpu/followup/README.md) contains the subsequent real-meme training and confirmation results.

| Artifact | Editable source |
| --- | --- |
| [Final report](Final_Report.pdf) | [LaTeX](source/report/) |
| [Proposal](Proposal.pdf) | [LaTeX](source/proposal/) |
| [Presentation](Final_Presentation.pptx) | [PowerPoint generator](source/presentation/) and [speaker guide](Speaker_Guide.md) |

The figures use aggregate results and illustrative artwork. The public teaching examples use project renderings and schematics. Benchmark names and papers identify the evaluated datasets. [Aggregate results](evidence/) · [Figure sources](source/figures/)

## Build the PDFs

Use TeX Live or MacTeX. Run from this directory.

```sh
(cd source/report && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex)
(cd source/proposal && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build proposal.tex)
```

Use the figure sources to change the diagrams. The report tables use [the training and evaluation aggregate](evidence/full-results.json).

## Build the presentation

Use Node.js 24, `@oai/artifact-tool` 2.8.188 and `jszip` 3.10.1. Set `NODE_MODULES_DIR` to the directory containing those packages. Run from this directory with a fresh output folder.

```sh
ln -s "$NODE_MODULES_DIR" node_modules
work=$(mktemp -d)
node source/presentation/build.mjs evidence/full-results.json evidence/pilot \
  "$work/Final_Presentation.pptx" "$work/previews"
```

The generated PowerPoint contains editable text, tables, chart data and notes. The generator also writes the speaker guide.

[Experiment commands](../../README.md#reproduce) use authorized evaluation inputs for exact reruns.
