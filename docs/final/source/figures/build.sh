#!/bin/sh
set -eu
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"
cd "$(dirname "$0")"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/memevlm-tikz.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
for name in qwen3vl-lora matched-training evaluation-protocol; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" "$name.tex" > "$build_dir/compile.log"; then
        cat "$build_dir/compile.log" >&2
        exit 1
    fi
    pdfinfo "$build_dir/$name.pdf" | awk '$1 == "Pages:" && $2 == 1 {one_page=1} END {exit !one_page}'
    cp "$build_dir/$name.pdf" "$name.pdf"
    pdftoppm -r 300 -singlefile -png "$name.pdf" "$name"
    pdftocairo -svg "$name.pdf" "$name.svg"
done
