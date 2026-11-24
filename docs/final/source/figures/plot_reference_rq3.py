#!/usr/bin/env python3
"""Render paired intervention results from summarize_reference_rq3 JSON."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_LABELS = dict(zip(MODELS, ("4B", "8B"), strict=True))
MARKS = (("#003366", "o"), ("#986412", "D"))
INK = "#213444"
GREY = "#4D5F6F"
GRID = "#DEE5EB"
PCT = (0, 25, 50, 75, 100)


def rows_for(model, split):
    triples = model["triples"]
    exact = [
        (label, model["variants"][variant]["joint_correct"])
        for label, variant in (("Original", "original"), ("Target edit", "edited"), ("Sham", "sham"))
    ]
    exact.append(("Strict family", triples["strict_pass"]))
    consistency = [("Original vs. sham", triples["sham_preservation"])]
    permutation = model["permutation"]
    if split == "test" or permutation is not None:
        if permutation is None:
            raise ValueError("test summary lacks planned option-order repeats")
        consistency.append(("Option-order repeat", permutation["semantic_relation_agreement"]))
    return exact + consistency


def validate(metric):
    success, n, planned = (metric[key] for key in ("numerator", "denominator", "n_expected"))
    if any(type(value) is not int for value in (success, n, planned)) or not 0 <= success <= n <= planned:
        raise ValueError("metric requires integer counts 0 ≤ successful ≤ scored ≤ planned")
    if metric["n_unavailable"] != planned - n:
        raise ValueError("unavailable count disagrees with the metric denominator")
    estimate = metric["estimate"]
    if n == 0:
        if estimate is not None:
            raise ValueError("unavailable metric has an estimate")
    elif not isinstance(estimate, (int, float)) or not math.isclose(
        estimate, success / n, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("metric estimate disagrees with the exact counts")


def count_label(metric):
    n, planned = metric["denominator"], metric["n_expected"]
    label = f"{metric['numerator']}/{n}" if n else "NA/0"
    return f"{label} ({planned})" if n != planned else label


def draw(summary, split, models, destination):
    block = summary["splits"][split]
    first, second = (rows_for(block[model], split) for model in models)
    cell_rows = []
    for (label, left), (other_label, right) in zip(first, second, strict=True):
        if label != other_label:
            raise ValueError("models have different planned comparisons")
        for metric in (left, right):
            validate(metric)
        if left["n_expected"] != right["n_expected"]:
            raise ValueError("models have different planned family counts")
        cell_rows.append((label, (left, right)))

    partial = any(m["n_unavailable"] for _, metrics in cell_rows for m in metrics)
    y_values = (7.35, 6.33, 5.31, 4.29, 2.54, 1.52)[:len(cell_rows)]
    bottom = y_values[-1] - 0.53
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.15, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "memevlm-reference-rq3",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(4.87, 2.83 if split == "test" else 2.55))
        fig.subplots_adjust(left=0.025, right=0.985, top=0.965, bottom=0.055)
        ax.set_xlim(-62, 187)
        ax.set_ylim(bottom - (1.33 if partial else 0.62), 9.25)
        ax.set_axis_off()

        for x in PCT:
            ax.vlines(x, 3.83, 7.79, color=GRID, lw=0.65)
            ax.vlines(x, bottom, 2.98, color=GRID, lw=0.65)
            ax.text(x, bottom - 0.31, f"{x}%", color=GREY, ha="center", va="center", fontsize=7.5)

        ax.hlines(3.64, -62, 186, color=GRID, linewidth=0.7)
        for letter, title, y in (
            ("A", "Exact correctness", 8.65), ("B", "Semantic consistency", 3.29)
        ):
            ax.text(-62, y, letter, color=MARKS[0][0], weight="bold", va="center")
            ax.text(-45, y, title, color=INK, weight="bold", va="center")

        count_columns = (124, 165)
        for model, (color, marker), x in zip(models, MARKS, count_columns, strict=True):
            ax.scatter(x - 12, 8.65, c=color, s=22, marker=marker, linewidths=0)
            ax.text(x + 3.5, 8.65, MODEL_LABELS.get(model, model), color=color,
                    weight="bold", ha="center", va="center", fontsize=8)
        for (label, metrics), y in zip(cell_rows, y_values, strict=True):
            ax.text(-62, y, label, color=INK, va="center", fontsize=8)
            for metric, (color, marker), offset, x in zip(
                metrics, MARKS, (0.17, -0.17), count_columns, strict=True
            ):
                if metric["estimate"] is not None:
                    ax.scatter(100 * metric["estimate"], y + offset, c=color, s=29,
                               marker=marker, linewidths=0, zorder=3)
                ax.text(x, y, count_label(metric), color=color, ha="center", va="center", fontsize=8)
        if partial:
            ax.text(-62, bottom - 1.01, "Incomplete coverage. Parentheses give the planned count.",
                    color=GREY, va="center", fontsize=7.1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(destination.with_suffix(f".{extension}"), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--models", nargs=2, default=MODELS, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    try:
        if len(set(args.models)) != 2:
            raise ValueError("choose two distinct models")
        summary = json.loads(args.summary.read_bytes())
        draw(summary, args.split, args.models, args.out / f"rq3-{args.split}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
