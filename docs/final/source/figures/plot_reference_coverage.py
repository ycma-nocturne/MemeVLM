#!/usr/bin/env python3
"""Plot paired Chinese meme reference-coverage contrasts from reference_cli analyze JSON."""
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
LABELS = {
    MODELS[0]: "4B",
    MODELS[1]: "8B",
    "local-qwen3vl-2b-base": "2B base",
    "local-qwen3vl-2b-lora": "2B LoRA",
}
STYLES = (("#003366", "o", 0.15), ("#986412", "D", -0.15))
INK = "#213444"
GREY = "#4D5F6F"
GRID = "#DDE5EB"

PANELS = (
    ("A", "Image − OCR only", ("V", "I", "K", "R"), ("V", "I", "K", "R")),
    ("B", "Factual hint added", ("K", "R", "R_minus_K"),
     (r"$\Delta K$", r"$\Delta R$", r"$\Delta R-\Delta K$")),
)


def metric_values(metric, extent):
    count, expected = metric["denominator"], metric["n_expected"]
    if any(type(x) is not int for x in (count, expected)) or not 0 <= count <= expected:
        raise ValueError("invalid count of matched families")
    if metric["n_unavailable"] != expected - count or metric["n_clusters"] != count:
        raise ValueError("family count differs from the reported coverage or bootstrap count")
    delta, interval = metric["estimate"], metric["ci95"]
    if count == 0:
        if delta is not None or interval is not None:
            raise ValueError("an unmatched subset cannot have an estimate or interval")
    elif type(delta) not in (int, float) or not math.isfinite(delta) or abs(delta) > extent + 1e-12:
        raise ValueError("paired difference lies beyond its possible range")
    if interval is not None:
        if (count < 2 or not isinstance(interval, list) or len(interval) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
                or not -extent - 1e-12 <= interval[0] <= interval[1] <= extent + 1e-12):
            raise ValueError("invalid bootstrap interval for the paired family count")
    return count, expected, delta, interval


def panel_values(block, models, split):
    result = []
    for i, (_, _, keys, row_labels) in enumerate(PANELS):
        rows = []
        for key, label in zip(keys, row_labels, strict=True):
            values = []
            for model in models:
                data = block[model]
                if i == 0:
                    source = data["ocr_subgroups"]["available"] if split == "test" else data
                    section = source["condition_differences"]["image_only_minus_ocr_only"]
                else:
                    section = data["hint_gains"]
                values.append(metric_values(section[key], extent=2 if key == "R_minus_K" else 1))
            rows.append((label, values))
        result.append(rows)
    return result


def common_ticks(panels):
    values = [abs(value) for panel in panels for _, row in panel
              for _, _, delta, ci in row if delta is not None
              for value in (delta, *(ci or []))]
    maximum = max(values, default=0) * 100
    step = next(s for s in (10, 12.5, 20, 25, 40, 50, 75, 100) if maximum <= 2 * s + 1e-9)
    return 2 * step, (-2 * step, -step, 0, step, 2 * step)


def draw(analysis, split, models, destination):
    if analysis["bootstrap_unit"] != "family_id":
        raise ValueError("only family-level bootstrap analysis is supported")
    panel_data = panel_values(analysis["splits"][split]["explanation"]["models"], models, split)
    scale, ticks = common_ticks(panel_data)
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.3, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "memevlm-reference-coverage",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(ncols=2, figsize=(6.85, 2.23))
        fig.subplots_adjust(left=0.023, right=0.99, bottom=0.073, top=0.963, wspace=0.08)
        for ax, (panel, title, _, _), rows in zip(axes, PANELS, panel_data, strict=True):
            ax.set_xlim(-1.65, 2.06)
            ax.set_ylim(-0.96, 4.94)
            ax.set_axis_off()
            y_bottom, y_top = -0.24, 3.37
            ax.text(-1.62, 4.59, panel, va="center", color=STYLES[0][0], weight="bold")
            ax.text(-1.40, 4.59, title, va="center", color=INK, weight="bold")
            for tick in ticks:
                x = tick / scale
                ax.vlines(x, y_bottom, y_top, color=GREY if tick == 0 else GRID,
                          lw=0.74 if tick == 0 else 0.55, zorder=0)
                display = f"{tick:g}".replace("-", "−")
                ax.text(x, y_bottom - 0.18, display, color=GREY, va="top", ha="center", fontsize=7.4)
            ax.hlines(y_bottom, -1, 1, color=GREY, lw=0.5)
            for model, (color, marker, _), x in zip(models, STYLES, (1.23, 1.84), strict=True):
                ax.scatter(x, 4.05, color=color, s=19, marker=marker, linewidths=0)
                label = LABELS.get(model, model)
                ax.text(x, 3.75, label, color=color, weight="bold",
                        ha="center", va="center", fontsize=7.6 if len(label) <= 4 else 6.8)
            ax.text(1.535, 3.44, "Matched n/N", color=GREY, ha="center", va="center", fontsize=5.9)
            for y, (label, values) in zip(range(3, 3 - len(rows), -1), rows, strict=True):
                ax.text(-1.60, y, label, color=INK, va="center")
                for (count, expected, difference, ci), (color, marker, offset), x in zip(
                    values, STYLES, (1.23, 1.84), strict=True
                ):
                    y_point = y + offset
                    ax.text(x, y, f"{count}/{expected}" if count else f"NA\n0/{expected}",
                            color=color if count else GREY, ha="center", va="center",
                            fontsize=7.8 if count else 6.8, linespacing=1.03)
                    if difference is None:
                        continue
                    if ci is not None:
                        low, high = (100 * bound / scale for bound in ci)
                        ax.hlines(y_point, low, high, color=color, lw=1.14)
                        ax.vlines((low, high), y_point - 0.075, y_point + 0.075, color=color, lw=0.75)
                    ax.scatter(100 * difference / scale, y_point, color=color, s=22,
                               marker=marker, linewidths=0, zorder=3)
        fig.text(0.5, 0.046, "Coverage difference (percentage points)", va="center", ha="center",
                 fontsize=7.7, color=INK)

        destination.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(destination.with_suffix(f".{extension}"), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--models", nargs=2, default=MODELS, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    try:
        if len(set(args.models)) != 2:
            raise ValueError("choose two distinct models")
        draw(json.loads(args.analysis.read_bytes()), args.split, args.models,
             args.out / f"reference-coverage-{args.split}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
