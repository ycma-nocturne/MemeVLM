"""Plot the six recorded 4B training runs from the bundled numeric evidence."""

import argparse
import csv
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGURES = Path(__file__).resolve().parent
INPUT = FIGURES.parent.parent / "evidence/training-loss/curves.csv"
SEEDS = (42, 43, 44)
STEPS = tuple(range(1, 292))
CHECKPOINTS = (97, 194, 291)
WINDOW = 10
PROFILES = {
    "bilingual": ("Mixed corpus", "#2C6197", "-", "o"),
    "chinese": ("Replay", "#A25A26", "--", "s"),
}


def read_loss():
    runs = {(profile, seed): {} for profile in PROFILES for seed in SEEDS}
    with INPUT.open(newline="") as stream:
        for row in csv.DictReader(stream):
            run = runs[row["profile"], int(row["seed"])]
            step = int(row["step"])
            if step in run:
                raise ValueError(f"Duplicate optimizer step {step}")
            validation = row["validation_loss"]
            run[step] = (float(row["training_loss"]), float(validation) if validation else None)

    for arm, run in runs.items():
        if tuple(sorted(run)) != STEPS:
            raise ValueError(f"Incomplete training-loss steps in {arm}")
        evaluated = tuple(step for step in STEPS if run[step][1] is not None)
        if evaluated != CHECKPOINTS:
            raise ValueError(f"Unexpected validation steps in {arm}")
    return runs


def trailing_mean(values):
    return [mean(values[i - WINDOW : i]) for i in range(WINDOW, len(values) + 1)]


def draw(destination):
    runs = read_loss()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "text.color": "#243749",
        "axes.labelcolor": "#243749",
        "axes.edgecolor": "#8192A5",
        "axes.linewidth": 0.65,
        "xtick.color": "#526677",
        "ytick.color": "#526677",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "memevlm-training-diagnostics",
    })
    figure, (training, validation) = plt.subplots(1, 2, figsize=(175 / 25.4, 67 / 25.4))
    figure.subplots_adjust(left=0.096, right=0.974, bottom=0.235, top=0.745, wspace=0.41)
    for axis in (training, validation):
        axis.set_xlim(0, 306)
        axis.set_xticks(CHECKPOINTS)
        axis.set_xlabel("Optimizer step", labelpad=3)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E3EBF1", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(which="major", length=2.5, pad=2.5)
        axis.axvline(97, color="#6B7683", linewidth=0.85, linestyle=(0, (2, 2)), zorder=1)

    training.set_title("(a) Training", loc="left", fontweight="bold", pad=6)
    training.set_ylabel("Training loss (log scale)", labelpad=4)
    training.set_yscale("log")
    training.set_ylim(0.0001, 10)
    training.set_yticks((0.0001, 0.001, 0.01, 0.1, 1, 10))
    training.set_xticks((1, *CHECKPOINTS))
    validation.set_title("(b) Shared validation", loc="left", fontweight="bold", pad=6)
    validation.set_ylabel("Validation loss", labelpad=4)
    validation.set_ylim(0.5, 3.5)
    validation.set_yticks((0.5, 1, 1.5, 2, 2.5, 3))

    legend = []
    for profile, (label, color, linestyle, marker) in PROFILES.items():
        profile_runs = [runs[profile, seed] for seed in SEEDS]
        traces = [trailing_mean([run[step][0] for step in STEPS]) for run in profile_runs]
        plot_steps = STEPS[WINDOW - 1 :]
        for trace in traces:
            training.plot(plot_steps, trace, linestyle, color=color, alpha=0.27, linewidth=0.65)
        training.plot(plot_steps, [mean(row) for row in zip(*traces)], linestyle,
                      color=color, linewidth=1.8, zorder=3)

        values = [[run[step][1] for run in profile_runs] for step in CHECKPOINTS]
        for offset, losses in zip((-6, 0, 6), zip(*values)):
            validation.scatter([step + offset for step in CHECKPOINTS], losses,
                               s=13, marker=marker, color=color, alpha=0.32,
                               edgecolors="none", zorder=2)
        validation.errorbar(CHECKPOINTS, [mean(row) for row in values],
                            yerr=[stdev(row) for row in values], linestyle=linestyle,
                            color=color, marker=marker, markersize=4.5,
                            markerfacecolor="white", markeredgewidth=1.15,
                            linewidth=1.5, elinewidth=0.85, capsize=2.5, zorder=3)
        legend.append(Line2D([0], [0], label=label, color=color, linestyle=linestyle,
                             linewidth=1.5, marker=marker, markersize=4,
                             markerfacecolor="white"))
    legend.append(Line2D([0], [0], label="Selected checkpoint", color="#6B7683",
                         linewidth=0.85, linestyle=(0, (2, 2))))
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.53, 0.97),
                  frameon=False, ncol=3, columnspacing=1.7)

    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(destination / f"training-diagnostics.{extension}", dpi=300,
                       facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES)
    draw(parser.parse_args().out)
