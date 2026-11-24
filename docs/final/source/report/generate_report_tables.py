"""Validate the public aggregate and regenerate its LaTeX tables."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = (42, 43, 44)
PROFILES = ("bilingual", "chinese")
ARMS = ("base", *(f"{profile}-{seed}" for profile in PROFILES for seed in SEEDS))
CONTRASTS = (
    "bilingual_minus_base", "chinese_minus_base", "bilingual_minus_chinese",
)
SOURCE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def equal(actual, expected):
    return type(actual) in (int, float) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0)


def validate_cohort(cohort, images, questions, label):
    require((cohort["images"], cohort["questions"]) == (images, questions), f"{label}: cohort size")
    require(set(cohort["per_arm"]) == set(ARMS), f"{label}: missing or extra arms")
    rates = {}
    for arm, row in cohort["per_arm"].items():
        correct = row["correct"]
        require(count(correct, questions) and row["denominator"] == questions, f"{label}: {arm} counts")
        rates[arm] = correct / questions
        require(equal(row["accuracy"], rates[arm]), f"{label}: {arm} accuracy")

    averages = {}
    require(set(cohort["by_profile"]) == set(PROFILES), f"{label}: profile names")
    for profile in PROFILES:
        values = [rates[f"{profile}-{seed}"] for seed in SEEDS]
        stat = cohort["by_profile"][profile]
        averages[profile] = mean(values)
        require(equal(stat["mean"], averages[profile]) and equal(stat["seed_sample_std"], stdev(values)),
                f"{label}: {profile} seed summary")
    expected = (
        averages["bilingual"] - rates["base"], averages["chinese"] - rates["base"],
        averages["bilingual"] - averages["chinese"],
    )
    require(set(cohort["paired_contrasts"]) == set(CONTRASTS), f"{label}: contrast names")
    for name, delta in zip(CONTRASTS, expected):
        row = cohort["paired_contrasts"][name]
        lo, hi = row["paired_image_bootstrap_ci95"]
        require(equal(row["delta"], delta), f"{label}: {name} contrast")
        require(all(type(value) in (float, int) for value in (lo, hi)) and -1 <= lo <= hi <= 1,
                f"{label}: {name} interval")


def read_aggregate(results_path, protocol_path):
    protocol_bytes = protocol_path.read_bytes()
    plan = json.loads(protocol_bytes)
    aggregate = json.loads(results_path.read_bytes())
    require(hashlib.sha256(protocol_bytes).hexdigest() == aggregate["protocol_sha256"],
            "aggregate does not bind to this protocol")
    require([arm["name"] for arm in plan["arms"]] == list(ARMS), "unexpected protocol arms")
    require(plan["training"]["seeds"] == list(SEEDS) and plan["bootstrap"]["draws"] == 10_000,
            "unexpected protocol seeds or bootstrap draws")
    synthetic = aggregate["synthetic_strict_triples"]
    primary = aggregate["native_memeqa"]["unseen_pilot_images"]
    sensitivity = aggregate["native_memeqa"]["all_eligible"]
    validate_cohort(synthetic, 32, 32, "synthetic")
    validate_cohort(primary, 134, 816, "primary MemeQA")
    validate_cohort(sensitivity, 166, 1027, "all-eligible MemeQA")
    for cohort in (primary, sensitivity):
        valid = cohort["per_arm_valid"]
        require(set(valid) == set(ARMS), "MemeQA format counts do not cover every arm")
        for arm in ARMS:
            require(count(valid[arm], cohort["questions"]) and valid[arm] >= cohort["per_arm"][arm]["correct"],
                    f"invalid MemeQA format count for {arm}")
    for arm in ARMS:
        pilot_correct = sensitivity["per_arm"][arm]["correct"] - primary["per_arm"][arm]["correct"]
        pilot_valid = sensitivity["per_arm_valid"][arm] - primary["per_arm_valid"][arm]
        require(0 <= pilot_correct <= pilot_valid <= 211, f"inconsistent nested MemeQA cohorts for {arm}")
    detail = synthetic["per_arm_details"]
    require(set(detail) == set(ARMS), "synthetic diagnostics do not cover every arm")
    for arm, row in detail.items():
        require(all(count(row[key], 96) for key in ("valid", "cue_correct", "joint_correct"))
                and count(row["target_edit_correct"], 32), f"synthetic diagnostics for {arm}")
        require(3 * synthetic["per_arm"][arm]["correct"] <= row["joint_correct"] <= row["cue_correct"] <= row["valid"]
                and synthetic["per_arm"][arm]["correct"] <= row["target_edit_correct"] <= row["joint_correct"],
                f"inconsistent synthetic diagnostics for {arm}")
    return synthetic, primary, sensitivity


def percent(value, digits=2, signed=False):
    rounded = round(100 * value, digits) + 0.0
    label = f"{rounded:+.{digits}f}" if signed else f"{rounded:.{digits}f}"
    return label.replace("-", "{-}")


def interval(contrast, digits=2):
    lo, hi = contrast["paired_image_bootstrap_ci95"]
    return f"[{percent(lo, digits)}, {percent(hi, digits)}]"


def table_row(label, cells):
    return " & ".join((label, *cells)) + r"\\"


def score_table(cohorts):
    precision = (1, 2, 2)

    def arm_cells(arm):
        return [percent(c["per_arm"][arm]["accuracy"], digits) for c, digits in zip(cohorts, precision)]

    rows = [table_row("Frozen base", arm_cells("base"))]
    for profile, label in (("bilingual", "Mixed corpus"), ("chinese", "Replay")):
        rows.append(r"\midrule")
        for seed in SEEDS:
            rows.append(table_row(f"{label}, seed {seed}", arm_cells(f"{profile}-{seed}")))
        cells = []
        for cohort, digits in zip(cohorts, precision):
            stat = cohort["by_profile"][profile]
            average = percent(stat["mean"], digits)
            sd = percent(stat["seed_sample_std"], digits)
            cells.append(f"${average} \\pm {sd}$")
        rows.append(table_row(f"{label}, mean $\\pm$ SD", cells))
    return r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Arm & Synthetic strict triples & MemeQA primary & MemeQA all-eligible\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Accuracy (\%) by arm. Synthetic scoring uses 32 families, with all three variants required for a pass. MemeQA primary uses 816 questions on 134 images outside the pilot. All-eligible sensitivity uses 1{,}027 questions on 166 images. SD is the sample standard deviation across three seeds.}
\label{tab:full-scores}
\end{table*}
"""


def contrast_table(cohorts):
    rows = []
    for name, cohort in zip(("Synthetic strict triples", "MemeQA primary", "MemeQA all-eligible"), cohorts):
        contrasts = cohort["paired_contrasts"]
        rows.append(table_row(name, [f"${percent(contrasts[key]['delta'], signed=True)}\\;"
                                     f"{interval(contrasts[key])}$" for key in CONTRASTS]))
    return r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Cohort & Mixed corpus $-$ base & Replay $-$ base & Mixed corpus $-$ replay\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Paired differences in percentage points with 95\% percentile bootstrap intervals. Each draw resamples synthetic families or natural images, together with all their questions and arms. Intervals condition on the fitted adapters.}
\label{tab:full-contrasts}
\end{table*}
"""


def displayed_count(values, denominator):
    lo, hi = min(values), max(values)
    return f"{lo}/{denominator}" if lo == hi else f"{lo}--{hi} of {denominator}"


def summary_macros(synthetic, primary):
    natural = primary["paired_contrasts"]["bilingual_minus_base"]
    synth_contrast = synthetic["paired_contrasts"]["bilingual_minus_chinese"]
    details = synthetic["per_arm_details"]
    values = {
        "PrimaryBase": percent(primary["per_arm"]["base"]["accuracy"]),
        "PrimaryBilingual": percent(primary["by_profile"]["bilingual"]["mean"]),
        "PrimaryChinese": percent(primary["by_profile"]["chinese"]["mean"]),
        "PrimaryContrast": "$" + percent(natural["delta"], signed=True) + "$",
        "PrimaryInterval": "$" + interval(natural) + "$",
        "PrimaryValidSummary": f"{displayed_count(primary['per_arm_valid'].values(), 816)} valid answers",
        "SyntheticBase": percent(synthetic["per_arm"]["base"]["accuracy"], 1),
        "SyntheticBilingual": percent(synthetic["by_profile"]["bilingual"]["mean"], 1),
        "SyntheticChinese": percent(synthetic["by_profile"]["chinese"]["mean"], 1),
        "SyntheticContrast": "$" + percent(synth_contrast["delta"], signed=True) + "$",
        "SyntheticInterval": "$" + interval(synth_contrast) + "$",
        "SyntheticDiagnosticSummary": (
            f"each arm had {displayed_count([row['valid'] for row in details.values()], 96)} valid answers and "
            f"{displayed_count([row['cue_correct'] for row in details.values()], 96)} correct cues"
        ),
        "TargetBase": displayed_count([details["base"]["target_edit_correct"]], 32),
    }
    for profile, macro in (("bilingual", "TargetBilingual"), ("chinese", "TargetChinese")):
        values[macro] = displayed_count([details[f"{profile}-{seed}"]["target_edit_correct"] for seed in SEEDS], 32)
    return "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Verified full-results.json from aggregate.py")
    parser.add_argument("--protocol", type=Path, default=SOURCE / "protocol.json")
    parser.add_argument("--output", type=Path, default=SOURCE, help="Directory for generated LaTeX inputs")
    args = parser.parse_args()
    try:
        cohorts = read_aggregate(args.results, args.protocol)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    synthetic, primary, _ = cohorts
    files = {
        "expanded-summary.tex": summary_macros(synthetic, primary),
        "expanded-tables.tex": score_table(cohorts) + "\n" + contrast_table(cohorts),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, snippet in files.items():
        path = args.output / name
        path.write_text("% Generated from the frozen full-study aggregate.\n" + snippet, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
