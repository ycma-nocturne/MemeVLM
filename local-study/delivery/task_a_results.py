"""Render report tables directly from scored held-out Task A evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path


MODELS = ('local-qwen3vl-2b-base', 'local-qwen3vl-2b-lora')
CONDITIONS = (('image_only', 'Image only'), ('hinted', 'Image with hint'), ('ocr_only', 'OCR only'))


def read(folder, name):
    return json.loads((folder / name).read_bytes())


def cell(entry):
    value, denominator = entry['estimate'], entry['denominator']
    return f'{100 * value:+.1f} ({denominator})' if value is not None else '-- (0)'


def table(rows):
    return '\n'.join(' & '.join(row) + r' \\' for row in rows)


def render(strict, normalized):
    raw = read(strict, 'summary.json')
    clean = read(normalized, 'summary.json')
    analysis = read(normalized, 'analysis.json')
    if (raw['ledger_sha256'] != clean['ledger_sha256'] or raw['split'] != 'test'
            or clean['split'] != 'test' or not raw['terminal_coverage'] or not clean['terminal_coverage']
            or raw['planned_explanations'] != clean['planned_explanations'] or raw['planned_explanations'] != 240):
        raise ValueError('Task A sources must cover the same complete held-out matrix')
    rows = [json.loads(line) for line in (strict / 'scored.jsonl').read_text().splitlines()]
    candidates = Counter(row['model'] for row in rows if row['run_status'] == 'ok')
    truncations = Counter(row['model'] for row in rows if row['candidate_truncated'])
    scores = Counter(row['model'] for row in rows if (row['score'] or {}).get('status') == 'ok')
    row2 = [json.loads(line) for line in (normalized / 'scored.jsonl').read_text().splitlines()]
    available = Counter(row['model'] for row in row2 if (row['score'] or {}).get('status') == 'ok')
    explanation = analysis['splits']['test']['explanation']
    comparison = explanation['model_differences'][f'{MODELS[1]}_minus_{MODELS[0]}']['conditions']
    cross = [[label, *(cell(comparison[condition][dim]) for dim in 'VIKR')]
             for condition, label in CONDITIONS]
    within = []
    for model, label in zip(MODELS, ('Base', 'LoRA')):
        entry = explanation['models'][model]
        hint = entry['condition_differences']['hinted_minus_image_only']
        ocr = entry['ocr_subgroups']['available']['condition_differences']['image_only_minus_ocr_only']
        within.extend([[f'{label}, image $-$ OCR', cell(ocr['V']), cell(ocr['R'])],
                       [f'{label}, hint $-$ image', cell(hint['K']), cell(hint['R'])]])
    base, lora = MODELS
    return rf"""\subsection{{Task A. Explanation coverage}}

The base and LoRA returned {candidates[base]}/120 and {candidates[lora]}/120 planned explanations. The response limit was reached in {truncations[base]} and {truncations[lora]} cases, respectively. The independent judge produced {clean['judge_status'].get('ok', 0)}/240 judgments. Strict scoring accepted {scores[base]}/120 base and {scores[lora]}/120 LoRA judgments. Removing a sole outer JSON fence affected {clean['sole_json_fences_removed']} judgments and raised availability to {available[base]}/120 and {available[lora]}/120. Both policies require the same item IDs, decisions and exact evidence. Missing or invalid judgments remain unavailable.

Table~\ref{{tab:held-out-a}} compares the base and LoRA on shared, scorable families. Table~\ref{{tab:conditions-a}} compares the input conditions within each arm. The image--OCR contrast uses families with available OCR. Each cell gives the mean difference in percentage points and its matched family count. The analysis includes family-level bootstrap intervals, source-item diagnostics and coverage for each arm. The estimates describe available judge-scored pairs. Independent human ratings remain pending for the 20 exported held-out explanations.

\begin{{table}}[H]
\centering
\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Condition & V & I & K & R \\
\midrule
{table(cross)}
\bottomrule
\end{{tabular}}
\caption{{Task A. LoRA minus base after outer-fence normalization, in percentage points. Parentheses give the number of matched scorable families out of 40.}}
\label{{tab:held-out-a}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
Contrast & Visual or K & Reasoning \\
\midrule
{table(within)}
\bottomrule
\end{{tabular}}
\caption{{Task A. Within-model condition differences. Image vs. OCR measures visual and reasoning coverage. Hint vs. image measures knowledge (K) and reasoning coverage.}}
\label{{tab:conditions-a}}
\end{{table}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', required=True, type=Path)
    parser.add_argument('--normalized', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.strict, args.normalized), encoding='utf-8')


if __name__ == '__main__':
    main()
