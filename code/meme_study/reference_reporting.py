"""Auditable scientific tables/figures for the current reference-reuse protocol.

All statistics come from ``analyze_reference_scores``. This module checks score
bindings, exports its metrics and memberships, and draws them without refitting
or pooling checklist items. Test and development outputs are always separate.
The full export gate checks the current planned design and terminal candidates
and judges. Explicit partial exports disclose omissions; synthetic previews are
prominently marked and never represented as experimental evidence.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from .freeze import _publish
from .reference_analysis import analyze_reference_scores
from .reference_demo import _validate_bindings
from .reference_protocol import CONDITIONS, PROTOCOL_ID
from .reference_scoring import _check_checklist, score_checklist


_VARIANTS = ('original', 'edited', 'sham')
_TERMINAL = {'ok', 'failed', 'interrupted'}
_COLUMNS = ('metric_id', 'analysis_path', 'evidence_label', 'split', 'section', 'model', 'subgroup',
            'comparison', 'condition', 'variant', 'dimension', 'metric', 'estimate',
            'numerator', 'denominator', 'n_expected', 'n_unavailable', 'n_clusters',
            'ci95_low', 'ci95_high', 'estimate_status', 'ci95_status', 'membership_key')
_TITLES = {'explanation_conditions': 'A: reference coverage and score availability',
           'condition_differences': 'A: matched condition differences',
           'hint_gains': 'A: knowledge and reasoning hint gains',
           'item_repair_harm': 'A: item repair and harm (family-weighted)',
           'model_differences': 'A: matched model differences',
           'paired_variants': 'B: cue, relation and joint correctness',
           'paired_triples': 'B: complete triples and sham preservation',
           'paired_permutations': 'B: option-order semantic agreement'}
_CONDITION_LABELS = {'image_only': 'Image only', 'hinted': 'Image + hint', 'ocr_only': 'OCR only'}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def _explanation_bindings(rows):
    """Revalidate supplied item totals/quotes; original raw judge is not in rows."""
    for row in rows:
        if row['task_kind'] != 'explanation' or row.get('score') is None:
            continue
        score = row['score']
        if row.get('judge_status') != 'ok':
            raise ValueError('explanation score requires an explicitly successful judge')
        checklist = (row.get('reference') or {}).get('checklist')
        _, counts = _check_checklist(checklist)
        hashes = score.get('source_sha256', {})
        if set(hashes) != {'candidate', 'checklist', 'judge'} or any(
                not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in hashes.values()):
            raise ValueError('explanation score requires candidate/checklist/judge source hashes')
        if score['status'] == 'ok':
            judge = json.dumps({'items': [{k: item.get(k) for k in ('item_id', 'verdict', 'evidence')}
                                         for item in score.get('items', [])]})
            expected = score_checklist(row['candidate_text'], checklist, judge)
            if {k: v for k, v in expected.items() if k != 'source_sha256'} != {
                    k: v for k, v in score.items() if k != 'source_sha256'}:
                raise ValueError('explanation score differs from its checklist items, counts or evidence')
        else:
            if score.get('items') != [] or not score.get('failure_reason'):
                raise ValueError('unavailable explanation score must retain failure and no item scores')
            for dim, count in counts.items():
                if score['dimensions'][dim] != {'applicable': count, 'covered': None, 'uncertain': None,
                                                'coverage': None, 'uncertainty_rate': None}:
                    raise ValueError('unavailable explanation dimension differs from its checklist')


def _completeness(rows, preview):
    config_path = Path(__file__).parents[1] / 'configs/reference-reuse.json'
    config = json.loads(config_path.read_text())
    models = sorted({r['model'] for r in rows}) if preview else config['model_ids']
    slots = {(r['split'], r['family_id'], r['model'], r['task_kind'], r['condition']) for r in rows}
    missing, issues, observed = [], [], {}
    for split in ('test', 'dev'):
        explanation = {r['family_id'] for r in rows if r['split'] == split and r['task_kind'] == 'explanation'}
        paired = {r['family_id'] for r in rows if r['split'] == split and r['task_kind'] != 'explanation'}
        observed[split] = {'explanation_families': len(explanation), 'paired_families': len(paired)}
        if not paired <= explanation:
            issues.append(f'{split}: paired families absent from the explanation universe')
        for family in sorted(explanation | paired):
            for model in models:
                for condition in CONDITIONS:
                    slot = (split, family, model, 'explanation', condition)
                    if slot not in slots:
                        missing.append(list(slot))
        for family in sorted(paired):
            for model in models:
                for kind, condition in [('pair', v) for v in _VARIANTS] + (
                        [('permutation', 'original')] if split == 'test' else []):
                    slot = (split, family, model, kind, condition)
                    if slot not in slots:
                        missing.append(list(slot))
        if not preview:
            for key, count_key in [('explanation_families', split), ('paired_families', 'paired_' + split)]:
                if observed[split][key] != config['design'][count_key]:
                    issues.append(f'{split}: {key}={observed[split][key]}, planned={config["design"][count_key]}')
    if any(r['split'] == 'dev' and r['task_kind'] == 'permutation' for r in rows):
        issues.append('development permutations are outside the current protocol')
    if not preview and set(r['model'] for r in rows) != set(models):
        issues.append('model IDs differ from the two configured tested models')
    nonterminal, missing_scores = [], []
    for row in rows:
        reason = None
        if row['run_status'] not in _TERMINAL:
            reason = 'candidate ' + row['run_status']
        elif row['task_kind'] == 'explanation':
            judge = row.get('judge_status')
            if row['run_status'] == 'ok' and judge not in _TERMINAL:
                reason = 'judge ' + str(judge)
            elif row['run_status'] != 'ok' and judge != 'not_applicable':
                reason = 'failed candidate must have explicit not_applicable judge status'
            elif judge == 'ok' and row.get('score') is None:
                missing_scores.append(row['job_id'])
        elif row.get('score') is None and row['run_status'] == 'ok':
            missing_scores.append(row['job_id'])
        if reason:
            nonterminal.append({'job_id': row['job_id'], 'reason': reason})
    return {'complete': bool(rows) and not (missing or issues or nonterminal or missing_scores),
            'scope': 'synthetic supplied universe' if preview else 'current configured protocol',
            'observed': observed, 'expected_design': None if preview else config['design'],
            'missing_slots': missing, 'design_issues': issues,
            'nonterminal_jobs': nonterminal, 'missing_scores': missing_scores,
            'input_scope': 'Counts and supplied slots are checked; absent family identities cannot be recovered from rows alone.'}


def _metadata(path):
    data = dict.fromkeys(('section', 'model', 'subgroup', 'comparison', 'condition', 'variant',
                         'dimension', 'metric'), '')
    data['split'] = path[1]
    tail = path[3:]
    if tail[0] == 'model_differences':
        data.update(section='model_differences', comparison=tail[1], condition=tail[3],
                    dimension=tail[4], metric='coverage_difference')
        return data
    data['model'] = tail[1]
    tail = tail[2:]
    if tail[0] == 'ocr_subgroups':
        data['subgroup'] = 'ocr_' + tail[1]
        tail = tail[2:]
    if tail[0] == 'conditions':
        data.update(section='explanation_conditions', condition=tail[1],
                    dimension=tail[-1] if tail[2] == 'dimensions' else '',
                    metric='coverage' if tail[2] == 'dimensions' else tail[2])
    elif tail[0] == 'condition_differences':
        data.update(section='condition_differences', comparison=tail[1], dimension=tail[2],
                    metric='coverage_difference')
    elif tail[0] == 'hint_gains':
        if tail[1] == 'item_repair_harm':
            data.update(section='item_repair_harm', comparison='hinted_minus_image_only',
                        dimension=tail[2], metric=tail[3])
        else:
            data.update(section='hint_gains', comparison='hinted_minus_image_only', metric=tail[1],
                        dimension=tail[1] if tail[1] in ('K', 'R') else '')
    elif tail[0] == 'variants':
        data.update(section='paired_variants', variant=tail[1], metric=tail[2])
    elif tail[0] in ('triples', 'permutations'):
        data.update(section='paired_' + tail[0], metric=tail[1])
    else:
        raise ValueError(f'unsupported analytic metric path: {path}')
    return data


def _metrics(analysis):
    records, memberships = [], {}
    def visit(node, path):
        if not isinstance(node, dict):
            return
        if {'estimate', 'denominator', 'n_expected', 'contributions', 'ci95'} <= node.keys():
            identity = 'metric_' + _sha(json.dumps(path, ensure_ascii=False).encode())[:20]
            interval = node['ci95']
            row = dict(metric_id=identity, analysis_path=json.dumps(path, ensure_ascii=False),
                       **_metadata(path), **{k: node[k] for k in ('estimate', 'numerator', 'denominator',
                       'n_expected', 'n_unavailable', 'n_clusters')},
                       ci95_low=interval[0] if interval is not None else None,
                       ci95_high=interval[1] if interval is not None else None,
                       estimate_status='NA' if node['estimate'] is None else 'available',
                       ci95_status='available' if interval is not None else 'NA: fewer than two available families',
                       membership_key=identity)
            records.append(row)
            memberships[identity] = {'analysis_path': path, **node}
            return
        for key, child in node.items():
            visit(child, path + [key])
    visit(analysis['splits'], ['splits'])
    return records, memberships


def _csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
    writer.writeheader()
    writer.writerows({k: 'NA' if row[k] is None else row[k] for k in _COLUMNS} for row in rows)
    return stream.getvalue()


def _tex(value):
    substitutions = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                     '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(substitutions.get(c, c) for c in str(value))


def _number(value):
    return 'NA' if value is None else f'{value:.3f}'


def _measure(row):
    return ' / '.join(row[k] for k in ('subgroup', 'comparison', 'condition', 'variant', 'dimension', 'metric')
                      if row[k]).replace('_', ' ')


def _latex(rows, title, label):
    parts = ['% Requires longtable and array. All estimates weight families equally.',
             '% Full precision and memberships are in the sibling CSV and metric_memberships.json.',
             r'\begingroup\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{longtable}{>{\raggedright\arraybackslash}p{.17\linewidth}>{\raggedright\arraybackslash}p{.34\linewidth}rrrr}',
             r'\caption{' + _tex(title + '. ' + label) + r'}\\',
             r'Model & Measure & Estimate & $n/N$ & NA & 95\% CI \\ \hline', r'\endfirsthead',
             r'Model & Measure & Estimate & $n/N$ & NA & 95\% CI \\ \hline', r'\endhead']
    for row in rows:
        interval = ('NA' if row['ci95_low'] is None else
                    '[' + _number(row['ci95_low']) + ', ' + _number(row['ci95_high']) + ']')
        values = [row['model'] or row['comparison'], _measure(row), _number(row['estimate']),
                  f'{row["denominator"]}/{row["n_expected"]}', row['n_unavailable'], interval]
        escaped = [_tex(v) for v in values]
        escaped[0] = escaped[0].replace('/', r'/\allowbreak{}').replace('-', r'-\allowbreak{}')
        parts.append(' & '.join(escaped) + r' \\')
    parts += [r'\hline', r'\end{longtable}',
              r'\noindent $n/N$: available/expected families; NA: unavailable families. '
              r'Intervals: 95\% percentile bootstrap over families; NA when fewer than two families. '
              r'No item pooling. Valid-score coverage includes all expected families.', r'\endgroup', '']
    return '\n'.join(parts)


def _plot_specs(rows):
    """Select canonical metrics, avoiding duplicated OCR-subgroup copies."""
    base = [r for r in rows if not r['subgroup']]
    by_model = sorted({r['model'] for r in base if r['model']})
    groups = []
    for index, model in enumerate(by_model, 1):
        selected = [r for r in base if r['model'] == model]
        for name, section, panels in [
            ('coverage', 'explanation_conditions', ['V', 'I', 'K', 'R', 'valid_score_coverage']),
            ('condition_differences', 'condition_differences', ['V', 'I', 'K', 'R']),
            ('hint_gains', 'hint_gains', ['K', 'R', 'R_minus_K']),
            ('paired_variants', 'paired_variants', ['cue_correct', 'relation_correct', 'joint_correct']),
            ('paired_controls', 'paired_controls', ['all_three_joint', 'sham_preservation', 'semantic_agreement'])]:
            metrics = [r for r in selected if (r['section'] == section if section != 'paired_controls'
                                               else r['section'] in ('paired_triples', 'paired_permutations'))]
            if name == 'paired_variants':
                metrics = [r for r in metrics if r['metric'] in panels]
            groups.append((f'model_{index:02d}_{name}', model, name, panels, metrics))
    return groups


def _plot(staging, split, spec, label):
    name, model, group, panel_names, rows = spec
    record = {'split': split, 'model': model, 'kind': group, 'metric_ids': [r['metric_id'] for r in rows]}
    if not rows or not any(r['estimate'] is not None for r in rows):
        return {**record, 'status': 'omitted', 'reason': 'No available analytic estimates in this scope.', 'paths': []}
    # Defer dependency loading until inputs have passed all export gates.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    titles = {'valid_score_coverage': 'Valid score coverage', 'cue_correct': 'Cue correct',
              'relation_correct': 'Relation correct', 'joint_correct': 'Joint correct',
              'all_three_joint': 'All-three joint success', 'sham_preservation': 'Sham preservation',
              'semantic_agreement': 'Option-order agreement', 'R_minus_K': 'R gain minus K gain'}
    titles.update(V='Visual (V)', I='Identity (I)', K='Knowledge (K)', R='Reasoning (R)')
    panels = len(panel_names)
    figure = Figure(figsize=(max(8.4, panels * 2.8), 4.9), dpi=160)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, panels, squeeze=False)[0]
    for axis, panel in zip(axes, panel_names):
        selected = [r for r in rows if (r['dimension'] == panel if group in ('coverage', 'condition_differences')
                                       and panel != 'valid_score_coverage' else r['metric'] == panel)]
        for i, row in enumerate(selected):
            label_text = (_CONDITION_LABELS[row['condition']] if row['condition'] else
                          row['variant'].capitalize() if row['variant'] else
                          row['comparison'].replace('_minus_', ' − ').replace('image_only', 'image')
                          .replace('ocr_only', 'OCR') if group == 'condition_differences' else 'Families')
            estimate = row['estimate']
            if estimate is not None:
                axis.plot(estimate, i, 'o', color='#21618c', markersize=5, zorder=3)
                if row['ci95_low'] is not None:
                    axis.hlines(i, row['ci95_low'], row['ci95_high'], color='#21618c', linewidth=1.8)
                    axis.plot([row['ci95_low'], row['ci95_high']], [i, i], '|', color='#21618c')
            note = f'{_number(estimate)}; n={row["denominator"]}/{row["n_expected"]}\nNA={row["n_unavailable"]}'
            if row['ci95_low'] is None:
                note += '; CI NA'
            axis.text(.02, i + .28, note, transform=axis.get_yaxis_transform(), fontsize=7.3,
                      va='top', color='#444444')
            row['_plot_label'] = label_text
        axis.set_yticks(range(len(selected)), [r.pop('_plot_label') for r in selected], fontsize=8)
        axis.set_ylim(max(1, len(selected)) - .3, -.5)
        difference = group in ('condition_differences', 'hint_gains')
        limit = 2.05 if panel == 'R_minus_K' else 1.05
        axis.set_xlim((-limit, limit) if difference else (-.04, 1.04))
        axis.set_xticks([-2, -1, 0, 1, 2] if limit > 2 else [-1, 0, 1] if difference else [0, .5, 1])
        axis.axvline(0, color='#888888', linewidth=.7)
        axis.set_title(titles.get(panel, panel), fontsize=10, pad=12)
        axis.set_xlabel('Difference in family coverage' if difference else 'Family proportion', fontsize=8)
        axis.grid(axis='x', color='#dddddd', linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='y', length=0)
    group_titles = {'coverage': 'A: reference coverage and score availability',
                    'condition_differences': 'A: matched condition differences',
                    'hint_gains': 'A: knowledge versus reasoning hint gains (hinted − image only)',
                    'paired_variants': 'B: cue, relation and joint correctness by variant',
                    'paired_controls': 'B: complete triples, sham and option-order controls'}
    figure.suptitle(f'{group_titles[group]}\n{model} | {"PRIMARY TEST" if split == "test" else "DEVELOPMENT ONLY"}',
                   fontsize=12, y=.96)
    figure.text(.5, .035, 'Points: equal-weight family means; bars: 95% family bootstrap CI. '
                'n/N: available/expected families. NA is never plotted as zero.', ha='center', fontsize=8)
    if 'SYNTHETIC' in label or 'PARTIAL' in label:
        figure.text(.5, .105, label, ha='center', fontsize=11, weight='bold', color='#9b3027')
    figure.subplots_adjust(left=.08, right=.985, top=.73, bottom=.29, wspace=.72)
    paths = [f'{split}/figures/{name}.{extension}' for extension in ('pdf', 'png')]
    for relative in paths:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, format=target.suffix[1:], dpi=160,
                       bbox_inches='tight', pad_inches=.18,
                       metadata={'Creator': 'reference_reporting', 'Title': label})
    return {**record, 'status': 'written', 'paths': paths,
            'values_source': 'metric_memberships.json; no plot-specific statistics'}


def export_reference_results(rows, *, output_dir, seed=6000, draws=10000,
                             allow_partial=False, synthetic_preview=False):
    """Export fresh analysis, tidy CSV/LaTeX, vector PDF/PNG and bound provenance.

    Existing destinations (including dangling symlinks) are never overwritten.
    Raw judge texts are absent from scored rows: item/evidence/count consistency
    is checked, but semantic validity and upstream execution authenticity are
    not established by this operation. Use verified ``load_scores`` at the CLI.
    """
    if type(allow_partial) is not bool or type(synthetic_preview) is not bool:
        raise ValueError('allow_partial and synthetic_preview must be booleans')
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'output already exists: {destination}')
    # Round-trip before reading validates JSON serializability and avoids mutation.
    rows = json.loads(_json(rows))
    analysis = analyze_reference_scores(rows, seed=seed, draws=draws)
    _validate_bindings(rows)
    _explanation_bindings(rows)
    if not any(r['run_status'] == 'ok' for r in rows):
        raise ValueError('no successful candidate outputs; cannot publish result tables or figures')
    completeness = _completeness(rows, synthetic_preview)
    complete = completeness['complete']
    if not complete and not allow_partial:
        raise ValueError('study is incomplete; use allow_partial for an explicitly partial export')
    label = ('SYNTHETIC FIXTURE PREVIEW — NOT EXPERIMENTAL RESULTS' if synthetic_preview
             else 'DESCRIPTIVE AUTOMATIC SCORES') + (' | PARTIAL SUPPLIED RESULTS' if not complete else '')
    rows.sort(key=lambda r: r['job_id'])
    raw = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for r in rows)
    records, memberships = _metrics(analysis)
    for record in records:
        record['evidence_label'] = label
    parent = destination.parent.resolve(strict=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=parent))
    try:
        (staging / 'scored.jsonl').write_text(raw, encoding='utf-8')
        (staging / 'analysis.json').write_text(_json(analysis), encoding='utf-8')
        (staging / 'metric_memberships.json').write_text(_json({
            'schema_version': 1, 'label': label, 'metrics': memberships}), encoding='utf-8')
        tables, figures = [], []
        for split in ('test', 'dev'):
            folder = staging / split
            folder.mkdir()
            selected = [r for r in records if r['split'] == split]
            (folder / 'all_metrics.csv').write_text(_csv(selected), encoding='utf-8')
            tables.append({'split': split, 'section': 'all_metrics', 'paths': [f'{split}/all_metrics.csv'],
                           'metric_ids': [r['metric_id'] for r in selected]})
            for section in _TITLES:
                group = [r for r in selected if r['section'] == section]
                if not group:
                    continue
                for extension, content in [('csv', _csv(group)),
                        ('tex', _latex(group, _TITLES[section] + ' (' + split + ')', label))]:
                    (folder / f'{section}.{extension}').write_text(content, encoding='utf-8')
                tables.append({'split': split, 'section': section,
                               'paths': [f'{split}/{section}.csv', f'{split}/{section}.tex'],
                               'metric_ids': [r['metric_id'] for r in group]})
            for spec in _plot_specs(selected):
                figures.append(_plot(staging, split, spec, label))
            if not selected:
                figures.append({'split': split, 'kind': 'all', 'status': 'omitted', 'metric_ids': [],
                                'paths': [], 'reason': 'No supplied rows for this split.'})
        readme = (label + '\n\nFind primary results in test/ and development calibration in dev/. '
            'Tables and figures use analysis.json. CSV retains full precision. See metric_memberships.json '
            'for family counts, source jobs, and scores.\n\n'
            'Estimates weight available families equally. n/N gives available and expected families. '
            'Invalid or truncated pair answers count as incorrect. Family bootstrap intervals require '
            'at least two available families. Treat the ten test triples as exploratory.\n\n'
            'Task A scores source checklist coverage. Task B scores original, edited, and sham images. '
            'Use the prepared inputs and run ledger to trace missing jobs and review original judgments.\n\n'
            'LaTeX fragments require longtable and array. Figures come in PDF and PNG. '
            'See manifest.json for output files and checksums.\n')
        (staging / 'README.txt').write_text(readme, encoding='utf-8')
        code = Path(__file__).parent
        provenance = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'label': label,
                      'input_sha256': _sha(raw.encode()), 'input_rows': len(rows),
                      'prepared_sha256': sorted({r['prepared_sha256'] for r in rows if r.get('prepared_sha256')}),
                      'seed': seed, 'draws': draws, 'bootstrap_unit': 'family_id', 'model_calls': 0,
                      'code_sha256': {name: _sha((code / name).read_bytes()) for name in
                                     ('reference_reporting.py', 'reference_analysis.py', 'analysis.py',
                                      'reference_scoring.py', 'reference_demo.py', 'reference_protocol.py')},
                      'config_sha256': _sha((code.parent / 'configs/reference-reuse.json').read_bytes()),
                      'validation_scope': 'Internal row/score binding; not upstream execution authentication.'}
        provenance['artifact_sha256'] = {p.relative_to(staging).as_posix(): _sha(p.read_bytes())
                                        for p in sorted(staging.rglob('*')) if p.is_file()}
        (staging / 'provenance.json').write_text(_json(provenance), encoding='utf-8')
        manifest = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'primary_split': 'test',
                    'label': label, 'synthetic_preview': synthetic_preview, 'allow_partial': allow_partial,
                    'complete': complete, 'completeness': completeness, 'model_calls': 0,
                    'counts': {'rows': len(rows), 'run_status': dict(sorted(Counter(r['run_status'] for r in rows).items()))},
                    'seed': seed, 'draws': draws, 'tables': tables, 'figures': figures,
                    'table_columns': list(_COLUMNS),
                    'artifact_sha256': {p.relative_to(staging).as_posix(): _sha(p.read_bytes())
                                       for p in sorted(staging.rglob('*')) if p.is_file()}}
        (staging / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
        _publish(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
