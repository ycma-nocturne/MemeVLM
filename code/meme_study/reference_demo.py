"""Portable offline reference-reuse results, without model calls or auto-selection.

``rows`` are complete planned-job score records from ``score_reference``.
``selections`` explicitly names 6–8 family/model/task_kind/condition anchors,
each with a selection_reason. All supplied sibling conditions/variants appear
alongside each anchor; missing siblings are visible. Aggregate figures use all
supplied rows, never the chosen examples. A supplied analysis must exactly match
reanalysis of those rows with its declared seed/draws.

``synthetic_preview=True`` allows 1–8 anchors and prominently labels the entire
package as fixture-only, never measured results. It still requires actual raw
fixture text, valid image bytes and provenance. No text or scores are invented.
Hash checks establish internal consistency, not upstream execution authenticity.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlsplit

from .data import content_hash
from .freeze import _publish
from .image_io import decode_image_receipt
from .reference_analysis import analyze_reference_scores
from .reference_protocol import PROTOCOL_ID
from .reference_scoring import score_pair


_SELECTION_FIELDS = {'family_id', 'model', 'task_kind', 'condition', 'selection_reason'}
_CONDITIONS = ('image_only', 'hinted', 'ocr_only')
_VARIANTS = ('original', 'edited', 'sham')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _bytes(value):
    return (_json(value) + '\n').encode('utf-8')


def _escape(value):
    if isinstance(value, (dict, list)):
        value = _json(value)
    return html.escape(str(value), quote=True)


def _pre(value):
    return '<pre>' + _escape(value) + '</pre>'


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} requires a lowercase SHA-256 hash')
    return value


def _identity(row):
    return tuple(row[k] for k in ('family_id', 'model', 'task_kind', 'condition'))


def _validate_bindings(rows):
    for row in rows:
        _digest(row.get('source_input_sha256'), 'source input')
        if type(row.get('candidate_truncated')) is not bool:
            raise ValueError('candidate_truncated must be explicit boolean')
        text = row.get('candidate_text')
        if row['run_status'] == 'ok':
            if not isinstance(text, str):
                raise ValueError('successful candidate requires raw response text, possibly empty')
            _digest(row.get('candidate_output_sha256'), 'candidate output')
        elif row.get('score') is not None:
            raise ValueError('unsuccessful candidate cannot have a score')
        if text is not None and not isinstance(text, str):
            raise ValueError('candidate text must be string or null')
        score = row.get('score') or {}
        hashes = score.get('source_sha256', {})
        if not isinstance(hashes, dict):
            raise ValueError('score source hashes must be an object')
        for key in ('candidate', 'response'):
            if key in hashes and (text is None or hashes[key] != _sha(text.encode('utf-8'))):
                raise ValueError('candidate text differs from its scored hash')
        for key, value in (('checklist', (row.get('reference') or {}).get('checklist')),
                           ('options', row.get('options')), ('gold', row.get('gold'))):
            if key in hashes and (value is None or hashes[key] != content_hash(value)):
                raise ValueError(f'{key} differs from its scored hash')
        if row.get('candidate_text_sha256') is not None:
            if text is None or _sha(text.encode()) != row['candidate_text_sha256']:
                raise ValueError('candidate text hash mismatch')
        if row['task_kind'] != 'explanation' and row.get('score') is not None:
            rescored = score_pair(text, row.get('options'), row.get('gold'),
                                  truncated=row['candidate_truncated'])
            if row['score'] != rescored:
                raise ValueError('pair score differs from its raw response, truncation or accepted answers')


def _selected(rows, selections, preview):
    minimum = 1 if preview else 6
    if not isinstance(selections, list) or not minimum <= len(selections) <= 8:
        raise ValueError(f'export requires {minimum}–8 explicitly selected cases; production requires 6–8')
    indexed = {_identity(row): row for row in rows}
    seen, result = set(), []
    for selected in selections:
        if (not isinstance(selected, dict) or set(selected) != _SELECTION_FIELDS
                or any(not isinstance(v, str) or not v.strip() for v in selected.values())):
            raise ValueError('selection needs explicit family/model/task_kind/condition and selection_reason')
        identity = _identity(selected)
        if identity in seen or identity not in indexed:
            raise ValueError('selection is duplicate or does not match an existing score row')
        seen.add(identity)
        anchor = indexed[identity]
        group = 'explanation' if anchor['task_kind'] == 'explanation' else 'paired'
        siblings = [r for r in rows if r['family_id'] == anchor['family_id'] and r['model'] == anchor['model']
                    and (r['task_kind'] == 'explanation') == (group == 'explanation')]
        result.append({'selection': selected, 'anchor_job_id': anchor['job_id'],
                       'split': anchor['split'], 'group': group, 'rows': siblings})
    if not any(r['run_status'] == 'ok' for case in result for r in case['rows']):
        raise ValueError('no available response in selected comparisons; cannot export a results demo')
    return result


def _image(root, relative, expected):
    if (not isinstance(relative, str) or not relative.strip() or '\\' in relative
            or ':' in relative or '\x00' in relative or PurePosixPath(relative).is_absolute()
            or '..' in PurePosixPath(relative).parts):
        raise ValueError('image path must remain relative to data_root')
    _digest(expected, 'image')
    try:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('image path escapes data_root or is not a regular file')
        raw = path.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise ValueError('image path is not a readable local file') from exc
    receipt = decode_image_receipt(raw, source_name=relative, expected_sha256=expected)
    extension = {'jpeg': 'jpg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}[receipt['detected_format']]
    return raw, {'source_path': relative, 'asset_path': f'images/{expected}.{extension}', **receipt}


def _collect_images(cases, root):
    assets, receipts, bindings, checked = {}, [], {}, set()
    for case in cases:
        for row in case['rows']:
            if row['job_id'] in checked:
                continue
            checked.add(row['job_id'])
            source = row.get('source')
            if (not isinstance(source, dict)
                    or any(not source.get(k) for k in ('dataset', 'revision', 'source_id', 'url'))):
                raise ValueError('selected rows require inherited public source attribution')
            if not isinstance(row.get('image_provenance'), dict):
                raise ValueError('selected rows require explicit image provenance')
            provenance = row['image_provenance']
            is_a = row['task_kind'] == 'explanation'
            image_role = 'reference_image' if is_a and row['condition'] == 'ocr_only' else 'image'
            if (provenance.get('path') != row.get(image_role + '_path')
                    or provenance.get('sha256') != row.get(image_role + '_sha256')):
                raise ValueError('image provenance differs from the displayed input image')
            if is_a:
                if provenance.get('adapted') is not False:
                    raise ValueError('A requires explicit unadapted public original image provenance')
                if (provenance['path'] != row.get('reference_image_path')
                        or provenance['sha256'] != row.get('reference_image_sha256')):
                    raise ValueError('A original image differs from its public reference image')
            elif provenance.get('variant') != row['condition']:
                raise ValueError('B image provenance variant differs from the scored variant')
            if row['task_kind'] == 'explanation':
                reference = row.get('reference')
                if (not isinstance(reference, dict) or not isinstance(reference.get('checklist'), list)
                        or not isinstance(reference.get('explanation'), str)):
                    raise ValueError('selected A rows require the inherited reference and checklist')
                field = {'hinted': 'hint_text', 'ocr_only': 'ocr_text'}.get(row['condition'])
                if field and (field not in row or (row[field] is not None and not isinstance(row[field], str))):
                    raise ValueError('selected input is missing its hint/OCR text field')
                if field == 'hint_text' and (not row[field] or not row[field].strip()):
                    raise ValueError('hinted input requires its factual hint text')
            else:
                if any(not isinstance(row.get(field), str) or not row[field].strip()
                       for field in ('cue_question', 'relation_question')):
                    raise ValueError('selected B rows require actual cue and relation questions')
                # This deterministic local call validates materials even for missing output.
                score_pair('', row.get('options'), row.get('gold'))
            is_ocr = row['task_kind'] == 'explanation' and row['condition'] == 'ocr_only'
            if is_ocr and row.get('image_path') is not None:
                raise ValueError('OCR-only condition must not have a model input image')
            if not is_ocr and not row.get('image_path'):
                raise ValueError('visual condition requires its actual input image')
            if not row.get('reference_image_path'):
                raise ValueError('selected rows require a separate public reference image path')
            for field in ('image', 'reference_image'):
                relative = row.get(field + '_path')
                if relative is None:
                    continue
                raw, receipt = _image(root, relative, row.get(field + '_sha256'))
                assets[receipt['asset_path']] = raw
                bindings[(row['job_id'], field)] = receipt['asset_path']
                receipts.append({'job_id': row['job_id'], 'role': field, **receipt})
    return assets, receipts, bindings


def _counts(rows):
    return {'rows': len(rows), 'families': len({r['family_id'] for r in rows}),
            'run_status': dict(sorted(Counter(r['run_status'] for r in rows).items())),
            'candidate_available': sum(r['run_status'] == 'ok' for r in rows),
            'candidate_unavailable': sum(r['run_status'] != 'ok' for r in rows),
            'candidate_truncated': sum(r['candidate_truncated'] for r in rows),
            'explanation_scores_available': sum(r['task_kind'] == 'explanation'
                and (r.get('score') or {}).get('status') == 'ok' for r in rows),
            'invalid_pair_outputs': sum(r['task_kind'] != 'explanation'
                and (r.get('score') or {}).get('status') in {'invalid', 'truncated'} for r in rows)}


def _number(value):
    return 'NA' if value is None else f'{value:.3f}'


def _metric_row(label, metric):
    estimate, interval = metric['estimate'], metric['ci95']
    bar = (f'<meter min="0" max="1" value="{estimate}">{_number(estimate)}</meter>'
           if estimate is not None and 0 <= estimate <= 1 else '')
    ci = 'NA' if interval is None else ' to '.join(_number(v) for v in interval)
    return ('<tr><th>' + _escape(label) + '</th><td>' + _number(estimate) + bar
            + '</td><td>' + str(metric['denominator']) + '/' + str(metric['n_expected'])
            + '</td><td>' + str(metric['n_unavailable']) + '</td><td>' + ci + '</td></tr>')


def _metric_table(items):
    return ('<table><thead><tr><th>Metric</th><th>Estimate</th><th>Available / expected families</th>'
            '<th>Unavailable</th><th>95% family bootstrap interval</th></tr></thead><tbody>'
            + ''.join(_metric_row(label, metric) for label, metric in items) + '</tbody></table>')


def _analysis_html(analysis):
    sections = []
    for split in ('test', 'dev'):
        result = analysis['splits'][split]
        section = [f'<section><h2>{split.upper()} — all supplied planned rows</h2>',
                   f'<p>{result["n_rows"]} rows; {result["n_families"]} families. '
                   'These aggregates do not depend on example selection.</p>']
        for model, data in result['explanation']['models'].items():
            section.append('<h3>A · ' + _escape(model) + ' · V/I/K/R coverage</h3>')
            metrics = [(condition + ' · ' + dim, metric)
                       for condition, item in data['conditions'].items()
                       for dim, metric in item['dimensions'].items()]
            section.append(_metric_table(metrics))
            section.append('<details><summary>Paired condition changes</summary>')
            section.append(_metric_table([(contrast + ' · ' + dim, metric)
                for contrast, dimensions in data['condition_differences'].items()
                for dim, metric in dimensions.items()]))
            section.append('</details>')
        for model, data in result['paired']['models'].items():
            section.append('<h3>B · ' + _escape(model) + ' · cue / relation</h3>')
            labels = {'cue_correct': 'Cue correctness', 'relation_correct': 'Relation correctness',
                      'joint_correct': 'Cue + relation correct',
                      'cue_relation_discrepancy': 'Cue / relation discrepancy'}
            section.append(_metric_table([(variant + ' · ' + label, item[key])
                for variant, item in data['variants'].items() for key, label in labels.items()]))
            section.append(_metric_table([
                ('All three joint + sham relation preservation', data['triples']['all_three_joint']),
                ('Sham semantic relation preservation', data['triples']['sham_preservation']),
                ('Original / permutation semantic agreement', data['permutations']['semantic_agreement'])]))
        section.append('</section>')
        sections.extend(section)
    return ''.join(sections)


def _source_html(source):
    url = source['url']
    parts = urlsplit(url) if isinstance(url, str) else None
    link = ('<a href="' + _escape(url) + '" rel="noreferrer">Source record</a>'
            if parts and parts.scheme in {'http', 'https'} and parts.netloc else _escape(url))
    return ('<p>Inherited public reference: ' + _escape(source['dataset']) + ' · source '
            + _escape(source['source_id']) + ' · revision ' + _escape(source['revision'])
            + ' · ' + link + '</p>')


def _score_html(row):
    score = row.get('score')
    if score is None or score['status'] == 'unavailable':
        reason = ((score or {}).get('failure_reason') or
                  'judge ' + str(row.get('judge_status', 'unavailable')))
        return '<p class="warn">Unavailable score — ' + _escape(reason) + '. No zero imputation.</p>'
    if row['task_kind'] == 'explanation':
        parts = []
        for dim, value in score['dimensions'].items():
            parts.append(dim + ': ' + _number(value['coverage']) + ' (' + str(value['covered'])
                         + '/' + str(value['applicable']) + ' items; uncertain '
                         + str(value.get('uncertain', 'unknown')) + ')')
        result = '<p>' + _escape(' · '.join(parts)) + '</p>'
    else:
        result = '<p>' + _escape(' · '.join([
            'Score status: ' + score['status'], 'Cue correctness: ' + str(score['cue_correct']),
            'Relation correctness: ' + str(score['relation_correct']),
            'Joint: ' + str(score['joint_correct'])])) + '</p>'
        if score.get('reason'):
            result += '<p class="warn">' + _escape(score['reason']) + '</p>'
    return result + '<details><summary>Automatic score and evidence</summary>' + _pre(score) + '</details>'


def _row_html(row, bindings, anchor):
    is_a = row['task_kind'] == 'explanation'
    title = ('A public original · ' if is_a else 'B cleaned project · ') + row['condition']
    if row['task_kind'] == 'permutation':
        title += ' · option permutation'
    parts = ['<article><h4>' + _escape(title) + (' · selected anchor' if anchor else '') + '</h4>',
             '<p>Run status: <strong>' + _escape(row['run_status']) + '</strong></p>']
    if row['candidate_truncated']:
        parts.append('<p class="warn">Candidate was truncated. Raw text is retained as recorded.</p>')
    image = bindings.get((row['job_id'], 'image'))
    if image:
        parts.append('<figure><img src="' + image + '" alt="' + _escape(title)
                     + '"><figcaption>Actual model input image</figcaption></figure>')
    else:
        parts.append('<p>OCR-only input — no image was sent to the model.</p>')
    reference = bindings.get((row['job_id'], 'reference_image'))
    if reference and reference != image:
        parts.append('<details><summary>Public original for reference; separate from this input</summary>'
                     '<img src="' + reference + '" alt="Public reference original"></details>')
    if is_a:
        if row['condition'] == 'ocr_only':
            parts.append('<h5>OCR input</h5>' + _pre(row.get('ocr_text')))
        elif row['condition'] == 'hinted':
            parts.append('<h5>Factual hint input</h5>' + _pre(row.get('hint_text')))
    else:
        parts.append('<h5>Cue / relation questions and options</h5>' + _pre({k: row.get(k)
                     for k in ('cue_question', 'relation_question', 'options')}))
        parts.append('<details><summary>Variant-specific accepted answers</summary>'
                     + _pre(row.get('gold')) + '</details>')
    parts.append('<h5>Raw candidate response</h5>')
    text = row.get('candidate_text')
    if text is None:
        parts.append('<p class="warn">Unavailable response; no replacement text generated.</p>')
    elif text == '':
        parts.append('<p class="warn">Empty response (recorded as an empty string).</p>' + _pre(text))
    else:
        parts.append(_pre(text))
    parts.append(_score_html(row))
    parts.append(_source_html(row['source']))
    if row.get('reference') is not None:
        parts.append('<details><summary>Inherited reference and checklist</summary>'
                     + _pre(row['reference']) + '</details>')
    parts.append('<details><summary>Image preparation and attribution</summary>'
                 + _pre(row['image_provenance']) + '</details>')
    parts.append('<details><summary>Bound record identifiers</summary>'
                 + _pre({k: row.get(k) for k in ('job_id', 'source_input_sha256',
                         'candidate_output_sha256', 'judge_job_id', 'judge_output_sha256')}) + '</details></article>')
    return ''.join(parts)


def _case_html(case, number, bindings):
    selected = case['selection']
    parts = ['<section><h2>Example ' + str(number) + ' · ' + _escape(selected['family_id'])
             + ' · ' + _escape(selected['model']) + ' · ' + case['split'] + '</h2>',
             '<p><strong>Selection reason:</strong> ' + _escape(selected['selection_reason']) + '</p>',
             '<div class="comparisons">']
    indexed = {(r['task_kind'], r['condition']): r for r in case['rows']}
    slots = ([('explanation', c) for c in _CONDITIONS] if case['group'] == 'explanation'
             else [('pair', v) for v in _VARIANTS] +
             ([('permutation', 'original')] if ('permutation', 'original') in indexed else []))
    for kind, condition in slots:
        row = indexed.get((kind, condition))
        if row is None:
            parts.append('<article><h4>' + condition + '</h4><p class="warn">Unavailable: planned '
                         'comparison row was not supplied. No response or score inferred.</p></article>')
        else:
            parts.append(_row_html(row, bindings, row['job_id'] == case['anchor_job_id']))
    return ''.join(parts) + '</div></section>'


_STYLE = '''body{margin:0;background:#f5f7fa;color:#142536;font:16px/1.55 system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:32px}h1,h2,h3,h4{line-height:1.2}h5{font-size:1rem}
section{margin:28px 0;padding:24px;background:white;border:1px solid #d5dfe9;border-radius:12px}
.banner,.warn{background:#fff1d6;padding:12px;border-left:4px solid #bc6800}.banner{font-weight:700}
.comparisons{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
article{border:1px solid #d5dfe9;border-radius:8px;padding:18px;min-width:0}
img{display:block;max-width:100%;max-height:360px;object-fit:contain;margin:auto}figure{margin:12px 0}
figcaption{font-size:0.85rem;color:#54677a;text-align:center}pre{white-space:pre-wrap;overflow-wrap:anywhere;
background:#f1f4f8;padding:12px;border-radius:6px;font:0.9rem/1.55 system-ui}details{margin:12px 0}
summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse;margin:16px 0;font-size:.9rem}
th,td{text-align:left;padding:9px;border-bottom:1px solid #d5dfe9}td{font-variant-numeric:tabular-nums}
meter{display:block;width:90px;height:12px}a{color:#1251a0}footer{color:#54677a;font-size:.9rem}
@media(max-width:650px){main{padding:12px}section{padding:12px}table{display:block;overflow-x:auto}}
'''


def _page(cases, analysis, counts, bindings, preview):
    title = 'SYNTHETIC PREVIEW — fixture outputs only' if preview else 'Reference-reuse actual-response demonstration'
    intro = ('<p class="banner">' + title + '</p>' if preview else '')
    intro += ('<p>Descriptive automatic scores. Examples were explicitly chosen by the coordinator; '
              'they do not estimate prevalence. No model call or human rating was performed by this export. '
              'A uses public originals and inherited references; B uses separate project-cleaned '
              'original / edited / sham materials. Source attribution and preparation are shown per response.</p>'
              '<p>Missing judgments stay unavailable; invalid paired answers count wrong. Empty dimensions are NA. '
              'Test and development results stay separate. Intervals use family bootstrap resampling and are NA '
              'with fewer than two available families. Small B samples support exploratory comparisons only.</p>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; '
            'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            '<title>' + title + '</title><style>' + _STYLE + '</style></head><body><main><h1>' + title
            + '</h1>' + intro + '<h2>Actual supplied counts</h2>' + _pre(counts)
            + _analysis_html(analysis) + ''.join(_case_html(case, i, bindings) for i, case in enumerate(cases, 1))
            + '<footer>Offline package: local image bytes, unmodified raw score rows, analysis, explicit '
            'selections and a SHA-256 manifest. Source links require a connection; the demonstration itself '
            'does not. Input hash checks establish consistency, not authenticity of absent upstream jobs. '
            'A wholly omitted family or model cannot be recovered from supplied rows.</footer></main></body></html>\n').encode()


def export_demo(rows, *, data_root, output_dir, selections, analysis=None, synthetic_preview=False):
    """Atomically publish a new, portable offline results directory; never overwrite."""
    if type(synthetic_preview) is not bool:
        raise ValueError('synthetic_preview must be a boolean')
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'demo output already exists: {destination}')
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('data_root must be a directory')
    # Freeze a finite JSON snapshot before computing scores, rendering or hashing.
    rows, selections = json.loads(_json([rows, selections]))
    if analysis is not None and not isinstance(analysis, dict):
        raise ValueError('analysis must be a bound analysis object')
    seed, draws = (analysis.get('seed'), analysis.get('draws')) if analysis is not None else (6000, 10000)
    computed = analyze_reference_scores(rows, seed=seed, draws=draws)
    if analysis is not None and analysis != computed:
        raise ValueError('analysis differs from the supplied score rows')
    _validate_bindings(rows)
    cases = _selected(rows, selections, synthetic_preview)
    assets, receipts, bindings = _collect_images(cases, root)
    counts = _counts(rows)
    summary = {'protocol_id': PROTOCOL_ID, 'mode': 'synthetic_preview' if synthetic_preview else 'actual_responses',
               'selected_cases': len(cases), 'model_calls': 0, 'counts': counts}
    readme = (('SYNTHETIC PREVIEW — fixture outputs only; not experimental results.\n\n' if synthetic_preview else '')
              + 'Open index.html locally. Keep this directory together; images are local copies of verified bytes.\n'
              'All supplied planned rows are in scored.jsonl; selected.json records explicit selection reasons\n'
              'and comparison membership. analysis.json uses all supplied rows with development/test separated.\n'
              'manifest.json binds source rows, source images and exported artifacts with SHA-256 digests.\n'
              'Missing judgments are NA, not zero. Invalid paired responses remain wrong and visible.\n'
              'No model call, fabricated response, human acceptance or GPU measurement was made by export.\n'
              'Source references are inherited public annotations. Project image preparation is distinct.\n'
              'Hashes verify consistency, not upstream authenticity or reference correctness.\n')
    artifacts = {**assets, 'index.html': _page(cases, computed, counts, bindings, synthetic_preview),
                 'selected.json': _bytes(cases), 'analysis.json': _bytes(computed),
                 'scored.jsonl': ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                                        + '\n' for row in rows).encode(), 'README.txt': readme.encode()}
    manifest = {**summary, 'schema_version': 1, 'selections': selections,
                'source_rows_sha256': content_hash(rows), 'source_row_sha256': {r['job_id']: content_hash(r) for r in rows},
                'images': receipts, 'export_code_sha256': _sha(Path(__file__).read_bytes()),
                'artifact_sha256': {name: _sha(raw) for name, raw in artifacts.items()}}
    artifacts['manifest.json'] = _bytes(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=parent))
    try:
        for name, raw in artifacts.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        _publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
