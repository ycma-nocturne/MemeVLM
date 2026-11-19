"""Descriptive, family-paired statistics for the reference-reuse protocol.

``analyze_reference_scores`` accepts one row per *planned* job, including null
scores for pending/failed jobs. Missing condition/variant rows are additionally
made visible using the observed family union for each model. The API cannot
infer families or models wholly omitted by its caller; ``input_scope`` says so.

Output schema (version 1): ``splits.test`` is primary; ``splits.dev`` is separate.
Each split has row/family/job membership and ``explanation`` and ``paired``.
Explanation models contain conditions, condition_differences, hint_gains and
OCR subgroups; cross-model differences are under explanation.model_differences.
Paired models contain variants, triples and permutations. Every numeric metric
has estimate, numerator (sum of family values), denominator (available family
count), n_expected, n_unavailable, family IDs, unavailable family IDs, source
job IDs, per-family contributions, n_clusters, and ci95. Empty denominators and
intervals from fewer than two families are null. Valid-score coverage explicitly
uses all expected families, whereas outcome estimates use available families.

All outcome means and intervals weight families equally. Existing bootstrap
code is reused with family_id supplied as its cluster key, not template_group.
Item repair/harm additionally retain item counts, but their estimates likewise
average eligible per-family rates. These are descriptive automatic scores, not
human judgments, official benchmark scores, p-values, or category rankings.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from .analysis import cluster_bootstrap
from .reference_protocol import CONDITIONS, PROTOCOL_ID


_DIMENSIONS = ('V', 'I', 'K', 'R')
_VARIANTS = ('original', 'edited', 'sham')
_RUN_STATUSES = {'ok', 'failed', 'interrupted', 'missing', 'pending'}
_PAIR_STATUSES = {'ok', 'invalid', 'truncated'}
_CONTRASTS = (('hinted', 'image_only'), ('image_only', 'ocr_only'), ('hinted', 'ocr_only'))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _validate(rows):
    if not isinstance(rows, list):
        raise ValueError('rows must be a list of planned score records')
    jobs, slots, families, templates, ocr = set(), set(), {}, {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('every score row must be an object')
        for field in ('job_id', 'family_id', 'template_group', 'model'):
            if not _text(row.get(field)):
                raise ValueError(f'score row needs a nonempty {field}')
        kind, condition = row.get('task_kind'), row.get('condition')
        if kind not in {'explanation', 'pair', 'permutation'}:
            raise ValueError('unsupported task_kind')
        if kind == 'explanation':
            if condition not in CONDITIONS or row.get('variant') not in (None, 'original'):
                raise ValueError('explanation has an invalid condition or variant')
        elif (condition not in _VARIANTS or row.get('variant', condition) != condition
              or (kind == 'permutation' and condition != 'original')):
            raise ValueError('paired condition and variant are inconsistent')
        split = row.get('split')
        if split not in {'test', 'dev'}:
            raise ValueError('split must be test or dev')
        if row.get('run_status') not in _RUN_STATUSES:
            raise ValueError('run_status is missing or invalid')
        family, group = row['family_id'], row['template_group']
        if family in families and families[family] != (split, group):
            raise ValueError('family crosses splits or template assignments')
        if group in templates and templates[group] != split:
            raise ValueError('template group crosses splits')
        families[family], templates[group] = (split, group), split
        slot = (family, row['model'], kind, condition)
        if row['job_id'] in jobs or slot in slots:
            raise ValueError('duplicate job ID or family/model/task/condition slot')
        jobs.add(row['job_id']); slots.add(slot)
        if 'ocr_available' in row and row['ocr_available'] is not None:
            if type(row['ocr_available']) is not bool:
                raise ValueError('ocr_available must be boolean or null')
            if family in ocr and ocr[family] != row['ocr_available']:
                raise ValueError('family has inconsistent OCR availability')
            ocr[family] = row['ocr_available']
        score = row.get('score')
        if score is None:
            continue
        if not isinstance(score, dict):
            raise ValueError('score must be an object or null')
        if kind == 'explanation':
            if score.get('status') not in {'ok', 'unavailable'}:
                raise ValueError('explanation needs a checklist score')
            if score['status'] != 'ok':
                continue
            dimensions = score.get('dimensions')
            if not isinstance(dimensions, dict) or set(dimensions) != set(_DIMENSIONS):
                raise ValueError('checklist score requires V/I/K/R dimensions')
            for dim in _DIMENSIONS:
                entry = dimensions[dim]
                if not isinstance(entry, dict):
                    raise ValueError('dimension score must be an object')
                count, covered, coverage = (entry.get(k) for k in ('applicable', 'covered', 'coverage'))
                if (type(count) is not int or count < 0 or type(covered) is not int
                        or not 0 <= covered <= count):
                    raise ValueError('invalid dimension item counts')
                if count == 0:
                    if coverage is not None:
                        raise ValueError('empty dimension must have null coverage')
                elif (type(coverage) not in (int, float) or not math.isfinite(coverage)
                      or not math.isclose(coverage, covered / count, abs_tol=1e-12)):
                    raise ValueError('coverage must match covered/applicable')
            seen = set()
            for item in score.get('items', []):
                if (not isinstance(item, dict) or not _text(item.get('item_id'))
                        or item['item_id'] in seen or item.get('dimension') not in _DIMENSIONS
                        or item.get('verdict') not in {'covered', 'not_covered', 'uncertain'}):
                    raise ValueError('invalid or duplicate source item status')
                seen.add(item['item_id'])
        else:
            if score.get('status') not in _PAIR_STATUSES:
                raise ValueError('paired row needs an exact pair score')
            for field in ('cue_correct', 'relation_correct', 'joint_correct'):
                if type(score.get(field)) is not int or score[field] not in (0, 1):
                    raise ValueError('pair correctness must be binary')
            if score['joint_correct'] != score['cue_correct'] * score['relation_correct']:
                raise ValueError('joint score disagrees with cue/relation correctness')
            if score['status'] == 'ok' and not _text(score.get('semantic_relation')):
                raise ValueError('valid pair score needs semantic relation ID')
            if score['status'] != 'ok' and any(score[k] for k in
                    ('cue_correct', 'relation_correct', 'joint_correct')):
                raise ValueError('invalid or truncated pair must score zero')
    return ocr


def _available(row, kind):
    if row is None or row['run_status'] != 'ok' or not isinstance(row.get('score'), dict):
        return False
    if kind == 'explanation':
        return (row['score']['status'] == 'ok'
                and row.get('judge_status', 'ok') in (None, 'ok'))
    return row['score']['status'] in _PAIR_STATUSES


def _counts(indexed, expected):
    rows = [indexed[f] for f in sorted(expected) if f in indexed]
    return {'n_expected': len(expected), 'n_rows': len(rows),
            'missing_rows': len(expected) - len(rows),
            'run_status': dict(sorted(Counter(r['run_status'] for r in rows).items())),
            'score_status': dict(sorted(Counter((r.get('score') or {}).get('status', 'missing')
                                               for r in rows).items())),
            'judge_status': dict(sorted(Counter(str(r['judge_status']) for r in rows
                                               if 'judge_status' in r).items())),
            'failure_reasons': dict(sorted(Counter(str((r.get('score') or {}).get('failure_reason')
                                                       or (r.get('score') or {}).get('reason'))
                    for r in rows if (r.get('score') or {}).get('failure_reason')
                    or (r.get('score') or {}).get('reason')).items())),
            'job_ids': sorted(r['job_id'] for r in rows)}


class _Report:
    def __init__(self, seed, draws):
        self.seed, self.draws, self.cache = seed, draws, {}

    def metric(self, values, expected, members):
        names = sorted(values)
        # Family names do not alter a resample once its sorted value order is fixed.
        key = tuple(values[family] for family in names)
        if key not in self.cache:
            self.cache[key] = cluster_bootstrap(
                [{'template_group': family, 'difference': values[family]} for family in names],
                seed=self.seed, repeats=self.draws)
        bootstrap = self.cache[key]
        contributions = [{'family_id': family, 'value': values[family],
                          'job_ids': sorted(members.get(family, []))} for family in names]
        return {'estimate': bootstrap['estimate'], 'numerator': sum(values.values()),
                'denominator': len(values), 'n_expected': len(expected),
                'n_unavailable': len(expected - values.keys()),
                'family_ids': names, 'unavailable_family_ids': sorted(expected - values.keys()),
                'job_ids': sorted({job for row in contributions for job in row['job_ids']}),
                'contributions': contributions, 'n_clusters': bootstrap['n_clusters'],
                'ci95': bootstrap['ci']}

    def explanation_condition(self, indexed, expected):
        members = {f: [r['job_id']] for f, r in indexed.items() if f in expected}
        valid = {f: int(_available(indexed.get(f), 'explanation')) for f in expected}
        dimensions = {}
        for dim in _DIMENSIONS:
            values = {f: indexed[f]['score']['dimensions'][dim]['coverage']
                      for f in expected if valid[f]
                      and indexed[f]['score']['dimensions'][dim]['coverage'] is not None}
            dimensions[dim] = self.metric(values, expected, members)
        return {'counts': _counts(indexed, expected),
                'valid_score_coverage': self.metric(valid, expected, members),
                'dimensions': dimensions}

    def differences(self, after, before, expected):
        members = {f: [r['job_id'] for r in (before.get(f), after.get(f)) if r]
                   for f in expected}
        dimensions = {}
        for dim in _DIMENSIONS:
            values = {}
            for family in sorted(expected):
                a, b = after.get(family), before.get(family)
                if not _available(a, 'explanation') or not _available(b, 'explanation'):
                    continue
                av, bv = (r['score']['dimensions'][dim]['coverage'] for r in (a, b))
                if av is not None and bv is not None:
                    values[family] = av - bv
            dimensions[dim] = self.metric(values, expected, members)
        return dimensions

    def item_changes(self, baseline, hinted, expected):
        results = {}
        for dim in _DIMENSIONS:
            repairs, harms, members, item_records = {}, {}, {}, []
            totals = dict.fromkeys(('repaired', 'repairable', 'harmed', 'harmable'), 0)
            for family in sorted(expected):
                before, after = baseline.get(family), hinted.get(family)
                if not _available(before, 'explanation') or not _available(after, 'explanation'):
                    continue
                b = {item['item_id']: item for item in before['score'].get('items', [])
                     if item['dimension'] == dim}
                a = {item['item_id']: item for item in after['score'].get('items', [])
                     if item['dimension'] == dim}
                if not b or b.keys() != a.keys():
                    continue
                counts = dict.fromkeys(totals, 0)
                members[family] = [before['job_id'], after['job_id']]
                for item_id in sorted(b):
                    old, new = b[item_id]['verdict'] == 'covered', a[item_id]['verdict'] == 'covered'
                    counts['harmable' if old else 'repairable'] += 1
                    counts['harmed'] += int(old and not new)
                    counts['repaired'] += int(not old and new)
                    item_records.append({'family_id': family, 'item_id': item_id,
                                         'before': b[item_id]['verdict'], 'after': a[item_id]['verdict'],
                                         'job_ids': sorted(members[family])})
                for key, count in counts.items():
                    totals[key] += count
                if counts['repairable']:
                    repairs[family] = counts['repaired'] / counts['repairable']
                if counts['harmable']:
                    harms[family] = counts['harmed'] / counts['harmable']
            results[dim] = {'repair': self.metric(repairs, expected, members),
                            'harm': self.metric(harms, expected, members),
                            'source_item_counts': totals, 'source_items': item_records,
                            'matched_item_family_ids': sorted(members)}
        return results

    def explanation_summary(self, indexes, expected):
        differences = {f'{a}_minus_{b}': self.differences(indexes[a], indexes[b], expected)
                       for a, b in _CONTRASTS}
        hint = differences['hinted_minus_image_only']
        gains = {dim: {r['family_id']: r['value'] for r in hint[dim]['contributions']}
                 for dim in ('K', 'R')}
        shared = gains['K'].keys() & gains['R'].keys()
        members = {f: [indexes[c][f]['job_id'] for c in ('image_only', 'hinted')]
                   for f in shared}
        return {'family_ids': sorted(expected),
                'conditions': {c: self.explanation_condition(indexes[c], expected) for c in CONDITIONS},
                'condition_differences': differences,
                'hint_gains': {'K': hint['K'], 'R': hint['R'],
                              'R_minus_K': self.metric({f: gains['R'][f] - gains['K'][f]
                                                        for f in shared}, expected, members),
                              'directions': {dim: {'improved': sum(v > 0 for v in vals.values()),
                                                   'unchanged': sum(v == 0 for v in vals.values()),
                                                   'worsened': sum(v < 0 for v in vals.values()),
                                                   'n_pairs': len(vals)} for dim, vals in gains.items()},
                              'item_repair_harm': self.item_changes(indexes['image_only'],
                                                                    indexes['hinted'], expected)}}

    def explanations(self, rows, ocr):
        models, indexes, universes = {}, {}, {}
        for model in sorted({r['model'] for r in rows}):
            selected = [r for r in rows if r['model'] == model]
            expected = {r['family_id'] for r in selected}
            indexed = {c: {r['family_id']: r for r in selected if r['condition'] == c}
                       for c in CONDITIONS}
            indexes[model], universes[model] = indexed, expected
            result = self.explanation_summary(indexed, expected)
            result['ocr_subgroups'] = {
                name: self.explanation_summary(indexed, {f for f in expected if ocr.get(f) is value})
                for name, value in (('available', True), ('unavailable', False), ('unknown', None))}
            models[model] = result
        model_differences = {}
        for before, after in combinations(sorted(models), 2):
            expected = universes[before] | universes[after]
            model_differences[f'{after}_minus_{before}'] = {
                'model_after': after, 'model_before': before,
                'conditions': {c: self.differences(indexes[after][c], indexes[before][c], expected)
                               for c in CONDITIONS}}
        return {'models': models, 'model_differences': model_differences}

    def variant(self, indexed, expected):
        valid = {f: r for f, r in indexed.items() if _available(r, 'pair')}
        members = {f: [r['job_id']] for f, r in valid.items()}
        result = {'counts': _counts(indexed, expected)}
        for key in ('cue_correct', 'relation_correct', 'joint_correct'):
            result[key] = self.metric({f: r['score'][key] for f, r in valid.items()}, expected, members)
        result['cue_relation_discrepancy'] = self.metric(
            {f: int(r['score']['cue_correct'] != r['score']['relation_correct'])
             for f, r in valid.items()}, expected, members)
        return result

    def triples(self, indexes, expected):
        complete, preserved, all_three = {}, {}, {}
        sham_members = {}
        for family in sorted(expected):
            original, edited, sham = (indexes[v].get(family) for v in _VARIANTS)
            if _available(original, 'pair') and _available(sham, 'pair'):
                a, b = original['score'], sham['score']
                preserved[family] = int(a['status'] == b['status'] == 'ok'
                                        and a['semantic_relation'] == b['semantic_relation'])
                sham_members[family] = [original['job_id'], sham['job_id']]
            if not all(_available(r, 'pair') for r in (original, edited, sham)):
                continue
            complete[family] = [r['job_id'] for r in (original, edited, sham)]
            all_three[family] = int(all(r['score']['joint_correct'] for r in (original, edited, sham))
                                    and preserved[family])
        return {'all_three_joint': self.metric(all_three, expected, complete),
                'sham_preservation': self.metric(preserved, expected, sham_members),
                'complete_family_ids': sorted(complete),
                'incomplete_family_ids': sorted(expected - complete.keys()),
                'interpretation': 'Exploratory family-level estimates; no mechanism rankings.'}

    def permutations(self, original, repeated, expected):
        complete, valid, members, agreement = [], [], {}, {}
        for family in sorted(expected):
            a, b = original.get(family), repeated.get(family)
            if not _available(a, 'pair') or not _available(b, 'pair'):
                continue
            complete.append(family)
            if a['score']['status'] != 'ok' or b['score']['status'] != 'ok':
                continue
            valid.append(family)
            members[family] = [a['job_id'], b['job_id']]
            agreement[family] = int(a['score']['semantic_relation'] == b['score']['semantic_relation'])
        return {'semantic_agreement': self.metric(agreement, expected, members),
                'complete_answer_pair_family_ids': complete, 'valid_answer_pair_family_ids': valid,
                'invalid_answer_pair_family_ids': sorted(set(complete) - set(valid)),
                'incomplete_answer_pair_family_ids': sorted(expected - set(complete)),
                'original_counts': _counts(original, expected),
                'permutation_counts': _counts(repeated, expected)}

    def paired(self, rows, split):
        models = {}
        for model in sorted({r['model'] for r in rows}):
            selected = [r for r in rows if r['model'] == model]
            expected = {r['family_id'] for r in selected}
            indexed = {v: {r['family_id']: r for r in selected
                           if r['task_kind'] == 'pair' and r['condition'] == v} for v in _VARIANTS}
            repeats = {r['family_id']: r for r in selected if r['task_kind'] == 'permutation'}
            # Every test triple has a planned original permutation. Dev does not.
            permutation_expected = expected if split == 'test' else set(repeats)
            models[model] = {'family_ids': sorted(expected),
                             'variants': {v: self.variant(indexed[v], expected) for v in _VARIANTS},
                             'triples': self.triples(indexed, expected),
                             'permutations': self.permutations(indexed['original'], repeats,
                                                              permutation_expected)}
        return {'models': models}


def analyze_reference_scores(rows: list[dict], *, seed=6000, draws=10000) -> dict:
    """Analyze all supplied planned rows; test is primary and dev is separate."""
    if type(seed) is not int or type(draws) is not int or draws < 1:
        raise ValueError('seed must be integer and draws must be a positive integer')
    ocr = _validate(rows)
    engine, splits = _Report(seed, draws), {}
    for split in ('test', 'dev'):
        selected = sorted((r for r in rows if r['split'] == split), key=lambda r: r['job_id'])
        families = sorted({r['family_id'] for r in selected})
        splits[split] = {'n_rows': len(selected), 'n_families': len(families),
                         'family_ids': families, 'job_ids': [r['job_id'] for r in selected],
                         'explanation': engine.explanations(
                             [r for r in selected if r['task_kind'] == 'explanation'], ocr),
                         'paired': engine.paired(
                             [r for r in selected if r['task_kind'] != 'explanation'], split)}
    return {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'primary_split': 'test',
            'seed': seed, 'draws': draws, 'bootstrap_unit': 'family_id',
            'input_scope': 'Supplied planned rows; wholly absent families/models cannot be inferred.',
            'measurement_scope': 'Descriptive automatic scores; no human or GPU validation implied.',
            'n_input_rows': len(rows), 'splits': splits}
