"""Score reference checklist coverage and paired answers.

Mark judge failures as unavailable. Check the presence of cited spans in the
candidate text. Assign zero correctness credit to malformed paired responses.
"""
from __future__ import annotations

import hashlib
import json


_DIMENSIONS = ('V', 'I', 'K', 'R')
_VERDICTS = {'covered', 'not_covered', 'uncertain'}
_CUES = {'yes', 'no', 'insufficient'}
_LABELS = {'A', 'B', 'C', 'D'}


def _sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None


def _sha_value(value):
    return _sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(',', ':'), allow_nan=False))


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f'{name} must contain exactly {", ".join(sorted(fields))}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'non-JSON constant: {value}')


def _decode(text):
    if not isinstance(text, str):
        raise ValueError('response must be a string')
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _check_truncated(truncated):
    if type(truncated) is not bool:
        raise ValueError('truncated must be a boolean')


def _check_checklist(checklist):
    if not isinstance(checklist, list):
        raise ValueError('checklist must be a list')
    seen, counts = set(), dict.fromkeys(_DIMENSIONS, 0)
    for item in checklist:
        _object(item, {'item_id', 'dimension', 'text'}, 'checklist item')
        if not _nonempty_string(item['item_id']) or item['item_id'] in seen:
            raise ValueError('checklist item IDs must be nonempty and unique')
        if not isinstance(item['dimension'], str) or item['dimension'] not in _DIMENSIONS:
            raise ValueError('checklist dimension must be V, I, K or R')
        if not _nonempty_string(item['text']):
            raise ValueError('checklist text must be a nonempty string')
        seen.add(item['item_id'])
        counts[item['dimension']] += 1
    return seen, counts


def _unavailable(counts, reason, hashes):
    return {'status': 'unavailable', 'failure_reason': reason, 'items': [],
            'dimensions': {
                dimension: {'applicable': counts[dimension], 'covered': None,
                            'uncertain': None, 'coverage': None, 'uncertainty_rate': None}
                for dimension in _DIMENSIONS},
            'source_sha256': hashes}


def score_checklist(candidate_text: str, checklist: list[dict], judge_text: str,
                    *, truncated=False) -> dict:
    """Validate all judge items atomically, then compute dimension coverage.

    Coverage is covered/applicable; uncertain items remain in that denominator
    and also contribute to uncertainty_rate. Empty dimensions are NA. All quote
    spans, including those on uncertain/not_covered items, must occur verbatim.
    Invalid caller-supplied checklist metadata raises ValueError; invalid judge
    output produces an unavailable record with no partial coverage credit.
    """
    _check_truncated(truncated)
    if not isinstance(candidate_text, str):
        raise ValueError('candidate_text must be a string')
    expected, counts = _check_checklist(checklist)
    hashes = {'candidate': _sha_text(candidate_text), 'checklist': _sha_value(checklist),
              'judge': _sha_text(judge_text)}
    if truncated:
        return _unavailable(counts, 'judge response was truncated', hashes)
    try:
        value = _decode(judge_text)
        _object(value, {'items'}, 'judge response')
        if not isinstance(value['items'], list):
            raise ValueError('judge items must be a list')
        if len(value['items']) != len(expected):
            raise ValueError('judge must return every expected item exactly once')
        by_id = {}
        for item in value['items']:
            _object(item, {'item_id', 'verdict', 'evidence'}, 'judge item')
            item_id = item['item_id']
            if not isinstance(item_id, str) or item_id not in expected:
                raise ValueError('judge returned an unknown item ID')
            if item_id in by_id:
                raise ValueError('judge repeated an item ID')
            verdict = item['verdict']
            if not isinstance(verdict, str) or verdict not in _VERDICTS:
                raise ValueError('judge verdict must be covered, not_covered or uncertain')
            evidence = item['evidence']
            if not isinstance(evidence, list):
                raise ValueError('judge evidence must be a list of verbatim spans')
            if verdict == 'covered' and not evidence:
                raise ValueError('covered requires at least one nonempty verbatim span')
            for span in evidence:
                if not _nonempty_string(span) or span not in candidate_text:
                    raise ValueError('judge evidence is empty, mistyped or absent from candidate text')
            by_id[item_id] = item
        if set(by_id) != expected:
            raise ValueError('judge item IDs differ from the expected checklist')
    except (ValueError, RecursionError) as exc:
        return _unavailable(counts, str(exc), hashes)

    dimensions = {
        dimension: {'applicable': counts[dimension], 'covered': 0, 'uncertain': 0,
                    'coverage': None, 'uncertainty_rate': None}
        for dimension in _DIMENSIONS}
    items = []
    for reference in checklist:
        item = by_id[reference['item_id']]
        dimension = reference['dimension']
        if item['verdict'] in {'covered', 'uncertain'}:
            dimensions[dimension][item['verdict']] += 1
        items.append({'item_id': reference['item_id'], 'dimension': dimension,
                      'verdict': item['verdict'], 'evidence': list(item['evidence'])})
    for row in dimensions.values():
        if row['applicable']:
            row['coverage'] = row['covered'] / row['applicable']
            row['uncertainty_rate'] = row['uncertain'] / row['applicable']
    return {'status': 'ok', 'failure_reason': None, 'items': items,
            'dimensions': dimensions, 'source_sha256': hashes}


def _check_pair_materials(options, gold):
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError('options must contain exactly four records')
    by_label, semantics = {}, set()
    for option in options:
        _object(option, {'label', 'semantic_id', 'text'}, 'option')
        label, semantic_id = option['label'], option['semantic_id']
        if not isinstance(label, str) or label not in _LABELS or label in by_label:
            raise ValueError('option labels must contain A, B, C and D exactly once')
        if not _nonempty_string(semantic_id) or semantic_id in semantics:
            raise ValueError('option semantic IDs must be nonempty and unique')
        if not _nonempty_string(option['text']):
            raise ValueError('option text must be a nonempty string')
        by_label[label] = semantic_id
        semantics.add(semantic_id)
    _object(gold, {'cue_answers', 'relation_answers'}, 'gold')
    for name, allowed in (('cue_answers', _CUES), ('relation_answers', semantics)):
        answers = gold[name]
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or answer not in allowed for answer in answers)
                or len(set(answers)) != len(answers)):
            raise ValueError(f'{name} must be a nonempty list of distinct permitted answers')
    return by_label


def _pair_error(status, reason, hashes):
    return {'status': status, 'cue_correct': 0, 'relation_correct': 0,
            'joint_correct': 0, 'cue': None, 'semantic_relation': None,
            'explanation': None, 'reason': reason, 'source_sha256': hashes}


def score_pair(raw_text: str, options: list[dict], gold: dict, *, truncated=False) -> dict:
    """Score one independent cue/relation answer by accepted semantic IDs.

    Options and gold are validated even for malformed or truncated responses;
    material configuration mistakes raise ValueError rather than becoming model
    errors. A malformed response receives zero for both components, with its
    failure retained separately. Correct cue alone never earns joint success.
    """
    _check_truncated(truncated)
    by_label = _check_pair_materials(options, gold)
    hashes = {'response': _sha_text(raw_text), 'options': _sha_value(options),
              'gold': _sha_value(gold)}
    if truncated:
        return _pair_error('truncated', 'candidate response was truncated', hashes)
    try:
        value = _decode(raw_text)
        _object(value, {'cue', 'relation', 'explanation'}, 'paired response')
        if not isinstance(value['cue'], str) or value['cue'] not in _CUES:
            raise ValueError('cue must be yes, no or insufficient')
        if not isinstance(value['relation'], str) or value['relation'] not in by_label:
            raise ValueError('relation must be A, B, C or D')
        if not _nonempty_string(value['explanation']):
            raise ValueError('explanation must be a nonempty string')
    except (ValueError, RecursionError) as exc:
        return _pair_error('invalid', str(exc), hashes)
    semantic_relation = by_label[value['relation']]
    cue_correct = int(value['cue'] in gold['cue_answers'])
    relation_correct = int(semantic_relation in gold['relation_answers'])
    return {'status': 'ok', 'cue_correct': cue_correct, 'relation_correct': relation_correct,
            'joint_correct': cue_correct * relation_correct, 'cue': value['cue'],
            'semantic_relation': semantic_relation, 'explanation': value['explanation'],
            'reason': None, 'source_sha256': hashes}
