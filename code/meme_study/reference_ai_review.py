"""Adopt AI material recommendations with content-bound evidence.

Record an adopted correction only when the current pair matches the documented edit.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from .data import content_hash
from .reference_protocol import _families, _validate_pair, pair_content_hash
from .reference_materials import _bytes, _destination, _snapshot, _write

MODEL = 'gpt-6.1-terra'
FIELDS = ['family_id', 'title', 'split', 'content_sha256', 'ai_model',
          'initial_ai_recommendation', 'ai_recommendation', 'confidence',
          'suggested_action', 'review_type']


def _capture(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'base64': base64.b64encode(raw).decode('ascii')}


def _raw(obj):
    if not isinstance(obj, dict) or set(obj) != {'sha256', 'base64'}:
        raise ValueError('invalid evidence envelope')
    raw = base64.b64decode(obj['base64'], validate=True)
    if hashlib.sha256(raw).hexdigest() != obj['sha256']:
        raise ValueError('evidence hash mismatch')
    return raw


def _rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    if reader.fieldnames != FIELDS:
        raise ValueError('unexpected AI review columns')
    rows = list(reader)
    if any(set(row) != set(FIELDS) or any(v is None for v in row.values()) for row in rows):
        raise ValueError('malformed AI review row')
    ids = [r['family_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate AI review family')
    return {r['family_id']: r for r in rows}


def _verify(pair, check):
    if (set(check) != {'role', 'model', 'prior_recommendation', 'decision', 'content_sha256', 'evidence'} or
            check.get('role') != 'ai' or check.get('model') != MODEL or
            check.get('content_sha256') != pair_content_hash(pair)):
        raise ValueError('invalid AI identity or current content binding')
    ev = check['evidence']
    if not isinstance(ev, dict): raise ValueError('missing evidence')
    rows = _rows(_raw(ev['review_csv']))
    old_pairs = json.loads(_raw(ev['reviewed_pairs']).decode('utf-8'))
    old = _families(old_pairs).get(pair['family_id'])
    row = rows.get(pair['family_id'])
    if not old or not row or old.get('material_check') is not None:
        raise ValueError('missing original reviewed pair')
    if (row['content_sha256'] != pair_content_hash(old) or row['split'] != old['split'] or
            row['ai_model'] != MODEL or row['review_type'] != 'AI-assisted recommendation, not human acceptance' or
            row['initial_ai_recommendation'] not in {'accept', 'revise', 'reject'} or
            not row['confidence'].strip() or
            (row['ai_recommendation'] == 'revise' and not row['suggested_action'].strip()) or
            row['ai_recommendation'] not in {'accept', 'revise', 'reject'} or
            check.get('prior_recommendation') != row['ai_recommendation']):
        raise ValueError('invalid or stale original AI recommendation')
    current = {k: v for k, v in pair.items() if k != 'material_check'}
    prior = {k: v for k, v in old.items() if k != 'material_check'}
    if row['ai_recommendation'] == 'accept':
        if check.get('decision') != 'adopted' or current != prior or set(ev) != {'review_csv', 'reviewed_pairs'}:
            raise ValueError('accepted recommendation does not match current pair')
    elif row['ai_recommendation'] == 'revise':
        if check.get('decision') != 'adopted_correction_applied' or set(ev) != {
                'review_csv', 'reviewed_pairs', 'correction_candidates', 'correction_notes', 'change_record'}:
            raise ValueError('revise requires explicit correction evidence')
        candidates = json.loads(_raw(ev['correction_candidates']).decode('utf-8'))
        notes = _raw(ev['correction_notes']).decode('utf-8')
        change = json.loads(_raw(ev['change_record']).decode('utf-8'))['pair_change']
        matches = [c for c in candidates['changes'] if c == change and c.get('scope') == 'B'
                   and c.get('source_id') == old.get('source_id')]
        if len(matches) != 1 or str(old['source_id']) not in notes:
            raise ValueError('correction evidence does not match source')
        field = change.get('field', '')
        import re
        match = re.fullmatch(r'options\[semantic_id=([A-Za-z0-9_-]+)\]\.text', field)
        if not match or match.group(1) not in notes or not change.get('reason'):
            raise ValueError('correction field lacks evidence')
        corrected = copy.deepcopy(prior)
        opts = [o for o in corrected['options'] if o['semantic_id'] == match.group(1)]
        if len(opts) != 1 or opts[0]['text'] != change.get('before') or not change.get('after'):
            raise ValueError('correction before text mismatch')
        opts[0]['text'] = change['after']
        if corrected != current:
            raise ValueError('current pair includes unreviewed changes')
    else:
        raise ValueError('reject cannot be adopted')
    return True


def valid_ai_check(pair, check):
    try:
        return _verify(pair, check)
    except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError, csv.Error):
        return False


def import_ai_review(*, review_csv, reviewed_pairs, current_pairs, output_dir,
                     correction_candidates=None, correction_notes=None, change_record=None):
    destination = _destination(output_dir)
    source_csv, source_pairs = _capture(review_csv), _capture(reviewed_pairs)
    rows = _rows(_raw(source_csv))
    old = _snapshot(json.loads(_raw(source_pairs).decode('utf-8')))
    pairs = _snapshot(json.loads(Path(current_pairs).read_text(encoding='utf-8')))
    if any(p.get('material_check') is not None for p in pairs):
        raise ValueError('existing checks cannot be overwritten')
    if set(rows) != set(_families(old)) or set(rows) != set(_families(pairs)):
        raise ValueError('reviews must cover exactly the old and current pairs')
    correction_paths = (correction_candidates, correction_notes, change_record)
    if any(correction_paths) and not all(correction_paths):
        raise ValueError('all three correction evidence files required')
    correction = dict(zip(('correction_candidates','correction_notes','change_record'),
                          map(_capture, correction_paths))) if all(correction_paths) else {}
    for pair in pairs:
        row = rows[pair['family_id']]
        recommendation = row['ai_recommendation']
        ev = {'review_csv': source_csv, 'reviewed_pairs': source_pairs}
        if recommendation == 'revise': ev.update(correction)
        check = {'role':'ai', 'model':MODEL, 'prior_recommendation':recommendation,
                 'decision':'adopted_correction_applied' if recommendation == 'revise' else 'adopted',
                 'content_sha256':pair_content_hash(pair), 'evidence':ev}
        pair['material_check'] = check
        if not valid_ai_check(pair, check):
            raise ValueError(f"{pair['family_id']}: stale, rejected, or unresolved AI review")
    manifest = {'schema_version':1, 'mode':'ai_material_adoption', 'model':MODEL,
                'pair_count':len(pairs), 'ready_pair_count':sum(_validate_pair(p) for p in pairs),
                'source_csv_sha256':source_csv['sha256'], 'reviewed_pairs_sha256':source_pairs['sha256'],
                'current_pairs_content_sha256':content_hash(json.loads(Path(current_pairs).read_text(encoding='utf-8'))),
                'human_decisions_created':0, 'model_calls':0}
    return _write(destination, {'pairs.json':_bytes(pairs)}, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('review-csv','reviewed-pairs','current-pairs','output-dir'):
        parser.add_argument('--'+name, required=True)
    for name in ('correction-candidates','correction-notes','change-record'):
        parser.add_argument('--'+name)
    try:
        result = import_ai_review(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({'error':str(exc)}), file=sys.stderr); return 2
    print(json.dumps(result, ensure_ascii=False)); return 0

if __name__ == '__main__':
    raise SystemExit(main())
