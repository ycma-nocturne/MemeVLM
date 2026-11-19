"""Apply documented corrections to reference explanations and missing OCR values.

Preserve source identities and checklists. Use source values when the manifest
is absent.
"""
from __future__ import annotations

import copy
import re

from .data import content_hash


SCHEMA = 'project_reference_errata_v1'
_FIELDS = {'reference.explanation', 'ocr_text'}


def _digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{' + str(length) + r'}', value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _indexed_entries(records, manifest):
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'metadata_sha256', 'source_revision', 'entries'}:
        raise ValueError('source_errata requires schema, metadata_sha256, source_revision and entries')
    if (manifest['schema'] != SCHEMA or not _digest(manifest['metadata_sha256'])
            or not _digest(manifest['source_revision'], 40)):
        raise ValueError('source_errata schema or pinned metadata identity is invalid')
    entries = manifest['entries']
    if not isinstance(entries, list) or not entries:
        raise ValueError('source_errata entries must be a nonempty list')
    by_source = {r['source_id']: r for r in records}
    result, seen = {}, set()
    keys = {'source_id', 'source_record_sha256', 'field', 'before', 'after', 'reason', 'evidence'}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != keys:
            raise ValueError('each source erratum requires exact field, old/new values, reason and evidence')
        source, field = entry['source_id'], entry['field']
        if type(source) is not int or source not in by_source or not isinstance(field, str) or field not in _FIELDS:
            raise ValueError('source erratum has unknown source ID or unsupported field')
        if (source, field) in seen:
            raise ValueError('duplicate source erratum field')
        seen.add((source, field))
        identity = by_source[source]['source']
        if (not _digest(entry['source_record_sha256'])
                or entry['source_record_sha256'] != identity['record_sha256']
                or manifest['metadata_sha256'] != identity['metadata_sha256']
                or manifest['source_revision'] != identity['revision']):
            raise ValueError('source erratum is bound to a different source record or release')
        if not _text(entry['reason']) or not _text(entry['evidence']):
            raise ValueError('source erratum requires an explicit reason and evidence pointer')
        if field == 'reference.explanation':
            if not _text(entry['before']) or not _text(entry['after']) or entry['before'] == entry['after']:
                raise ValueError('reference erratum requires distinct nonempty explanations')
        elif (not _text(entry['before']) or entry['before'].strip().lower() != 'null'
              or entry['after'] is not None):
            raise ValueError('OCR erratum may only map an explicitly audited null sentinel to missing')
        result.setdefault(source, []).append(entry)
    return result


def _value(row, field):
    return row['ocr_text'] if field == 'ocr_text' else row['reference']['explanation']


def _receipt(manifest, entries):
    return {'schema': SCHEMA, 'manifest_sha256': content_hash(manifest), 'entries': copy.deepcopy(entries)}


def apply_source_errata(records, manifest=None):
    """Return adapted copies only after matching exact unmodified source values."""
    if manifest is None:
        return records
    grouped = _indexed_entries(records, manifest)
    result = copy.deepcopy(records)
    for row in result:
        if 'project_errata' in row['provenance']:
            raise ValueError('source errata cannot be applied twice')
        entries = grouped.get(row['source_id'], [])
        for entry in entries:
            if _value(row, entry['field']) != entry['before']:
                raise ValueError('source erratum before value differs from the public source')
            if entry['field'] == 'ocr_text':
                row['ocr_text'] = entry['after']
            else:
                row['reference']['explanation'] = entry['after']
        if entries:
            row['provenance']['project_errata'] = _receipt(manifest, entries)
    return result


def verify_applied_errata(records, manifest=None):
    """Check packaged records against the same adaptation contract at runtime."""
    grouped = {} if manifest is None else _indexed_entries(records, manifest)
    for row in records:
        entries = grouped.get(row['source_id'], [])
        receipt = row['provenance'].get('project_errata')
        if not entries:
            if receipt is not None:
                raise ValueError('record contains project errata absent from configuration')
            continue
        if receipt != _receipt(manifest, entries):
            raise ValueError('project errata receipt does not match the configured corrections')
        for entry in entries:
            if _value(row, entry['field']) != entry['after']:
                raise ValueError('effective source value does not match its recorded correction')
