"""Export B triples and a blank review form with one row per triple.

Import the supplied CSV decisions and declared reviewer names.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote

from .data import content_hash
from .freeze import _publish
from .image_io import decode_image_receipt
from .intake import REPOSITORY
from .reference_protocol import PROTOCOL_ID, _families, _validate_pair, pair_content_hash


_FIELDS = ('family_id', 'split', 'content_sha256', 'decision', 'reviewer', 'notes')
_VARIANTS = ('original', 'edited', 'sham')
_LABELS = {'original': '原图 · B 原始版本', 'edited': '目标编辑 · edited', 'sham': '对照编辑 · sham'}
_RUBRIC = '''每组由一名真实的第二成员检查一次，并在 review.csv 中填一行。
accept：三图可读，目标事实与关系变化有图像依据；sham 保留原关系；各版本的题目、选项和拟定答案均可辩护，没有影响判断的明显伪影或歧义。
revise：有可修复问题，例如目标仍残留、文字泄漏答案、sham 改变含义、答案不唯一或合成痕迹影响判断；notes 写清问题及修订建议。
reject：该组三图无法形成可解释的对照或现有证据无法支持拟定答案；notes 写明原因，后续替换材料。
不要为了凑齐15组而接受不明确材料。reviewer 填实际检查者姓名或稳定标识；不预填或代填成员决定。'''


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _bytes(value):
    return (_json(value) + '\n').encode('utf-8')


def _escape(value):
    return html.escape(_json(value) if isinstance(value, (dict, list)) else str(value), quote=True)


def _snapshot(pairs):
    pairs = json.loads(_json(pairs))
    _families(pairs)
    for pair in pairs:
        _validate_pair(pair)
        if type(pair.get('source_id')) is not int or pair['source_id'] < 1:
            raise ValueError('pairs require positive integer source_id attribution')
        if not isinstance(pair.get('preparation'), dict):
            raise ValueError('pairs require retained preparation notes')
    return pairs


def _destination(output_dir):
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f'material output already exists: {destination}')
    parent = destination.parent.resolve(strict=True)
    return parent / destination.name


def _write(destination, artifacts, manifest):
    manifest['artifact_sha256'] = {name: _sha(raw) for name, raw in artifacts.items()}
    artifacts['manifest.json'] = _bytes(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
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
    return manifest


def _local_bytes(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or ':' in relative or '\x00' in relative or PurePosixPath(relative).is_absolute()
            or '..' in PurePosixPath(relative).parts):
        raise ValueError('source path must remain relative to data_root')
    try:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('source file escapes data_root or is not a regular file')
        return path.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'unreadable local source: {relative}') from exc


def _sources(pairs, root):
    """Use the existing pinned acquisition receipt, without fetching any source."""
    base = 'candidates/raw/memebench-metadata'
    receipt_raw = _local_bytes(root, base + '.receipt.json')
    receipt = json.loads(receipt_raw)
    raw = _local_bytes(root, base + '.jsonl')
    if _sha(raw) != receipt.get('sha256'):
        raise ValueError('public metadata sha256 mismatch')
    prefix = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/'
    match = re.fullmatch(re.escape(prefix) + r'([0-9a-f]{40})/metadata\.jsonl', receipt.get('source_url', ''))
    if match is None:
        raise ValueError('source receipt requires the pinned MemeBench metadata URL')
    revision = match.group(1)
    indexed = {}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item = row.get('id') if isinstance(row, dict) else None
        if type(item) is not int or item < 1 or item in indexed:
            raise ValueError('public metadata has missing or duplicate source IDs')
        indexed[item] = row
    sources = {}
    for pair in pairs:
        item = pair['source_id']
        if item not in indexed:
            raise ValueError(f'missing public source ID: {item}')
        source = indexed[item]
        path = source.get('image_path')
        if (not isinstance(path, str) or not path.startswith('images/') or '\\' in path
                or ':' in path or '..' in PurePosixPath(path).parts):
            raise ValueError('invalid public image attribution path')
        sources[pair['family_id']] = {
            'dataset': REPOSITORY, 'source_id': item, 'revision': revision,
            'metadata_sha256': _sha(raw), 'record_sha256': content_hash(source),
            'acquisition_receipt_sha256': _sha(receipt_raw),
            'url': f'https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/metadata.jsonl',
            'image_url': prefix + revision + '/' + quote(path),
            'image_scope': 'B project-prepared triple; not the A unadapted public original',
        }
    return sources


def _notes(pair):
    notes = pair['preparation'].get('draft_notes', {})
    if not isinstance(notes, dict):
        notes = {}
    parts = []
    for key, label in (('proposed_edit', '目标编辑说明'), ('proposed_sham', 'sham 说明')):
        value = notes.get(key)
        if value is None:
            retained = [(name, value[key]) for name, value in notes.items()
                        if isinstance(value, dict) and key in value]
            if retained:
                value = {name: text for name, text in retained}
                label += '（留存设计，须对照当前三图）'
            else:
                value = notes.get('replacement_control',
                                  '未单列说明；请结合当前三图、风险和下方完整制作记录检查。')
        parts.append(f'<p><strong>{label}：</strong>{_escape(value)}</p>')
    for key, label in (('risk', '已有风险'), ('local_composite_limits', '制作局限'),
                       ('caption_leakage', '文字线索'), ('competing_readings', '其他读法')):
        if notes.get(key):
            parts.append(f'<p><strong>{label}：</strong>{_escape(notes[key])}</p>')
    return ''.join(parts)


def _page(pairs, sources, images):
    bindings = {(r['family_id'], r['variant']): r for r in images}
    nav, sections = [], []
    for index, pair in enumerate(pairs, 1):
        family = pair['family_id']
        title = pair['preparation'].get('draft_notes', {}).get('title', family)
        nav.append(f'<a href="#pair-{index}">{index:02d} · {_escape(title)}</a>')
        options = {o['semantic_id']: o for o in pair['options']}
        figures = []
        for variant in _VARIANTS:
            item = pair['variants'][variant]
            receipt = bindings[(family, variant)]
            answers = '；'.join(f'{options[a]["label"]} · {a} · {options[a]["text"]}'
                               for a in item['relation_answers'])
            cue = ' / '.join({'yes': 'yes（是）', 'no': 'no（否）',
                             'insufficient': 'insufficient（证据不足）'}[a] for a in item['cue_answers'])
            asset = _escape(receipt['asset_path'])
            figures.append(f'<figure><figcaption>{_LABELS[variant]}</figcaption>'
                           f'<a href="{asset}" target="_blank" rel="noopener"><img src="{asset}" '
                           f'alt="{_escape(family)} {_LABELS[variant]}"></a>'
                           f'<p class="small">{receipt["width"]} × {receipt["height"]} · 点击查看原尺寸</p>'
                           f'<div class="answer"><strong>拟定事实答案：</strong>{cue}<br>'
                           f'<strong>拟定关系答案：</strong>{_escape(answers)}</div></figure>')
        option_rows = ''.join(f'<li><strong>{_escape(o["label"])} · {_escape(o["semantic_id"])}：</strong>'
                              f'{_escape(o["text"])}</li>' for o in sorted(pair['options'], key=lambda x: x['label']))
        source = sources[family]
        sections.append(f'<article id="pair-{index}"><p class="tag">{index:02d} / {len(pairs)} · '
                        f'{_escape(pair["split"])} · {_escape(family)}</p><h2>{_escape(title)}</h2>'
                        f'<p><strong>事实问题：</strong>{_escape(pair["cue_question"])}</p>'
                        f'<p><strong>关系问题：</strong>{_escape(pair["relation_question"])}</p>'
                        f'<ol class="options">{option_rows}</ol><div class="triple">{"".join(figures)}</div>'
                        '<div class="notes"><h3>制作说明与待检查风险</h3>' + _notes(pair) + '</div>'
                        f'<p class="source">公开来源：MemeBench · source_id {source["source_id"]} · '
                        f'<a href="{_escape(source["url"])}">固定版本元数据</a> · '
                        f'<a href="{_escape(source["image_url"])}">公开原始图像出处</a>。'
                        '上方只展示本项目 B 三图，可能包含既有清理、裁切或重编码；A 的公共原图另行保留。</p>'
                        f'<details><summary>来源、内容校验值与完整制作记录</summary><p>版本：'
                        f'{source["revision"]}</p><p>content_sha256：<code>{pair_content_hash(pair)}</code></p>'
                        f'<pre>{_escape({"source": source, "preparation": pair["preparation"], "material_check": pair.get("material_check")})}</pre></details>'
                        '<p class="decision">在 CSV 同名行填写：decision = accept / revise / reject；'
                        'reviewer = 实际检查者；notes = 问题与建议（revise / reject 必填）。</p></article>')
    counts = Counter(p['split'] for p in pairs)
    return ('''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B 配对材料 · 第二成员检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:#18252d;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
main{max-width:1520px;margin:auto;padding:32px 28px 60px}h1{font-size:32px;margin:0 0 12px}h2{margin:4px 0 18px;font-size:25px}h3{font-size:18px;margin:0 0 10px}p{margin:9px 0}a{color:#126b72;text-underline-offset:3px}header,article{background:#fff;border:1px solid #d9e2e5;border-radius:14px;padding:25px;margin:0 0 25px}article{scroll-margin-top:20px}.scope{font-size:18px}.badge,.tag{color:#526873;font-weight:600}.rubric{white-space:pre-line;background:#eef6f5;border-left:4px solid #288c85;padding:14px 20px;margin:18px 0}nav{display:flex;flex-wrap:wrap;gap:8px 20px;margin:18px 0 0}nav a{font-size:14px}.options{list-style:none;padding:12px 18px;background:#f3f6f8;border-radius:8px}.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:20px 0}figure{margin:0;min-width:0;border:1px solid #dce3e7;border-radius:10px;padding:12px;align-self:start}figcaption{font-weight:700;margin-bottom:12px}img{display:block;width:100%;height:auto;max-height:520px;object-fit:contain;background:#fff}.small{font-size:12px;color:#657880}.answer{background:#f0f5f8;padding:10px 12px;font-size:14px}.notes{padding:16px 20px;background:#faf8f1;border-radius:8px;overflow-wrap:anywhere}.notes p{white-space:pre-wrap}.source{font-size:14px;color:#526873;margin-top:18px}details{border-top:1px solid #e2e7ea;padding:12px 0;font-size:13px}summary{cursor:pointer;color:#526873}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f8;padding:16px}code{overflow-wrap:anywhere}.decision{font-size:14px;border-top:1px solid #e2e7ea;padding-top:12px}.boundary{color:#596d76;font-size:14px}@media(max-width:850px){main{padding:18px 12px}.triple{grid-template-columns:1fr}header,article{padding:18px}h1{font-size:27px}}@media print{body{background:#fff}nav{display:none}article{break-before:page}.triple{grid-template-columns:repeat(3,minmax(0,1fr))}details{display:none}}
</style></head><body><main><header><p class="badge">REFERENCE REUSE · B MATERIAL CHECK</p>
<h1>只检查这批配对材料</h1>'''
            + f'<p class="scope">{len(pairs)} 组三图 · {counts["dev"]} dev / {counts["test"]} test · '
            f'{len(images)} 张原字节图像</p>'
            + '<p>本页是给真实第二成员的材料检查包。60 张 A 公共原图无需重新全量验收；'
            '30 条模型解释抽查是另一个阶段，待真实输出后进行。</p>'
            + '<p>当前答案是待核验草稿。只检查图像、问题、选项、答案与对照关系；'
            '答案改变本身不等于干预有效，也不要求原有幽默全部消失。</p>'
            + '<div class="rubric">' + _escape(_RUBRIC) + '</div>'
            + '<p><a href="review.csv" download>下载空白 review.csv</a> · '
            '<a href="pairs.json">查看材料快照</a> · <a href="README.txt">填写与回传说明</a></p>'
            + '<p class="boundary">离线查看，无模型推理、自动评分或自动人工接受。图片只是逐字节复制；'
            '来源、已有风险和历史制作记录均保留。导入审核表仅记录提交者声明，不认证人员身份。</p>'
            + '<nav>' + ''.join(nav) + '</nav></header>' + ''.join(sections)
            + '</main></body></html>').encode('utf-8')


def export_pair_review(pairs, *, data_root, output_dir):
    """Create a new portable handoff; return its manifest. No acceptance is added.

    data_root contains pair images and the existing candidates/raw pinned
    MemeBench metadata/receipt. Output parent must exist; output must not exist.
    """
    destination = _destination(output_dir)
    pairs = _snapshot(pairs)
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('data_root must be a directory')
    sources = _sources(pairs, root)
    assets, images = {}, []
    for index, pair in enumerate(pairs, 1):
        for variant in _VARIANTS:
            item = pair['variants'][variant]
            raw = _local_bytes(root, item['image_path'])
            receipt = decode_image_receipt(raw, source_name=item['image_path'],
                                           expected_sha256=item['image_sha256'])
            extension = {'jpeg': 'jpg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}[receipt['detected_format']]
            asset = f'images/{index:02d}-{variant}.{extension}'
            assets[asset] = raw
            images.append({'family_id': pair['family_id'], 'variant': variant,
                           'source_path': item['image_path'], 'asset_path': asset, **receipt})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=_FIELDS)
    writer.writeheader()
    for pair in pairs:
        writer.writerow({'family_id': pair['family_id'], 'split': pair['split'],
                         'content_sha256': pair_content_hash(pair),
                         'decision': '', 'reviewer': '', 'notes': ''})
    readme = ('B 配对材料第二成员检查\n\n打开 index.html；保持整个目录完整即可离线查看。\n'
              + _RUBRIC + '\n\n只填 review.csv 的 decision、reviewer、notes 三列；其余列保持不变。\n'
              '保存为 UTF-8 CSV，回传填写完成的文件。每组只需一个真实第二成员决定；不能用 AI 检查代替。\n'
              '全部行须有明确决定；导入会拒绝空白、重复、缺失、过期内容或有歧义的记录。\n'
              'revise/reject 保持待处理，不会被当作 accept。修订时创建材料新副本，重新导出并核验新内容。\n'
              '导入生成新的 pairs.json 和原始 CSV 字节副本及校验值；不改源数据或准备包。\n'
              '协调者须另行重新 prepare 新目录；仅 accept 且当前内容匹配的 B 材料可通过现有执行门槛。\n'
              '姓名是提交者声明，工具不认证身份，也不能验证检查者确为独立第二成员。\n\n'
              '60 张 A 公共原图不在本次人工验收队列；30 条解释抽查待真实输出后进行。\n'
              '当前包没有模型回答，没有产生新的人工作答或接受记录。答案与语义风险保持原稿。\n'
              '本地图片来源为 MemeBench 及其项目派生材料；保留固定来源版本与制作出处。\n'
              '仅供本地课程材料检查，导出不授予新传播许可；公开元数据出处不等于图片权利证明。\n')
    artifacts = {**assets, 'index.html': _page(pairs, sources, images),
                 'review.csv': stream.getvalue().encode('utf-8-sig'),
                 'pairs.json': _bytes(pairs), 'sources.json': _bytes(sources),
                 'README.txt': readme.encode('utf-8')}
    manifest = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'mode': 'pair_material_handoff',
                'pair_count': len(pairs), 'split_counts': dict(Counter(p['split'] for p in pairs)),
                'image_count': len(images), 'images': images, 'model_calls': 0,
                'human_decisions_created': 0, 'source_pairs_sha256': content_hash(pairs),
                'pair_content_sha256': {p['family_id']: pair_content_hash(p) for p in pairs},
                'existing_material_checks': sum(p.get('material_check') is not None for p in pairs),
                'export_code_sha256': _sha(Path(__file__).read_bytes())}
    return _write(destination, artifacts, manifest)


def import_pair_review(pairs, *, review_csv, output_dir):
    """Record one complete user-filled CSV into a new pairs copy and receipt.

    This does not authenticate names, perform semantic checking, or prepare/run
    jobs. Re-import onto already checked pairs is rejected, including revise and
    reject. The original CSV bytes are preserved even if UTF-8 BOM is present.
    """
    destination = _destination(output_dir)
    pairs = _snapshot(pairs)
    if any(p.get('material_check') is not None for p in pairs):
        raise ValueError('repeated or existing material checks cannot be overwritten')
    raw = Path(review_csv).read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
        if reader.fieldnames != list(_FIELDS):
            raise ValueError('review CSV must retain the exact unique exported columns')
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError('review CSV must be valid UTF-8 CSV') from exc
    indexed, supplied = {p['family_id']: p for p in pairs}, {}
    for row in rows:
        if set(row) != set(_FIELDS) or any(v is None for v in row.values()):
            raise ValueError('review CSV row has missing or extra cells')
        family = row['family_id']
        if family not in indexed or family in supplied:
            raise ValueError('review CSV has unknown or repeated family_id')
        pair = indexed[family]
        if row['split'] != pair['split'] or row['content_sha256'] != pair_content_hash(pair):
            raise ValueError(f'{family}: stale or mismatched material content')
        decision, reviewer, notes = (row[k].strip() for k in ('decision', 'reviewer', 'notes'))
        if decision not in {'accept', 'revise', 'reject'}:
            raise ValueError(f'{family}: an unambiguous accept/revise/reject decision is required')
        if not reviewer or (decision != 'accept' and not notes):
            raise ValueError(f'{family}: reviewer is required; revise/reject also require notes')
        supplied[family] = {'decision': decision, 'reviewer': reviewer, 'notes': notes,
                            'role': 'human', 'content_sha256': pair_content_hash(pair),
                            'source_csv_sha256': _sha(raw), 'identity_verification': 'not_performed'}
    if set(supplied) != set(indexed):
        raise ValueError('review CSV must contain exactly one completed row per pair')
    original_hash = content_hash(pairs)
    for pair in pairs:
        pair['material_check'] = supplied[pair['family_id']]
    manifest = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'mode': 'pair_material_import',
                'pair_count': len(pairs), 'source_pairs_sha256': original_hash,
                'source_csv_sha256': _sha(raw), 'output_pairs_sha256': content_hash(pairs),
                'decision_counts': dict(Counter(c['decision'] for c in supplied.values())),
                'identity_verification': 'not_performed', 'model_calls': 0,
                'ready_pair_count': sum(_validate_pair(p) for p in pairs),
                'import_code_sha256': _sha(Path(__file__).read_bytes())}
    return _write(destination, {'pairs.json': _bytes(pairs), 'source-review.csv': raw}, manifest)
