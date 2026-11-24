from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from threading import Lock

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from app.models import DesignCode, DesignPackageInfo, MainImageContent
from app.services.design_content import content_signature, dump
from app.services.media import thumbnail_cache_path
from app.services.package_identity import package_root
from app.services.ticket_lookup import lookup_league_team_codes, lookup_product_catalog
from app.services.visionmatch import file_hash

lock = Lock()
IGNORED_WORDS = {'YOUR NAME', 'NAME', 'YOUR NUMBER', 'NUMBER', 'TEAM NAME', 'CUSTOM NAME'}
PREFIX = re.compile(r'[A-Z]{1,12}')
OFFICIAL_TEAM_PREFIX = re.compile(r'[A-Z][A-Z0-9]{0,11}')


def catalog_path():
    return os.getenv('DESIGN_MATCHER_CODING_CATALOG')


def package_tokens(root):
    name = re.sub(r'^（已完成）', '', package_root(Path(root)).name)
    parts = [part.strip() for part in re.split(r'[-－–—]', name, maxsplit=3)[:3]]
    return dict(zip(('theme', 'subtheme', 'category'), (parts + ['', '', ''])[:3]))


def key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def safe_stem(text):
    return bool(text and len(text) < 220 and text == text.strip(' .')
                and not re.search(r'[\\/:*?"<>|\x00-\x1f]', text))


def number_prefix(prefixes):
    return f"{prefixes['category']}-{prefixes['theme']}"


def recommend(name, used):
    words = re.findall(r'[A-Z]+', unicodedata.normalize('NFKC', name).upper())
    letters = ''.join(words)
    if not letters:
        return None
    seed = (''.join(word[0] for word in words) if len(words) > 1 else letters[:3])[:6]
    for candidate in [seed, *(letters[:width] for width in range(2, min(7, len(letters) + 1))),
                      *(seed + suffix for suffix in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')]:
        if len(candidate) >= 2 and not used[candidate]:
            return candidate
    return None


def identification_content(row):
    if not row or not (row.final_json or row.ai_json):
        return None, '请先完成主图内容识别。'
    data = json.loads(row.final_json or row.ai_json)
    if not row.final_json and (data.get('confidence', 0) < .85 or not data.get('evidence')):
        return None, '图像识别证据不足，请人工确认主体或可见文字。'
    return data, None


def official_team_candidates(filename, data, tokens):
    if not data:
        return set()
    explicit = {key(name) for name in data.get('team_names', []) if name.strip()}
    if explicit:
        return explicit
    stem = key(Path(filename).stem)
    omitted = {key(word) for word in (*IGNORED_WORDS, *tokens.values())}
    # The filename supplies a full-name candidate. Confirm its leading place
    # name against the artwork, then require an exact match in the same league.
    for text in data.get('visible_text', []):
        visible = key(text)
        if visible not in omitted and len(visible) >= 3 and stem.startswith(visible + ' '):
            return {stem}
    return set()


def visual_identity(data, subjects, tokens):
    tiers = [data.get('team_names', []), data.get('animal_names', []), data.get('visible_text', []), data.get('main_subject', [])]
    for values in tiers:
        names = {key(text) for text in values if text.strip()}
        matches = [subject for subject, aliases in subjects if aliases & names]
        if matches:
            return (matches[0], None) if len(matches) == 1 else (None, '图上出现多个已有缩写候选，请人工确认。')
    omit = {key(value) for value in [*IGNORED_WORDS, *tokens.values()]}
    for values in (tiers[0], tiers[1], tiers[2], tiers[3]):
        names = {key(text): text.strip() for text in values if re.search(r'[A-Za-z]', text) and key(text) not in omit}
        if names:
            if len(names) == 1:
                return {'name': next(iter(names.values())), 'prefix': None}, None
            break
    return None, '无法唯一确定新缩写对应的主体，请人工确认。'


def official_options(mains, contents, prior, tokens):
    url = os.getenv('DESIGN_MATCHER_TICKET_DB_URL')
    if not url or tokens['theme'] != '球迷主题' or not tokens['subtheme']:
        return {}, {}, None
    candidates = {}
    for main in mains:
        previous = prior.get(main.id)
        filename = previous.original_filename if previous else main.filename
        data, _ = identification_content(contents.get(main.id))
        names = official_team_candidates(filename, data, tokens)
        if names:
            candidates[main.id] = names
    if not candidates:
        return {}, {}, None
    try:
        parsed = make_url(url)
        if parsed.drivername != 'mysql+pymysql':
            raise ValueError('需要 MySQL 只读查询配置')
        rows = lookup_league_team_codes(parsed, set().union(*candidates.values()),
                                        theme_type=tokens['theme'], league=tokens['subtheme'])
    except (ImportError, ValueError, SQLAlchemyError):
        note = '正式球队缩写查询不可用，需核对连接后重新预览。'
        return {}, dict.fromkeys(candidates, note), note
    by_name = defaultdict(set)
    for row in rows:
        full_name = (row['subtheme_name'] or '').strip()
        prefix = (row['abbreviation'] or '').strip().upper()
        if full_name and OFFICIAL_TEAM_PREFIX.fullmatch(prefix):
            by_name[key(full_name)].add((full_name, prefix))
    found, ambiguous = {}, {}
    for main_id, names in candidates.items():
        choices = set().union(*(by_name[name] for name in names))
        if len(choices) == 1:
            name, prefix = next(iter(choices))
            found[main_id] = {'name': name, 'prefix': prefix}
        elif choices:
            ambiguous[main_id] = '正式库有多个球队缩写候选，需人工核对。'
    return found, ambiguous, None


def preview_company_catalog(db, batch, mains, url):
    tokens = package_tokens(batch.root_path)
    rows = [{'main_image_id': main.id, 'original_filename': main.filename} for main in mains]
    state = {'mode': 'COMPANY_PREVIEW', 'tokens': tokens, 'rows': rows}
    profile = db.get(DesignPackageInfo, batch.id)
    manual = json.loads(profile.manual_json or '{}') if profile else {}
    product_name = manual.get('product_name') or tokens['category']
    try:
        product, catalog, history = lookup_product_catalog(
            make_url(url), product_name=key(product_name), theme_type=key(tokens['theme']),
        )
    except (ImportError, ValueError, SQLAlchemyError):
        return {**state, 'error': '无法读取公司 ticket_system 编码目录。'}
    if not product:
        return {**state, 'error': f'公司库未能唯一匹配品类「{product_name}」。请在设计包信息中填写公司库产品全名。'}

    last_issued = history['last_issued_at']
    state['ticket_catalog'] = {
        'product_name': product['product_name'],
        'product_prefix': product['prefix'],
        'current_number': product['current_number'],
        'last_issued_number': history['last_issued_number'],
        'last_issued_at': last_issued.isoformat(sep=' ') if last_issued else None,
    }
    groups = defaultdict(list)
    for entry in catalog:
        groups[entry['theme_name'] or ''].append(entry)
    term = key(tokens['subtheme'])
    accepted_names = {term, f'{term.removesuffix("球队").strip()}球队'} if tokens['theme'] == '球迷主题' else {term}
    matching = [name for name, items in groups.items() if term and (
        key(name) in accepted_names or any(key(x['child_theme_name'] or '') == term for x in items)
    )]
    if len(matching) != 1:
        return {**state, 'theme_choices': sorted(groups),
                'error': f'公司库未能唯一匹配「{tokens["subtheme"]}」对应的主题。'}
    theme = matching[0]
    subjects = set()
    prefixes = set()
    for entry in groups[theme]:
        pair = ((entry['child_theme_name'] or '').strip(), (entry['child_theme_prefix'] or '').strip().upper())
        if pair[0] and OFFICIAL_TEAM_PREFIX.fullmatch(pair[1]):
            subjects.add(pair)
        if entry['medium_code']:
            prefixes.add(entry['medium_code'])
    state['ticket_catalog']['theme_name'] = theme
    if len(prefixes) == 1:
        state['source_code_prefix'] = next(iter(prefixes))
    if len(prefixes) != 1:
        state['error'] = '公司库中该主题的号段前缀不唯一，需核对。'
    contents = {content.main_image_id: content for content in db.scalars(
        select(MainImageContent).where(MainImageContent.main_image_id.in_([main.id for main in mains]))
    )}
    for main, row in zip(mains, rows):
        content, error = identification_content(contents.get(main.id))
        if error:
            row['error'] = error
            continue
        if tokens['theme'] == '球迷主题':
            candidates = official_team_candidates(main.filename, content, tokens)
            choices = {(name, prefix) for name, prefix in subjects if key(name) in candidates}
        else:
            candidates = {key(text) for text in content.get('main_subject', [])}
            choices = {(name, prefix) for name, prefix in subjects if key(name) in candidates}
        if len(choices) != 1:
            row['error'] = '需人工核对公司库中的小主题全名。'
            continue
        name, prefix = choices.pop()
        if len({key(n) for n, p in subjects if p == prefix}) != 1:
            row['error'] = '公司库中该缩写对应多个小主题，需人工核对。'
            continue
        row.update(subject_name=name, abbreviation=prefix, decision='OFFICIAL')
    return state


def read_catalog(path, tokens):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        context = db.execute('''SELECT id, theme_prefix, subtheme_prefix, category_prefix,
                                 range_start, range_end, number_width FROM coding_context
                                 WHERE theme=? AND subtheme=? AND category=?''', tuple(tokens.values())).fetchone()
        subjects = db.execute('SELECT * FROM coding_subject').fetchall()
        history = db.execute('SELECT context_id, image_sha256, subject_name, abbreviation, source_code FROM coding_history').fetchall()
    return context, subjects, history


def proposal(row):
    return {**{name: getattr(row, name) for name in ('main_image_id', 'original_filename', 'abbreviation',
                                                     'source_code', 'operation_code', 'target_stem', 'decision', 'applied')},
            'artist_code': row.target_stem}


def prepare(db, batch, mains):
    tokens = package_tokens(batch.root_path)
    state = {'mode': 'ISOLATED', 'tokens': tokens, 'prefixes': None,
             'rows': [{'main_image_id': main.id, 'original_filename': main.filename} for main in mains]}
    try:
        context, subjects, history = read_catalog(catalog_path(), tokens)
    except (OSError, sqlite3.Error):
        state['error'] = '隔离编码目录不可用，请核对主题、品类及历史号段数据。'
        return {}, state
    if not context:
        state['error'] = '编码目录没有对应的主题、小主题和品类组合，请核对设计包前三段。'
        return {}, state
    prefixes = {name: context[name + '_prefix'].upper() for name in tokens}
    state['prefixes'] = prefixes
    context_key = dump(tokens)
    if not all(PREFIX.fullmatch(value) for value in prefixes.values()):
        state['error'] = '编码目录中的主题、小主题或品类缩写格式无效。'
        return {}, state
    source_prefix = number_prefix(prefixes)
    width = context['number_width']
    if len(source_prefix) + width > 40 or len(str(context['range_end'])) > width:
        state['error'] = '隔离编码目录的流水号宽度无效。'
        return {}, state
    source_pattern = re.compile(re.escape(source_prefix) + rf'[0-9]{{{width}}}')
    state['source_code_prefix'] = source_prefix

    records = list(db.scalars(select(DesignCode)))
    contents = dict(db.execute(select(MainImageContent.main_image_id, MainImageContent).where(
        MainImageContent.main_image_id.in_([main.id for main in mains]))).all())
    prior = {row.main_image_id: row for row in records}
    official, lookup_blockers, lookup_error = official_options(mains, contents, prior, tokens)
    if lookup_error:
        state['official_lookup_error'] = lookup_error
    official_owners = defaultdict(set)
    for subject in official.values():
        official_owners[subject['prefix']].add(key(subject['name']))
    used = defaultdict(set)
    occupied = defaultdict(set)
    for subject in subjects:
        used[subject['prefix'].upper()].add(key(subject['name']))
    for historical in history:
        used[historical['abbreviation'].upper()].add(key(historical['subject_name']))
        occupied[historical['source_code'].upper()].add(historical['image_sha256'])
    used.update({prefix: owners.copy() for prefix, owners in official_owners.items()})
    for row in records:
        if row.applied or row.abbreviation not in official_owners:
            used[row.abbreviation].add(row.subject_key)
        occupied[row.source_code].add(row.source_sha256)

    options = [(subject, {key(subject['name']), *(key(alias) for alias in json.loads(subject['aliases_json']))})
               for subject in subjects if subject['context_id'] == context['id']]
    ready = {}
    used_in_batch = set()
    next_number = context['range_start']
    for main, result in zip(mains, state['rows']):
        try:
            sha256 = file_hash(main.file_path)
        except OSError:
            result['error'] = '主图不可读取。'
            continue
        existing = prior.get(main.id)
        row = contents.get(main.id)
        signature = content_signature(row)
        official_subject = official.get(main.id)
        if main.id in lookup_blockers and not (existing and existing.applied):
            result['error'] = lookup_blockers[main.id]
            continue
        available_draft = None
        if existing:
            if existing.context_key != context_key or existing.source_sha256 != sha256:
                result['error'] = '主图或设计包分类发生变化，原编码仍保留，需人工核对。'
                continue
            complete_code = source_pattern.fullmatch(existing.source_code)
            if existing.applied and not complete_code:
                result['error'] = '此文件已经按旧号段命名，请在隔离副本中重新生成。'
                continue
            if existing.source_code in used_in_batch:
                result['error'] = '同批次有两张主图对应同一设计编码。'
                continue
            draft_matches_official = (not official_subject or (
                existing.decision == 'OFFICIAL' and existing.subject_key == key(official_subject['name'])
                and existing.abbreviation == official_subject['prefix']))
            draft_prefix_available = not (official_owners.get(existing.abbreviation, set()) - {existing.subject_key})
            if complete_code and (existing.applied or (
                    existing.content_signature == signature and draft_matches_official and draft_prefix_available)):
                ready[main.id] = existing
                used_in_batch.add(existing.source_code)
                result.update(proposal(existing))
                continue
            if official_subject and complete_code and not existing.applied:
                available_draft = existing.source_code
        if row and row.source_sha256 and row.source_sha256 != sha256:
            result['error'] = '主图已变化，请先重新识别内容。'
            continue
        data, error = identification_content(row)
        if not error:
            subject, error = (official_subject, None) if official_subject else visual_identity(data, options, tokens)
        if error:
            result['error'] = error
            continue
        name = subject['name']
        name_key = key(name)
        prefix = (subject['prefix'] or '').upper()
        matching_history = [entry for entry in history if entry['context_id'] == context['id'] and entry['image_sha256'] == sha256]
        matching_local = {entry.source_code for entry in records if entry.context_key == context_key and
                          entry.subject_key == name_key and entry.source_sha256 == sha256 and
                          source_pattern.fullmatch(entry.source_code)}
        if matching_history:
            if len(matching_history) != 1 or key(matching_history[0]['subject_name']) != name_key:
                result['error'] = '历史图像的主体或号段有冲突，请人工核对。'
                continue
            historical_prefix = matching_history[0]['abbreviation'].upper()
            if prefix and prefix != historical_prefix:
                result['error'] = '目录缩写与历史编码不一致，请人工核对。'
                continue
            prefix = historical_prefix
            matching_local.add(matching_history[0]['source_code'].upper())
        suggested = not prefix
        if not prefix:
            reserved = [value for value, subjects_for_prefix in used.items() if subjects_for_prefix == {name_key}]
            prefix = reserved[0] if len(reserved) == 1 else recommend(name, used)
        prefix_pattern = OFFICIAL_TEAM_PREFIX if official_subject else PREFIX
        if not prefix or not prefix_pattern.fullmatch(prefix) or used[prefix] - {name_key}:
            result['error'] = '主体缩写有重名冲突或无可用字母缩写。'
            continue
        if len(matching_local) > 1:
            result['error'] = '同一图像存在多个历史编码，请人工核对。'
            continue
        source_code = next(iter(matching_local), available_draft)
        if source_code:
            if not source_pattern.fullmatch(source_code) or occupied[source_code] - {sha256}:
                result['error'] = '历史编码格式或归属与目录不一致。'
                continue
        else:
            while next_number <= context['range_end']:
                candidate = f'{source_prefix}{next_number:0{width}d}'
                next_number += 1
                if not occupied[candidate]:
                    source_code = candidate
                    break
            if not source_code:
                result['error'] = '隔离号段已用完。'
                continue
        stem = name + source_code
        if not safe_stem(stem) or source_code in used_in_batch:
            result['error'] = '目标名称不合法或本批次已使用同一编码。'
            continue
        if official_subject:
            decision = 'OFFICIAL'
        else:
            decision = existing.decision if existing and existing.source_code == source_code else (
                'INHERITED' if matching_local else 'RECOMMENDED' if suggested else 'MAPPED')
        entry = existing or DesignCode(main_image_id=main.id, original_filename=main.filename, applied=False)
        entry.context_key, entry.subject_key = context_key, name_key
        entry.abbreviation, entry.source_code, entry.operation_code = prefix, source_code, f'{prefix}-{source_code}'
        entry.target_stem, entry.source_sha256, entry.content_signature = stem, sha256, signature
        entry.decision = decision
        if not existing:
            db.add(entry)
            records.append(entry)
        ready[main.id] = entry
        result.update(proposal(entry))
        used[prefix].add(name_key)
        occupied[source_code].add(sha256)
        used_in_batch.add(source_code)
    return ready, state


def apply_main_path(batch, main, target):
    main.filename = target.name
    main.file_path = str(target)
    main.thumbnail_path = str(thumbnail_cache_path(Path(batch.root_path), batch.id, target))
