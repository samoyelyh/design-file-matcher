from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from sqlalchemy import select

from app.models import Batch, DesignCode, DesignPackageInfo, MainImage, MainImageContent, MatchedFile
from app.services.package_identity import IDENTITY_FIELDS, package_root, parse_package
from app.services.review_scope import ignored_review_categories, is_review_confirmed

LIST_FIELDS = (
    'main_subject', 'secondary_subjects', 'visible_text', 'team_names', 'league_names',
    'animal_names', 'sports_categories', 'pattern_types', 'theme_candidates', 'theme_types',
    'subtheme_candidates', 'product_category_hints', 'uncertain_items', 'evidence',
)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def content_signature(row):
    text = (row.final_json or row.ai_json or '') if row else ''
    return hashlib.sha256(text.encode()).hexdigest()


def utc_iso(value):
    if value is None:
        return None
    return (value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()


def new_package(batch):
    name, parsed = parse_package(Path(batch.root_path))
    return DesignPackageInfo(batch_id=batch.id, package_uuid=str(uuid4()), original_name=name, parsed_json=dump(parsed))


def package_dict(info):
    parsed = json.loads(info.parsed_json)
    manual = json.loads(info.manual_json or '{}')
    fields = {**parsed['fields'], **manual}
    return {'package_id': info.package_uuid, 'package_name': info.original_name, **fields,
            'parse_result': parsed['fields'], 'parse_confidence': parsed['confidence'],
            'parse_uncertain_fields': parsed['uncertain_fields'], 'manual_fields': sorted(manual),
            'uncertain_fields': [key for key in IDENTITY_FIELDS if not fields[key]]}


def content_dict(row):
    raw = json.loads(row.ai_json) if row and row.ai_json else None
    final = json.loads(row.final_json) if row and row.final_json else None
    return {'recognized_content': final if final is not None else raw, 'ai_content': raw,
            'confirmed_content': final, 'content_source': 'HUMAN' if final is not None else 'AI' if raw else None,
            'recognition_status': row.recognition_status if row else 'PENDING',
            'review_status': 'CONFIRMED' if final is not None else 'PENDING' if raw else 'NOT_READY',
            'model_name': row.model_name if row else None, 'recognized_at': utc_iso(row.recognized_at) if row else None,
            'reviewed_at': utc_iso(row.reviewed_at) if row else None, 'source_sha256': row.source_sha256 if row else None,
            'recognition_error': row.error if row else None}


def relative(path, root):
    # Export the actual relative path, including the unchanged historical name.
    # Paths outside this import cannot be expressed as design-package assets.
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return None


def asset_dict(item, root, ignored):
    return {'asset_id': item.id, 'relative_path': relative(item.current_path, root),
            'original_relative_path': relative(item.original_path, root),
            'filename': item.current_filename, 'original_filename': item.original_filename,
            'category': item.category, 'file_type': item.extension.lstrip('.').lower(),
            'match_method': item.match_method, 'match_confidence': item.algorithm_score,
            'match_status': item.status, 'review_status': item.review_status,
            'association_confirmed': is_review_confirmed(item), 'category_ignored': item.category in ignored}


def snapshot(db, batch: Batch, info: DesignPackageInfo):
    root = package_root(Path(batch.root_path))
    package = package_dict(info)
    assets = defaultdict(list)
    ignored = ignored_review_categories(batch)
    query = select(MatchedFile).where(MatchedFile.batch_id == batch.id).order_by(MatchedFile.id)
    for item in db.scalars(query):
        assets[item.matched_main_id].append(asset_dict(item, root, ignored))
    groups = []
    counts = Counter()
    content = db.execute(select(MainImage, MainImageContent, DesignCode).join(
        MainImageContent, MainImage.id == MainImageContent.main_image_id, isouter=True).join(
        DesignCode, MainImage.id == DesignCode.main_image_id, isouter=True).where(
            MainImage.batch_id == batch.id).order_by(MainImage.id))
    for main, row, code in content:
        group = content_dict(row)
        counts['recognized'] += group['ai_content'] is not None
        counts['pending'] += group['recognition_status'] == 'PENDING'
        counts['failed'] += group['recognition_status'] == 'FAILED'
        counts['awaiting_review'] += group['review_status'] == 'PENDING'
        counts['confirmed'] += group['review_status'] == 'CONFIRMED'
        group_assets = assets.pop(main.id, [])
        groups.append({'design_group_id': str(uuid5(UUID(info.package_uuid), str(main.id))),
                       'main_image_id': main.id,
                       'main_image': {'main_image_id': main.id, 'filename': main.filename,
                                       'original_filename': code.original_filename if code else main.filename,
                                       'relative_path': relative(main.file_path, root)},
                       'assets': group_assets, 'coding': None if not code or (
                           not code.applied and code.content_signature != content_signature(row)) else {
                           'mode': 'TICKET_API' if code.decision == 'TICKET_ALLOCATED' else 'ISOLATED',
                           'abbreviation': code.abbreviation, 'source_code': code.source_code, 'artist_code': code.target_stem,
                           'operation_code': code.operation_code, 'decision': code.decision, 'applied': code.applied,
                       }, **group})
    return {'schema_version': '1.0', 'source_system': 'design-file-matcher', 'package': package,
            'design_groups': groups, 'unassigned_assets': [item for rows in assets.values() for item in rows],
            'counts': {'main_images': len(groups), **{key: counts[key] for key in (
                'recognized', 'pending', 'failed', 'awaiting_review', 'confirmed')}}}


def csv_export(data, assets=False):
    stream = io.StringIO(newline='')
    group_fields = ('design_group_id', 'main_image_id')
    package_fields = ('package_id', 'package_name', *IDENTITY_FIELDS)
    asset_fields = ('asset_id', 'relative_path', 'original_relative_path', 'filename', 'original_filename', 'category',
                    'file_type', 'match_method', 'match_confidence', 'match_status', 'review_status',
                    'association_confirmed', 'category_ignored')
    fields = (*package_fields, *group_fields, *asset_fields) if assets else (
        *package_fields, *group_fields, 'main_filename', 'main_relative_path', 'asset_count', 'association_confirmed_count',
        'abbreviation', 'source_code', 'artist_code', 'operation_code', 'coding_decision', 'coding_applied', 'coding_mode',
        'recognition_status', 'review_status', 'content_source', 'model_name', 'recognized_at', 'reviewed_at',
        *LIST_FIELDS, 'confidence', 'ai_content', 'confirmed_content')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()

    def write(values):
        # Keep formula-like raw values intact in JSON and treat them as literal
        # text when the companion CSV is opened in a spreadsheet.
        row = {key: dump(value) if isinstance(value, (dict, list, bool)) else value for key, value in values.items()}
        writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(
            ('=', '+', '-', '@', '\t', '\r')) else value for key, value in row.items()})

    package = data['package']
    for group in data['design_groups']:
        identity = {key: group[key] for key in group_fields}
        if assets:
            for item in group['assets']:
                write({**package, **identity, **item})
        else:
            main = group['main_image']
            coding = group['coding'] or {}
            write({**package, **group, **(group['recognized_content'] or {}), 'main_filename': main['original_filename'],
                   **coding, 'coding_decision': coding.get('decision'), 'coding_applied': coding.get('applied'),
                   'coding_mode': coding.get('mode'),
                   'main_relative_path': main['relative_path'], 'asset_count': len(group['assets']),
                   'association_confirmed_count': sum(item['association_confirmed'] for item in group['assets'])})
    if assets:
        for item in data['unassigned_assets']:
            write({**package, **item})
    return stream.getvalue().encode('utf-8-sig')
