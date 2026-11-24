from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, create_model
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db import SessionLocal, get_db
from app.models import Batch, ContentRun, DesignPackageInfo, MainImage, MainImageContent, utcnow
from app.services import visionmatch
from app.services.design_content import LIST_FIELDS, csv_export, dump, new_package, package_dict, snapshot, utc_iso

router = APIRouter(prefix='/api/design-content')
logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='main-content')
write_lock = Lock()
SHEET_LIMIT = 8


class PackageEdit(BaseModel):
    product_name: str | None = Field(None, max_length=255)
    target_category: str | None = Field(None, max_length=255)
    designer: str | None = Field(None, max_length=255)


ContentEdit = create_model('ContentEdit', **{
    **{name: (list[Annotated[str, Field(max_length=1000)]], Field(default_factory=list, max_length=50)) for name in LIST_FIELDS},
    'confidence': (float | None, Field(None, ge=0, le=1)),
})


class Start(BaseModel):
    confirmed_budget: bool | None = None
    max_calls: int | None = Field(None, ge=1)
    main_image_ids: list[int] | None = Field(None, min_length=1)
    force: bool = False


def same_origin(request: Request):
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, '拒绝跨来源提交。')


def get_batch(db, batch_id):
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(404, '批次不存在。')
    return batch


def ensure_package(db, batch):
    # For existing batches, materialize the package identity once, even if
    # the first use is an export. Fresh imports create it in create_batch.
    with write_lock:
        row = db.get(DesignPackageInfo, batch.id)
        if row is None:
            row = new_package(batch)
            db.add(row)
            db.commit()
        return row


def terra_request(method, endpoint, body=None):
    url = os.getenv('DESIGN_MATCHER_CONTENT_TERRA_URL', visionmatch.URL)
    with httpx.Client(timeout=httpx.Timeout(345, connect=3), trust_env=False) as client:
        response = client.request(method, url.rstrip('/') + '/api/main-content/' + endpoint, json=body)
    if response.is_error:
        detail = response.json().get('detail', response.reason_phrase)
        if response.status_code == 404:
            detail = '当前 Terra 后端未接入主图内容识别。'
        return {'error': str(detail), 'stop': response.status_code in (404, 409, 503), 'real_calls': 0}
    return response.json()


def run_dict(run):
    ids = json.loads(run.main_ids_json)
    sheets = json.loads(run.sheets_json)
    succeeded = sum(len(item['succeeded']) for item in sheets)
    failed = sum(len(item['failed']) for item in sheets)
    return {'id': run.id, 'status': run.status, 'main_image_ids': ids, 'max_calls': run.max_calls, 'sheets': sheets,
            'real_calls': sum(item['real_calls'] for item in sheets),
            'unknown_usage': (run.status == 'INTERRUPTED' or any(item.get('usage_unknown') for item in sheets)),
            'succeeded': succeeded, 'failed': failed, 'remaining': len(ids) - succeeded - failed,
            'created_at': utc_iso(run.created_at), 'finished_at': utc_iso(run.finished_at)}


def current_sources(db, batch, ids):
    rows = db.scalars(select(MainImage).where(MainImage.batch_id == batch.id, MainImage.id.in_(ids)))
    root = Path(batch.root_path).resolve()
    sources, errors = {}, dict.fromkeys(ids, '主图已从当前批次移除。')
    for main in rows:
        path = Path(main.file_path)
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError('主图位于设计包以外。')
            sources[main.id] = {'main_image_id': main.id, 'path': str(path), 'sha256': visionmatch.file_hash(path)}
            errors.pop(main.id)
        except (OSError, ValueError):
            errors[main.id] = '无法读取原始主图，请检查图片或共享盘。'
    return sources, errors


def recognize_sheet(db, batch, ids, force):
    sources, errors = current_sources(db, batch, ids)
    if not sources:
        return {}, errors, {}
    try:
        answer = terra_request('POST', 'recognize', {'request_id': str(uuid4()), 'confirmed_budget': True,
            'force': force, 'sources': [sources[key] for key in ids if key in sources]})
    except (httpx.HTTPError, ValueError):
        answer = {'error': '未能确认 Terra 返回结果，已停止后续调用。请查看运行记录后手动重试。',
                  'usage_unknown': True, 'stop': True}
    if answer.get('error'):
        errors.update(dict.fromkeys(sources, answer['error']))
        return {}, errors, answer
    results = {row['main_image_id']: row for row in answer.get('images', [])}
    if len(results) != len(answer.get('images', [])) or set(results) != set(sources):
        errors.update(dict.fromkeys(sources, 'Terra 主图编号与本次提交不一致。'))
        return {}, errors, answer
    db.expire_all()
    latest, changed = current_sources(db, batch, ids)
    for key, source in sources.items():
        if latest.get(key) != source:
            errors[key] = changed.get(key, '识别期间主图发生变化，请重新识别。')
            results.pop(key)
    return {key: (results[key], sources[key]['sha256']) for key in results}, errors, answer


def execute(run_id, force):
    try:
        with SessionLocal() as db:
            run = db.get(ContentRun, run_id)
            if run is None:
                return
            ids = json.loads(run.main_ids_json)
            batch = db.get(Batch, run.batch_id)
            run.status = 'RUNNING'
            db.commit()
            sheets = []
            for offset in range(0, len(ids), SHEET_LIMIT):
                part = ids[offset:offset + SHEET_LIMIT]
                results, errors, answer = recognize_sheet(db, batch, part, force)
                with write_lock:
                    for main_id, (result, sha256) in results.items():
                        row = db.get(MainImageContent, main_id)
                        row.ai_json = dump({key: value for key, value in result.items() if key != 'main_image_id'})
                        row.source_sha256 = sha256
                        row.model_name = answer['model_name']
                        row.recognized_at = utcnow()
                        row.recognition_status, row.error = 'RECOGNIZED', None
                    for main_id, error in errors.items():
                        row = db.get(MainImageContent, main_id)
                        if row is not None:
                            row.recognition_status, row.error = 'FAILED', error
                    sheets.append({'main_image_ids': part, 'succeeded': list(results), 'failed': list(errors),
                                   **{key: answer.get(key) for key in ('error', 'usage', 'log_key', 'model_name',
                                                                     'elapsed_seconds', 'cache_hit', 'usage_unknown')},
                                   'real_calls': answer.get('real_calls', 0)})
                    run.sheets_json = dump(sheets)
                    db.commit()
                if answer.get('stop'):
                    break
            completed = sum(len(sheet['succeeded']) for sheet in sheets)
            run.status = 'COMPLETE' if completed == len(ids) else 'PARTIAL' if completed else 'FAILED'
            run.finished_at = utcnow()
            db.commit()
    except Exception:
        logger.exception('Main content run interrupted: %s', run_id)
        with SessionLocal() as db:
            db.execute(update(ContentRun).where(ContentRun.id == run_id).values(status='INTERRUPTED', finished_at=utcnow()))
            db.commit()


@router.get('/batches')
def batches(db: Session = Depends(get_db)):
    rows = db.scalars(select(Batch).where(select(MainImage.id).where(
        MainImage.batch_id == Batch.id).exists()).order_by(Batch.id.desc()))
    return [{'id': row.id, 'name': Path(row.root_path).name} for row in rows]


@router.get('/batches/{batch_id}')
def workspace(batch_id: int, db: Session = Depends(get_db)):
    batch = get_batch(db, batch_id)
    return snapshot(db, batch, ensure_package(db, batch))


@router.get('/capabilities')
def capabilities():
    try:
        result = terra_request('GET', 'capabilities')
    except (httpx.HTTPError, ValueError):
        return {'ready': False, 'sheet_limit': SHEET_LIMIT, 'error': '未连接到 Terra 内容识别接口。'}
    return {'ready': not result.get('error'), 'sheet_limit': SHEET_LIMIT, **result}


@router.patch('/batches/{batch_id}/package', dependencies=[Depends(same_origin)])
def edit_package(batch_id: int, edit: PackageEdit, db: Session = Depends(get_db)):
    info = ensure_package(db, get_batch(db, batch_id))
    with write_lock:
        db.refresh(info)
        fields = json.loads(info.manual_json or '{}')
        fields.update({key: (value or '').strip() or None for key, value in edit.model_dump(exclude_unset=True).items()})
        info.manual_json = dump(fields)
        db.commit()
    return package_dict(info)


@router.put('/batches/{batch_id}/mains/{main_image_id}/confirmation', dependencies=[Depends(same_origin)])
def confirm(batch_id: int, main_image_id: int, edit: ContentEdit, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_image_id)
    if main is None or main.batch_id != batch_id:
        raise HTTPException(404, '该批次没有这张主图。')
    with write_lock:
        row = db.get(MainImageContent, main_image_id)
        if row is None:
            row = MainImageContent(main_image_id=main_image_id)
            db.add(row)
        row.final_json = dump(edit.model_dump())
        row.reviewed_at = utcnow()
        db.commit()
    return {'review_status': 'CONFIRMED'}


@router.get('/batches/{batch_id}/runs')
def runs(batch_id: int, db: Session = Depends(get_db)):
    get_batch(db, batch_id)
    rows = db.scalars(select(ContentRun).where(ContentRun.batch_id == batch_id).order_by(ContentRun.created_at.desc()).limit(15))
    return [run_dict(run) for run in rows]


@router.post('/batches/{batch_id}/runs', status_code=202, dependencies=[Depends(same_origin)])
def start(batch_id: int, body: Start | None = None, db: Session = Depends(get_db)):
    batch = get_batch(db, batch_id)
    body = body or Start()
    if body.confirmed_budget is False:
        raise HTTPException(400, '本次请求已禁用 Terra 调用。')
    with write_lock:
        if db.scalar(select(ContentRun.id).where(ContentRun.status.in_(['QUEUED', 'RUNNING'])).limit(1)):
            raise HTTPException(409, '已有主图内容任务运行中。')
        rows = list(db.execute(select(MainImage.id, MainImageContent).outerjoin(MainImageContent).where(
            MainImage.batch_id == batch.id).order_by(MainImage.id)))
        mains = dict(rows)
        ids = sorted(body.main_image_ids) if body.main_image_ids is not None else [
            key for key, row in rows if not row or (
                row.recognition_status == 'PENDING' and not row.ai_json and not row.final_json)]
        if body.main_image_ids is None and not ids:
            raise HTTPException(400, '没有待识别的主图。')
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= mains.keys():
            raise HTTPException(400, '请选择该批次的有效主图，每张只选择一次。')
        if body.force and len(ids) != 1:
            raise HTTPException(400, '重新识别请单张提交。')
        required = math.ceil(len(ids) / SHEET_LIMIT)
        if body.max_calls is not None and required > body.max_calls:
            raise HTTPException(400, f'本次需要最多 {required} 次 Terra 调用，超出请求指定的上限。')
        for key in ids:
            if mains[key] is None:
                db.add(MainImageContent(main_image_id=key))
        run = ContentRun(id=str(uuid4()), batch_id=batch_id, main_ids_json=dump(ids), max_calls=required)
        db.add(run)
        db.commit()
        executor.submit(execute, run.id, body.force)
        return run_dict(run)


@router.get('/batches/{batch_id}/export/{format}')
def export(batch_id: int, format: str, db: Session = Depends(get_db)):
    data = workspace(batch_id, db)
    if format == 'json':
        return JSONResponse(jsonable_encoder(data), headers={'Content-Disposition': 'attachment; filename="design_package.json"'})
    if format in ('design_groups.csv', 'design_group_assets.csv'):
        return Response(csv_export(data, assets=format == 'design_group_assets.csv'), media_type='text/csv',
                        headers={'Content-Disposition': f'attachment; filename="{format}"'})
    raise HTTPException(404, '没有这种导出格式。')
