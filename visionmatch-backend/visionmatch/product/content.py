from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import provider
from ..ultra.requests import Client
from ..ultra.sheets import digest, make_sheet
from .store import decode

MODEL = 'gpt-6.1-terra'
SHEET_LIMIT = 8
KIND = 'MAIN_CONTENT'
LIST_FIELDS = (
    'main_subject', 'secondary_subjects', 'visible_text', 'team_names', 'league_names',
    'animal_names', 'sports_categories', 'pattern_types', 'theme_candidates',
    'theme_types', 'subtheme_candidates', 'product_category_hints', 'uncertain_items', 'evidence',
)
PROMPT = '''图片是一张主图拼图。每格上方是主图 ID（M 加数字），它只是返回结果的键。
逐格独立观察。每个提供的 ID 必须恰好返回一条结果，即使看不清。禁止借用相邻格的结论，禁止联网或使用工具。
main_subject 写印花或造型的主体，如球队、动物、人物、Logo、文字图案。
secondary_subjects 写辅助印花、装饰、标语、数字和设计本身的背景图案，忽略拍摄背景、模特和编号。
visible_text 逐字抄录图上可辨认文字，保留英文拼写、大小写和语言，勿翻译、补全或把拼图编号抄进去。
team_names、league_names、animal_names、sports_categories、pattern_types 分别填写可见证据支持的身份或类别。
theme_candidates 写主题描述，theme_types 写主题类型，subtheme_candidates 写子主题候选。
product_category_hints 只写从外观可见的品类线索。主题候选不代表正式编码，不生成编码、缩写或号段。
evidence 写可复查的图案位置、形状、文字等证据。confidence 为 0 到 1 的视觉识别把握。
文字太小、遮挡、标记模糊或身份不能确定时，对应字段留空并在 uncertain_items 写明原因。禁止编造。
只返回指定 JSON。'''


class Source(BaseModel):
    main_image_id: int = Field(gt=0)
    path: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class SheetRequest(BaseModel):
    request_id: UUID
    confirmed_budget: bool
    force: bool = False
    sources: list[Source] = Field(min_length=1, max_length=SHEET_LIMIT)


def content_schema(labels):
    fields = {
        'image_id': {'type': 'string', 'enum': labels},
        **{field: {'type': 'array', 'items': {'type': 'string'}} for field in LIST_FIELDS},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    }
    return {'type': 'object', 'properties': {'images': {'type': 'array', 'items': {
        'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False,
    }}}, 'required': ['images'], 'additionalProperties': False}


def require_coverage(result, labels):
    if Counter(row['image_id'] for row in result['images']) != Counter(labels):
        raise ValueError('Terra 返回的主图编号有遗漏或重复，未保存这组结果。')


def recognize(data_root, payload):
    sources = [{'id': f'M{item.main_image_id}', 'path': item.path, 'sha256': item.sha256} for item in payload.sources]
    labels = [item['id'] for item in sources]
    schema = content_schema(labels)
    root = Path(data_root) / 'main_content'
    prepared = make_sheet(sources, root / 'sheets', cell=700)
    # The existing Client caches validated answers and persists each attempt and
    # its usage. An explicit single-image rerun opens a new request directory.
    binary = Path(provider.executable())
    key = digest([prepared['cache_key'], PROMPT, schema, MODEL, str(binary), binary.stat().st_mtime_ns,
                  str(payload.request_id) if payload.force else None])
    attempt_root = root / 'calls' / key
    attempt_root.mkdir(parents=True, exist_ok=True)
    client = Client(attempt_root, approved_plan=str(payload.request_id), max_calls=1)
    previous = None
    try:
        previous = client.db.execute('SELECT status,metadata FROM calls').fetchone()
        if previous and previous[0] != 'SUCCESS':
            raise ValueError('相同拼图已有失败或中断记录，请单张重新识别以授权一次新调用。')
        answer = client.call([prepared['path']], PROMPT, schema, str(payload.request_id), timeout=300,
                             kind=KIND, validator=lambda result: require_coverage(result, labels))
    except Exception as exc:
        if previous is None:
            recorded = client.db.execute('SELECT metadata FROM calls').fetchone()
            attempt = decode(recorded[0], {}) if recorded else {}
        else:
            attempt = {}
        detail = provider.scrub(str(exc))[:800]
        _, stop = provider.classify_error(detail)
        return {'error': detail, 'real_calls': int(bool(attempt.get('cli_started'))),
                'usage_unknown': 'TIMEOUT_REMOTE_UNKNOWN' in detail, 'stop': stop,
                'log_key': key, 'model_name': MODEL}
    finally:
        client.close()
    label_to_main = dict(zip(labels, (item.main_image_id for item in payload.sources)))
    return {'images': [{'main_image_id': label_to_main[row['image_id']],
                        **{k: v for k, v in row.items() if k != 'image_id'}} for row in answer['result']['images']],
            'model_name': MODEL, 'real_calls': int(not answer.get('cache_hit')), 'cache_hit': bool(answer.get('cache_hit')),
            'usage': answer.get('usage'), 'elapsed_seconds': answer.get('elapsed_seconds'), 'log_key': key}


def content_router(queue):
    router = APIRouter(prefix='/api/main-content')

    @router.get('/capabilities')
    def capabilities():
        try:
            provider.executable()
        except provider.ProviderError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {'model_name': MODEL, 'sheet_limit': SHEET_LIMIT}

    @router.post('/recognize')
    def run(payload: SheetRequest):
        if not payload.confirmed_budget:
            raise HTTPException(400, '请确认本张拼图的 1 次 Terra 调用预算。')
        if len({item.main_image_id for item in payload.sources}) != len(payload.sources):
            raise HTTPException(400, '同一张主图不能重复提交。')
        if not queue.claim_content():
            raise HTTPException(409, 'Terra 正在处理图片匹配或其它内容识别，请稍后再试。')
        try:
            return recognize(queue.store.root, payload)
        except (ValueError, OSError, provider.ProviderError) as exc:
            raise HTTPException(422, provider.scrub(str(exc))[:800]) from exc
        finally:
            queue.release_content()

    return router
