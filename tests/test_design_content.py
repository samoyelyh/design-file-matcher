import csv
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app import design_content
from app.db import SessionLocal
from app.main import app
from app.models import Batch, MainImage, MainImageContent, MatchedFile
from app.services.package_identity import parse_package
from app.services.scanner import scan_batch


def test_parses_existing_package_layout_and_leaves_unrecognized_fields_blank(tmp_path):
    name = '球迷主题-MLB-刀锋鞋-城市剪影-球类纹路-成人款-王小明'
    original, parsed = parse_package(tmp_path / name / (name + '-效果图'))
    assert original == name
    assert parsed['fields'] == {'product_name': '刀锋鞋', 'target_category': '成人款', 'designer': '王小明'}
    _, parsed = parse_package(tmp_path / 'NFL-3D火焰+城市名元素儿童帽衫单面定制-王小明')
    assert parsed['fields'] == {'product_name': '帽衫', 'target_category': '儿童款', 'designer': '王小明'}
    _, parsed = parse_package(tmp_path / '主题-帽衫-圣诞节')
    assert parsed['uncertain_fields'] == ['target_category', 'designer']


@pytest.fixture
def workspace(tmp_path):
    name = '球迷主题-NFL-帽衫-儿童款-王小明'
    root = tmp_path / name / (name + '-效果图')
    (root / '主图').mkdir(parents=True)
    (root / '细节图').mkdir()
    for index in (1, 2):
        Image.new('RGB', (32, 32), (index * 40, 25, 55)).save(root / '主图' / f'main{index}.png')
        Image.new('RGB', (32, 32), (index * 40, 35, 55)).save(root / '细节图' / f'asset{index}.png')
    with TestClient(app) as client, SessionLocal() as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        ids = db.scalars(select(MainImage.id).where(MainImage.batch_id == batch.id).order_by(MainImage.id)).all()
        assets = db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch.id).order_by(MatchedFile.id)).all()
        for asset, main_id in zip(assets, ids):
            asset.matched_main_id = main_id
            asset.matched_main_basename = db.get(MainImage, main_id).basename
            asset.match_method = 'VISIONMATCH'
        assets[0].review_status = 'CONFIRMED'
        db.commit()
        yield client, db, batch, ids


def test_verified_main_ids_manual_priority_rescan_and_exports(workspace, monkeypatch):
    client, db, batch, ids = workspace
    path = f'/api/design-content/batches/{batch.id}'
    scheduled = []
    monkeypatch.setattr(design_content.executor, 'submit', lambda worker, *args: scheduled.append((worker, args)))
    prompts = []

    def offline(method, endpoint, body):
        prompts.append(body)
        return {'model_name': 'offline-test', 'real_calls': 0, 'images': [
            {'main_image_id': source['main_image_id'], 'main_subject': [f"visual-{source['main_image_id']}-{len(prompts)}"],
             'visible_text': ['CaseSensitive'], 'confidence': 0.9}
            for source in reversed(body['sources'])]}

    monkeypatch.setattr(design_content, 'terra_request', offline)
    initial = client.get(path).json()
    assert client.patch(path + '/package', json={'designer': '人工填写'}).json()['designer'] == '人工填写'
    scan_batch(db, batch.id)
    assert client.get(path).json()['package']['designer'] == '人工填写'
    request = {'main_image_ids': ids, 'max_calls': 1, 'confirmed_budget': False}
    assert client.post(path + '/runs', json=request).status_code == 400
    assert client.post(path + '/runs', json={**request, 'confirmed_budget': True, 'main_image_ids': ids + [99999]}).status_code == 400
    assert scheduled == []
    assert client.post(path + '/runs', json={**request, 'confirmed_budget': True}).status_code == 202
    worker, args = scheduled.pop()
    worker(*args)
    assert client.get(path + '/runs').json()[0]['succeeded'] == 2
    for group, ident in zip(client.get(path).json()['design_groups'], ids):
        assert group['ai_content']['main_subject'] == [f'visual-{ident}-1']
    assert client.put(path + f'/mains/{ids[0]}/confirmation', json={'main_subject': ['人工确认']}).status_code == 200
    assert client.post(path + '/runs', json={**request, 'confirmed_budget': True, 'main_image_ids': [ids[0]], 'force': True}).status_code == 202
    worker, args = scheduled.pop()
    worker(*args)
    export = client.get(path + '/export/json').json()
    first, second = export['design_groups']
    assert prompts[1]['force']
    assert first['recognized_content']['main_subject'] == ['人工确认']
    assert first['ai_content']['main_subject'] == [f'visual-{ids[0]}-2']
    assert first['design_group_id'] == initial['design_groups'][0]['design_group_id']
    assert first['main_image']['relative_path'].count('/') == 2
    assert first['assets'][0]['association_confirmed'] is True
    assert second['assets'][0]['association_confirmed'] is False
    for filename in ('design_groups.csv', 'design_group_assets.csv'):
        downloaded = client.get(path + '/export/' + filename)
        rows = list(csv.DictReader(io.StringIO(downloaded.content.decode('utf-8-sig'))))
        assert [row['main_image_id'] for row in rows] == [str(ident) for ident in ids]


def test_selection_spans_sheets_and_respects_call_cap(workspace, monkeypatch):
    client, db, batch, ids = workspace
    path = f'/api/design-content/batches/{batch.id}/runs'
    original = db.get(MainImage, ids[0])
    for index in range(7):
        db.add(MainImage(batch_id=batch.id, basename=f'extra{index}', filename=f'extra{index}.png', file_path=original.file_path))
    db.commit()
    queued = []
    calls = []
    monkeypatch.setattr(design_content.executor, 'submit', lambda fn, *args: queued.append((fn, args)))

    def offline(method, endpoint, body):
        calls.append(len(body['sources']))
        return {'error': '离线失败测试', 'real_calls': 0}

    monkeypatch.setattr(design_content, 'terra_request', offline)
    assert client.post(path, json={'confirmed_budget': True, 'max_calls': 1}).status_code == 400
    assert not queued
    assert client.post(path, json={'confirmed_budget': True, 'max_calls': 2}).status_code == 202
    worker, args = queued.pop()
    worker(*args)
    assert calls == [8, 1]
    assert client.get(path).json()[0]['failed'] == 9


def test_auto_runs_five_sheets_and_skips_resolved_or_failed_mains(workspace, monkeypatch):
    client, db, batch, ids = workspace
    path = f'/api/design-content/batches/{batch.id}/runs'
    original = db.get(MainImage, ids[0])
    db.add(MainImageContent(main_image_id=ids[0], final_json='{}'))
    db.add(MainImageContent(main_image_id=ids[1], recognition_status='FAILED', error='用量未知'))
    pending = [MainImage(batch_id=batch.id, basename=f'new{i}', filename=f'new{i}.png',
                         file_path=original.file_path) for i in range(33)]
    db.add_all(pending)
    db.commit()
    pending_ids = [main.id for main in pending]
    queued, calls = [], []
    monkeypatch.setattr(design_content.executor, 'submit', lambda fn, *args: queued.append((fn, args)))

    def offline(method, endpoint, body):
        sheet_ids = [source['main_image_id'] for source in body['sources']]
        calls.append(sheet_ids)
        return {'model_name': 'offline-test', 'real_calls': 0, 'images': [
            {'main_image_id': ident, 'main_subject': ['测试图案']} for ident in sheet_ids]}

    monkeypatch.setattr(design_content, 'terra_request', offline)
    response = client.post(path)
    assert response.status_code == 202
    assert response.json()['max_calls'] == 5
    assert response.json()['main_image_ids'] == pending_ids
    assert client.post(path, json={}).status_code == 409
    worker, args = queued.pop()
    worker(*args)
    assert [len(sheet) for sheet in calls] == [8, 8, 8, 8, 1]
    assert [ident for sheet in calls for ident in sheet] == pending_ids
    assert client.get(path).json()[0]['succeeded'] == 33
    assert client.post(path).json()['detail'] == '没有待识别的主图。'
    assert not queued
