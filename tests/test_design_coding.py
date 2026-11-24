import json
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, DesignCode, DesignCodingPlan, MainImage, MainImageContent, MatchedFile
from app.services import design_coding, ticket_coding
from app.services.design_coding import file_hash
from app.services.design_content import new_package
from app.services.rename_plan import RenamePlanService


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / '主题样例-目录小主题样例-品类样例-夜景'
    catalog = tmp_path / 'catalog.sqlite'
    monkeypatch.setenv('DESIGN_MATCHER_CODING_CATALOG', str(catalog))
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db, closing(sqlite3.connect(catalog)) as source:
        source.executescript(Path('docs/coding-catalog.sql').read_text(encoding='utf-8'))
        source.execute("INSERT INTO coding_context VALUES (1,'主题样例','目录小主题样例','品类样例','FD','XZT','YLWY',346,370,7)")
        source.execute("INSERT INTO coding_context VALUES (2,'经典','圣诞','卫衣','CL','XM','SW',10,30,4)")
        source.execute("INSERT INTO coding_subject VALUES (1,'披萨','PZ','[]'),(1,'DALLAS','DAL','[]'),(2,'New Year','NY','[]')")
        batch = Batch(root_path=str(root), status='COMPLETED')
        db.add(batch)
        db.flush()
        for index, text in enumerate(('披萨', 'DALLAS', 'NEW YORK', ''), 1):
            main_path = root / '主图' / f'old{index}.png'
            detail_path = root / '细节图' / f'old{index}.png'
            for path in (main_path, detail_path):
                path.parent.mkdir(exist_ok=True, parents=True)
                Image.new('RGB', (32, 32), (index * 30, 0, 0)).save(path)
            main = MainImage(batch_id=batch.id, basename=main_path.stem, filename=main_path.name, file_path=str(main_path))
            db.add(main)
            db.flush()
            db.add(MainImageContent(main_image_id=main.id, source_sha256=file_hash(main_path), ai_json=json.dumps({
                'confidence': 0.99, 'evidence': ['visible'], 'visible_text': [text, 'YOUR NAME', '00'],
            }) if text else None))
            db.add(MatchedFile(batch_id=batch.id, category='细节图', original_filename=detail_path.name,
                current_filename=detail_path.name, original_path=str(detail_path), current_path=str(detail_path),
                extension='.png', matched_main_id=main.id, matched_main_basename=main.basename,
                status='MATCHED', review_status='PENDING' if index == 2 else 'CONFIRMED'))
            if index == 1:
                source.execute("INSERT INTO coding_history VALUES (1,?,'披萨','PZ','YLWY-FD0000345')", (file_hash(main_path),))
        source.execute("INSERT INTO coding_history VALUES (1,?,'DALLAS','DAL','YLWY-FD0000346')", ('2' * 64,))
        source.commit()
        db.commit()
        yield db, batch, root
    engine.dispose()


def test_coding_preview_reuses_history_checks_prefix_collision_and_uses_existing_confirmation(workspace):
    db, batch, root = workspace
    service = RenamePlanService()
    previous = service.generate(db, batch.id, with_codes=True)
    current = service.generate(db, batch.id, with_codes=True)
    assert current['coding'] == previous['coding']
    assert current['coding']['prefixes'] == {'theme': 'FD', 'subtheme': 'XZT', 'category': 'YLWY'}
    assert current['coding']['source_code_prefix'] == 'YLWY-FD'
    first, second, third, skipped = current['coding']['rows']
    assert (first['source_code'], first['artist_code'], first['operation_code'], first['decision']) == (
        'YLWY-FD0000345', '披萨YLWY-FD0000345', 'PZ-YLWY-FD0000345', 'INHERITED')
    assert (second['source_code'], second['artist_code'], second['decision']) == (
        'YLWY-FD0000347', 'DALLASYLWY-FD0000347', 'MAPPED')
    assert third['abbreviation'] != 'NY' and third['decision'] == 'RECOMMENDED'
    assert 'error' in skipped
    assert not list(root.glob('*.txt'))
    assert (root / '主图' / 'old1.png').is_file()
    with pytest.raises(ValueError, match='预览已更新'):
        service.execute(db, batch.id, previous['plan_group'], confirm_coding=True)
    result = service.execute(db, batch.id, current['plan_group'], confirm_coding=True)
    assert result['counts'] == {'RENAMED': 5, 'SKIPPED': 2}
    assert [item['line_count'] for item in result['code_text_files']] == [3, 3]
    assert (root / '美工使用号段.txt').read_text(encoding='utf-8').splitlines() == [
        item['artist_code'] for item in (first, second, third)]
    assert (root / '运营使用号段.txt').read_text(encoding='utf-8').splitlines() == [
        item['operation_code'] for item in (first, second, third)]
    assert (root / '主图' / '披萨YLWY-FD0000345.png').is_file()
    assert (root / '细节图' / '披萨YLWY-FD0000345.png').is_file()
    assert (root / '细节图' / 'old2.png').is_file()
    assert (root / '主图' / 'old4.png').is_file()
    assets = db.scalars(select(MatchedFile).order_by(MatchedFile.id)).all()
    assert assets[0].matched_main_id == first['main_image_id'] and assets[1].review_status == 'PENDING'


def test_default_rename_and_explicit_coding_confirmation_are_separate(workspace):
    db, batch, root = workspace
    service = RenamePlanService()
    ordinary = service.generate(db, batch.id)
    coded = service.generate(db, batch.id, with_codes=True)
    assert 'coding' not in ordinary and service.latest(db, batch.id)['plan_group'] == ordinary['plan_group']
    with pytest.raises(ValueError, match='对应的更名预览'):
        service.execute(db, batch.id, coded['plan_group'])
    with pytest.raises(ValueError, match='对应的更名预览'):
        service.execute(db, batch.id, ordinary['plan_group'], confirm_coding=True)
    assert (root / '主图' / 'old1.png').is_file()
    assert not list(root.glob('*.txt'))

    assert 'code_text_files' not in service.execute(db, batch.id, ordinary['plan_group'])
    assert (root / '主图' / 'old1.png').is_file()
    assert not list(root.glob('*.txt'))
    service.execute(db, batch.id, coded['plan_group'], confirm_coding=True)
    after = service.generate(db, batch.id)
    first = after['plans'][0]
    assert first['rename_status'] == 'NOOP'
    assert first['current_filename'] == first['target_filename'] == '披萨YLWY-FD0000345.png'
    assert Path(first['source_path']).is_file()


def test_company_catalog_uses_official_mappings_without_drafting_codes(workspace, tmp_path, monkeypatch):
    from app.main import company_coding_preview

    db, batch, _ = workspace
    batch.root_path = str(tmp_path / '球迷主题-NHL-帽衫-节日')
    profile = new_package(batch)
    profile.manual_json = '{"product_name":"常规帽衫"}'
    db.add(profile)
    mains = list(db.scalars(select(MainImage).order_by(MainImage.id)))
    teams = [('New York Islanders', 'NYI'), ('New York Rangers', 'NYR')]
    for main, (name, _) in zip(mains, teams):
        db.get(MainImageContent, main.id).final_json = json.dumps({'team_names': [name]})
    db.commit()
    monkeypatch.setenv('DESIGN_MATCHER_TICKET_DB_URL', 'mysql+pymysql://localhost/ticket_system')

    def catalog(url, *, product_name, theme_type):
        assert url.database == 'ticket_system'
        assert (product_name, theme_type) == ('常规帽衫', '球迷主题')
        return (
            {'id': 32, 'product_name': product_name, 'prefix': 'HODI', 'current_number': 11526},
            [{'theme_name': 'NHL球队', 'medium_code': 'HODI-NH', 'child_theme_name': name,
              'child_theme_prefix': prefix} for name, prefix in teams],
            {'last_issued_number': 11526, 'last_issued_at': None},
        )

    monkeypatch.setattr(design_coding, 'lookup_product_catalog', catalog)
    result = company_coding_preview(batch.id, db)
    assert result['coding']['source_code_prefix'] == 'HODI-NH'
    assert [row['abbreviation'] for row in result['coding']['rows'][:2]] == ['NYI', 'NYR']
    assert 'plan_group' not in result and not result['plans']
    assert db.scalar(select(DesignCode)) is None
    assert db.scalar(select(DesignCodingPlan)) is None


@pytest.fixture
def ticket_workspace(workspace, monkeypatch):
    db, batch, root = workspace
    monkeypatch.setenv("DESIGN_MATCHER_TICKET_API_URL", "http://127.0.0.1:9099/api")
    monkeypatch.setenv("DESIGN_MATCHER_TICKET_USER_ID", "42")
    for main_id, name in [(1, "披萨"), (2, "DALLAS")]:
        db.get(MainImageContent, main_id).final_json = json.dumps({"main_subject": [name]})
    db.commit()
    calls = []
    transport = {"timeout_after": None}

    def provider(method, base, path, **kwargs):
        assert base == "http://127.0.0.1:9099/api"
        if method == "GET":
            return {
                "/products": [{"id": 7, "productName": "品类样例", "prefix": "YLWY"}],
                "/themes/by-product": [{"id": 1, "themeType": "目录小主题样例", "themeTypePrefix": "FD"}],
                "/themes/by-product-and-type-name": [
                    {"id": 4, "childThemeName": "披萨", "childThemePrefix": "PZ"},
                    {"id": 9, "childThemeName": "DALLAS", "childThemePrefix": "DAL"},
                ],
            }[path]
        assert (method, path) == ("POST", "/ticket/claim")
        calls.append(kwargs["json"])
        if len(calls) == transport["timeout_after"]:
            raise httpx.ReadTimeout("lost confirmation")
        code = f"YLWY-FD{8000100 + len(calls)}"
        return {"startCode": code, "endCode": code}

    monkeypatch.setattr(ticket_coding, "request", provider)
    return db, batch, root, calls, transport


def test_ticket_preview_confirmation_and_local_retry_use_same_receipts(ticket_workspace, monkeypatch):
    from app.services import rename_plan

    db, batch, root, calls, _ = ticket_workspace
    service = RenamePlanService()
    preview = service.generate(db, batch.id, with_codes=True)
    assert preview["coding"]["ticket_quantity"] == 2
    assert all("artist_code" not in row for row in preview["coding"]["rows"])
    assert not calls and db.scalar(select(DesignCode)) is None
    with pytest.raises(ValueError, match="对应的更名预览"):
        service.execute(db, batch.id, preview["plan_group"])
    assert not calls and not list(root.glob("*.txt"))

    publish = rename_plan.publish_code_texts

    def cannot_write(*_args):
        raise OSError("fixture write failure")

    monkeypatch.setattr(rename_plan, "publish_code_texts", cannot_write)
    with pytest.raises(ValueError, match="已尝试恢复"):
        service.execute(db, batch.id, preview["plan_group"], confirm_coding=True)
    assert (root / "主图" / "old1.png").exists()
    assert calls == [{"productId": 7, "userId": 42, "themeId": i, "quantity": 1} for i in (4, 9)]
    retry = service.generate(db, batch.id, with_codes=True)
    assert retry["plan_group"] == preview["plan_group"]
    assert retry["coding"]["ticket_quantity"] == 0
    monkeypatch.setattr(rename_plan, "publish_code_texts", publish)
    result = service.execute(db, batch.id, retry["plan_group"], confirm_coding=True)
    assert len(calls) == 2 and result["counts"]["RENAMED"] == 3
    assert all(row["applied"] for row in result["coding"]["rows"][:2])
    assert (root / "美工使用号段.txt").read_text(encoding="utf-8").splitlines()[0] == "披萨YLWY-FD8000101"
    assert (root / "运营使用号段.txt").read_text(encoding="utf-8").splitlines()[0] == "PZ-YLWY-FD8000101"
    assert (root / "细节图" / "old2.png").exists()
    service.execute(db, batch.id, retry["plan_group"], confirm_coding=True)
    assert len(calls) == 2


def test_lost_ticket_response_blocks_reclaim_and_preserves_returned_numbers(ticket_workspace):
    db, batch, root, calls, status = ticket_workspace
    service = RenamePlanService()
    preview = service.generate(db, batch.id, with_codes=True)
    status["timeout_after"] = 2
    with pytest.raises(ValueError, match="领号结果无法确认"):
        service.execute(db, batch.id, preview["plan_group"], confirm_coding=True)
    retry = service.generate(db, batch.id, with_codes=True)
    assert not retry["coding"]["ticket_ready"]
    assert retry["plan_group"] == preview["plan_group"]
    assert retry["coding"]["rows"][0]["operation_code"] == "PZ-YLWY-FD8000101"
    with pytest.raises(ValueError, match="领号结果未确认"):
        service.execute(db, batch.id, retry["plan_group"], confirm_coding=True)
    assert len(calls) == 2 and not list(root.glob("*.txt"))
    assert (root / "主图" / "old1.png").exists()


def test_auto_abbreviations_avoid_other_theme_prefixes_for_the_same_product(ticket_workspace, monkeypatch):
    from app.main import CompleteTicketAbbreviationsRequest, complete_ticket_abbreviations

    db, batch, root, claims, _ = ticket_workspace
    monkeypatch.setenv("DESIGN_MATCHER_TICKET_THEME_CREATE", "1")
    for main_id, name in ((3, "New River"), (4, "Night Rain")):
        db.get(MainImageContent, main_id).final_json = json.dumps({"main_subject": [name]})
    db.commit()
    types = [{"id": 1, "productIds": [7]}, {"id": 2, "productIds": [7]}, {"id": 3, "productIds": [88]}]
    children = {
        1: [{"id": 4, "name": "披萨", "prefix": "PZ"}, {"id": 9, "name": "DALLAS", "prefix": "DAL"}],
        2: [{"id": 5, "name": "North Range", "prefix": "NR"}],
        3: [{"id": 6, "name": "Unrelated", "prefix": "NE"}],
    }
    additions = []
    provider = ticket_coding.request

    def catalog(method, base, path, **kwargs):
        if path == "/themes/types":
            assert method == "GET"
            return types
        if path.startswith("/themes/types/"):
            group = int(path.split("/")[3])
            if method == "GET":
                assert group in {1, 2}
                return list(children[group])
            assert method == "POST" and group == 1
            new = kwargs["json"]
            additions.append(new)
            children[group].append({"id": 100 + len(additions), **new})
            return None
        if path == "/themes/by-product-and-type-name":
            return [{"id": row["id"], "childThemeName": row["name"], "childThemePrefix": row["prefix"]}
                    for row in children[1]]
        return provider(method, base, path, **kwargs)

    monkeypatch.setattr(ticket_coding, "request", catalog)
    service = RenamePlanService()
    plan = service.generate(db, batch.id, with_codes=True)
    assert [row["new_theme_name"] for row in plan["coding"]["rows"] if row.get("new_theme_name")] == ["New River", "Night Rain"]
    assert not additions and not claims
    # Another catalog user finished this one after the matcher preview.
    children[1].append({"id": 78, "name": "Night Rain", "prefix": "NRA"})
    result = complete_ticket_abbreviations(
        batch.id, CompleteTicketAbbreviationsRequest(plan_group=plan["plan_group"]), db,
    )
    assert additions == [{"name": "New River", "prefix": "NE"}]
    assert [(item["prefix"], item["status"]) for item in result["results"]] == [("NE", "CREATED"), ("NRA", "EXISTS")]
    assert [row["abbreviation"] for row in result["preview"]["coding"]["rows"]] == ["PZ", "DAL", "NE", "NRA"]
    assert children[2][0]["prefix"] == "NR"
    assert not claims and not list(root.glob("*.txt"))
    assert db.scalar(select(DesignCode)) is None


def test_number_files_use_design_library_folders_and_preserve_existing_codes(workspace):
    db, batch, root = workspace
    (root / 'PSD').mkdir()
    (root / '包效果图').mkdir()
    artist = root / 'PSD' / '美工使用号段.txt'
    operation = root / '包效果图' / '运营使用号段.txt'
    artist.write_bytes('既有号段YLWY-FD0000001\r\n'.encode('utf-8-sig'))
    service = RenamePlanService()
    plan = service.generate(db, batch.id, with_codes=True)
    result = service.execute(db, batch.id, plan['plan_group'], confirm_coding=True)
    assert [f['relative_path'] for f in result['code_text_files']] == ['PSD/美工使用号段.txt', '包效果图/运营使用号段.txt']
    lines = artist.read_text(encoding='utf-8-sig').splitlines()
    assert lines[0] == '既有号段YLWY-FD0000001' and len(lines) == 4
    assert len(operation.read_text(encoding='utf-8').splitlines()) == 3
    snapshot = (artist.read_bytes(), operation.read_bytes())
    service.execute(db, batch.id, plan['plan_group'], confirm_coding=True)
    assert snapshot == (artist.read_bytes(), operation.read_bytes())


def test_number_file_failure_rolls_back_renames_and_first_number_file(workspace, monkeypatch):
    from app.services import code_texts

    db, batch, root = workspace
    service = RenamePlanService()
    plan = service.generate(db, batch.id, with_codes=True)
    replace = code_texts._replace_file

    def fail_on_second(path, content):
        if path.name == '运营使用号段.txt':
            raise OSError('write blocked')
        replace(path, content)

    monkeypatch.setattr(code_texts, '_replace_file', fail_on_second)
    with pytest.raises(ValueError, match='命名或写入号段文件失败'):
        service.execute(db, batch.id, plan['plan_group'], confirm_coding=True)
    assert not list(root.glob('*.txt'))
    assert (root / '主图' / 'old1.png').is_file()
    assert (root / '细节图' / 'old1.png').is_file()
    assert service.latest(db, batch.id, with_codes=True)['counts'] == plan['counts']


def test_updated_content_and_external_conflicts_block_confirmation(workspace):
    db, batch, root = workspace
    service = RenamePlanService()
    plan = service.generate(db, batch.id, with_codes=True)
    main_id = plan['coding']['rows'][1]['main_image_id']
    db.get(MainImageContent, main_id).final_json = json.dumps({'visible_text': ['NEW WORD']})
    db.commit()
    with pytest.raises(ValueError, match='内容审核结果已更新'):
        service.execute(db, batch.id, plan['plan_group'], confirm_coding=True)
    latest = service.generate(db, batch.id, with_codes=True)
    assert latest['coding']['rows'][1]['target_stem'].startswith('NEW WORD')
    blocked = Path(latest['main_plans'][0]['target_path'])
    blocked.write_bytes(b'outside file')
    with pytest.raises(ValueError, match='已被占用'):
        service.execute(db, batch.id, latest['plan_group'], confirm_coding=True)
    assert blocked.read_bytes() == b'outside file'
    assert (root / '细节图' / 'old1.png').is_file()


@pytest.mark.parametrize('source_code', ['FD0000345', 'FD-YLWY0000345'])
def test_invalid_historical_source_code_is_not_reused(workspace, tmp_path, source_code):
    db, batch, _ = workspace
    with closing(sqlite3.connect(tmp_path / 'catalog.sqlite')) as catalog:
        catalog.execute("UPDATE coding_history SET source_code=? WHERE subject_name='披萨'", (source_code,))
        catalog.commit()
    result = RenamePlanService().generate(db, batch.id, with_codes=True)
    first = result['coding']['rows'][0]
    assert '历史编码格式' in first['error']
    assert 'artist_code' not in first


def test_confirmation_rejects_preview_saved_with_old_segment_order(workspace):
    db, batch, root = workspace
    service = RenamePlanService()
    plan = service.generate(db, batch.id, with_codes=True)
    saved = db.scalar(select(DesignCodingPlan).where(DesignCodingPlan.plan_group == plan['plan_group']))
    saved.summary_json = json.dumps({**plan['coding'], 'source_code_prefix': 'FD-YLWY'})
    db.commit()
    with pytest.raises(ValueError, match='号段顺序已更新'):
        service.execute(db, batch.id, plan['plan_group'], confirm_coding=True)
    assert (root / '主图' / 'old1.png').is_file()


@pytest.mark.parametrize(('full_names', 'locations', 'prefixes'), [
    (('New York Islanders', 'New York Rangers'), ('NEW YORK', 'NEW YORK'), ('NYI', 'NYR')),
    (('Colorado Avalanche', 'Columbus Blue Jackets'), ('COLORADO', 'COLUMBUS'), ('COL1', 'COL2')),
])
def test_official_team_full_names_replace_isolated_draft_without_changing_test_number(
        workspace, tmp_path, monkeypatch, full_names, locations, prefixes):
    db, batch, _ = workspace
    batch.root_path = str(tmp_path / '球迷主题-NHL-品类样例-夜景')
    local_prefixes = {location: 'TMP' + suffix for location, suffix in zip(dict.fromkeys(locations), ('A', 'B'))}
    with closing(sqlite3.connect(tmp_path / 'catalog.sqlite')) as catalog:
        catalog.execute("UPDATE coding_context SET theme='球迷主题',subtheme='NHL' WHERE id=1")
        catalog.executemany("INSERT INTO coding_subject VALUES (1,?,?,'[]')", local_prefixes.items())
        catalog.commit()
    mains = db.scalars(select(MainImage).order_by(MainImage.id)).all()[2:]
    for main, full_name, location in zip(mains, full_names, locations):
        main.filename = full_name + '.png'
        row = db.get(MainImageContent, main.id)
        row.ai_json = json.dumps({'visible_text': [location, 'HOCKEY'], 'confidence': 0.99, 'evidence': ['visible']})
    db.commit()
    old, _ = design_coding.prepare(db, batch, mains)
    old_numbers = {main_id: row.source_code for main_id, row in old.items()}
    assert {row.abbreviation for row in old.values()} == set(local_prefixes.values())
    db.commit()

    def lookup(_url, names, *, theme_type, league):
        assert names == {name.lower() for name in full_names}
        assert (theme_type, league) == ('球迷主题', 'NHL')
        return [{'subtheme_name': name, 'abbreviation': prefix} for name, prefix in zip(full_names, prefixes)]

    monkeypatch.setenv('DESIGN_MATCHER_TICKET_DB_URL', 'mysql+pymysql://localhost/ticket_system')
    monkeypatch.setattr(design_coding, 'lookup_league_team_codes', lookup)
    new, _ = design_coding.prepare(db, batch, mains)
    assert [new[main.id].abbreviation for main in mains] == list(prefixes)
    assert set(row.decision for row in new.values()) == {'OFFICIAL'}
    assert {main_id: row.source_code for main_id, row in new.items()} == old_numbers
    db.commit()

    def offline(*args, **kwargs):
        from sqlalchemy.exc import SQLAlchemyError

        raise SQLAlchemyError('database unavailable')

    monkeypatch.setattr(design_coding, 'lookup_league_team_codes', offline)
    blocked, summary = design_coding.prepare(db, batch, mains)
    assert not blocked
    assert all('正式球队缩写查询不可用' in row['error'] for row in summary['rows'])
