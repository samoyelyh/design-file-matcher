import importlib
import json
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.models import Batch, MainImage, MatchedFile, RenamePlan
from app.services.rename_plan import RenamePlanService


@pytest.fixture
def example(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'scope.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    root = tmp_path / "package"
    with sessions() as db:
        batch = Batch(root_path=str(root), status="COMPLETED", match_config_snapshot=json.dumps({"version": "preserve-me"}))
        db.add(batch)
        db.flush()
        mains, items, originals = [], [], {}
        for index, category in enumerate(("场景图", "场景图", "问题图")):
            if index < 2:
                main = MainImage(batch_id=batch.id, basename=f"SKU{index}", filename=f"SKU{index}.png", file_path=f"main{index}.png")
                db.add(main)
                db.flush()
                mains.append(main.id)
            path = root / category / f"original{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), (index * 80, 20, 30)).save(path)
            originals[str(path)] = path.read_bytes()
            item = MatchedFile(batch_id=batch.id, category=category, original_filename=path.name,
                               current_filename=path.name, original_path=str(path), current_path=str(path),
                               extension=".png", matched_main_id=mains[index if index < 2 else 0],
                               matched_main_basename=f"SKU{index if index < 2 else 0}", status="MATCHED", review_status="PENDING")
            db.add(item)
            db.flush()
            items.append(item.id)
        cache = root / ".design_matcher" / "keep.bin"
        cache.parent.mkdir()
        cache.write_bytes(b"needed for later review")
        db.commit()
        batch_id = batch.id
    yield sessions, batch_id, mains, items, originals, cache
    engine.dispose()


def test_partial_rename_skips_unconfirmed_and_can_continue_later(example):
    sessions, bid, _, ids, originals, cache = example
    with sessions() as db:
        item = db.get(MatchedFile, ids[0])
        item.review_status = "CONFIRMED"
        db.commit()
        plan = RenamePlanService().generate(db, bid)
        assert plan["counts"] == {"READY": 1, "SKIPPED": 2}
        result = RenamePlanService().execute(db, bid, plan["plan_group"])
        assert result["counts"] == {"RENAMED": 1, "SKIPPED": 2}
        assert cache.exists() and "workspace_cleanup" not in result
        assert Path(item.current_path).read_bytes() == originals[item.original_path]
        for file_id in ids[1:]:
            pending = db.get(MatchedFile, file_id)
            assert Path(pending.current_path).read_bytes() == originals[pending.original_path]
            assert pending.current_path == pending.original_path
            assert pending.review_status == "PENDING"
            pending.review_status = "CONFIRMED"
        db.commit()
        later = RenamePlanService().generate(db, bid)
        assert later["counts"] == {"NOOP": 1, "READY": 2}
        assert RenamePlanService().execute(db, bid, later["plan_group"])["counts"] == {"RENAMED": 3}
        assert not cache.exists()


def test_old_preview_never_renames_an_unconfirmed_file(example):
    sessions, bid, _, ids, originals, _ = example
    with sessions() as db:
        plan = RenamePlanService().generate(db, bid)
        for row in db.scalars(select(RenamePlan)):
            row.rename_status = "READY"  # A preview created by the old application.
        db.commit()
        assert RenamePlanService().latest(db, bid)["counts"] == {"SKIPPED": 3}
        assert RenamePlanService().execute(db, bid, plan["plan_group"])["counts"] == {"SKIPPED": 3}
        for file_id in ids:
            item = db.get(MatchedFile, file_id)
            assert Path(item.original_path).read_bytes() == originals[item.original_path]


def test_skipped_target_does_not_create_a_false_collision(example):
    sessions, bid, _, ids, originals, _ = example
    with sessions() as db:
        confirmed, pending = [db.get(MatchedFile, i) for i in ids[:2]]
        confirmed.review_status = "CONFIRMED"
        pending.matched_main_id = confirmed.matched_main_id
        pending.matched_main_basename = confirmed.matched_main_basename
        db.commit()
        plan = RenamePlanService().generate(db, bid)
        assert plan["counts"] == {"READY": 1, "SKIPPED": 2}
        RenamePlanService().execute(db, bid, plan["plan_group"])
        assert Path(pending.original_path).read_bytes() == originals[pending.original_path]


def test_changed_confirmed_assignment_is_rejected_before_any_rename(example):
    sessions, bid, mains, ids, originals, _ = example
    with sessions() as db:
        item = db.get(MatchedFile, ids[0])
        item.review_status = "CONFIRMED"
        db.commit()
        plan = RenamePlanService().generate(db, bid)
        item.matched_main_id = mains[1]
        item.matched_main_basename = "SKU1"
        db.commit()
        with pytest.raises(ValueError, match="审核归属已变化"):
            RenamePlanService().execute(db, bid, plan["plan_group"])
        assert Path(item.original_path).read_bytes() == originals[item.original_path]


def test_ignore_restore_and_confirm_only_active_categories(example, monkeypatch):
    sessions, bid, mains, ids, originals, cache = example
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    monkeypatch.setattr(main, "learn_confirmed_references", lambda *_: 0)
    monkeypatch.setattr(main, "settings", replace(main.settings, upload_root=str(cache.parents[2])))

    def override_db():
        with sessions() as db:
            yield db

    main.app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(main.app) as client:
            url = f"/api/batches/{bid}"
            assert client.get(url + "/review").json()["summary"]["unmatched"] == 1
            ignored = client.post(url + "/review-categories", json={"category": "问题图", "ignored": True})
            assert ignored.status_code == 200
            old_group = ignored.json()["rename_plan"]["plan_group"]
            review = client.get(url + "/review").json()
            assert review["summary"].get("unmatched", 0) == 0
            assert all(slot["category"] != "问题图" for group in review["groups"] for slot in group["categories"])
            assert next(x for x in review["review_categories"] if x["category"] == "问题图")["ignored"]
            confirmed = client.post(f"/api/main-images/{mains[0]}/confirm").json()
            assert confirmed["confirmed"] == 1
            assert confirmed["rename_plan"]["counts"] == {"READY": 1, "SKIPPED": 1, "IGNORED": 1}
            assert client.post(url + "/rename-plan/execute", json={"plan_group": old_group}).status_code == 409
            result = client.post(url + "/rename-plan/execute", json={"plan_group": confirmed["rename_plan"]["plan_group"]})
            assert result.status_code == 200
            assert result.json()["counts"] == {"RENAMED": 1, "SKIPPED": 1, "IGNORED": 1}
            assert cache.exists()
            assert client.post(f"/api/files/{ids[2]}/confirm", json={}).status_code == 409
            assert client.post(url + "/review-categories", json={"category": "主图", "ignored": True}).status_code == 400
            confirmed = client.post(url + "/confirm-groups", json={"main_image_ids": mains}).json()
            assert confirmed["confirmed_files"] == 2
            with sessions() as db:
                problem = db.get(MatchedFile, ids[2])
                assert problem.review_status == "PENDING" and problem.matched_main_id == mains[0]
                assert Path(problem.current_path).read_bytes() == originals[problem.original_path]
                assert json.loads(db.get(Batch, bid).match_config_snapshot)["version"] == "preserve-me"
            restored = client.post(url + "/review-categories", json={"category": "问题图", "ignored": False})
            assert restored.status_code == 200 and restored.json()["ignored_categories"] == []
            assert client.get(url + "/review").json()["summary"]["unmatched"] == 1
            with sessions() as db:
                db.get(Batch, bid).status = "MATCHING"
                db.commit()
            assert client.post(url + "/review-categories", json={"category": "问题图", "ignored": True}).status_code == 409
    finally:
        main.app.dependency_overrides.clear()


@contextmanager
def conflict_client(example, monkeypatch):
    sessions, bid, mains, ids, originals, cache = example
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    monkeypatch.setattr(main, "learn_confirmed_references", lambda *_: 0)
    monkeypatch.setattr(main, "settings", replace(main.settings, upload_root=str(cache.parents[2])))
    with sessions() as db:
        for ident in ids[:2]:
            row = db.get(MatchedFile, ident)
            row.matched_main_id, row.matched_main_basename = mains[0], "SKU0"
            row.status, row.review_status = "MANUAL_CONFIRMED", "CONFIRMED"
        db.commit()
    def override_db():
        with sessions() as db:
            yield db
    main.app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(main.app) as client:
            yield client
    finally:
        main.app.dependency_overrides.clear()


def test_conflict_mark_preview_preserves_paths_and_safely_names_both_images(example, monkeypatch):
    sessions, bid, _, ids, originals, _ = example
    with conflict_client(example, monkeypatch) as client:
        url = f"/api/batches/{bid}"
        plan = client.post(url + "/rename-plan").json()
        assert plan["counts"] == {"RENAME_CONFLICT": 2, "SKIPPED": 1}
        assert all(row["source_directory"] and row["can_mark_conflict"] for row in plan["plans"][:2])
        first = client.post(url + "/rename-conflict-marks", json={"file_ids": [ids[0]]})
        assert first.status_code == 200
        rows = first.json()["rename_plan"]["plans"]
        assert rows[0]["target_filename"] == "A-original0-冲突图片1.png"
        assert rows[1]["can_mark_conflict"]  # Still visible after the other image is marked.
        second = client.post(url + "/rename-conflict-marks", json={"file_ids": ids[:2]})
        assert second.status_code == 200
        preview = second.json()["rename_plan"]
        assert preview["plans"][1]["target_filename"] == "A-original1-冲突图片2.png"
        with sessions() as db:
            for ident in ids[:2]:
                row = db.get(MatchedFile, ident)
                assert row.matched_main_id is None and row.status == "CONFLICT_MARKED"
                assert Path(row.current_path).read_bytes() == originals[row.original_path]
            assert json.loads(db.get(Batch, bid).match_config_snapshot)["version"] == "preserve-me"
        assert client.post(url + "/rename-plan/execute", json={"plan_group": plan["plan_group"]}).status_code == 409
        result = client.post(url + "/rename-plan/execute", json={"plan_group": preview["plan_group"]})
        assert result.status_code == 200
        assert result.json()["counts"] == {"RENAMED": 2, "SKIPPED": 1}
        with sessions() as db:
            for index, ident in enumerate(ids[:2]):
                row = db.get(MatchedFile, ident)
                assert row.current_filename == f"A-original{index}-冲突图片{index + 1}.png"
                assert Path(row.current_path).read_bytes() == originals[row.original_path]
                assert row.matched_main_id is None
        again = client.post(url + "/rename-plan").json()
        assert again["counts"] == {"NOOP": 2, "SKIPPED": 1}
        assert not any("A-A-" in row["target_filename"] for row in again["plans"] if row["target_filename"])


def test_unmark_does_not_recreate_duplicate_association(example, monkeypatch):
    sessions, bid, _, ids, originals, _ = example
    with conflict_client(example, monkeypatch) as client:
        url = f"/api/batches/{bid}"
        client.post(url + "/rename-plan")
        assert client.post(url + "/rename-conflict-marks", json={"file_ids": [ids[0]]}).status_code == 200
        result = client.post(url + "/rename-conflict-marks", json={"file_ids": []})
        assert result.status_code == 200
        with sessions() as db:
            row = db.get(MatchedFile, ids[0])
            assert row.matched_main_id is None and row.status == "UNMATCHED"
            assert Path(row.current_path).read_bytes() == originals[row.original_path]


def test_marked_name_never_overwrites_an_external_file(example, monkeypatch):
    sessions, bid, _, ids, originals, _ = example
    with conflict_client(example, monkeypatch) as client:
        url = f"/api/batches/{bid}"
        client.post(url + "/rename-plan")
        with sessions() as db:
            row = db.get(MatchedFile, ids[0])
            occupied = Path(row.current_path).with_name("A-original0-冲突图片1.png")
            occupied.write_bytes(b"external contents")
        preview = client.post(url + "/rename-conflict-marks", json={"file_ids": [ids[0]]}).json()["rename_plan"]
        assert preview["counts"]["RENAME_CONFLICT"] == 1
        assert client.post(url + "/rename-plan/execute", json={"plan_group": preview["plan_group"]}).status_code == 409
        assert occupied.read_bytes() == b"external contents"


def test_duplicate_positions_cannot_be_confirmed_individually_or_in_bulk(example, monkeypatch):
    _, bid, mains, ids, _, _ = example
    with conflict_client(example, monkeypatch) as client:
        for url, body in [(f"/api/main-images/{mains[0]}/confirm", {}),
                          (f"/api/batches/{bid}/confirm-groups", {"main_image_ids": mains}),
                          (f"/api/files/{ids[0]}/confirm", {})]:
            response = client.post(url, json=body)
            assert response.status_code == 409
            assert "original0.png" in response.json()["detail"] and "original1.png" in response.json()["detail"]
        review = client.get(f"/api/batches/{bid}/review").json()
        group = next(row for row in review["groups"] if row["main"]["id"] == mains[0])
        assert group["flags"]["conflict"] and not group["flags"]["confirmed"]
        assert client.post(f"/api/batches/{bid}/rename-conflict-marks", json={"file_ids": [999999]}).status_code == 400


def test_ignored_type_keeps_conflict_locations_and_restore_reveals_mark_choices(example, monkeypatch):
    _, bid, _, ids, _, _ = example
    with conflict_client(example, monkeypatch) as client:
        url = f"/api/batches/{bid}"
        result = client.post(url + "/review-categories", json={"category": "场景图", "ignored": True})
        assert result.status_code == 200
        problems = [row for row in result.json()["rename_plan"]["plans"] if row["normal_position_conflict"]]
        assert len(problems) == 2
        assert all(row["category_ignored"] and row["source_directory"] and not row["can_mark_conflict"] for row in problems)
        restored = client.post(url + "/review-categories", json={"category": "场景图", "ignored": False})
        assert restored.status_code == 200
        assert all(row["can_mark_conflict"] for row in restored.json()["rename_plan"]["plans"][:2])


def test_reassign_cannot_add_a_third_image_to_an_occupied_position(example, monkeypatch):
    sessions, bid, mains, ids, _, _ = example
    with conflict_client(example, monkeypatch) as client:
        with sessions() as db:
            row = db.get(MatchedFile, ids[2])
            row.category, row.matched_main_id, row.matched_main_basename = "场景图", mains[1], "SKU1"
            db.commit()
        response = client.post(f"/api/files/{ids[2]}/reassign", json={"main_image_id": mains[0]})
        assert response.status_code == 409
        with sessions() as db:
            assert db.get(MatchedFile, ids[2]).matched_main_id == mains[1]


def test_unassigned_duplicate_can_be_marked_without_occupying_another_main(example, monkeypatch):
    sessions, bid, _, ids, _, _ = example
    with conflict_client(example, monkeypatch) as client:
        with sessions() as db:
            row = db.get(MatchedFile, ids[0])
            row.matched_main_id, row.matched_main_basename = None, None
            row.review_status, row.status = "PENDING", "UNMATCHED"
            row.match_details_json = json.dumps({"duplicate_target_ids": ids[:2]})
            db.commit()
        preview = client.post(f"/api/batches/{bid}/rename-plan").json()
        assert preview["plans"][0]["can_mark_conflict"]
        result = client.post(f"/api/batches/{bid}/rename-conflict-marks", json={"file_ids": [ids[0]]})
        assert result.status_code == 200
        assert result.json()["rename_plan"]["plans"][0]["target_filename"] == "A-original0-冲突图片1.png"
