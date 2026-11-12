from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base, get_db
from app.models import Batch, DerivedImageAsset, MatchedFile, MatchingRun, MainImage
from app.services import visionmatch as vm
from app.services.matcher import match_batch
from app.services.masks import mask_readiness
from app.services.phash import PHashProvider
from app.services.psd import PsdPreviewExtractor
from app.services.rename_plan import RenamePlanService
from app.services.scanner import scan_batch
from tests.support import configure_manual_masks


def design(reverse=False):
    image = Image.new("RGB", (96, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 7, 34, 72), fill="black")
    draw.ellipse((43, 13, 87, 47), fill="navy")
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if reverse else image


@pytest.fixture
def psd_case(tmp_path):
    root = tmp_path / "package"
    main_dir = root / "效果图" / "主图"
    main_dir.mkdir(parents=True)
    design().save(main_dir / "SKU-A.png")
    design(True).save(main_dir / "SKU-B.png")
    source = root / "psd" / "未命名设计.psd"
    source.parent.mkdir()
    PSDImage.frompil(design()).save(source)
    engine = create_engine(f"sqlite:///{tmp_path / 'psd.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        yield db, batch, source
    engine.dispose()


def test_psd_scan_indexes_without_decoding_and_preserves_existing_results(psd_case, monkeypatch):
    db, batch, source = psd_case
    monkeypatch.setattr("app.services.psd.open_visual", lambda *_: pytest.fail("scan must not decode PSD"))
    result = scan_batch(db, batch.id)
    assert result["categories"] == {"PSD": 1} and result["psd_directories"] == ["psd"]
    item = db.scalar(select(MatchedFile))
    item.status, item.review_status = "MANUAL_CONFIRMED", "CONFIRMED"
    db.commit()
    again = scan_batch(db, batch.id)
    assert again["preserved_file_count"] == 1 and again["added_file_count"] == 0
    assert item.review_status == "CONFIRMED" and Path(item.current_path) == source


def test_psd_preview_matches_without_masks_and_only_renames_after_confirmation(psd_case):
    db, batch, source = psd_case
    original = source.read_bytes()
    scan_batch(db, batch.id)
    assert mask_readiness(db, batch.id)["ready"]
    result = match_batch(db, batch.id)
    item = db.scalar(select(MatchedFile))
    assert result["psd_matched"] == 1 and result["matched"] == 1
    assert item.matched_main_basename == "SKU-A"
    assert item.match_method == "PSD_PREVIEW_PHASH" and item.phash_distance == 0
    assert db.scalar(select(MatchingRun)).mask_profile_id is None
    assert RenamePlanService().latest(db, batch.id)["counts"] == {"SKIPPED": 1}
    assert source.read_bytes() == original
    item.review_status, item.status = "CONFIRMED", "MANUAL_CONFIRMED"
    db.commit()
    plan = RenamePlanService().generate(db, batch.id)
    assert plan["counts"] == {"READY": 1}
    assert RenamePlanService().execute(db, batch.id, plan["plan_group"])["counts"] == {"RENAMED": 1}
    assert not source.exists() and source.with_name("SKU-A.psd").read_bytes() == original


def test_psd_preview_cache_reuses_preview_and_invalidates_when_source_changes(psd_case, monkeypatch):
    db, batch, source = psd_case
    scan_batch(db, batch.id)
    item = db.scalar(select(MatchedFile))
    extractor = PsdPreviewExtractor()
    first = extractor.extract(db, batch, item)
    db.commit()
    original_open = importlib.import_module("app.services.psd").open_visual
    monkeypatch.setattr("app.services.psd.open_visual", lambda *_: pytest.fail("unchanged PSD should use cached preview"))
    assert extractor.extract(db, batch, item).id == first.id
    monkeypatch.setattr("app.services.psd.open_visual", original_open)
    PSDImage.frompil(design(True)).save(source)
    changed = extractor.extract(db, batch, item)
    assert changed.id != first.id and changed.asset_path != first.asset_path
    with Image.open(changed.asset_path) as preview:
        assert preview.tobytes() == design(True).tobytes()


def test_visionmatch_stages_only_preview_and_writes_back_original_psd_identity(psd_case, tmp_path):
    db, batch, source = psd_case
    original = source.read_bytes()
    scan_batch(db, batch.id)
    item = db.scalar(select(MatchedFile))
    mains = list(db.scalars(select(MainImage).order_by(MainImage.basename)))
    mapping = {**vm.stage(mains, tmp_path / "main", "main"),
               **vm.stage([item], tmp_path / "target", "file", db=db, batch=batch)}
    images = [{"id": n, "path": path, "sha256": row["hash"], "local_id": ("A" if row["kind"] == "main" else "B") + str(n)}
              for n, (path, row) in enumerate(mapping.items(), 1)]
    target = next(row for row in images if row["local_id"].startswith("B"))
    assert Path(target["path"]).suffix == ".png"
    state = {"images": images, "matches": [{"target_id": target["id"], "assigned_source_id": 1,
             "status": "MATCHED", "suggestion": {"decision": "MATCH"}, "candidates": []}]}
    assert vm.apply_result(db, batch.id, state, mapping, {"remote_batch_id": "test", "run_id": "test"}) == 1
    assert item.matched_main_basename == "SKU-A" and item.extension == ".psd"
    assert source.read_bytes() == original
    assert mapping[target["path"].casefold()]["source_hash"] == hashlib.sha256(original).hexdigest()


def test_psd_preview_thumbnail_candidates_and_ignored_confirmation_api(psd_case, monkeypatch):
    db, batch, source = psd_case
    scan_batch(db, batch.id)
    match_batch(db, batch.id)
    item = db.scalar(select(MatchedFile))
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    main.app.dependency_overrides[get_db] = lambda: db
    try:
        with TestClient(main.app) as client:
            for kind in ("preview", "thumbnail"):
                response = client.get(f"/api/files/{item.id}/{kind}")
                assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
            candidates = client.get(f"/api/files/{item.id}/product-candidates?exclude_confirmed=false").json()
            assert len(candidates) == 2 and candidates[0]["basename"] == "SKU-A"
            ignored = client.post(f"/api/batches/{batch.id}/review-categories", json={"category": "PSD", "ignored": True})
            assert ignored.status_code == 200 and ignored.json()["rename_plan"]["counts"] == {"IGNORED": 1}
            assert client.post(f"/api/files/{item.id}/confirm", json={}).status_code == 409
            assert source.exists()
    finally:
        main.app.dependency_overrides.clear()


def test_unreadable_psd_is_not_silently_matched_or_renamed(psd_case):
    db, batch, source = psd_case
    source.write_bytes(b"broken PSD")
    scan_batch(db, batch.id)
    with pytest.raises(Exception):
        match_batch(db, batch.id)
    assert batch.status == "FAILED" and source.read_bytes() == b"broken PSD"
    assert db.scalar(select(DerivedImageAsset)) is None


def test_phash_hamming_distance_is_zero_for_same_visual(tmp_path):
    path = tmp_path / "image.png"
    design().save(path)
    provider = PHashProvider()
    value = provider.compute_path(path)
    assert len(value) == 16 and provider.distance(value, value) == 0


def test_psd_and_regular_images_share_matching_review_and_confirmed_naming(psd_case):
    db, batch, source = psd_case
    scene = Path(batch.root_path) / "效果图" / "场景图" / "scene.png"
    scene.parent.mkdir()
    design().save(scene)
    scan_batch(db, batch.id)
    configure_manual_masks(db, batch.id)
    result = match_batch(db, batch.id)
    assert result["matched"] == 2 and result["psd_matched"] == 1
    files = list(db.scalars(select(MatchedFile)))
    assert {item.matched_main_basename for item in files} == {"SKU-A"}
    assert RenamePlanService().latest(db, batch.id)["counts"] == {"SKIPPED": 2}
    assert source.exists() and scene.exists()


def test_psd_defaults_to_visionmatch_and_needs_explicit_call_budget(psd_case, monkeypatch):
    db, batch, _ = psd_case
    scan_batch(db, batch.id)
    main = importlib.import_module("app.main")
    assert main.BatchCreate(root_path="example").match_engine == "VISIONMATCH"
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    monkeypatch.setattr(main.visionmatch, "status", lambda: {"id": "VISIONMATCH", "selectable": True})
    submitted = []
    monkeypatch.setattr(main.executor, "submit", lambda *args: submitted.append(args))
    main.app.dependency_overrides[get_db] = lambda: db
    try:
        with TestClient(main.app) as client:
            config = client.get("/api/workspace-config").json()
            assert config["default_match_engine"] == "VISIONMATCH"
            assert [engine["id"] for engine in config["matching_engines"]] == ["VISIONMATCH"]
            retired = client.post(f"/api/batches/{batch.id}/match", json={"folder_engines": {"PSD": {"engine": "PHASH"}}})
            assert retired.status_code == 409 and not submitted
            body = {"folder_engines": {"PSD": {"engine": "VISIONMATCH"}}}
            response = client.post(f"/api/batches/{batch.id}/match", json=body)
            assert response.status_code == 409 and "预算" in response.json()["detail"]
            body.update(visionmatch_confirmed_budget=True, visionmatch_max_calls=1)
            response = client.post(f"/api/batches/{batch.id}/match", json=body)
            assert response.status_code == 200 and response.json()["status"] == "QUEUED"
            assert submitted and response.json()["folder_engines"]["PSD"]["engine"] == "VISIONMATCH"
    finally:
        main.app.dependency_overrides.clear()
