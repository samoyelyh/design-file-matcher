import importlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base, _configure_sqlite_connection, get_db
from app.models import Batch, DerivedImageAsset, MainImage, MatchedFile, ProductReference
from app.services.cropper import MaskedCropProcessor
from tests.support import configure_manual_masks

main_module = importlib.import_module("app.main")


@pytest.fixture
def isolated_app(monkeypatch):
    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(main_module.qwen_lab, "status", lambda: {"reachable": False})
    yield main_module.app
    main_module.app.dependency_overrides.clear()


def test_concurrent_bulk_and_single_confirmation_keep_references_and_allow_readers(tmp_path, monkeypatch, isolated_app):
    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent.db'}", connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _configure_sqlite_connection)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, autoflush=False, expire_on_commit=False)
    source_bytes = {}
    with sessions() as db:
        assert db.connection().exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert db.connection().exec_driver_sql("PRAGMA busy_timeout").scalar() == 30000
        assert db.connection().exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        batch = Batch(root_path=str(tmp_path), status="COMPLETED")
        db.add(batch)
        db.flush()
        mains, files = [], []
        for index, color in enumerate(("red", "blue")):
            main_path = tmp_path / f"SKU{index}.png"
            target_path = tmp_path / "details" / f"IMG{index}.png"
            target_path.parent.mkdir(exist_ok=True)
            for path in (main_path, target_path):
                Image.new("RGB", (48, 48), color).save(path)
                source_bytes[path] = path.read_bytes()
            main = MainImage(batch_id=batch.id, basename=main_path.stem, filename=main_path.name, file_path=str(main_path))
            db.add(main)
            db.flush()
            item = MatchedFile(batch_id=batch.id, category="details", original_filename=target_path.name,
                               current_filename=target_path.name, original_path=str(target_path),
                               current_path=str(target_path), extension=".png", matched_main_id=main.id,
                               matched_main_basename=main.basename, status="AMBIGUOUS", review_status="PENDING")
            db.add(item)
            db.flush()
            mains.append(main.id)
            files.append(item.id)
        db.commit()
        batch_id = batch.id
        configure_manual_masks(db, batch_id)

    def override_db():
        with sessions() as db:
            yield db

    isolated_app.dependency_overrides[get_db] = override_db
    written, release = Event(), Event()
    original_process = MaskedCropProcessor.process_file

    def slow_crop(self, *args, **kwargs):
        result = original_process(self, *args, **kwargs)
        if not written.is_set():
            written.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(MaskedCropProcessor, "process_file", slow_crop)
    original_learn = main_module.learn_confirmed_references
    mutex = Lock()
    counts = {"active": 0, "peak": 0}

    def learn(*args, **kwargs):
        with mutex:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            return original_learn(*args, **kwargs)
        finally:
            with mutex:
                counts["active"] -= 1

    monkeypatch.setattr(main_module, "learn_confirmed_references", learn)
    with TestClient(isolated_app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        bulk = pool.submit(client.post, f"/api/batches/{batch_id}/confirm-groups", json={"main_image_ids": mains})
        try:
            assert written.wait(5)
            single = pool.submit(client.post, f"/api/files/{files[0]}/confirm", json={})
            with sessions() as reader:
                assert reader.scalar(select(func.count()).select_from(MatchedFile)) == 2
        finally:
            release.set()
        assert bulk.result(timeout=10).status_code == 200
        assert single.result(timeout=10).status_code == 200
    assert counts["peak"] == 1
    with sessions() as db:
        assert all(row.review_status == "CONFIRMED" for row in db.scalars(select(MatchedFile)))
        assert db.scalar(select(func.count()).select_from(ProductReference)) == 2
        assert db.scalar(select(func.count()).select_from(DerivedImageAsset).where(DerivedImageAsset.asset_type == "MASKED_CROP")) == 2
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    engine.dispose()


def test_locked_database_returns_json_and_releases_review_guard(isolated_app):
    def failing_db():
        raise OperationalError("UPDATE matched_files", {}, sqlite3.OperationalError("database is locked"))
        yield

    isolated_app.dependency_overrides[get_db] = failing_db
    with TestClient(isolated_app) as client:
        response = client.post("/api/files/1/confirm", json={})
    assert response.status_code == 409
    assert "无需重新匹配" in response.json()["detail"]
    assert not main_module._review_write_lock.locked()
