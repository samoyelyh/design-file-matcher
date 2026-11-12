"""A verification run must not reset matching tasks in the running application."""
import importlib
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.db import Base, SessionLocal, engine
from app.models import Batch, BatchProgress


def test_testclient_startup_recovers_only_the_isolated_test_database(monkeypatch):
    database = Path(engine.url.database).resolve()
    assert database.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert database.parent.name.startswith("design-matcher-tests-")
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        batch = Batch(root_path="isolated startup sentinel", status="MATCHING")
        db.add(batch)
        db.flush()
        db.add(BatchProgress(batch_id=batch.id, stage="VISIONMATCH", percent=50))
        db.commit()
        batch_id = batch.id
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
    with SessionLocal() as db:
        assert db.get(Batch, batch_id).status == "SCANNED"
        assert db.get(BatchProgress, batch_id).stage == "READY"
