from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models import Batch, MainImage, MatchedFile


def test_health_and_frontend_are_served():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        page = client.get("/")
        assert page.status_code == 200
        assert "设计文件匹配台" in page.text


def test_review_is_grouped_by_main_and_reports_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as db:
        batch = Batch(root_path=str(tmp_path), status="COMPLETED")
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="88421",
            filename="88421.jpg",
            file_path=str(tmp_path / "88421.jpg"),
        )
        db.add(main)
        db.flush()
        db.add(MatchedFile(
            batch_id=batch.id,
            category="观赛图",
            original_filename="IMG_A.jpg",
            original_path=str(tmp_path / "IMG_A.jpg"),
            current_filename="88421.jpg",
            current_path=str(tmp_path / "watch" / "88421.jpg"),
            extension=".jpg",
            matched_main_id=main.id,
            matched_main_basename=main.basename,
            top1_score=0.91,
            status="NORMAL",
        ))
        db.commit()
        batch_id = batch.id

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/batches/{batch_id}/review")
            assert response.status_code == 200
            payload = response.json()
            assert payload["groups"][0]["main"]["basename"] == "88421"
            assert payload["groups"][0]["flags"]["missing"] is True
            assert "PSD" in payload["groups"][0]["missing_categories"]
            assert payload["summary"]["missing"] == 1
    finally:
        app.dependency_overrides.clear()
