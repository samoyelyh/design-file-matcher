from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models import Batch, MainImage, MaskProfile, MatchedFile


def test_health_and_frontend_are_served():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        page = client.get("/")
        assert page.status_code == 200
        assert "图片自动命名" in page.text


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


def test_bulk_confirm_refreshes_rename_plan(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'bulk.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as db:
        batch = Batch(root_path=str(tmp_path), status="COMPLETED")
        db.add(batch)
        db.flush()
        main_ids = []
        for index in range(2):
            main = MainImage(
                batch_id=batch.id,
                basename=f"SKU-{index}",
                filename=f"SKU-{index}.jpg",
                file_path=str(tmp_path / f"main-{index}.jpg"),
            )
            db.add(main)
            db.flush()
            main_ids.append(main.id)
            source = tmp_path / "watch" / f"source-{index}.jpg"
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(b"source")
            db.add(MatchedFile(
                batch_id=batch.id,
                category="观赛图",
                original_filename=source.name,
                original_path=str(source),
                current_filename=source.name,
                current_path=str(source),
                extension=".jpg",
                matched_main_id=main.id,
                matched_main_basename=main.basename,
                status="LOW_CONFIDENCE",
                review_status="PENDING",
            ))
        db.commit()
        batch_id = batch.id

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/batches/{batch_id}/confirm-groups",
                json={"main_image_ids": main_ids},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["confirmed_groups"] == 2
            assert payload["confirmed_files"] == 2
            assert payload["rename_plan"]["counts"] == {"READY": 2}
            latest = client.get(f"/api/batches/{batch_id}/rename-plan").json()
            assert latest["plan_group"] == payload["rename_plan"]["plan_group"]
    finally:
        app.dependency_overrides.clear()


def test_completed_region_template_can_be_reused_in_new_batch(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reuse.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    mask_path = source_root / "mask.png"
    Image.new("L", (40, 40), 255).save(mask_path)
    reference_path = target_root / "scene.jpg"
    Image.new("RGB", (80, 60), "red").save(reference_path)

    with Session(engine) as db:
        source_batch = Batch(root_path=str(source_root), product_category="橄榄球衣")
        target_batch = Batch(root_path=str(target_root), product_category="橄榄球衣")
        db.add_all([source_batch, target_batch])
        db.flush()
        db.add(MaskProfile(
            batch_id=source_batch.id,
            product_category="橄榄球衣",
            image_category="观赛图",
            name="橄榄球衣 / 观赛图",
            version=1,
            reference_width=40,
            reference_height=40,
            mask_path=str(mask_path),
            crop_left=0.1,
            crop_top=0.2,
            crop_right=0.9,
            crop_bottom=0.8,
            creation_method="MANUAL",
            enabled=True,
        ))
        db.add(MatchedFile(
            batch_id=target_batch.id,
            category="观赛图",
            original_filename="scene.jpg",
            original_path=str(reference_path),
            current_filename="scene.jpg",
            current_path=str(reference_path),
            extension=".jpg",
        ))
        db.commit()
        target_id = target_batch.id

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            listing = client.get(f"/api/batches/{target_id}/mask-profiles").json()
            reusable = listing["reusable_profiles"]
            assert len(reusable) == 1
            response = client.post(
                f"/api/batches/{target_id}/mask-profiles/观赛图/reuse",
                json={"source_profile_id": reusable[0]["id"]},
            )
            assert response.status_code == 200
            assert response.json()["crop_box"] == [0.1, 0.2, 0.9, 0.8]
        with Session(engine) as db:
            copied = db.scalar(select(MaskProfile).where(
                MaskProfile.batch_id == target_id
            ))
            assert copied is not None
            assert copied.creation_method == "MANUAL"
    finally:
        app.dependency_overrides.clear()
