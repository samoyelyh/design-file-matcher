import importlib
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models import Batch, MainImage, MatchedFile
from tests.support import mask_data_url


def test_main_body_api_requires_and_applies_manual_mask(tmp_path):
    root = tmp_path / "body-api"
    source = root / "主图" / "SKU.png"
    source.parent.mkdir(parents=True)
    image = Image.new("RGB", (160, 120), "white")
    ImageDraw.Draw(image).rectangle((45, 15, 125, 105), fill=(30, 80, 170))
    image.save(source)
    second_source = source.with_name("SKU2.png")
    image.save(second_source)
    engine = create_engine(f"sqlite:///{tmp_path / 'body-api.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), status="SCANNED")
        db.add(batch)
        db.flush()
        main = MainImage(
            batch_id=batch.id,
            basename="SKU",
            filename="SKU.png",
            file_path=str(source),
        )
        db.add(main)
        db.add(MainImage(
            batch_id=batch.id,
            basename="SKU2",
            filename="SKU2.png",
            file_path=str(second_source),
        ))
        db.commit()
        batch_id = batch.id
        main_id = main.id

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/batches/{batch_id}/main-bodies")
            assert response.status_code == 409
            assert "手动遮罩" in response.json()["detail"]

            profiles = client.get(f"/api/batches/{batch_id}/mask-profiles")
            assert profiles.status_code == 200
            assert profiles.json()["categories"][0] == "主图"
            assert "主图" in profiles.json()["readiness"]["missing_categories"]
            references = client.get(
                f"/api/batches/{batch_id}/category-files/%E4%B8%BB%E5%9B%BE"
            )
            assert references.status_code == 200
            assert len(references.json()) == 2

            applied = client.post(
                f"/api/batches/{batch_id}/mask-profiles/%E4%B8%BB%E5%9B%BE",
                json={
                    "mask_data_url": mask_data_url(
                        (160, 120),
                        (50, 20, 110, 100),
                    ),
                    "reference_main_id": main_id,
                    "crop_box": [0.2, 0.1, 0.9, 0.95],
                    "name": "主图",
                },
            )
            assert applied.status_code == 200
            assert applied.json()["image_category"] == "主图"
            assert applied.json()["reference_main_id"] == main_id
            batch_items = client.get(
                f"/api/batches/{batch_id}/main-bodies"
            ).json()["items"]
            assert {item["method"] for item in batch_items} == {"MANUAL_MASK"}
            assert {tuple(item["bbox"]) for item in batch_items} == {
                (0.2, 0.1, 0.9, 0.95)
            }
            assert all(item["body_phash"] for item in batch_items)
    finally:
        app.dependency_overrides.clear()


def test_health_and_frontend_are_served():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        page = client.get("/")
        assert page.status_code == 200
        assert "图片自动命名" in page.text


def test_batch_creation_is_limited_to_configured_upload_root(tmp_path, monkeypatch):
    main_module = importlib.import_module("app.main")
    upload_root = tmp_path / "public-upload"
    allowed = upload_root / "package-a"
    outside = tmp_path / "private-package"
    allowed.mkdir(parents=True)
    outside.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'upload-root.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        replace(main_module.settings, upload_root=str(upload_root)),
    )

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            config = client.get("/api/workspace-config")
            assert config.json()["upload_root"] == str(upload_root)
            listing = client.get("/api/upload-folders")
            assert listing.status_code == 200
            assert listing.json()["is_root"] is True
            assert listing.json()["folders"] == [
                {"name": allowed.name, "path": str(allowed.resolve())}
            ]
            nested = client.get(
                "/api/upload-folders",
                params={"path": str(allowed)},
            )
            assert nested.json()["parent_path"] == str(upload_root.resolve())
            forbidden_listing = client.get(
                "/api/upload-folders",
                params={"path": str(outside)},
            )
            assert forbidden_listing.status_code == 400
            root_only = client.post(
                "/api/batches",
                json={"root_path": str(upload_root)},
            )
            assert root_only.status_code == 400
            assert "具体的文件包子目录" in root_only.json()["detail"]
            rejected = client.post(
                "/api/batches",
                json={"root_path": str(outside)},
            )
            assert rejected.status_code == 400
            assert "请先将文件包上传到公共盘目录" in rejected.json()["detail"]
            created = client.post(
                "/api/batches",
                json={"root_path": str(allowed)},
            )
            assert created.status_code == 200
            assert created.json()["root_path"] == str(allowed.resolve())
    finally:
        app.dependency_overrides.clear()


def test_review_uses_only_actual_folder_categories(tmp_path):
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
            original_path=str(tmp_path / "观赛图" / "IMG_A.jpg"),
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
            group = payload["groups"][0]
            assert group["flags"]["missing"] is False
            assert group["missing_categories"] == []
            assert [slot["name"] for slot in group["categories"]] == ["观赛图"]
            assert group["categories"][0]["category"] == "观赛图"
            assert payload["summary"].get("missing", 0) == 0
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
            available = client.get(
                f"/api/batches/{batch_id}/main-images/search",
                params={"exclude_confirmed": True},
            )
            assert available.status_code == 200
            assert available.json() == []
    finally:
        app.dependency_overrides.clear()


def test_reassign_swaps_existing_category_occupant(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'swap.db'}")
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as db:
        batch = Batch(root_path=str(tmp_path), status="COMPLETED")
        db.add(batch)
        db.flush()
        first_main = MainImage(
            batch_id=batch.id,
            basename="Atlanta",
            filename="Atlanta.jpg",
            file_path=str(tmp_path / "Atlanta.jpg"),
        )
        second_main = MainImage(
            batch_id=batch.id,
            basename="Miami",
            filename="Miami.jpg",
            file_path=str(tmp_path / "Miami.jpg"),
        )
        db.add_all([first_main, second_main])
        db.flush()
        first_file = MatchedFile(
            batch_id=batch.id,
            category="观赛图",
            original_filename="one.jpg",
            original_path=str(tmp_path / "one.jpg"),
            current_filename="one.jpg",
            current_path=str(tmp_path / "one.jpg"),
            extension=".jpg",
            matched_main_id=first_main.id,
            matched_main_basename=first_main.basename,
        )
        second_file = MatchedFile(
            batch_id=batch.id,
            category="观赛图",
            original_filename="two.jpg",
            original_path=str(tmp_path / "two.jpg"),
            current_filename="two.jpg",
            current_path=str(tmp_path / "two.jpg"),
            extension=".jpg",
            matched_main_id=second_main.id,
            matched_main_basename=second_main.basename,
        )
        db.add_all([first_file, second_file])
        db.commit()
        first_file_id = first_file.id
        second_file_id = second_file.id
        second_main_id = second_main.id
        first_main_id = first_main.id

    def override_db():
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/files/{first_file_id}/reassign",
                json={"main_image_id": second_main_id},
            )
            assert response.status_code == 200
            assert response.json()["swapped_file_id"] == second_file_id
        with Session(engine) as db:
            first = db.get(MatchedFile, first_file_id)
            second = db.get(MatchedFile, second_file_id)
            assert first.matched_main_id == second_main_id
            assert second.matched_main_id == first_main_id
            assert first.review_status == second.review_status == "CORRECTED"
    finally:
        app.dependency_overrides.clear()
