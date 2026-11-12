from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import (
    Batch,
    BatchProgress,
    MainImage,
    MatchCandidate,
    MatchedFile,
    RenameOperation,
)
from app.services.matcher import match_batch
from app.services.renamer import restore_original
from app.services.scanner import scan_batch
from tests.support import configure_manual_masks


class ColorEmbedder:
    def encode(self, paths):
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            rows.append(rgb / max(np.linalg.norm(rgb), 1e-12))
        return np.asarray(rows, dtype=np.float32)


def make_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), color)
    channel = int(np.argmax(color))
    boxes = [(4, 6, 18, 40), (28, 6, 42, 40), (14, 14, 34, 34)]
    ImageDraw.Draw(image).rectangle(boxes[channel], fill=(10, 10, 10))
    image.save(path)


def database(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return engine


def test_scan_match_top3_without_renaming_source_files(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "88421.jpg", (250, 5, 5))
    make_image(root / "主图" / "88422.jpg", (5, 250, 5))
    make_image(root / "主图" / "88423.jpg", (5, 5, 250))
    make_image(root / "观赛图" / "IMG_A.jpg", (220, 20, 10))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        result = scan_batch(db, batch.id)
        assert result["main_count"] == 3
        assert result["categories"] == {"观赛图": 1}

        configure_manual_masks(db, batch.id)
        matched = match_batch(db, batch.id, ColorEmbedder())
        assert matched["matched"] == 1
        item = db.scalar(select(MatchedFile))
        assert item.matched_main_basename == "88421"
        assert item.current_filename == "IMG_A.jpg"
        assert Path(item.current_path).exists()
        assert (root / "观赛图" / "IMG_A.jpg").exists()
        assert len(db.scalars(select(MatchCandidate)).all()) == 3
        assert not db.scalars(select(RenameOperation)).all()
        progress = db.get(BatchProgress, batch.id)
        assert progress.stage == "COMPLETED"
        assert progress.percent == 100
        assert progress.current == progress.total == 1

        restore_original(db, item)
        assert item.current_filename == "IMG_A.jpg"
        assert (root / "观赛图" / "IMG_A.jpg").exists()


def test_scan_preserves_visual_folder_names_as_categories(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (250, 5, 5))
    make_image(root / "细节图" / "detail.jpg", (220, 20, 10))
    make_image(root / "自定义-背面展示" / "back.jpg", (210, 30, 10))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()

        result = scan_batch(db, batch.id)

        assert result["categories"] == {"细节图": 1, "自定义-背面展示": 1}
        categories = set(db.scalars(select(MatchedFile.category)).all())
        assert categories == {"细节图", "自定义-背面展示"}


def test_incremental_scan_restores_file_and_preserves_latest_match_data(tmp_path):
    root = tmp_path / "设计目录"
    main_path = root / "主图" / "SKU.jpg"
    original = root / "观赛图" / "original.jpg"
    make_image(main_path, (250, 5, 5))
    make_image(original, (220, 20, 10))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root), status="CREATED")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        item = db.scalar(select(MatchedFile))
        main = db.scalar(select(MainImage))
        original_item_id = item.id
        original_main_id = main.id
        missing_current = original.parent / "SKU.jpg"
        original.rename(missing_current)
        item.current_path = str(missing_current)
        item.current_filename = missing_current.name
        item.matched_main_id = main.id
        item.matched_main_basename = main.basename
        item.status = "MANUAL_CONFIRMED"
        item.review_status = "CORRECTED"
        batch.status = "COMPLETED"
        db.commit()

        missing_current.unlink()
        make_image(original, (220, 20, 10))
        result = scan_batch(db, batch.id)

        db.refresh(item)
        db.refresh(batch)
        assert result["incremental"] is True
        assert result["restored_file_count"] == 1
        assert result["added_file_count"] == 0
        assert result["matching_data_preserved"] is True
        assert item.id == original_item_id
        assert main.id == original_main_id
        assert item.current_path == str(original)
        assert item.matched_main_id == main.id
        assert item.status == "MANUAL_CONFIRMED"
        assert item.review_status == "CORRECTED"
        assert batch.status == "COMPLETED"


def test_scan_ignores_white_background_folders_only_when_explicitly_configured(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.IGNORED_IMAGE_CATEGORY_KEYWORDS", ("白底",))
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (250, 5, 5))
    white_model = root / "白底模特图" / "white.jpg"
    white_plain = root / "白底图" / "plain.jpg"
    make_image(white_model, (220, 20, 10))
    make_image(white_plain, (210, 30, 10))
    make_image(root / "观赛图" / "watch.jpg", (200, 40, 10))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()

        result = scan_batch(db, batch.id)

        assert result["categories"] == {"观赛图": 1}
        assert set(db.scalars(select(MatchedFile.category))) == {"观赛图"}
        assert white_model.is_file()
        assert white_plain.is_file()


def test_incremental_scan_includes_white_folder_and_preserves_existing_confirmation(tmp_path):
    from app.services.masks import mask_readiness
    from app.services.visionmatch import plan

    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (250, 5, 5))
    make_image(root / "观赛图" / "watch.jpg", (220, 20, 10))
    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root), match_engine="VISIONMATCH")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        main = db.scalar(select(MainImage))
        previous = db.scalar(select(MatchedFile))
        previous.matched_main_id = previous.manual_product_id = main.id
        previous.matched_main_basename = main.basename
        previous.status = "MANUAL_CONFIRMED"
        previous.review_status = "CONFIRMED"
        batch.status = "COMPLETED"
        db.commit()
        before = (previous.id, previous.current_path, previous.matched_main_id, previous.review_status)

        white = root / "白底模特图" / "white.jpg"
        make_image(white, (210, 30, 10))
        result = scan_batch(db, batch.id)
        db.refresh(previous)

        assert result["categories"] == {"白底模特图": 1, "观赛图": 1}
        assert result["added_file_count"] == 1 and result["preserved_file_count"] == 1
        assert (previous.id, previous.current_path, previous.matched_main_id, previous.review_status) == before
        assert "白底模特图" in mask_readiness(db, batch.id)["required_categories"]
        pending = plan(db, batch.id)
        assert pending["folders"] == [{"category": "白底模特图", "mains": 1, "files": 1}]
        assert pending["calls_upper_bound"] == 1
        assert white.is_file()


def test_match_requires_manual_mask_before_model_loading(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (250, 5, 5))
    make_image(root / "观赛图" / "scene.jpg", (220, 20, 10))
    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        with pytest.raises(ValueError, match="请先完成手绘 Mask 和裁剪框"):
            match_batch(db, batch.id, ColorEmbedder())
        db.refresh(batch)
        assert batch.status == "SCANNED"
        assert not db.scalars(select(MatchCandidate)).all()


def test_global_assignment_prevents_duplicate_match(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "RED.jpg", (255, 0, 0))
    make_image(root / "主图" / "GREEN.jpg", (0, 255, 0))
    make_image(root / "观赛图" / "one.jpg", (250, 2, 2))
    make_image(root / "观赛图" / "two.jpg", (245, 4, 3))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        configure_manual_masks(db, batch.id)
        match_batch(db, batch.id, ColorEmbedder())
        items = db.scalars(select(MatchedFile).order_by(MatchedFile.id)).all()
        assert sum(item.matched_main_basename == "RED" for item in items) == 1
        assigned = [item.matched_main_id for item in items if item.matched_main_id]
        assert len(assigned) == len(set(assigned))
        assert not any(item.is_conflict for item in items)
        assert (root / "观赛图" / "one.jpg").exists()
        assert (root / "观赛图" / "two.jpg").exists()
        assert not (root / "观赛图" / "RED.jpg").exists()


def test_existing_untracked_target_is_never_overwritten(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (255, 0, 0))
    make_image(root / "细节展示图" / "incoming.png", (250, 1, 1))
    make_image(root / "细节展示图" / "SKU.png", (0, 0, 255))

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        configure_manual_masks(db, batch.id)
        match_batch(db, batch.id, ColorEmbedder())
        items = db.scalars(select(MatchedFile)).all()
        assert len(items) == 2
        assert not any(item.is_conflict for item in items)
        with Image.open(root / "细节展示图" / "SKU.png") as existing:
            assert existing.getpixel((0, 0))[2] > 200


def test_psd_scan_indexes_without_opening_file(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (255, 0, 0))
    psd_path = root / "psd" / "source.psd"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"not opened by scanner")

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        result = scan_batch(db, batch.id)
        assert result["categories"] == {"PSD": 1}
        assert result["psd_directories"] == ["psd"]
        assert db.scalar(select(MatchedFile)).category == "PSD"
        assert psd_path.read_bytes() == b"not opened by scanner"


def test_real_nested_effect_folder_and_sibling_psd_are_discovered(tmp_path):
    root = tmp_path / "球迷主题-NFL-女款"
    effect_root = root / "球迷主题-NFL-女款-效果图"
    make_image(effect_root / "主图" / "88421.jpg", (255, 0, 0))
    make_image(effect_root / "双人模特图" / "model.jpg", (240, 10, 10))
    make_image(effect_root / "观赛图" / "watch.jpg", (230, 15, 10))
    psd_path = root / "psd" / "source.psd"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"not opened by scanner")

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        result = scan_batch(db, batch.id)

        assert result["effect_root"] == "球迷主题-NFL-女款-效果图"
        assert result["main_directory"] == "球迷主题-NFL-女款-效果图\\主图"
        assert result["psd_directories"] == ["psd"]
        assert result["categories"] == {"双人模特图": 1, "观赛图": 1, "PSD": 1}
        categories = set(db.scalars(select(MatchedFile.category)).all())
        assert categories == {"双人模特图", "观赛图", "PSD"}
