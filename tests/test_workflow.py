from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.main import _thumbnail
from app.models import (
    Batch,
    BatchProgress,
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
    Image.new("RGB", (48, 48), color).save(path)


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


def test_psd_scan_only_records_path_and_thumbnail_is_lazy(tmp_path):
    root = tmp_path / "设计目录"
    make_image(root / "主图" / "SKU.jpg", (255, 0, 0))
    psd_path = root / "psd" / "source.psd"
    psd_path.parent.mkdir(parents=True)
    PSDImage.new("RGB", (64, 48), color=(200, 20, 10)).save(psd_path)

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        result = scan_batch(db, batch.id)
        assert result["categories"] == {"PSD": 1}
        item = db.scalar(select(MatchedFile))
        assert item.extension == ".psd"
        assert not Path(item.thumbnail_path).exists()
        assert item.current_path == str(psd_path)
        response = _thumbnail(item.current_path, item.thumbnail_path)
        assert response.media_type == "image/jpeg"
        with Image.open(item.thumbnail_path) as thumbnail:
            assert thumbnail.size == (64, 48)


def test_real_nested_effect_folder_and_sibling_psd_are_discovered(tmp_path):
    root = tmp_path / "球迷主题-NFL-女款"
    effect_root = root / "球迷主题-NFL-女款-效果图"
    make_image(effect_root / "主图" / "88421.jpg", (255, 0, 0))
    make_image(effect_root / "双人模特图" / "model.jpg", (240, 10, 10))
    make_image(effect_root / "观赛图" / "watch.jpg", (230, 15, 10))
    psd_path = root / "psd" / "source.psd"
    psd_path.parent.mkdir(parents=True)
    PSDImage.new("RGB", (32, 32), color=(200, 20, 10)).save(psd_path)

    with Session(database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        result = scan_batch(db, batch.id)

        assert result["effect_root"] == "球迷主题-NFL-女款-效果图"
        assert result["main_directory"] == "球迷主题-NFL-女款-效果图\\主图"
        assert result["psd_directories"] == ["psd"]
        assert result["categories"] == {"PSD": 1, "双人模特图": 1, "观赛图": 1}
        categories = set(db.scalars(select(MatchedFile.category)).all())
        assert categories == {"PSD", "双人模特图", "观赛图"}
