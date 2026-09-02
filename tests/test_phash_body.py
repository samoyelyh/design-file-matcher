from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import (
    Batch,
    ImagePHashRegion,
    MainImage,
    MainImageFeature,
    MatchedFile,
    MatchingRun,
)
from app.services.detail_phash import build_detail_region_matrix
from app.services.main_body import EXPECTED_REGION_TYPES, ensure_main_body
from app.services.matcher import match_batch
from app.services.phash import PHashProvider
from app.services.scanner import scan_batch
from tests.support import configure_manual_masks


def _patterned_main(path: Path, *, side: str = "right") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (200, 160), "white")
    draw = ImageDraw.Draw(image)
    if side == "right":
        draw.rectangle((90, 20, 180, 145), fill=(35, 70, 150))
        draw.ellipse((112, 35, 165, 88), fill=(240, 210, 30))
        draw.polygon(((100, 130), (145, 92), (175, 140)), fill=(220, 40, 40))
    else:
        draw.rectangle((20, 15, 110, 145), fill=(35, 150, 70))
        draw.line((25, 25, 100, 135), fill=(20, 20, 20), width=12)
        draw.line((100, 25, 25, 135), fill=(20, 20, 20), width=12)
    image.save(path)


def _database(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'body.db'}")
    Base.metadata.create_all(engine)
    return engine


def test_scan_extracts_and_caches_main_body_phash(tmp_path):
    root = tmp_path / "project"
    _patterned_main(root / "主图" / "SKU.png")
    detail_path = root / "细节展示图" / "detail.png"
    detail_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 40), "blue").save(detail_path)

    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)

        main = db.scalar(select(MainImage))
        assert main.body_crop_method == "AUTO_WHITE_BG"
        assert Path(main.body_mask_path).is_file()
        assert Path(main.body_crop_path).is_file()
        assert 0.35 < main.body_bbox_x1 < 0.5
        assert main.body_bbox_x2 > 0.9

        feature_types = set(db.scalars(select(MainImageFeature.feature_type)))
        assert {"FULL_PHASH", "BODY_PHASH"} <= feature_types
        regions = list(db.scalars(select(ImagePHashRegion)))
        assert {row.region_type for row in regions} == EXPECTED_REGION_TYPES

        cached = ensure_main_body(db, batch, main)
        assert cached.cached is True


def test_manual_main_body_roi_recomputes_body_and_regions(tmp_path):
    root = tmp_path / "project"
    _patterned_main(root / "主图" / "SKU.png")
    detail_path = root / "细节展示图" / "detail.png"
    detail_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 40), "blue").save(detail_path)

    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        main = db.scalar(select(MainImage))

        result = ensure_main_body(
            db,
            batch,
            main,
            manual_box=(0.4, 0.1, 0.95, 0.95),
        )
        db.commit()

        assert main.body_crop_method == "MANUAL_ROI"
        assert result.cached is False
        assert tuple(round(value, 2) for value in (
            main.body_bbox_x1,
            main.body_bbox_y1,
            main.body_bbox_x2,
            main.body_bbox_y2,
        )) == (0.4, 0.1, 0.95, 0.95)
        assert set(result.regions) == EXPECTED_REGION_TYPES


def test_detail_phash_prefers_matching_main_region(tmp_path):
    root = tmp_path / "project"
    first_path = root / "主图" / "A.png"
    second_path = root / "主图" / "B.png"
    _patterned_main(first_path, side="right")
    _patterned_main(second_path, side="left")
    detail_path = root / "细节展示图" / "detail.png"
    detail_path.parent.mkdir(parents=True)
    with Image.open(first_path) as first:
        first.crop((90, 0, 200, 160)).save(detail_path)

    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        mains = list(db.scalars(select(MainImage).order_by(MainImage.basename)))
        for main in mains:
            ensure_main_body(db, batch, main, manual_box=(0.0, 0.0, 1.0, 1.0))
        item = db.scalar(select(MatchedFile))

        matrix = build_detail_region_matrix(db, [item], mains)
        provider = PHashProvider()
        query_hash = provider.compute_path(detail_path)
        first_full = db.scalar(select(ImagePHashRegion).where(
            ImagePHashRegion.main_image_id == mains[0].id,
            ImagePHashRegion.region_type == "FULL",
        ))

        assert matrix[0, 0] > matrix[0, 1]
        assert item.phash_distance < provider.distance(query_hash, first_full.phash_value)
        assert item.phash_best_region_type != "FULL"
        assert item.match_method == "PHASH_DETAIL_REGIONS"


def test_match_batch_routes_detail_folder_to_masked_region_phash(tmp_path):
    root = tmp_path / "project"
    first_path = root / "主图" / "A.png"
    second_path = root / "主图" / "B.png"
    _patterned_main(first_path, side="right")
    _patterned_main(second_path, side="left")
    detail_path = root / "细节展示图" / "detail.png"
    detail_path.parent.mkdir(parents=True)
    with Image.open(first_path) as first:
        first.crop((90, 0, 200, 160)).save(detail_path)

    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        mains = list(db.scalars(select(MainImage).order_by(MainImage.basename)))
        for main in mains:
            ensure_main_body(db, batch, main, manual_box=(0.0, 0.0, 1.0, 1.0))
        configure_manual_masks(db, batch.id)

        match_batch(db, batch.id)

        item = db.scalar(select(MatchedFile))
        run = db.scalar(select(MatchingRun))
        assert item.matched_main_basename == "A"
        assert item.match_method == "PHASH_DETAIL_REGIONS"
        assert item.phash_best_region_type != "FULL"
        assert run.algorithm_version == "detail_region_phash_v1"
        assert run.model_name == "pHash64"
