from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import (
    Batch,
    DerivedImageAsset,
    ImagePHashRegion,
    MainImage,
    MainImageFeature,
    MatchedFile,
    MatchingRun,
)
from app.services.detail_phash import build_detail_region_matrix
from app.services.main_body import EXPECTED_REGION_TYPES, ensure_main_body
from app.services.masks import save_main_profile_version, save_profile_version
from app.services.matcher import match_batch
from app.services.phash import PHashProvider
from app.services.scanner import scan_batch
from tests.support import mask_data_url


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


def test_scan_waits_for_manual_main_mask_before_generating_phash(tmp_path):
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
        assert main.body_crop_method is None
        assert main.body_mask_path is None
        assert main.body_crop_path is None
        assert not db.scalars(select(MainImageFeature)).all()
        assert not db.scalars(select(ImagePHashRegion)).all()


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


def test_manual_main_body_mask_keeps_only_body_hash_and_is_cached(tmp_path):
    root = tmp_path / "project"
    _patterned_main(root / "主图" / "SKU.png")

    with Session(_database(tmp_path)) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        main = db.scalar(select(MainImage))
        painted = Image.new("L", (200, 160), 0)
        ImageDraw.Draw(painted).ellipse((70, 25, 175, 145), fill=255)

        result = ensure_main_body(
            db,
            batch,
            main,
            manual_box=(0.4, 0.1, 0.95, 0.95),
            manual_mask=painted,
        )
        db.commit()

        assert main.body_crop_method == "MANUAL_MASK"
        assert result.cached is False
        assert set(result.regions) == {"BODY"}
        assert main.body_mask_path is None
        assert main.body_crop_path is None

        cached = ensure_main_body(db, batch, main)
        assert cached.cached is True
        assert cached.main.body_crop_method == "MANUAL_MASK"


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


def test_match_batch_uses_manual_masks_for_detail_and_main(tmp_path):
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
        main_profile, main_mask = save_main_profile_version(
            db,
            batch.id,
            mask_data_url((200, 160), (90, 0, 200, 160)),
            mains[0].id,
            (0.45, 0.0, 1.0, 1.0),
        )
        for main in mains:
            ensure_main_body(
                db,
                batch,
                main,
                manual_box=(0.45, 0.0, 1.0, 1.0),
                manual_mask=main_mask,
                manual_mask_path=main_profile.mask_path,
            )
        item = db.scalar(select(MatchedFile))
        file_profile = save_profile_version(
            db,
            batch.id,
            "细节展示图",
            mask_data_url((110, 160)),
            item.id,
            (0.0, 0.0, 1.0, 1.0),
        )

        match_batch(db, batch.id)

        run = db.scalar(select(MatchingRun))
        assert item.matched_main_basename == "A"
        assert item.match_method == "PHASH_MANUAL_MASK"
        assert item.phash_best_region_type == "MANUAL_MASK"
        assert run.algorithm_version == "manual_mask_product_reference_hungarian_v1"
        assert run.mask_profile_id == file_profile.id
        assert f"main_v{main_profile.version}" in run.feature_version
        assert run.model_name == "pHash64"
        assert item.candidates
        assert item.candidates[0].distance is not None
        assert item.candidates[0].best_distance == item.candidates[0].distance
        assert item.candidates[0].aggregated_score is not None
        wrong_main_crops = db.scalars(select(DerivedImageAsset).where(
            DerivedImageAsset.parent_main_image_id.is_not(None),
            DerivedImageAsset.mask_profile_id == file_profile.id,
            DerivedImageAsset.asset_type == "MASKED_CROP",
        )).all()
        assert not wrong_main_crops
