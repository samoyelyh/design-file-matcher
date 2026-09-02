from __future__ import annotations

from PIL import Image
from psd_tools import PSDImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import (
    Batch,
    DerivedImageAsset,
    MainImageFeature,
    MatchedFile,
    MatchingRun,
)
from app.services.matcher import match_batch
from app.services.psd import PHashProvider
from app.services.scanner import scan_batch
from tests.support import configure_manual_masks


def test_psd_is_linked_by_filename_without_visual_matching(tmp_path):
    root = tmp_path / "project"
    main_path = root / "主图" / "SKU.jpg"
    main_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 20, 10)).save(main_path)
    psd_path = root / "psd" / "SKU.psd"
    psd_path.parent.mkdir(parents=True)
    PSDImage.new("RGB", (64, 48), color=(200, 20, 10)).save(psd_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'psd.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        configure_manual_masks(db, batch.id)
        result = match_batch(db, batch.id)
        item = db.scalar(select(MatchedFile))
        assert item.matched_main_basename == "SKU"
        assert item.status == "MATCHED"
        assert item.review_status == "CONFIRMED"
        assert item.phash_distance is None
        assert item.phash_value is None
        assert item.algorithm_score is None
        assert item.match_method == "FILENAME_EXACT"
        assert item.algorithm_version == "psd_filename_exact_v1"
        assert result["matched"] == 0
        assert result["psd_filename_linked"] == 1
        preview = db.scalar(select(DerivedImageAsset).where(
            DerivedImageAsset.asset_type == "PSD_PREVIEW"
        ))
        assert preview is None
        assert db.scalar(select(MatchingRun).where(
            MatchingRun.category == "PSD"
        )) is None
        features = db.scalars(select(MainImageFeature)).all()
        assert {feature.feature_type for feature in features} == {
            "FULL_PHASH", "BODY_PHASH"
        }


def test_psd_with_different_filename_is_not_forced_to_a_main(tmp_path):
    root = tmp_path / "project"
    main_path = root / "主图" / "SKU.jpg"
    main_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 20, 10)).save(main_path)
    psd_path = root / "psd" / "OTHER.psd"
    psd_path.parent.mkdir(parents=True)
    PSDImage.new("RGB", (64, 48), color=(200, 20, 10)).save(psd_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'psd-mismatch.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        batch = Batch(root_path=str(root))
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        configure_manual_masks(db, batch.id)
        result = match_batch(db, batch.id)

        item = db.scalar(select(MatchedFile))
        assert item.matched_main_id is None
        assert item.status == "UNMATCHED"
        assert item.match_method == "FILENAME_EXACT"
        assert result["psd_filename_linked"] == 0


def test_phash_hamming_distance_is_zero_for_same_visual(tmp_path):
    path = tmp_path / "image.png"
    image = Image.new("RGB", (40, 40), "white")
    for x in range(20):
        for y in range(40):
            image.putpixel((x, y), (20, 20, 20))
    image.save(path)
    provider = PHashProvider()
    value = provider.compute_path(path)
    assert len(value) == 16
    assert provider.distance(value, value) == 0
