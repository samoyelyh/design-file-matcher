from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, DerivedImageAsset, MainImageFeature, MatchedFile
from app.services.matcher import match_batch
from app.services.media import open_visual
from app.services.psd import PHashProvider
from app.services.scanner import scan_batch


class VisualColorProvider:
    device = "TEST/CPU"

    def encode(self, paths):
        rows = []
        for path in paths:
            image = open_visual(Path(path))
            rgb = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            rows.append(rgb / max(float(np.linalg.norm(rgb)), 1e-12))
        return np.asarray(rows)


def test_psd_preview_uses_embedding_and_keeps_phash_traceable(tmp_path):
    root = tmp_path / "project"
    main_path = root / "主图" / "SKU.jpg"
    main_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 20, 10)).save(main_path)
    psd_path = root / "psd" / "source.psd"
    psd_path.parent.mkdir(parents=True)
    PSDImage.new("RGB", (64, 48), color=(200, 20, 10)).save(psd_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'psd.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        batch = Batch(root_path=str(root), product_category="橄榄球衣")
        db.add(batch)
        db.commit()
        scan_batch(db, batch.id)
        match_batch(db, batch.id, VisualColorProvider())
        item = db.scalar(select(MatchedFile))
        assert item.matched_main_basename == "SKU"
        assert item.phash_distance == 0
        assert item.match_method == "DINO_COSINE_PSD"
        preview = db.scalar(select(DerivedImageAsset).where(
            DerivedImageAsset.asset_type == "PSD_PREVIEW"
        ))
        assert preview.parent_file_id == item.id
        assert Path(preview.asset_path).is_file()
        features = db.scalars(select(MainImageFeature)).all()
        assert {feature.feature_type for feature in features} == {
            "FULL_PHASH", "FULL_EMBEDDING"
        }


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
