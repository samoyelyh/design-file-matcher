from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Batch,
    DerivedImageAsset,
    MainImage,
    MainImageFeature,
    MatchedFile,
)

from .media import open_visual
from .phash import PHashProvider
from .phash_matching import ensure_asset_phash, similarities_from_hashes


class PsdPreviewExtractor:
    feature_version = "psd_preview_v1"

    def extract(
        self,
        db: Session,
        batch: Batch,
        item: MatchedFile,
    ) -> DerivedImageAsset:
        existing = db.scalar(select(DerivedImageAsset).where(
            DerivedImageAsset.parent_file_id == item.id,
            DerivedImageAsset.asset_type == "PSD_PREVIEW",
            DerivedImageAsset.feature_version == self.feature_version,
        ))
        if existing and Path(existing.asset_path).is_file():
            item.preview_asset_id = existing.id
            return existing
        target = (
            Path(batch.root_path)
            / ".design_matcher"
            / str(batch.id)
            / "derived"
            / "psd_previews"
            / f"{item.id}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        open_visual(Path(item.current_path)).save(target, "PNG", optimize=True)
        asset = DerivedImageAsset(
            batch_id=batch.id,
            parent_file_id=item.id,
            asset_type="PSD_PREVIEW",
            asset_path=str(target),
            feature_version=self.feature_version,
        )
        db.add(asset)
        db.flush()
        item.preview_asset_id = asset.id
        return asset


def build_psd_matrix(
    db: Session,
    batch: Batch,
    psd_files: list[MatchedFile],
    mains: list[MainImage],
) -> tuple[np.ndarray, list[DerivedImageAsset]]:
    provider = PHashProvider()
    extractor = PsdPreviewExtractor()
    previews = [extractor.extract(db, batch, item) for item in psd_files]
    cached_hashes = dict(db.execute(select(
        MainImageFeature.main_image_id,
        MainImageFeature.phash_value,
    ).where(
        MainImageFeature.main_image_id.in_([main.id for main in mains]),
        MainImageFeature.feature_type == "FULL_PHASH",
        MainImageFeature.feature_version == provider.feature_version,
        MainImageFeature.phash_value.is_not(None),
    )).all())
    main_hashes = [
        cached_hashes.get(main.id) or provider.compute_path(Path(main.file_path))
        for main in mains
    ]
    preview_hashes = [ensure_asset_phash(asset, provider) for asset in previews]
    distances = provider.distance_matrix(preview_hashes, main_hashes)
    phash_similarity = similarities_from_hashes(preview_hashes, main_hashes)

    for row, item in enumerate(psd_files):
        order = np.argsort(distances[row])
        best_distance = int(distances[row, int(order[0])])
        item.phash_distance = best_distance
        item.match_method = "PSD_PREVIEW_PHASH"

    existing_phash = set(db.scalars(select(MainImageFeature.main_image_id).where(
        MainImageFeature.feature_type == "FULL_PHASH",
        MainImageFeature.feature_version == provider.feature_version,
    )).all())
    for main, value in zip(mains, main_hashes):
        if main.id not in existing_phash:
            db.add(MainImageFeature(
                main_image_id=main.id,
                feature_type="FULL_PHASH",
                feature_version=provider.feature_version,
                phash_value=value,
            ))
    db.flush()
    return phash_similarity, previews
