from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Batch,
    DerivedImageAsset,
    ImagePHashRegion,
    MainImage,
    MainImageFeature,
    MaskProfile,
    MatchedFile,
)

from .cropper import MaskedCropProcessor
from .phash import PHashProvider, source_fingerprint


def similarities_from_hashes(
    query_hashes: list[str], reference_hashes: list[str]
) -> np.ndarray:
    distances = PHashProvider().distance_matrix(query_hashes, reference_hashes)
    return 1.0 - distances.astype(np.float32) / 64.0


def ensure_file_full_phash(item: MatchedFile, provider: PHashProvider) -> str:
    path = Path(item.current_path)
    size, mtime_ns = source_fingerprint(path)
    if (
        item.phash_value
        and item.phash_source_size == size
        and item.phash_source_mtime_ns == mtime_ns
    ):
        return item.phash_value
    value = provider.compute_path(path)
    item.phash_value = value
    item.phash_source_size = size
    item.phash_source_mtime_ns = mtime_ns
    return value


def ensure_asset_phash(asset: DerivedImageAsset, provider: PHashProvider) -> str:
    path = Path(asset.asset_path)
    size, mtime_ns = source_fingerprint(path)
    if (
        asset.phash_value
        and asset.phash_source_size == size
        and asset.phash_source_mtime_ns == mtime_ns
    ):
        return asset.phash_value
    value = provider.compute_path(path)
    asset.phash_value = value
    asset.phash_source_size = size
    asset.phash_source_mtime_ns = mtime_ns
    return value


def _main_region_hashes(
    db: Session, mains: list[MainImage], region_type: str
) -> list[str]:
    rows = dict(db.execute(select(
        ImagePHashRegion.main_image_id,
        ImagePHashRegion.phash_value,
    ).where(
        ImagePHashRegion.main_image_id.in_([main.id for main in mains]),
        ImagePHashRegion.region_type == region_type,
    )).all())
    missing = [main.basename for main in mains if main.id not in rows]
    if missing:
        raise ValueError(f"主图尚未生成 {region_type} pHash：{missing[0]}")
    return [rows[main.id] for main in mains]


def build_full_phash_matrix(
    db: Session,
    files: list[MatchedFile],
    mains: list[MainImage],
    *,
    main_region: str = "FULL",
) -> np.ndarray:
    provider = PHashProvider()
    query_hashes = [ensure_file_full_phash(item, provider) for item in files]
    main_hashes = _main_region_hashes(db, mains, main_region)
    return similarities_from_hashes(query_hashes, main_hashes)


def _record_masked_main_feature(
    db: Session,
    main: MainImage,
    profile: MaskProfile,
    crop_asset: DerivedImageAsset,
    phash_value: str,
) -> None:
    version = f"masked_phash_v1:mask_v{profile.version}"
    feature = db.scalar(select(MainImageFeature).where(
        MainImageFeature.main_image_id == main.id,
        MainImageFeature.feature_type == "MASKED_PHASH",
        MainImageFeature.feature_version == version,
        MainImageFeature.mask_profile_id == profile.id,
    ))
    if not feature:
        feature = MainImageFeature(
            main_image_id=main.id,
            feature_type="MASKED_PHASH",
            feature_version=version,
            mask_profile_id=profile.id,
        )
        db.add(feature)
    feature.phash_value = phash_value
    feature.crop_asset_id = crop_asset.id


def build_masked_phash_matrix(
    db: Session,
    batch: Batch,
    files: list[MatchedFile],
    mains: list[MainImage],
    profile: MaskProfile,
) -> np.ndarray:
    provider = PHashProvider()
    processor = MaskedCropProcessor()
    file_assets = [
        processor.process_file(db, batch, item, profile).crop for item in files
    ]
    main_assets = [
        processor.process_main(db, batch, main, profile).crop for main in mains
    ]
    query_hashes = [ensure_asset_phash(asset, provider) for asset in file_assets]
    main_hashes = [ensure_asset_phash(asset, provider) for asset in main_assets]
    for item in files:
        item.match_method = "MASKED_PHASH"
        item.phash_best_region_type = "MASKED"
    for main, asset, value in zip(mains, main_assets, main_hashes):
        _record_masked_main_feature(db, main, profile, asset, value)
    matrix = similarities_from_hashes(query_hashes, main_hashes)
    for row, item in enumerate(files):
        item.phash_distance = round((1.0 - float(np.max(matrix[row]))) * 64)
    db.flush()
    return matrix
