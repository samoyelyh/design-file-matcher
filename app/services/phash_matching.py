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


def main_region_hashes(
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


def masked_query_hashes(
    db: Session,
    batch: Batch,
    files: list[MatchedFile],
    profile: MaskProfile,
) -> list[str]:
    provider = PHashProvider()
    processor = MaskedCropProcessor()
    assets = [
        processor.process_file(db, batch, item, profile).crop for item in files
    ]
    return [ensure_asset_phash(asset, provider) for asset in assets]


def build_full_phash_matrix(
    db: Session,
    files: list[MatchedFile],
    mains: list[MainImage],
    *,
    main_region: str = "FULL",
) -> np.ndarray:
    provider = PHashProvider()
    query_hashes = [ensure_file_full_phash(item, provider) for item in files]
    main_hashes = main_region_hashes(db, mains, main_region)
    return similarities_from_hashes(query_hashes, main_hashes)


def build_masked_phash_matrix(
    db: Session,
    batch: Batch,
    files: list[MatchedFile],
    mains: list[MainImage],
    profile: MaskProfile,
    *,
    query_hashes: list[str] | None = None,
) -> np.ndarray:
    query_hashes = query_hashes or masked_query_hashes(db, batch, files, profile)
    # The main-image crop is configured independently by the operator. Applying
    # an attachment folder's coordinates to a main image points at a different
    # part of the canvas and was the cause of the recent matching regression.
    main_hashes = main_region_hashes(db, mains, "BODY")
    for item in files:
        item.match_method = "PHASH_MANUAL_MASK"
        item.phash_best_region_type = "MANUAL_MASK"
    matrix = similarities_from_hashes(query_hashes, main_hashes)
    for row, item in enumerate(files):
        item.phash_distance = round((1.0 - float(np.max(matrix[row]))) * 64)
    db.flush()
    return matrix
