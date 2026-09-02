from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ImagePHashRegion, MainImage, MatchedFile

from .phash import PHashProvider, source_fingerprint


def is_detail_category(category: str) -> bool:
    folded = category.casefold().replace(" ", "")
    return any(keyword.replace(" ", "") in folded for keyword in settings.detail_category_keywords)


def ensure_file_phash(item: MatchedFile, provider: PHashProvider) -> str:
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


def build_detail_region_matrix(
    db: Session,
    detail_files: list[MatchedFile],
    mains: list[MainImage],
) -> np.ndarray:
    provider = PHashProvider()
    rows = list(db.scalars(select(ImagePHashRegion).where(
        ImagePHashRegion.main_image_id.in_([main.id for main in mains]),
        ImagePHashRegion.feature_version == settings.body_phash_version,
    )))
    by_main: dict[int, list[ImagePHashRegion]] = defaultdict(list)
    for row in rows:
        by_main[row.main_image_id].append(row)
    missing = [main.basename for main in mains if not by_main.get(main.id)]
    if missing:
        raise ValueError(f"主图尚未生成局部 pHash：{missing[0]}")

    matrix = np.zeros((len(detail_files), len(mains)), dtype=np.float32)
    best_types: list[list[str]] = []
    best_distances: list[list[int]] = []
    query_hashes = [ensure_file_phash(item, provider) for item in detail_files]
    for row_index, (item, query_hash) in enumerate(zip(detail_files, query_hashes)):
        row_types: list[str] = []
        row_distances: list[int] = []
        for column, main in enumerate(mains):
            distances = [
                (provider.distance(query_hash, region.phash_value), region.region_type)
                for region in by_main[main.id]
            ]
            best_distance, best_type = min(distances, key=lambda value: value[0])
            matrix[row_index, column] = 1.0 - best_distance / 64.0
            row_types.append(best_type)
            row_distances.append(best_distance)
        best_types.append(row_types)
        best_distances.append(row_distances)

    for row_index, item in enumerate(detail_files):
        best_column = int(np.argmax(matrix[row_index]))
        item.phash_distance = best_distances[row_index][best_column]
        item.phash_best_region_type = best_types[row_index][best_column]
        item.match_method = "PHASH_DETAIL_REGIONS"
    db.flush()
    return matrix
