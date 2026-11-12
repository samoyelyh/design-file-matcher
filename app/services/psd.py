from __future__ import annotations

from pathlib import Path
import uuid

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
from .phash import source_fingerprint
from .phash_matching import ensure_asset_phash, similarities_from_hashes


class PsdPreviewExtractor:
    feature_version = "psd_preview_v2"

    def extract(
        self,
        db: Session,
        batch: Batch,
        item: MatchedFile,
    ) -> DerivedImageAsset:
        source = Path(item.current_path)
        size, mtime = source_fingerprint(source)
        version = f"{self.feature_version}:{size}:{mtime}"
        existing = db.scalar(select(DerivedImageAsset).where(
            DerivedImageAsset.parent_file_id == item.id,
            DerivedImageAsset.asset_type == "PSD_PREVIEW",
            DerivedImageAsset.feature_version == version,
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
            / f"{item.id}-{size}-{mtime}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}-{uuid.uuid4().hex}.tmp")
        try:
            open_visual(source).save(temporary, "PNG", optimize=True)
            if source_fingerprint(source) != (size, mtime):
                raise ValueError(f"PSD 在提取预览期间发生变化：{source.name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        asset = DerivedImageAsset(
            batch_id=batch.id,
            parent_file_id=item.id,
            asset_type="PSD_PREVIEW",
            asset_path=str(target),
            feature_version=version,
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
    main_hashes = [
        provider.compute_path(Path(main.file_path))
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


def build_psd_output(db, batch, files, mains):
    """Adapt the existing full-preview matrix to the shared review pipeline."""
    from app.config import settings
    from .match_engines.base import EngineOutput
    from .references import ProductEvidence, ProductScoreMatrix

    matrix, _ = build_psd_matrix(db, batch, files, mains)
    evidence = [[ProductEvidence(
        aggregated_score=float(score), best_distance=int(round((1 - score) * 64)),
        second_distance=None, third_distance=None, support_count=1,
        reference_id=None, reference_category="PSD_PREVIEW",
    ) for score in row] for row in matrix]
    return EngineOutput(
        category_matrices={"PSD": matrix},
        product_scores={"PSD": ProductScoreMatrix(matrix, evidence)},
        algorithm_version=settings.psd_algorithm_version, model_name="PSD Preview pHash64",
        min_score=settings.match_min_score, min_margin=settings.match_min_margin,
        unmatched_score=settings.unmatched_score, device_label="CPU / PSD预览pHash",
        assignment_input="complete_psd_preview_phash_matrix", candidate_input="PSD_PREVIEW",
        feature_version_prefix=PsdPreviewExtractor.feature_version,
    )
