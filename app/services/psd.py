from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    DerivedImageAsset,
    MainImage,
    MainImageFeature,
    MatchedFile,
)

from .media import open_visual


class PHashProvider:
    feature_version = "phash64_v1"

    def compute_image(self, image: Image.Image) -> str:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        low = dctn(pixels, type=2, norm="ortho")[:8, :8]
        median = float(np.median(low.flatten()[1:]))
        bits = (low > median).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:016x}"

    def compute_path(self, path: Path) -> str:
        return self.compute_image(open_visual(path))

    @staticmethod
    def distance(left: str, right: str) -> int:
        return (int(left, 16) ^ int(right, 16)).bit_count()

    def distance_matrix(self, queries: list[str], mains: list[str]) -> np.ndarray:
        return np.asarray([
            [self.distance(query, main) for main in mains]
            for query in queries
        ], dtype=np.int16)


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
    embedding_matrix: np.ndarray,
) -> tuple[np.ndarray, list[DerivedImageAsset]]:
    provider = PHashProvider()
    extractor = PsdPreviewExtractor()
    previews = [extractor.extract(db, batch, item) for item in psd_files]
    main_hashes = [provider.compute_path(Path(main.file_path)) for main in mains]
    preview_hashes = [provider.compute_path(Path(asset.asset_path)) for asset in previews]
    distances = provider.distance_matrix(preview_hashes, main_hashes)
    phash_similarity = 1.0 - distances.astype(np.float32) / 64.0
    hybrid = 0.65 * phash_similarity + 0.35 * embedding_matrix

    for row, item in enumerate(psd_files):
        order = np.argsort(distances[row])
        best_distance = int(distances[row, int(order[0])])
        second_distance = int(distances[row, int(order[1])]) if len(order) > 1 else 64
        clear = (
            best_distance <= settings.phash_max_distance
            and second_distance - best_distance >= settings.phash_min_margin
        )
        item.phash_distance = best_distance
        item.match_method = (
            "PHASH_AND_EMBEDDING" if clear else "PHASH_TIE_BREAK_EMBEDDING"
        )
        if not clear:
            # pHash is still the first stage. When multiple products share the
            # same/near-same structural hash, keep that nearest shortlist and
            # let the full-image embedding break the tie. If every product is
            # tied, this intentionally becomes the full embedding baseline.
            shortlist = distances[row] <= best_distance + settings.phash_min_margin
            hybrid[row] = np.where(shortlist, embedding_matrix[row], -1.0)

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
    return hybrid, previews
