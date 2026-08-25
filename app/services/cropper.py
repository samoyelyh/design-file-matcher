from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    DerivedImageAsset,
    MainImage,
    MaskProfile,
    MatchedFile,
)

from .media import open_visual


@dataclass(frozen=True)
class CropArtifacts:
    mask: DerivedImageAsset
    crop: DerivedImageAsset
    bounding_box: tuple[int, int, int, int]


class MaskedCropProcessor:
    feature_version = "masked_crop_v1"

    def __init__(self, padding_ratio: float | None = None) -> None:
        self.padding_ratio = (
            settings.mask_crop_padding_ratio
            if padding_ratio is None else padding_ratio
        )

    def _asset_paths(
        self,
        batch: Batch,
        profile: MaskProfile,
        parent_key: str,
    ) -> tuple[Path, Path]:
        digest = hashlib.sha1(parent_key.encode("utf-8")).hexdigest()[:20]
        root = (
            Path(batch.root_path)
            / ".design_matcher"
            / str(batch.id)
            / "derived"
            / f"mask_{profile.id}_v{profile.version}"
        )
        return root / "masks" / f"{digest}.png", root / "crops" / f"{digest}.jpg"

    def _existing(
        self,
        db: Session,
        profile_id: int,
        parent_file_id: int | None,
        parent_main_image_id: int | None,
    ) -> CropArtifacts | None:
        rows = list(db.scalars(select(DerivedImageAsset).where(
            DerivedImageAsset.mask_profile_id == profile_id,
            DerivedImageAsset.parent_file_id == parent_file_id,
            DerivedImageAsset.parent_main_image_id == parent_main_image_id,
            DerivedImageAsset.asset_type.in_(["MASK", "MASKED_CROP"]),
        )))
        by_type = {row.asset_type: row for row in rows if Path(row.asset_path).is_file()}
        if set(by_type) == {"MASK", "MASKED_CROP"}:
            with Image.open(by_type["MASK"].asset_path) as mask:
                box = mask.getbbox()
            if box:
                return CropArtifacts(by_type["MASK"], by_type["MASKED_CROP"], box)
        return None

    def process_file(
        self,
        db: Session,
        batch: Batch,
        item: MatchedFile,
        profile: MaskProfile,
    ) -> CropArtifacts:
        return self._process(
            db,
            batch,
            Path(item.current_path),
            profile,
            parent_file_id=item.id,
            parent_main_image_id=None,
            parent_key=f"file:{item.id}:{item.current_path}",
        )

    def process_main(
        self,
        db: Session,
        batch: Batch,
        main: MainImage,
        profile: MaskProfile,
    ) -> CropArtifacts:
        return self._process(
            db,
            batch,
            Path(main.file_path),
            profile,
            parent_file_id=None,
            parent_main_image_id=main.id,
            parent_key=f"main:{main.id}:{main.file_path}",
        )

    def _process(
        self,
        db: Session,
        batch: Batch,
        source_path: Path,
        profile: MaskProfile,
        parent_file_id: int | None,
        parent_main_image_id: int | None,
        parent_key: str,
    ) -> CropArtifacts:
        cached = self._existing(
            db, profile.id, parent_file_id, parent_main_image_id
        )
        if cached:
            return cached
        source = open_visual(source_path)
        with Image.open(profile.mask_path) as template:
            mask = template.convert("L").resize(source.size, Image.Resampling.NEAREST)
        mask = mask.point(lambda value: 255 if value >= 128 else 0)
        box = mask.getbbox()
        if not box:
            raise ValueError(f"Mask Profile v{profile.version}没有有效区域")
        left, top, right, bottom = box
        padding = int(max(right - left, bottom - top) * self.padding_ratio)
        padded = (
            max(0, left - padding),
            max(0, top - padding),
            min(source.width, right + padding),
            min(source.height, bottom + padding),
        )
        neutral = Image.new("RGB", source.size, (245, 245, 245))
        neutral.paste(source, mask=mask)
        crop = neutral.crop(padded)
        mask_path, crop_path = self._asset_paths(batch, profile, parent_key)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path, "PNG")
        crop.save(crop_path, "JPEG", quality=95, optimize=True)
        version = f"{self.feature_version}:mask_v{profile.version}"
        mask_asset = DerivedImageAsset(
            batch_id=batch.id,
            parent_file_id=parent_file_id,
            parent_main_image_id=parent_main_image_id,
            asset_type="MASK",
            asset_path=str(mask_path),
            mask_profile_id=profile.id,
            feature_version=version,
        )
        crop_asset = DerivedImageAsset(
            batch_id=batch.id,
            parent_file_id=parent_file_id,
            parent_main_image_id=parent_main_image_id,
            asset_type="MASKED_CROP",
            asset_path=str(crop_path),
            mask_profile_id=profile.id,
            feature_version=version,
        )
        db.add_all([mask_asset, crop_asset])
        db.flush()
        return CropArtifacts(mask_asset, crop_asset, padded)
