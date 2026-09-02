from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    DerivedImageAsset,
    ImagePHashRegion,
    MainImage,
    MainImageFeature,
)

from .media import open_visual
from .phash import PHashProvider, source_fingerprint


REGION_BOXES: dict[str, tuple[float, float, float, float]] = {
    "BODY": (0.0, 0.0, 1.0, 1.0),
    "CENTER": (0.25, 0.25, 0.75, 0.75),
    "TOP": (0.0, 0.0, 1.0, 0.55),
    "BOTTOM": (0.0, 0.45, 1.0, 1.0),
    "LEFT": (0.0, 0.0, 0.55, 1.0),
    "RIGHT": (0.45, 0.0, 1.0, 1.0),
    "TOP_LEFT": (0.0, 0.0, 0.6, 0.6),
    "TOP_RIGHT": (0.4, 0.0, 1.0, 0.6),
    "BOTTOM_LEFT": (0.0, 0.4, 0.6, 1.0),
    "BOTTOM_RIGHT": (0.4, 0.4, 1.0, 1.0),
}
EXPECTED_REGION_TYPES = {"FULL", *REGION_BOXES}


@dataclass(frozen=True)
class MainBodyArtifacts:
    main: MainImage
    full_phash: str
    body_phash: str
    regions: dict[str, ImagePHashRegion]
    cached: bool


def _full_image_mask(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Keep the full main image, including its white background, by default."""
    return Image.new("L", image.size, 255), (0, 0, image.width, image.height)


def _pixel_box(
    normalized: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = normalized
    return (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )


def _normalized_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        box[0] / width,
        box[1] / height,
        box[2] / width,
        box[3] / height,
    )


def _asset_paths(batch: Batch, main: MainImage) -> tuple[Path, Path]:
    root = (
        Path(batch.root_path)
        / ".design_matcher"
        / str(batch.id)
        / "derived"
        / "main_bodies"
        / str(main.id)
    )
    return root / "body_mask.png", root / "body_crop.jpg"


def _upsert_asset(
    db: Session,
    batch: Batch,
    main: MainImage,
    asset_type: str,
    path: Path,
) -> DerivedImageAsset:
    asset = db.scalar(
        select(DerivedImageAsset)
        .where(
            DerivedImageAsset.parent_main_image_id == main.id,
            DerivedImageAsset.asset_type == asset_type,
        )
        .order_by(DerivedImageAsset.id.desc())
    )
    if not asset:
        asset = DerivedImageAsset(
            batch_id=batch.id,
            parent_main_image_id=main.id,
            asset_type=asset_type,
            asset_path=str(path),
            feature_version=settings.body_phash_version,
        )
        db.add(asset)
    else:
        asset.asset_path = str(path)
        asset.feature_version = settings.body_phash_version
    db.flush()
    return asset


def _upsert_main_feature(
    db: Session,
    main: MainImage,
    feature_type: str,
    phash_value: str,
    crop_asset_id: int | None = None,
) -> None:
    feature = db.scalar(
        select(MainImageFeature)
        .where(
            MainImageFeature.main_image_id == main.id,
            MainImageFeature.feature_type == feature_type,
            MainImageFeature.feature_version == PHashProvider.feature_version,
        )
        .order_by(MainImageFeature.id.desc())
    )
    if not feature:
        feature = MainImageFeature(
            main_image_id=main.id,
            feature_type=feature_type,
            feature_version=PHashProvider.feature_version,
        )
        db.add(feature)
    feature.phash_value = phash_value
    feature.crop_asset_id = crop_asset_id


def _relative_region_box(
    body_box: tuple[float, float, float, float],
    region_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    bx1, by1, bx2, by2 = body_box
    rx1, ry1, rx2, ry2 = region_box
    width = bx2 - bx1
    height = by2 - by1
    return (
        bx1 + rx1 * width,
        by1 + ry1 * height,
        bx1 + rx2 * width,
        by1 + ry2 * height,
    )


def _store_regions(
    db: Session,
    main: MainImage,
    image: Image.Image,
    body_crop: Image.Image,
    body_box: tuple[float, float, float, float],
    provider: PHashProvider,
) -> dict[str, ImagePHashRegion]:
    rows: list[tuple[str, tuple[float, float, float, float], str]] = [
        ("FULL", (0.0, 0.0, 1.0, 1.0), provider.compute_image(image))
    ]
    for region_type, relative_box in REGION_BOXES.items():
        crop_box = _pixel_box(relative_box, body_crop.width, body_crop.height)
        region_image = body_crop.crop(crop_box)
        rows.append(
            (
                region_type,
                _relative_region_box(body_box, relative_box),
                provider.compute_image(region_image),
            )
        )

    db.execute(delete(ImagePHashRegion).where(ImagePHashRegion.main_image_id == main.id))
    stored: dict[str, ImagePHashRegion] = {}
    for region_type, coordinates, phash_value in rows:
        row = ImagePHashRegion(
            main_image_id=main.id,
            region_type=region_type,
            x1=coordinates[0],
            y1=coordinates[1],
            x2=coordinates[2],
            y2=coordinates[3],
            phash_value=phash_value,
            feature_version=settings.body_phash_version,
        )
        db.add(row)
        stored[region_type] = row
    db.flush()
    return stored


def _cached_artifacts(
    db: Session,
    main: MainImage,
    size: int,
    mtime_ns: int,
) -> MainBodyArtifacts | None:
    if not (
        main.body_source_size == size
        and main.body_source_mtime_ns == mtime_ns
        and main.body_feature_version == settings.body_phash_version
        and main.body_mask_path
        and main.body_crop_path
        and Path(main.body_mask_path).is_file()
        and Path(main.body_crop_path).is_file()
    ):
        return None
    rows = list(db.scalars(select(ImagePHashRegion).where(
        ImagePHashRegion.main_image_id == main.id,
        ImagePHashRegion.feature_version == settings.body_phash_version,
    )))
    regions = {row.region_type: row for row in rows}
    if set(regions) != EXPECTED_REGION_TYPES:
        return None
    return MainBodyArtifacts(
        main=main,
        full_phash=regions["FULL"].phash_value,
        body_phash=regions["BODY"].phash_value,
        regions=regions,
        cached=True,
    )


def ensure_main_body(
    db: Session,
    batch: Batch,
    main: MainImage,
    *,
    force_auto: bool = False,
    manual_box: tuple[float, float, float, float] | None = None,
    manual_mask: Image.Image | None = None,
) -> MainBodyArtifacts:
    source_path = Path(main.file_path)
    size, mtime_ns = source_fingerprint(source_path)
    if not force_auto and manual_box is None and manual_mask is None:
        cached = _cached_artifacts(db, main, size, mtime_ns)
        if cached:
            return cached

    image = open_visual(source_path)
    if manual_mask is not None:
        if manual_box is None:
            raise ValueError("手绘遮罩必须同时提供裁剪框")
        box = _pixel_box(manual_box, image.width, image.height)
        mask = manual_mask.convert("L").resize(
            image.size, Image.Resampling.NEAREST
        )
        mask = mask.point(lambda value: 255 if value >= 128 else 0)
        clipped = Image.new("L", image.size, 0)
        clipped.paste(mask.crop(box), box[:2])
        if not clipped.getbbox():
            raise ValueError("主图识别区域不能为空")
        mask = clipped
        method = "MANUAL_MASK"
    elif manual_box is not None:
        box = _pixel_box(manual_box, image.width, image.height)
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box)
        method = "MANUAL_ROI"
    elif (
        main.body_crop_method == "MANUAL_MASK"
        and not force_auto
        and main.body_mask_path
        and Path(main.body_mask_path).is_file()
        and all(
            value is not None
            for value in (
                main.body_bbox_x1,
                main.body_bbox_y1,
                main.body_bbox_x2,
                main.body_bbox_y2,
            )
        )
    ):
        normalized = (
            float(main.body_bbox_x1),
            float(main.body_bbox_y1),
            float(main.body_bbox_x2),
            float(main.body_bbox_y2),
        )
        box = _pixel_box(normalized, image.width, image.height)
        with Image.open(main.body_mask_path) as stored:
            mask = stored.convert("L").resize(
                image.size, Image.Resampling.NEAREST
            )
            mask.load()
        mask = mask.point(lambda value: 255 if value >= 128 else 0)
        clipped = Image.new("L", image.size, 0)
        clipped.paste(mask.crop(box), box[:2])
        if not clipped.getbbox():
            raise ValueError("已保存的主图遮罩为空，请重新绘制")
        mask = clipped
        method = "MANUAL_MASK"
    elif main.body_crop_method == "MANUAL_ROI" and not force_auto and all(
        value is not None
        for value in (
            main.body_bbox_x1,
            main.body_bbox_y1,
            main.body_bbox_x2,
            main.body_bbox_y2,
        )
    ):
        normalized = (
            float(main.body_bbox_x1),
            float(main.body_bbox_y1),
            float(main.body_bbox_x2),
            float(main.body_bbox_y2),
        )
        box = _pixel_box(normalized, image.width, image.height)
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box)
        method = "MANUAL_ROI"
    else:
        mask, box = _full_image_mask(image)
        method = "FULL_IMAGE"

    neutral = Image.new("RGB", image.size, (255, 255, 255))
    neutral.paste(image, mask=mask)
    body_crop = neutral.crop(box)
    body_box = _normalized_box(box, image.width, image.height)

    mask_path, crop_path = _asset_paths(batch, main)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path, "PNG", optimize=True)
    body_crop.save(crop_path, "JPEG", quality=95, optimize=True)
    _upsert_asset(db, batch, main, "MAIN_BODY_MASK", mask_path)
    crop_asset = _upsert_asset(db, batch, main, "MAIN_BODY_CROP", crop_path)

    provider = PHashProvider()
    regions = _store_regions(db, main, image, body_crop, body_box, provider)
    _upsert_main_feature(db, main, "FULL_PHASH", regions["FULL"].phash_value)
    _upsert_main_feature(
        db,
        main,
        "BODY_PHASH",
        regions["BODY"].phash_value,
        crop_asset.id,
    )

    main.body_bbox_x1, main.body_bbox_y1, main.body_bbox_x2, main.body_bbox_y2 = body_box
    main.body_mask_path = str(mask_path)
    main.body_crop_path = str(crop_path)
    main.body_crop_method = method
    main.body_source_size = size
    main.body_source_mtime_ns = mtime_ns
    main.body_feature_version = settings.body_phash_version
    main.body_updated_at = datetime.now(timezone.utc)
    db.flush()
    return MainBodyArtifacts(
        main=main,
        full_phash=regions["FULL"].phash_value,
        body_phash=regions["BODY"].phash_value,
        regions=regions,
        cached=False,
    )


def ensure_batch_main_bodies(
    db: Session,
    batch: Batch,
    mains: list[MainImage],
    *,
    force_auto: bool = False,
) -> list[MainBodyArtifacts]:
    return [
        ensure_main_body(db, batch, main, force_auto=force_auto)
        for main in mains
    ]


def main_body_dict(artifacts: MainBodyArtifacts) -> dict:
    main = artifacts.main
    return {
        "main_id": main.id,
        "basename": main.basename,
        "filename": main.filename,
        "method": main.body_crop_method,
        "bbox": [
            main.body_bbox_x1,
            main.body_bbox_y1,
            main.body_bbox_x2,
            main.body_bbox_y2,
        ],
        "full_phash": artifacts.full_phash,
        "body_phash": artifacts.body_phash,
        "cached": artifacts.cached,
        "updated_at": main.body_updated_at,
        "original_url": f"/api/main-images/{main.id}/preview",
        "mask_url": f"/api/main-images/{main.id}/body-mask",
        "crop_url": f"/api/main-images/{main.id}/body-crop",
        "regions": [
            {
                "region_type": row.region_type,
                "bbox": [row.x1, row.y1, row.x2, row.y2],
                "phash": row.phash_value,
            }
            for row in sorted(
                artifacts.regions.values(), key=lambda value: value.region_type
            )
        ],
    }
