from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Batch, MainImage, MaskProfile, MatchedFile


MAIN_IMAGE_CATEGORY = "主图"


def _scope_key(product_category: str, image_category: str) -> str:
    raw = f"{product_category}\0{image_category}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _mask_path(batch: Batch, image_category: str, version: int) -> Path:
    key = _scope_key(batch.product_category, image_category)
    return (
        Path(batch.root_path)
        / ".design_matcher"
        / "masks"
        / key
        / f"mask_v{version}.png"
    )


def active_profile(db: Session, batch_id: int, image_category: str) -> MaskProfile | None:
    return db.scalar(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.image_category == image_category,
        MaskProfile.enabled.is_(True),
    ).order_by(MaskProfile.version.desc()))


def active_manual_profile(
    db: Session,
    batch_id: int,
    image_category: str,
) -> MaskProfile | None:
    return db.scalar(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.image_category == image_category,
        MaskProfile.creation_method == "MANUAL",
        MaskProfile.enabled.is_(True),
    ).order_by(MaskProfile.version.desc()))


def profile_history(db: Session, batch_id: int) -> list[MaskProfile]:
    return list(db.scalars(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id
    ).order_by(MaskProfile.image_category, MaskProfile.version.desc())))


def _reference_file(
    db: Session,
    batch_id: int,
    image_category: str,
    reference_file_id: int | None = None,
) -> MatchedFile:
    if reference_file_id is not None:
        item = db.get(MatchedFile, reference_file_id)
        if not item or item.batch_id != batch_id or item.category != image_category:
            raise ValueError("参考图不属于当前批次和分类")
        return item
    item = db.scalar(select(MatchedFile).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category == image_category,
    ).order_by(MatchedFile.id))
    if not item:
        raise ValueError("该分类没有可用参考图")
    return item


def mask_readiness(db: Session, batch_id: int) -> dict:
    discovered = list(db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category != "PSD",
    ).distinct().order_by(MatchedFile.category)).all())
    attachment_categories = [
        category
        for category in discovered
        if not any(
            keyword.replace(" ", "")
            in category.casefold().replace(" ", "")
            for keyword in settings.detail_category_keywords
        )
    ]
    has_mains = db.scalar(select(MainImage.id).where(
        MainImage.batch_id == batch_id
    ).limit(1)) is not None
    categories = ([MAIN_IMAGE_CATEGORY] if has_mains else []) + attachment_categories
    profiles = list(db.scalars(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.creation_method == "MANUAL",
        MaskProfile.enabled.is_(True),
    )))
    by_category = {profile.image_category: profile for profile in profiles}
    missing_masks = [category for category in categories if category not in by_category]
    missing_crops = [
        category for category in categories
        if category in by_category and _profile_crop_box(by_category[category]) is None
    ]
    missing = [
        category for category in categories
        if category in set(missing_masks) | set(missing_crops)
    ]
    return {
        "ready": not missing,
        "required_categories": categories,
        "configured_categories": [
            category for category in categories
            if category in by_category and category not in missing_crops
        ],
        "missing_categories": missing,
        "missing_mask_categories": missing_masks,
        "missing_crop_categories": missing_crops,
    }


def require_manual_profiles(db: Session, batch_id: int) -> list[MaskProfile]:
    readiness = mask_readiness(db, batch_id)
    if readiness["missing_categories"]:
        missing = "、".join(readiness["missing_categories"])
        raise ValueError(f"请先完成手绘 Mask 和裁剪框：{missing}")
    profiles = [
        active_manual_profile(db, batch_id, category)
        for category in readiness["required_categories"]
    ]
    return [profile for profile in profiles if profile is not None]


def _validate_crop_box(
    crop_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(crop_box) != 4:
        raise ValueError("裁剪框数据无效")
    left, top, right, bottom = (float(value) for value in crop_box)
    if not all(0 <= value <= 1 for value in (left, top, right, bottom)):
        raise ValueError("裁剪框必须位于图片范围内")
    if right - left < 0.02 or bottom - top < 0.02:
        raise ValueError("裁剪框太小，请重新框选")
    return left, top, right, bottom


def decode_mask_data_url(mask_data_url: str) -> Image.Image:
    try:
        encoded = mask_data_url.split(",", 1)[1]
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as incoming:
            mask = incoming.convert("L")
            mask.load()
        return mask
    except Exception as exc:
        raise ValueError("Mask PNG 数据无效") from exc


def _profile_crop_box(profile: MaskProfile) -> tuple[float, float, float, float] | None:
    values = (profile.crop_left, profile.crop_top, profile.crop_right, profile.crop_bottom)
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)


def save_profile_version(
    db: Session,
    batch_id: int,
    image_category: str,
    mask_data_url: str,
    reference_file_id: int,
    crop_box: tuple[float, float, float, float],
    name: str | None = None,
) -> MaskProfile:
    if image_category == "PSD":
        raise ValueError("PSD 使用整图匹配，不配置 Mask")
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    reference = _reference_file(db, batch_id, image_category, reference_file_id)
    mask = decode_mask_data_url(mask_data_url)
    width, height = mask.size
    left, top, right, bottom = _validate_crop_box(crop_box)
    pixel_box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )
    clipped_mask = Image.new("L", mask.size, 0)
    clipped_mask.paste(mask.crop(pixel_box), pixel_box[:2])
    mask = clipped_mask
    if not mask.getbbox():
        raise ValueError("识别区域不能为空")

    current = active_profile(db, batch_id, image_category)
    max_version = db.scalar(select(func.max(MaskProfile.version)).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.image_category == image_category,
    )) or 0
    if current:
        current.enabled = False
    version = max_version + 1
    path = _mask_path(batch, image_category, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path, "PNG")
    profile = MaskProfile(
        batch_id=batch.id,
        product_category=batch.product_category,
        image_category=image_category,
        name=name or (current.name if current else f"{batch.product_category} / {image_category}"),
        version=version,
        reference_width=width,
        reference_height=height,
        reference_file_id=reference.id,
        mask_path=str(path),
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        creation_method="MANUAL",
        enabled=True,
    )
    db.add(profile)
    db.commit()
    return profile


def save_main_profile_version(
    db: Session,
    batch_id: int,
    mask_data_url: str,
    reference_main_id: int,
    crop_box: tuple[float, float, float, float],
) -> tuple[MaskProfile, Image.Image]:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    reference = db.get(MainImage, reference_main_id)
    if not reference or reference.batch_id != batch_id:
        raise ValueError("参考主图不属于当前批次")
    mask = decode_mask_data_url(mask_data_url)
    width, height = mask.size
    left, top, right, bottom = _validate_crop_box(crop_box)
    pixel_box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )
    clipped_mask = Image.new("L", mask.size, 0)
    clipped_mask.paste(mask.crop(pixel_box), pixel_box[:2])
    if not clipped_mask.getbbox():
        raise ValueError("识别区域不能为空")

    current = active_profile(db, batch_id, MAIN_IMAGE_CATEGORY)
    max_version = db.scalar(select(func.max(MaskProfile.version)).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.image_category == MAIN_IMAGE_CATEGORY,
    )) or 0
    if current:
        current.enabled = False
    version = max_version + 1
    path = _mask_path(batch, MAIN_IMAGE_CATEGORY, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped_mask.save(path, "PNG")
    profile = MaskProfile(
        batch_id=batch.id,
        product_category=batch.product_category,
        image_category=MAIN_IMAGE_CATEGORY,
        name=MAIN_IMAGE_CATEGORY,
        version=version,
        reference_width=width,
        reference_height=height,
        reference_file_id=None,
        reference_main_id=reference.id,
        mask_path=str(path),
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        creation_method="MANUAL",
        enabled=True,
    )
    db.add(profile)
    db.flush()
    return profile, clipped_mask


def profile_dict(profile: MaskProfile) -> dict:
    crop_box = _profile_crop_box(profile)
    return {
        "id": profile.id,
        "batch_id": profile.batch_id,
        "product_category": profile.product_category,
        "image_category": profile.image_category,
        "name": profile.name,
        "version": profile.version,
        "reference_width": profile.reference_width,
        "reference_height": profile.reference_height,
        "reference_file_id": profile.reference_file_id,
        "reference_main_id": profile.reference_main_id,
        "enabled": profile.enabled,
        "creation_method": profile.creation_method,
        "crop_box": list(crop_box) if crop_box else None,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "mask_url": f"/api/mask-profiles/{profile.id}/mask",
        "reference_url": (
            f"/api/files/{profile.reference_file_id}/preview"
            if profile.reference_file_id else None
        ) or (
            f"/api/main-images/{profile.reference_main_id}/preview"
            if profile.reference_main_id else None
        ),
        "padding_ratio": settings.mask_crop_padding_ratio,
    }
