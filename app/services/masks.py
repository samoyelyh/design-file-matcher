from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Batch, MaskProfile, MatchedFile


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


def profile_history(db: Session, batch_id: int) -> list[MaskProfile]:
    return list(db.scalars(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id
    ).order_by(MaskProfile.image_category, MaskProfile.version.desc())))


def _default_mask(category: str, width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    def rectangle(box: tuple[float, float, float, float]) -> None:
        draw.rounded_rectangle(
            tuple(int(value * size) for value, size in zip(box, (width, height, width, height))),
            radius=max(4, int(min(width, height) * 0.025)),
            fill=255,
        )

    if category == "双人模特图":
        rectangle((0.08, 0.20, 0.48, 0.82))
        rectangle((0.52, 0.20, 0.92, 0.82))
    elif category == "白底模特图":
        rectangle((0.22, 0.12, 0.78, 0.90))
    elif category == "细节展示图":
        rectangle((0.10, 0.08, 0.90, 0.92))
    else:
        # Scene photos in the current product line place the garment torso in
        # the central band. Administrators can refine this editable seed.
        rectangle((0.25, 0.20, 0.75, 0.82))
    return mask


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


def create_default_profile(
    db: Session,
    batch_id: int,
    image_category: str,
    reference_file_id: int | None = None,
) -> MaskProfile:
    if image_category == "PSD":
        raise ValueError("PSD 使用整图匹配，不配置 Mask")
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    existing = active_profile(db, batch_id, image_category)
    if existing:
        return existing
    reference = _reference_file(db, batch_id, image_category, reference_file_id)
    with Image.open(reference.thumbnail_path or reference.current_path) as image:
        width, height = image.size
    version = 1
    path = _mask_path(batch, image_category, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    _default_mask(image_category, width, height).save(path, "PNG")
    profile = MaskProfile(
        batch_id=batch.id,
        product_category=batch.product_category,
        image_category=image_category,
        name=f"{batch.product_category} / {image_category}",
        version=version,
        reference_width=width,
        reference_height=height,
        reference_file_id=reference.id,
        mask_path=str(path),
        enabled=True,
    )
    db.add(profile)
    db.commit()
    return profile


def ensure_default_profiles(db: Session, batch_id: int) -> list[MaskProfile]:
    categories = db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category != "PSD",
    ).distinct()).all()
    return [create_default_profile(db, batch_id, category) for category in categories]


def save_profile_version(
    db: Session,
    batch_id: int,
    image_category: str,
    mask_data_url: str,
    reference_file_id: int,
    name: str | None = None,
) -> MaskProfile:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    reference = _reference_file(db, batch_id, image_category, reference_file_id)
    try:
        encoded = mask_data_url.split(",", 1)[1]
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as incoming:
            mask = incoming.convert("L")
            width, height = mask.size
    except Exception as exc:
        raise ValueError("Mask PNG 数据无效") from exc
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
        enabled=True,
    )
    db.add(profile)
    db.commit()
    return profile


def profile_dict(profile: MaskProfile) -> dict:
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
        "enabled": profile.enabled,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "mask_url": f"/api/mask-profiles/{profile.id}/mask",
        "reference_url": (
            f"/api/files/{profile.reference_file_id}/preview"
            if profile.reference_file_id else None
        ),
        "padding_ratio": settings.mask_crop_padding_ratio,
    }
