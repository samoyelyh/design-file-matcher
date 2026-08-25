from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image
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
    categories = list(db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category != "PSD",
    ).distinct().order_by(MatchedFile.category)).all())
    profiles = list(db.scalars(select(MaskProfile).where(
        MaskProfile.batch_id == batch_id,
        MaskProfile.creation_method == "MANUAL",
        MaskProfile.enabled.is_(True),
    )))
    by_category = {profile.image_category: profile for profile in profiles}
    missing = [category for category in categories if category not in by_category]
    return {
        "ready": not missing,
        "required_categories": categories,
        "configured_categories": [
            category for category in categories if category in by_category
        ],
        "missing_categories": missing,
    }


def require_manual_profiles(db: Session, batch_id: int) -> list[MaskProfile]:
    readiness = mask_readiness(db, batch_id)
    if readiness["missing_categories"]:
        missing = "、".join(readiness["missing_categories"])
        raise ValueError(f"请先手绘并保存 Mask：{missing}")
    return [
        active_manual_profile(db, batch_id, category)
        for category in readiness["required_categories"]
    ]


def save_profile_version(
    db: Session,
    batch_id: int,
    image_category: str,
    mask_data_url: str,
    reference_file_id: int,
    name: str | None = None,
) -> MaskProfile:
    if image_category == "PSD":
        raise ValueError("PSD 使用整图匹配，不配置 Mask")
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
        creation_method="MANUAL",
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
        "creation_method": profile.creation_method,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "mask_url": f"/api/mask-profiles/{profile.id}/mask",
        "reference_url": (
            f"/api/files/{profile.reference_file_id}/preview"
            if profile.reference_file_id else None
        ),
        "padding_ratio": settings.mask_crop_padding_ratio,
    }
