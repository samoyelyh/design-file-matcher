from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import is_ignored_image_category, settings
from app.models import (
    DerivedImageAsset,
    MainImage,
    MatchedFile,
    ProductReference,
)

from .cropper import MaskedCropProcessor
from .masks import active_manual_profile
from .phash import PHashProvider
from .phash_matching import ensure_asset_phash, main_region_hashes


@dataclass(frozen=True)
class ProductEvidence:
    aggregated_score: float
    best_distance: int | None
    second_distance: int | None
    third_distance: int | None
    support_count: int
    reference_id: int | None
    reference_category: str | None
    details: dict | None = None


@dataclass(frozen=True)
class ProductScoreMatrix:
    matrix: np.ndarray
    evidence: list[list[ProductEvidence]]


def _product_key(value: str) -> str:
    return value.strip().casefold()


def ensure_main_references(
    db: Session,
    mains: list[MainImage],
) -> list[ProductReference]:
    """Persist the active, manually-cropped main hash as each product's anchor."""
    hashes = main_region_hashes(db, mains, "BODY")
    result: list[ProductReference] = []
    for main, phash in zip(mains, hashes, strict=True):
        row = db.scalar(select(ProductReference).where(
            ProductReference.main_image_id == main.id,
            ProductReference.category == "MAIN",
            ProductReference.source_type == "MAIN",
        ).order_by(ProductReference.id.desc()))
        if row is None:
            row = ProductReference(
                main_image_id=main.id,
                category="MAIN",
                phash=phash,
                source_type="MAIN",
                is_manual_confirmed=False,
                is_active=True,
            )
            db.add(row)
        else:
            # A newly saved main-image crop changes the production feature.
            # Updating its anchor is safe; manually confirmed rows are immutable.
            row.phash = phash
            row.is_active = True
        result.append(row)
    db.flush()
    return result


def _reference_hash_for_file(db: Session, item: MatchedFile) -> str | None:
    if item.category == "PSD" or is_ignored_image_category(item.category):
        return None
    asset = db.scalar(select(DerivedImageAsset).where(
        DerivedImageAsset.parent_file_id == item.id,
        DerivedImageAsset.asset_type == "MASKED_CROP",
    ).order_by(DerivedImageAsset.id.desc()))
    try:
        if asset is None or not Path(asset.asset_path).is_file():
            profile = active_manual_profile(db, item.batch_id, item.category)
            if profile is None or not Path(profile.mask_path).is_file():
                return None
            asset = MaskedCropProcessor().process_file(db, item.batch, item, profile).crop
        return ensure_asset_phash(asset, PHashProvider())
    except (OSError, ValueError):
        # Historical shared-drive caches may have been cleaned after renaming.
        # They are not reliable reference material and must not block review.
        return None


def learn_confirmed_reference(
    db: Session,
    item: MatchedFile,
    *,
    source_type: str = "MANUAL_CONFIRMED",
) -> ProductReference | None:
    """Turn one explicit operator confirmation into an immutable reference."""
    main_id = item.manual_product_id or item.matched_main_id
    if main_id is None:
        return None
    main = db.get(MainImage, main_id)
    if main is None:
        return None
    phash = _reference_hash_for_file(db, item)
    if phash is None:
        return None

    existing = db.scalar(select(ProductReference).where(
        ProductReference.source_file_id == item.id,
        ProductReference.category == item.category,
        ProductReference.is_manual_confirmed.is_(True),
    ).order_by(ProductReference.id.desc()))
    if existing is not None:
        # Reassignment may move an already learned file to the corrected product.
        existing.main_image_id = main.id
        existing.phash = phash
        existing.source_type = source_type
        existing.is_active = True
        return existing

    # References are reusable across batches by stable product basename. Never
    # evict an operator-confirmed row automatically when the cap is reached.
    rows = list(db.execute(select(ProductReference, MainImage.basename).join(
        MainImage, ProductReference.main_image_id == MainImage.id
    ).where(
        ProductReference.category == item.category,
        ProductReference.is_manual_confirmed.is_(True),
        ProductReference.is_active.is_(True),
    )).all())
    active_for_product = [
        row for row, basename in rows
        if _product_key(basename) == _product_key(main.basename)
    ]
    if len(active_for_product) >= settings.max_references_per_category:
        return None
    row = ProductReference(
        main_image_id=main.id,
        source_file_id=item.id,
        category=item.category,
        phash=phash,
        source_type=source_type,
        is_manual_confirmed=True,
        is_active=True,
    )
    db.add(row)
    db.flush()
    return row


def learn_confirmed_references(
    db: Session,
    items: list[MatchedFile],
) -> int:
    learned = 0
    for item in items:
        if learn_confirmed_reference(db, item) is not None:
            learned += 1
    return learned


def backfill_confirmed_references(db: Session) -> dict[str, int]:
    """Seed the reference set from existing explicit review confirmations."""
    items = list(db.scalars(select(MatchedFile).where(
        MatchedFile.confirmed_at.is_not(None),
        MatchedFile.matched_main_id.is_not(None),
        MatchedFile.status == "MANUAL_CONFIRMED",
    ).order_by(MatchedFile.confirmed_at.desc(), MatchedFile.id.desc())))
    existing_source_ids = set(db.scalars(select(
        ProductReference.source_file_id
    ).where(ProductReference.source_file_id.is_not(None))))
    added = 0
    skipped = 0
    for item in items:
        if item.id in existing_source_ids:
            skipped += 1
            continue
        reference = learn_confirmed_reference(db, item)
        if reference is None:
            skipped += 1
        else:
            added += 1
            existing_source_ids.add(item.id)
    db.flush()
    return {"considered": len(items), "added": added, "skipped": skipped}


def _active_references_by_product(
    db: Session,
) -> dict[str, list[ProductReference]]:
    rows = list(db.execute(select(ProductReference, MainImage.basename).join(
        MainImage, ProductReference.main_image_id == MainImage.id
    ).where(ProductReference.is_active.is_(True)).order_by(
        ProductReference.is_manual_confirmed.desc(),
        ProductReference.created_at.desc(),
        ProductReference.id.desc(),
    )).all())
    result: dict[str, list[ProductReference]] = {}
    for reference, basename in rows:
        result.setdefault(_product_key(basename), []).append(reference)
    return result


def build_product_score_matrix(
    db: Session,
    query_hashes: list[str],
    category: str,
    mains: list[MainImage],
    *,
    max_manual_references: int | None = None,
    query_file_ids: list[int] | None = None,
) -> ProductScoreMatrix:
    """Score Query vs Product using one main anchor plus confirmed references."""
    anchors = ensure_main_references(db, mains)
    cap = (
        settings.max_references_per_category
        if max_manual_references is None
        else max(0, max_manual_references)
    )
    by_product = _active_references_by_product(db) if cap else {}
    provider = PHashProvider()
    weights = (
        settings.reference_weight_1,
        settings.reference_weight_2,
        settings.reference_weight_3,
    )
    score_rows: list[list[float]] = []
    all_evidence: list[list[ProductEvidence]] = []
    for query_index, query_hash in enumerate(query_hashes):
        score_row: list[float] = []
        evidence_row: list[ProductEvidence] = []
        for main, anchor in zip(mains, anchors, strict=True):
            available = [
                reference
                for reference in by_product.get(_product_key(main.basename), [])
                if reference.is_manual_confirmed
                and not (
                    query_file_ids
                    and reference.source_file_id == query_file_ids[query_index]
                )
            ]
            # Keep at most N samples from each folder type. Same-folder samples
            # are considered first, then recent references from other folders.
            available.sort(
                key=lambda reference: reference.category != category
            )
            per_category: dict[str, int] = {}
            manual: list[ProductReference] = []
            for reference in available:
                used = per_category.get(reference.category, 0)
                if used >= cap:
                    continue
                manual.append(reference)
                per_category[reference.category] = used + 1
            references = [anchor, *manual]
            measured = sorted(
                [
                    (
                        provider.distance(query_hash, reference.phash),
                        reference,
                    )
                    for reference in references
                ],
                key=lambda value: (value[0], value[1].id or 0),
            )
            distances = [distance for distance, _reference in measured]
            best_three = measured[:3]
            used_weights = weights[:len(best_three)]
            weight_total = sum(used_weights) or 1.0
            aggregate = sum(
                (1.0 - distance / 64.0) * weight
                for (distance, _reference), weight in zip(
                    best_three, used_weights, strict=True
                )
            ) / weight_total
            support_count = sum(
                distance <= settings.reference_support_max_distance
                for distance in distances
            )
            best_reference = measured[0][1]
            evidence_row.append(ProductEvidence(
                aggregated_score=float(aggregate),
                best_distance=distances[0],
                second_distance=distances[1] if len(distances) > 1 else None,
                third_distance=distances[2] if len(distances) > 2 else None,
                support_count=int(support_count),
                reference_id=best_reference.id,
                reference_category=best_reference.category,
            ))
            score_row.append(float(aggregate))
        score_rows.append(score_row)
        all_evidence.append(evidence_row)
    return ProductScoreMatrix(
        matrix=np.asarray(score_rows, dtype=np.float32),
        evidence=all_evidence,
    )
