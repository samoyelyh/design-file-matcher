from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Batch, ImagePHashRegion, MainImage, MatchedFile
from app.services.detail_phash import ensure_file_phash, is_detail_category
from app.services.cropper import MaskedCropProcessor
from app.services.main_body import ensure_batch_main_bodies
from app.services.masks import active_manual_profile
from app.services.phash import PHashProvider
from app.services.phash_matching import ensure_asset_phash


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def evaluate(batch_id: int) -> dict:
    init_db()
    with SessionLocal() as db:
        batch = db.get(Batch, batch_id)
        if not batch:
            raise ValueError(f"批次不存在: {batch_id}")
        mains = list(db.scalars(select(MainImage).where(
            MainImage.batch_id == batch_id
        ).order_by(MainImage.id)))
        samples = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id,
            MatchedFile.manual_product_id.is_not(None),
            MatchedFile.review_status.in_(["CONFIRMED", "CORRECTED"]),
        ).order_by(MatchedFile.id)))
        ensure_batch_main_bodies(db, batch, mains)
        region_rows = list(db.scalars(select(ImagePHashRegion).where(
            ImagePHashRegion.main_image_id.in_([main.id for main in mains]),
            ImagePHashRegion.feature_version == settings.body_phash_version,
        )))
        db.commit()

        regions: dict[int, dict[str, str]] = defaultdict(dict)
        for row in region_rows:
            regions[row.main_image_id][row.region_type] = row.phash_value
        provider = PHashProvider()
        totals = Counter()
        by_category: dict[str, Counter] = defaultdict(Counter)

        for item in samples:
            query_hash = ensure_file_phash(item, provider)
            full_rank = sorted(
                mains,
                key=lambda main: provider.distance(
                    query_hash, regions[main.id]["FULL"]
                ),
            )
            body_rank = sorted(
                mains,
                key=lambda main: provider.distance(
                    query_hash, regions[main.id]["BODY"]
                ),
            )
            correct_id = item.manual_product_id
            totals["samples"] += 1
            by_category[item.category]["samples"] += 1
            if full_rank[0].id == correct_id:
                totals["full_correct"] += 1
                by_category[item.category]["full_correct"] += 1
            if body_rank[0].id == correct_id:
                totals["body_correct"] += 1
                by_category[item.category]["body_correct"] += 1

            if is_detail_category(item.category):
                if full_rank[0].id == correct_id:
                    totals["detail_full_correct"] += 1
                    by_category[item.category]["detail_full_correct"] += 1
                profile = active_manual_profile(db, batch_id, item.category)
                detail_hash = query_hash
                if profile:
                    crop = MaskedCropProcessor().process_file(
                        db, batch, item, profile
                    ).crop
                    detail_hash = ensure_asset_phash(crop, provider)
                unmasked_local_rank = sorted(
                    mains,
                    key=lambda main: min(
                        provider.distance(query_hash, value)
                        for value in regions[main.id].values()
                    ),
                )
                masked_local_rank = sorted(
                    mains,
                    key=lambda main: min(
                        provider.distance(detail_hash, value)
                        for value in regions[main.id].values()
                    ),
                )
                totals["detail_samples"] += 1
                by_category[item.category]["detail_samples"] += 1
                if unmasked_local_rank[0].id == correct_id:
                    totals["detail_unmasked_local_correct"] += 1
                if masked_local_rank[0].id == correct_id:
                    totals["detail_masked_local_correct"] += 1
                if unmasked_local_rank[0].id == correct_id:
                    totals["detail_local_correct"] += 1
                    by_category[item.category]["detail_local_correct"] += 1

        db.commit()
        return {
            "batch_id": batch_id,
            "ground_truth_samples": totals["samples"],
            "full_phash_accuracy": _accuracy(
                totals["full_correct"], totals["samples"]
            ),
            "body_phash_accuracy": _accuracy(
                totals["body_correct"], totals["samples"]
            ),
            "body_vs_full_delta": _accuracy(
                totals["body_correct"] - totals["full_correct"], totals["samples"]
            ),
            "detail_samples": totals["detail_samples"],
            "detail_full_phash_accuracy": _accuracy(
                totals["detail_full_correct"], totals["detail_samples"]
            ),
            "detail_region_phash_accuracy": _accuracy(
                totals["detail_local_correct"], totals["detail_samples"]
            ),
            "detail_unmasked_region_phash_accuracy": _accuracy(
                totals["detail_unmasked_local_correct"], totals["detail_samples"]
            ),
            "detail_masked_region_phash_accuracy": _accuracy(
                totals["detail_masked_local_correct"], totals["detail_samples"]
            ),
            "detail_region_vs_full_delta": _accuracy(
                totals["detail_local_correct"] - totals["detail_full_correct"],
                totals["detail_samples"],
            ),
            "detail_query_mode": "full_detail_image_vs_main_regions",
            "by_category": {
                category: {
                    "samples": values["samples"],
                    "full_phash_accuracy": _accuracy(
                        values["full_correct"], values["samples"]
                    ),
                    "body_phash_accuracy": _accuracy(
                        values["body_correct"], values["samples"]
                    ),
                    "detail_region_phash_accuracy": _accuracy(
                        values["detail_local_correct"], values["detail_samples"]
                    ),
                }
                for category, values in sorted(by_category.items())
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full, body, and detail-region pHash with reviewed data"
    )
    parser.add_argument("--batch-id", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.batch_id)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
