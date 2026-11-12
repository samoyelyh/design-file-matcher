from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import func, select

from app.config import is_ignored_image_category, settings
from app.db import SessionLocal, init_db
from app.models import DerivedImageAsset, ImagePHashRegion, MainImage, MatchedFile
from app.services.phash import PHashProvider
from app.services.phash_matching import ensure_asset_phash
from app.services.references import backfill_confirmed_references


SCENARIOS = {"main_only": 0, "plus_1": 1, "plus_3": 3, "plus_5": 5}


def _key(value: str) -> str:
    return value.strip().casefold()


def _aggregate(distances: list[int]) -> float:
    weights = (
        settings.reference_weight_1,
        settings.reference_weight_2,
        settings.reference_weight_3,
    )
    best = sorted(distances)[:3]
    used = weights[:len(best)]
    return sum(
        (1.0 - distance / 64.0) * weight
        for distance, weight in zip(best, used, strict=True)
    ) / (sum(used) or 1.0)


def _latest_assets(db) -> dict[int, DerivedImageAsset]:
    assets = list(db.scalars(select(DerivedImageAsset).where(
        DerivedImageAsset.parent_file_id.is_not(None),
        DerivedImageAsset.asset_type == "MASKED_CROP",
    ).order_by(DerivedImageAsset.id.desc())))
    result: dict[int, DerivedImageAsset] = {}
    for asset in assets:
        if asset.parent_file_id not in result and Path(asset.asset_path).is_file():
            result[asset.parent_file_id] = asset
    return result


def _eligible_batches(
    db,
    assets: dict[int, DerivedImageAsset],
    truth_status: str,
) -> list[int]:
    main_counts = dict(db.execute(select(
        MainImage.batch_id, func.count(MainImage.id)
    ).group_by(MainImage.batch_id)).all())
    body_counts = dict(db.execute(select(
        MainImage.batch_id, func.count(ImagePHashRegion.id)
    ).join(MainImage, ImagePHashRegion.main_image_id == MainImage.id).where(
        ImagePHashRegion.region_type == "BODY"
    ).group_by(MainImage.batch_id)).all())
    confirmed_counts: dict[int, int] = defaultdict(int)
    criteria = [
        MatchedFile.confirmed_at.is_not(None),
        MatchedFile.matched_main_id.is_not(None),
        MatchedFile.status == "MANUAL_CONFIRMED",
        MatchedFile.category != "PSD",
    ]
    if truth_status == "corrected":
        criteria.append(MatchedFile.review_status == "CORRECTED")
    rows = db.execute(select(MatchedFile.id, MatchedFile.batch_id).where(*criteria)).all()
    for file_id, batch_id in rows:
        if file_id in assets:
            confirmed_counts[batch_id] += 1
    return sorted(
        batch_id for batch_id, count in confirmed_counts.items()
        if count >= 10 and body_counts.get(batch_id) == main_counts.get(batch_id)
    )


def evaluate(
    batch_ids: list[int] | None = None,
    truth_status: str = "confirmed",
    reference_scope: str = "all_categories",
) -> dict:
    init_db()
    provider = PHashProvider()
    with SessionLocal() as db:
        assets = _latest_assets(db)
        eligible = _eligible_batches(db, assets, truth_status)
        selected = batch_ids or eligible
        selected = [batch_id for batch_id in selected if batch_id in eligible]
        if not selected:
            raise RuntimeError("没有同时具备主图 BODY 哈希和人工确认裁剪样本的批次")

        all_confirmed = list(db.scalars(select(MatchedFile).where(
            MatchedFile.confirmed_at.is_not(None),
            MatchedFile.matched_main_id.is_not(None),
            MatchedFile.status == "MANUAL_CONFIRMED",
            MatchedFile.category != "PSD",
        ).order_by(MatchedFile.confirmed_at.desc(), MatchedFile.id.desc())))
        reference_pool: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        hash_by_file: dict[int, str] = {}
        for item in all_confirmed:
            if is_ignored_image_category(item.category):
                continue
            asset = assets.get(item.id)
            if asset is None:
                continue
            value = ensure_asset_phash(asset, provider)
            hash_by_file[item.id] = value
            reference_pool[_key(item.matched_main_basename or "")].append(
                (value, item.category, item.id)
            )

        totals = {
            scenario: {"hits1": 0, "hits3": 0, "hits5": 0, "count": 0, "refs": 0}
            for scenario in SCENARIOS
        }
        categories: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {
                scenario: {"hits1": 0, "hits3": 0, "hits5": 0, "count": 0, "refs": 0}
                for scenario in SCENARIOS
            }
        )

        for batch_id in selected:
            mains = list(db.scalars(select(MainImage).where(
                MainImage.batch_id == batch_id
            ).order_by(MainImage.id)))
            body_rows = dict(db.execute(select(
                ImagePHashRegion.main_image_id, ImagePHashRegion.phash_value
            ).where(
                ImagePHashRegion.main_image_id.in_([main.id for main in mains]),
                ImagePHashRegion.region_type == "BODY",
            )).all())
            queries = [
                item for item in all_confirmed
                if item.batch_id == batch_id
                and (
                    truth_status == "confirmed"
                    or item.review_status == "CORRECTED"
                )
            ]
            for query in queries:
                query_hash = hash_by_file.get(query.id)
                truth = _key(query.matched_main_basename or "")
                if not query_hash or not truth:
                    continue
                for scenario, limit in SCENARIOS.items():
                    ranked: list[tuple[float, str, int]] = []
                    truth_ref_count = 0
                    for main in mains:
                        product = _key(main.basename)
                        unique: list[tuple[str, str, int]] = []
                        seen_hashes = {query_hash}
                        pool = reference_pool.get(product, [])
                        if reference_scope == "same_category":
                            pool = [row for row in pool if row[1] == query.category]
                        candidates = [] if limit == 0 else sorted(
                            pool,
                            key=lambda row: row[1] != query.category,
                        )
                        for value, category, source_id in candidates:
                            if source_id == query.id or value in seen_hashes:
                                continue
                            seen_hashes.add(value)
                            unique.append((value, category, source_id))
                            if len(unique) >= limit:
                                break
                        distances = [provider.distance(query_hash, body_rows[main.id])]
                        distances.extend(
                            provider.distance(query_hash, value)
                            for value, _category, _source_id in unique
                        )
                        if product == truth:
                            truth_ref_count = len(unique)
                        ranked.append((_aggregate(distances), product, len(unique)))
                    ranked.sort(reverse=True)
                    order = [product for _score, product, _count in ranked]
                    bucket = categories[query.category][scenario]
                    for target in (totals[scenario], bucket):
                        target["count"] += 1
                        target["refs"] += truth_ref_count
                        target["hits1"] += int(truth in order[:1])
                        target["hits3"] += int(truth in order[:3])
                        target["hits5"] += int(truth in order[:5])

        def metrics(raw):
            count = raw["count"]
            return {
                "samples": count,
                "average_extra_references": round(raw["refs"] / count, 2) if count else 0,
                "top1_accuracy": round(raw["hits1"] / count, 4) if count else None,
                "top3_recall": round(raw["hits3"] / count, 4) if count else None,
                "top5_recall": round(raw["hits5"] / count, 4) if count else None,
            }

        backfill = backfill_confirmed_references(db)
        db.commit()
        return {
            "feature": "manual_mask_crop_phash64",
            "query_batches": selected,
            "eligible_batches": eligible,
            "truth_status": truth_status,
            "reference_scope": reference_scope,
            "leakage_guard": "exclude query file and identical pHash references",
            "weights": [
                settings.reference_weight_1,
                settings.reference_weight_2,
                settings.reference_weight_3,
            ],
            "overall": {scenario: metrics(raw) for scenario, raw in totals.items()},
            "by_category": {
                category: {scenario: metrics(raw) for scenario, raw in values.items()}
                for category, values in sorted(categories.items())
            },
            "reference_backfill": backfill,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", type=int, action="append")
    parser.add_argument(
        "--truth-status", choices=("confirmed", "corrected"), default="confirmed"
    )
    parser.add_argument(
        "--reference-scope",
        choices=("all_categories", "same_category"),
        default="all_categories",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.batch_id, args.truth_status, args.reference_scope)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
