from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import MainImage, MatchCandidate, MatchedFile
from app.services.matcher import match_batch


def category_report(files: list[MatchedFile], main_count: int) -> dict:
    rows = defaultdict(list)
    for item in files:
        rows[item.category].append(item)
    report = {}
    for category, items in sorted(rows.items()):
        assigned = [item for item in items if item.matched_main_id is not None]
        automatic = [item for item in assigned if item.status != "MANUAL_CONFIRMED"]
        manual = [item for item in assigned if item.status == "MANUAL_CONFIRMED"]
        counts = Counter(item.status for item in items)
        product_counts = Counter(item.matched_main_id for item in assigned)
        automatic_counts = Counter(item.matched_main_id for item in automatic)
        manual_counts = Counter(item.matched_main_id for item in manual)
        algorithm_counts = Counter(item.algorithm_status for item in items)
        algorithm_products = Counter(
            item.algorithm_main_id for item in items if item.algorithm_main_id is not None
        )
        report[category] = {
            "source_files": len(items),
            "MATCHED": counts["MATCHED"],
            "AMBIGUOUS": counts["AMBIGUOUS"],
            "LOW_CONFIDENCE": counts["LOW_CONFIDENCE"],
            "UNMATCHED": counts["UNMATCHED"],
            "MANUAL_CONFIRMED": counts["MANUAL_CONFIRMED"],
            "unique_products": len(product_counts),
            "duplicate_assignments": sum(count - 1 for count in product_counts.values()),
            "automatic_duplicate_assignments": sum(
                count - 1 for count in automatic_counts.values()
            ),
            "manual_duplicate_assignments": sum(
                count - 1 for count in manual_counts.values()
            ),
            "algorithm_status_counts": dict(algorithm_counts),
            "algorithm_unique_products": len(algorithm_products),
            "algorithm_duplicate_assignments": sum(
                count - 1 for count in algorithm_products.values()
            ),
            "products_without_image": main_count - len(product_counts),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    init_db()
    with SessionLocal() as db:
        before = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == args.batch
        )))
        mains = list(db.scalars(select(MainImage).where(
            MainImage.batch_id == args.batch
        )))
        paths_before = sum(Path(item.current_path).is_file() for item in before)
        old = category_report(before, len(mains))
        result = match_batch(db, args.batch)
        after = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == args.batch
        )))
        paths_after = sum(Path(item.current_path).is_file() for item in after)
        spectator_37 = next(
            (
                item for item in after
                if item.category == "观赛图" and item.original_filename == "37.jpg"
            ),
            None,
        )
        detail = None
        if spectator_37:
            candidates = list(db.scalars(select(MatchCandidate).where(
                MatchCandidate.matched_file_id == spectator_37.id
            ).order_by(MatchCandidate.rank)))
            candidate_names = {
                main.id: main.basename for main in mains
            }
            detail = {
                "file_id": spectator_37.id,
                "final_product": spectator_37.matched_main_basename,
                "algorithm_product": candidate_names.get(spectator_37.algorithm_main_id),
                "assigned_score": spectator_37.algorithm_score,
                "original_top1": {
                    "product": candidate_names[candidates[0].main_image_id],
                    "score": candidates[0].similarity,
                } if candidates else None,
                "original_top2": {
                    "product": candidate_names[candidates[1].main_image_id],
                    "score": candidates[1].similarity,
                } if len(candidates) > 1 else None,
                "row_margin": spectator_37.row_margin,
                "column_margin": spectator_37.column_margin,
                "assigned_rank": spectator_37.assigned_rank,
                "status": spectator_37.status,
                "algorithm_status": spectator_37.algorithm_status,
            }
        arizona = next((main for main in mains if main.basename.strip() == "Arizona"), None)
        payload = {
            "match_result": result,
            "physical_files_before": paths_before,
            "physical_files_after": paths_after,
            "old_snapshot": old,
            "new": category_report(after, len(mains)),
            "spectator_arizona_count": sum(
                item.category == "观赛图" and item.matched_main_id == arizona.id
                for item in after
            ) if arizona else None,
            "spectator_arizona_algorithm_count": sum(
                item.category == "观赛图" and item.algorithm_main_id == arizona.id
                for item in after
            ) if arizona else None,
            "spectator_37": detail,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
