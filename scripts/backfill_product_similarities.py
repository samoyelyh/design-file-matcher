from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import delete, insert, select

from app.config import is_ignored_image_category
from app.db import SessionLocal, init_db
from app.models import Batch, MainImage, MatchedFile, MatchingRun, ProductSimilarity
from app.services.dinov3 import DINO_MODELS, build_dino_product_matrices, require_model_ready
from app.services.masks import require_manual_profiles


def backfill(batch_id: int) -> dict:
    init_db()
    with SessionLocal() as db:
        batch = db.get(Batch, batch_id)
        if batch is None:
            raise ValueError("批次不存在")
        if batch.matching_model not in DINO_MODELS:
            raise ValueError("只需要为旧 DINO 批次补齐完整矩阵")
        require_model_ready(batch.matching_model)
        mains = list(db.scalars(select(MainImage).where(
            MainImage.batch_id == batch.id
        ).order_by(MainImage.id)))
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch.id,
            MatchedFile.category != "PSD",
        ).order_by(MatchedFile.id)))
        files = [item for item in files if not is_ignored_image_category(item.category)]
        profiles = {
            profile.image_category: profile
            for profile in require_manual_profiles(db, batch.id)
        }
        files_by_category: dict[str, list[MatchedFile]] = defaultdict(list)
        for item in files:
            files_by_category[item.category].append(item)
        matrices, scores = build_dino_product_matrices(
            db, batch, mains, dict(files_by_category), profiles, batch.matching_model
        )
        db.execute(delete(ProductSimilarity).where(ProductSimilarity.batch_id == batch.id))
        created = datetime.now(timezone.utc)
        rows = []
        for category, category_files in files_by_category.items():
            run = db.scalar(select(MatchingRun).where(
                MatchingRun.batch_id == batch.id,
                MatchingRun.category == category,
            ).order_by(MatchingRun.id.desc()))
            if run is None:
                raise ValueError(f"{category} 没有可用的匹配运行记录")
            matrix = matrices[category]
            evidence = scores[category].evidence
            for row_index, item in enumerate(category_files):
                item.matching_run_id = run.id
                for column, main in enumerate(mains):
                    cell = evidence[row_index][column]
                    rows.append({
                        "batch_id": batch.id,
                        "matching_run_id": run.id,
                        "matched_file_id": item.id,
                        "main_image_id": main.id,
                        "model_name": run.model_name,
                        "score": float(matrix[row_index, column]),
                        "best_distance": cell.best_distance,
                        "second_distance": cell.second_distance,
                        "third_distance": cell.third_distance,
                        "support_count": cell.support_count,
                        "created_at": created,
                    })
        if rows:
            db.execute(insert(ProductSimilarity), rows)
        db.commit()
        return {
            "batch_id": batch.id,
            "model": batch.matching_model,
            "files": len(files),
            "products": len(mains),
            "matrix_cells": len(rows),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", type=int, required=True)
    args = parser.parse_args()
    print(backfill(args.batch_id))


if __name__ == "__main__":
    main()
