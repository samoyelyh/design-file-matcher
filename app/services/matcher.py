from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    BatchProgress,
    MainImage,
    MaskProfile,
    MatchCandidate,
    MatchedFile,
    MatchingRun,
)

from .assignment import GlobalAssignmentSolver, MatchConfidenceEvaluator
from .detail_phash import build_detail_region_matrix, is_detail_category
from .masks import require_manual_profiles
from .main_body import ensure_batch_main_bodies
from .phash_matching import build_full_phash_matrix, build_masked_phash_matrix
from .psd import build_psd_matrix
from .rename_plan import RenamePlanService


def _set_progress(db: Session, batch_id: int, stage: str, current: int, total: int,
                  percent: float, message: str, device: str | None = None) -> None:
    progress = db.get(BatchProgress, batch_id)
    if not progress:
        progress = BatchProgress(batch_id=batch_id)
        db.add(progress)
    progress.stage = stage
    progress.current = current
    progress.total = total
    progress.percent = max(0.0, min(100.0, percent))
    progress.message = message
    if device is not None:
        progress.device = device
    db.commit()




def _is_manual(item: MatchedFile) -> bool:
    return bool(
        item.confirmed_at
        or item.status == "MANUAL_CONFIRMED"
        or item.review_status in {"CONFIRMED", "CORRECTED"}
    )


def _apply_diagnostics(
    item: MatchedFile,
    result,
    mains: list[MainImage],
    run: MatchingRun,
) -> None:
    item.algorithm_main_id = (
        mains[result.image_best_column].id
        if result.status == "UNMATCHED"
        else None
    )
    item.algorithm_score = result.assigned_score
    item.image_best_score = result.image_best_score
    item.image_second_best_score = result.image_second_best_score
    item.image_best_main_id = mains[result.image_best_column].id
    item.assigned_rank = result.assigned_rank
    item.row_competitor_score = result.row_competitor_score
    item.row_margin = result.row_margin
    item.column_competitor_score = result.column_competitor_score
    item.column_margin = result.column_margin
    item.algorithm_status = result.status
    item.algorithm_version = run.algorithm_version
    item.matching_run_id = run.id


def _assign_categories(
    db: Session,
    batch: Batch,
    mains: list[MainImage],
    files: list[MatchedFile],
    category_matrices: dict[str, np.ndarray],
    baseline_matrices: dict[str, np.ndarray],
    profiles: dict[str, MaskProfile],
) -> Counter[str]:
    solver = GlobalAssignmentSolver(settings.unmatched_score)
    evaluator = MatchConfidenceEvaluator(
        settings.match_min_score, settings.match_min_margin
    )
    positions: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(files):
        positions[item.category].append(index)

    totals: Counter[str] = Counter()
    for category, global_rows in positions.items():
        category_files = [files[index] for index in global_rows]
        category_matrix = category_matrices[category]
        baseline_matrix = baseline_matrices[category]
        profile = profiles.get(category)
        is_psd = category == "PSD"
        is_detail = is_detail_category(category)
        run = MatchingRun(
            batch_id=batch.id,
            category=category,
            algorithm_version=(
                settings.psd_algorithm_version
                if is_psd
                else settings.detail_region_algorithm_version
                if is_detail
                else settings.masked_algorithm_version
            ),
            model_name="pHash64",
            match_min_score=settings.match_min_score,
            match_min_margin=settings.match_min_margin,
            unmatched_score=settings.unmatched_score,
            mask_profile_id=profile.id if profile and not is_detail else None,
            feature_version=(
                "phash64_v1"
                if is_psd
                else settings.body_phash_version
                if is_detail
                else f"masked_phash_v1:mask_v{profile.version}"
            ),
            image_count=len(category_files),
            product_count=len(mains),
        )
        db.add(run)
        db.flush()

        keep = min(5, len(mains))
        for row, item in enumerate(category_files):
            if not is_psd and not is_detail:
                item.match_method = "MASKED_PHASH"
            order = np.argsort(-category_matrix[row])[:keep]
            baseline_order = np.argsort(-baseline_matrix[row])[:2]
            item.top1_score = float(category_matrix[row, int(order[0])])
            item.top2_score = (
                float(category_matrix[row, int(order[1])]) if len(order) > 1 else None
            )
            item.top3_score = (
                float(category_matrix[row, int(order[2])]) if len(order) > 2 else None
            )
            item.baseline_top1_main_id = mains[int(baseline_order[0])].id
            item.baseline_top1_score = float(baseline_matrix[row, int(baseline_order[0])])
            item.baseline_top2_score = (
                float(baseline_matrix[row, int(baseline_order[1])])
                if len(baseline_order) > 1 else None
            )
            for rank, column in enumerate(order, 1):
                db.add(MatchCandidate(
                    matched_file_id=item.id,
                    main_image_id=mains[int(column)].id,
                    rank=rank,
                    similarity=float(category_matrix[row, int(column)]),
                ))

        assignments = solver.solve(category_matrix)
        evaluated = [
            (assignment, evaluator.evaluate(category_matrix, assignment))
            for assignment in assignments
        ]
        overflow_count = max(0, len(category_files) - len(mains))
        overflow_candidates = sorted(
            (
                (assignment.row, result.image_best_score)
                for assignment, result in evaluated
                if result.status == "UNMATCHED"
                and not _is_manual(category_files[assignment.row])
            ),
            key=lambda value: value[1],
        )
        ignored_extra_rows = {
            row for row, _ in overflow_candidates[:overflow_count]
        }
        artifact: list[dict] = []
        for assignment, result in evaluated:
            category_row = assignment.row
            assigned_column = assignment.column
            item = category_files[category_row]
            effective_status = (
                "EXTRA_IGNORED"
                if category_row in ignored_extra_rows
                else result.status
            )
            _apply_diagnostics(item, result, mains, run)
            item.algorithm_main_id = (
                mains[assigned_column].id if assigned_column is not None else None
            )
            item.is_conflict = False
            if _is_manual(item):
                item.status = "MANUAL_CONFIRMED"
                totals["MANUAL_CONFIRMED"] += 1
            else:
                item.is_low_confidence = effective_status == "LOW_CONFIDENCE"
                item.status = effective_status
                item.review_status = (
                    "IGNORED" if effective_status == "EXTRA_IGNORED" else "PENDING"
                )
                if assigned_column is None:
                    item.matched_main_id = None
                    item.matched_main_basename = None
                else:
                    winner = mains[assigned_column]
                    item.matched_main_id = winner.id
                    item.matched_main_basename = winner.basename
                totals[effective_status] += 1
            artifact_row = {
                "file_id": item.id,
                "filename": item.original_filename,
                "assigned_main_id": item.algorithm_main_id,
                "status": item.status,
                "algorithm_status": result.status,
                "assigned_score": result.assigned_score,
                "row_margin": result.row_margin,
                "column_margin": result.column_margin,
                "assigned_rank": result.assigned_rank,
                "baseline_top1_main_id": item.baseline_top1_main_id,
                "baseline_top1_score": item.baseline_top1_score,
                "baseline_top2_score": item.baseline_top2_score,
                "mask_profile_id": profile.id if profile else None,
                "mask_profile_version": profile.version if profile else None,
                "match_method": item.match_method,
                "top5": [
                    {
                        "main_id": mains[int(column)].id,
                        "score": float(category_matrix[category_row, int(column)]),
                    }
                    for column in np.argsort(-category_matrix[category_row])[:5]
                ],
            }
            if item.status == "MANUAL_CONFIRMED":
                artifact_row["manual_product_id"] = item.matched_main_id
            artifact.append(artifact_row)

        run.summary_json = json.dumps(
            {
                "counts": dict(Counter(row["status"] for row in artifact)),
                "results": artifact,
            },
            ensure_ascii=False,
        )
        run.finished_at = datetime.now(timezone.utc)
    return totals


def match_batch(db: Session, batch_id: int, _legacy_provider=None) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    if batch.status not in {"SCANNED", "QUEUED", "FAILED", "COMPLETED"}:
        raise ValueError(f"批次状态 {batch.status} 不能开始匹配")
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch.id
    ).order_by(MainImage.id)))
    files = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch.id
    ).order_by(MatchedFile.id)))
    if not mains:
        raise ValueError("没有主图，请先扫描")
    if not files:
        raise ValueError("没有待匹配文件")
    profile_rows = require_manual_profiles(db, batch.id)
    profiles = {profile.image_category: profile for profile in profile_rows}

    batch.status = "MATCHING"
    batch.error_message = None
    db.commit()
    try:
        _set_progress(
            db, batch.id, "MAIN_BODY_PHASH", 0, len(mains), 2,
            "正在提取主图主体并生成整图、主体和局部 pHash",
        )
        body_artifacts = ensure_batch_main_bodies(db, batch, mains)
        db.commit()
        device_label = "CPU / pHash"
        db.execute(delete(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_([item.id for item in files])
        ))
        _set_progress(
            db, batch.id, "FULL_PHASH", 0, len(files), 20,
            "正在生成并缓存整图 pHash", device_label,
        )
        full_similarities = build_full_phash_matrix(db, files, mains)
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(files):
            positions[item.category].append(index)
        baseline_matrices = {
            category: full_similarities[rows, :]
            for category, rows in positions.items()
        }
        category_matrices: dict[str, np.ndarray] = {}
        if "PSD" in positions:
            psd_files = [files[index] for index in positions["PSD"]]
            category_matrices["PSD"], _ = build_psd_matrix(
                db,
                batch,
                psd_files,
                mains,
            )

        detail_categories = [
            category
            for category in positions
            if category != "PSD" and is_detail_category(category)
        ]
        for category in detail_categories:
            detail_files = [files[index] for index in positions[category]]
            category_matrices[category] = build_detail_region_matrix(
                db, detail_files, mains
            )
        db.commit()

        visual_categories = [
            category
            for category in positions
            if category != "PSD" and category not in detail_categories
        ]
        for category_index, category in enumerate(visual_categories, 1):
            profile = profiles[category]
            category_files = [files[index] for index in positions[category]]
            _set_progress(
                db, batch.id, "MASKED_CROP", category_index - 1,
                len(visual_categories), 45 + 45 * (category_index - 1) / max(len(visual_categories), 1),
                f"正在生成 {category} Masked Crop 并计算 pHash（Mask v{profile.version}）",
                device_label,
            )
            category_matrices[category] = build_masked_phash_matrix(
                db, batch, category_files, mains, profile
            )
            db.commit()
        _set_progress(
            db, batch.id, "SIMILARITY", len(files), len(files), 92,
            "正在按分类执行纯 pHash 全局一对一匹配", device_label,
        )
        totals = _assign_categories(
            db,
            batch,
            mains,
            files,
            category_matrices,
            baseline_matrices,
            profiles,
        )
        db.commit()
        _set_progress(
            db, batch.id, "SAVING", len(files), len(files), 96,
            "正在保存诊断并生成 Rename Plan（尚不执行改名）", device_label,
        )
        rename_plan = RenamePlanService().generate(db, batch.id)
        batch.status = "COMPLETED"
        batch.finished_at = datetime.now(timezone.utc)
        _set_progress(
            db, batch.id, "COMPLETED", len(files), len(files), 100,
            "全局一对一匹配完成，源文件未改名", device_label,
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        batch = db.get(Batch, batch_id)
        batch.status = "FAILED"
        batch.error_message = str(exc)
        _set_progress(
            db, batch.id, "FAILED", 0, 0, 0,
            f"匹配失败：{exc}",
        )
        db.commit()
        raise

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "matched": len(files),
        "status_counts": dict(totals),
        "low_confidence": totals["LOW_CONFIDENCE"],
        "conflicts": 0,
        "rename_plan": {
            "plan_group": rename_plan["plan_group"],
            "counts": rename_plan["counts"],
        },
        "main_body_phash": {
            "total": len(body_artifacts),
            "cached": sum(artifact.cached for artifact in body_artifacts),
        },
    }
