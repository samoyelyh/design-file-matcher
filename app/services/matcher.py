from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.config import is_excluded_file_category, is_ignored_image_category
from app.models import (
    Batch,
    BatchProgress,
    MainImage,
    MaskProfile,
    MatchCandidate,
    MatchedFile,
    MatchingRun,
    ProductSimilarity,
)

from .assignment import (
    Assignment,
    MatchConfidenceEvaluator,
    solve_complete_matrix,
)
from .match_engines import EngineContext, EngineOutput, get_match_engine, normalize_engine_name
from .masks import MAIN_IMAGE_CATEGORY, require_manual_profiles
from .references import ProductScoreMatrix
from .rename_plan import RenamePlanService
from .filename_matching import EXACT, associate_filenames


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


def _ignore_unused_images(files: list[MatchedFile]) -> Counter[str]:
    """Keep ignored source folders untouched and out of review/rename gates."""
    totals: Counter[str] = Counter()
    for item in files:
        item.matched_main_id = None
        item.matched_main_basename = None
        item.algorithm_main_id = None
        item.algorithm_score = None
        item.image_best_main_id = None
        item.image_best_score = None
        item.image_second_best_score = None
        item.assigned_rank = None
        item.top1_score = None
        item.top2_score = None
        item.top3_score = None
        item.phash_distance = None
        item.phash_best_region_type = None
        item.match_method = "IGNORED_CATEGORY"
        item.match_details_json = None
        item.algorithm_version = None
        item.matching_run_id = None
        item.algorithm_status = "EXTRA_IGNORED"
        item.status = "EXTRA_IGNORED"
        item.review_status = "IGNORED"
        item.is_conflict = False
        item.is_low_confidence = False
        totals["EXTRA_IGNORED"] += 1
    return totals


def _assign_categories(
    db: Session,
    batch: Batch,
    mains: list[MainImage],
    files: list[MatchedFile],
    product_scores: dict[str, ProductScoreMatrix],
    profiles: dict[str, MaskProfile],
    engine_output: EngineOutput,
    *,
    category_outputs: dict[str, EngineOutput] | None = None,
    independent_categories: set[str] | None = None,
) -> Counter[str]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(files):
        positions[item.category].append(index)

    totals: Counter[str] = Counter()
    for category, global_rows in positions.items():
        category_files = [files[index] for index in global_rows]
        product_score = product_scores[category]
        category_output = (category_outputs or {}).get(category, engine_output)
        independent = category in (independent_categories or set())
        evaluator = MatchConfidenceEvaluator(
            category_output.min_score,
            category_output.min_margin,
        )
        candidate_matrix = product_score.matrix
        expected_shape = (len(category_files), len(mains))
        if candidate_matrix.shape != expected_shape:
            raise ValueError(
                f"{category} Product Score Matrix 维度错误："
                f"{candidate_matrix.shape} != {expected_shape}"
            )
        if not np.isfinite(candidate_matrix).all():
            raise ValueError(f"{category} Product Score Matrix 包含无效分数")
        # This is the complete Query x Product matrix. Top5 is derived only
        # for review after the global solution and never masks these columns.
        assignment_matrix = candidate_matrix
        profile = profiles.get(category)
        main_profile = profiles.get(MAIN_IMAGE_CATEGORY)
        run = MatchingRun(
            batch_id=batch.id,
            category=category,
            algorithm_version=category_output.algorithm_version,
            model_name=category_output.model_name,
            match_min_score=category_output.min_score,
            match_min_margin=category_output.min_margin,
            unmatched_score=category_output.unmatched_score,
            mask_profile_id=profile.id if profile else None,
            feature_version=(
                f"manual_masks:main_v{main_profile.version}:{category}_v{profile.version}"
                if profile and main_profile else category_output.feature_version_prefix
            ),
            image_count=len(category_files),
            product_count=len(mains),
        )
        db.add(run)
        db.flush()

        # Persist every cell of the complete matrix for fast, model-consistent
        # reassignment. MatchCandidate remains Top5-only for the review payload.
        full_score_rows = []
        for row, item in enumerate(category_files):
            for column, main in enumerate(mains):
                evidence = product_score.evidence[row][column]
                full_score_rows.append({
                    "batch_id": batch.id,
                    "matching_run_id": run.id,
                    "matched_file_id": item.id,
                    "main_image_id": main.id,
                    "model_name": run.model_name,
                    "score": float(candidate_matrix[row, column]),
                    "best_distance": evidence.best_distance,
                    "second_distance": evidence.second_distance,
                    "third_distance": evidence.third_distance,
                    "support_count": evidence.support_count,
                    "details_json": json.dumps(evidence.details, ensure_ascii=False) if evidence.details else None,
                    "created_at": datetime.now(timezone.utc),
                })
        if full_score_rows:
            db.execute(insert(ProductSimilarity), full_score_rows)

        main_column_by_id = {main.id: column for column, main in enumerate(mains)}
        locked_columns_by_row = {
            row: main_column_by_id[item.matched_main_id]
            for row, item in enumerate(category_files)
            if _is_manual(item) and item.matched_main_id in main_column_by_id
        }
        reserved = set(db.scalars(select(MatchedFile.matched_main_id).where(
            MatchedFile.batch_id == batch.id,
            MatchedFile.category == category,
            MatchedFile.match_method == EXACT,
            MatchedFile.matched_main_id.is_not(None),
        )))
        available = [column for column, main in enumerate(mains) if main.id not in reserved]
        reduced_locked = {row: available.index(column) for row, column in locked_columns_by_row.items()
                          if column in available}
        assignments = [Assignment(a.row, available[a.column] if a.column is not None else None)
                       for a in solve_complete_matrix(assignment_matrix[:, available],
                           category_output.unmatched_score, reduced_locked, force_assign=True)]
        evaluated = [
            (
                assignment,
                evaluator.evaluate(assignment_matrix, assignment),
            )
            for assignment in assignments
        ]

        # Persist only the five best products for review. The solver above has
        # already evaluated every product column, including a possible Top6+.
        keep = min(5, len(mains))
        for row, item in enumerate(category_files):
            order = np.argsort(-candidate_matrix[row])[:keep]
            item.top1_score = float(candidate_matrix[row, int(order[0])])
            item.top2_score = (
                float(candidate_matrix[row, int(order[1])]) if len(order) > 1 else None
            )
            item.top3_score = (
                float(candidate_matrix[row, int(order[2])]) if len(order) > 2 else None
            )
            item.baseline_top1_main_id = None
            item.baseline_top1_score = None
            item.baseline_top2_score = None
            for rank, column in enumerate(order, 1):
                evidence = product_score.evidence[row][int(column)]
                db.add(MatchCandidate(
                    matched_file_id=item.id,
                    main_image_id=mains[int(column)].id,
                    rank=rank,
                    similarity=evidence.aggregated_score,
                    distance=evidence.best_distance,
                    similarity_display=(
                        evidence.aggregated_score
                        if evidence.best_distance is None
                        else 1.0 - evidence.best_distance / 64.0
                    ),
                    reference_id=evidence.reference_id,
                    reference_category=evidence.reference_category,
                    best_distance=evidence.best_distance,
                    second_distance=evidence.second_distance,
                    third_distance=evidence.third_distance,
                    support_count=evidence.support_count,
                    aggregated_score=evidence.aggregated_score,
                    details_json=json.dumps(evidence.details, ensure_ascii=False) if evidence.details else None,
                ))

        overflow_count = 0 if independent else max(0, len(category_files) - len(mains))
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
            top_product_column = int(np.argmax(candidate_matrix[category_row]))
            top_product_evidence = product_score.evidence[category_row][top_product_column]
            item.phash_distance = top_product_evidence.best_distance
            selected_evidence = (
                product_score.evidence[category_row][assigned_column]
                if assigned_column is not None
                else top_product_evidence
            )
            item.match_details_json = (
                json.dumps(selected_evidence.details, ensure_ascii=False)
                if selected_evidence.details else None
            )
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
                "mask_profile_id": profile.id if profile else None,
                "mask_profile_version": profile.version if profile else None,
                "main_mask_profile_id": main_profile.id if main_profile else None,
                "main_mask_profile_version": main_profile.version if main_profile else None,
                "match_method": item.match_method,
                "top5": [
                    {
                        "main_id": mains[int(column)].id,
                        "aggregated_score": float(
                            candidate_matrix[category_row, int(column)]
                        ),
                        "best_distance": product_score.evidence[
                            category_row
                        ][int(column)].best_distance,
                        "second_distance": product_score.evidence[
                            category_row
                        ][int(column)].second_distance,
                        "third_distance": product_score.evidence[
                            category_row
                        ][int(column)].third_distance,
                        "support_count": product_score.evidence[
                            category_row
                        ][int(column)].support_count,
                    }
                    for column in np.argsort(-candidate_matrix[category_row])[:5]
                ],
            }
            if item.status == "MANUAL_CONFIRMED":
                artifact_row["manual_product_id"] = item.matched_main_id
            artifact.append(artifact_row)

        run.summary_json = json.dumps(
            {
                "counts": dict(Counter(row["status"] for row in artifact)),
                "match_engine": batch.match_engine,
                "assignment_input": category_output.assignment_input,
                "matrix_shape": [len(category_files), len(mains)],
                "top5_is_display_only": True,
                "unmatched_fill": "complete_matrix_hungarian",
                "manual_locked_rows": len(locked_columns_by_row),
                "candidate_input": category_output.candidate_input,
                "results": artifact,
            },
            ensure_ascii=False,
        )
        run.finished_at = datetime.now(timezone.utc)
    return totals


def match_batch(
    db: Session,
    batch_id: int,
    _legacy_provider=None,
    *,
    category_filter: set[str] | None = None,
    forced_engine: str | None = None,
    independent: bool = False,
    finalize: bool = True,
) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    if batch.status not in {"SCANNED", "QUEUED", "FAILED", "COMPLETED"} and not (
        category_filter and batch.status == "MATCHING"
    ):
        raise ValueError(f"批次状态 {batch.status} 不能开始匹配")
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch.id
    ).order_by(MainImage.id)))
    files = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch.id
    ).order_by(MatchedFile.id)))
    associate_filenames(db, batch_id)
    exact_files = [item for item in files if item.match_method == EXACT
                   and (not category_filter or item.category in category_filter)]
    selected_categories = set(category_filter or ())
    ignored_files = [
        item for item in files
        if is_ignored_image_category(item.category)
        and (not selected_categories or item.category in selected_categories)
    ]
    image_files = [
        item for item in files
        if not is_excluded_file_category(item.category)
        and item.category != "PSD"
        and item.match_method != EXACT
        and item.status != "CONFLICT_MARKED"
        and (not selected_categories or item.category in selected_categories)
    ]
    psd_files = [item for item in files if item.category == "PSD"
                 and not is_excluded_file_category(item.category)
                 and item.match_method != EXACT and item.status != "CONFLICT_MARKED"
                 and (not selected_categories or item.category in selected_categories)]
    if not mains:
        raise ValueError("没有主图，请先扫描")
    if not image_files and not psd_files:
        if exact_files:
            if finalize:
                RenamePlanService().generate(db, batch_id)
                batch.status = "COMPLETED"
                batch.finished_at = datetime.now(timezone.utc)
                _set_progress(db, batch_id, "COMPLETED", len(exact_files), len(exact_files), 100,
                              "文件名一致，已直接关联；已跳过视觉识别")
            db.commit()
            return {"batch_id": batch_id, "status": batch.status, "matched": len(exact_files),
                    "exact_filename_count": len(exact_files)}
        raise ValueError("没有待匹配图片")
    profile_rows = require_manual_profiles(db, batch.id, categories={item.category for item in image_files}) if image_files else []
    profiles = {profile.image_category: profile for profile in profile_rows}
    batch.status = "MATCHING"
    batch.error_message = None
    db.commit()
    try:
        original_engine = batch.match_engine
        if forced_engine:
            batch.match_engine = normalize_engine_name(forced_engine)
        engine = get_match_engine(batch)
        if not category_filter and not forced_engine:
            batch.match_engine = engine.name
            config_snapshot = engine.config_snapshot()
            batch.match_engine_version = str(config_snapshot.get("version") or "") or None
            batch.match_config_snapshot = json.dumps(config_snapshot, ensure_ascii=False)
        db.commit()
        matching_files = [*image_files, *psd_files]
        db.execute(delete(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_([item.id for item in matching_files])
        ))
        db.execute(delete(ProductSimilarity).where(
            ProductSimilarity.matched_file_id.in_([item.id for item in matching_files])
        ))
        def report_progress(stage: str, current: int, total: int, percent: float,
                            message: str, device: str | None = None) -> None:
            _set_progress(db, batch.id, stage, current, total, percent, message, device)

        engine_output = engine.build_score_matrices(EngineContext(
            db=db,
            batch=batch,
            mains=mains,
            files=image_files,
            profiles=profiles,
            progress=report_progress,
        )) if image_files else None
        category_outputs = {}
        if psd_files:
            from .psd import build_psd_output
            report_progress("PSD_PREVIEW", 0, len(psd_files), 60, "正在提取PSD合成预览并匹配主图", "CPU / PSD预览pHash")
            psd_output = build_psd_output(db, batch, psd_files, mains)
            category_outputs["PSD"] = psd_output
            if engine_output is None:
                engine_output = psd_output
        product_scores = dict(engine_output.product_scores)
        if psd_files:
            product_scores.update(psd_output.product_scores)
        device_label = engine_output.device_label
        body_artifacts = engine_output.artifacts
        _set_progress(
            db, batch.id, "SIMILARITY", len(image_files), len(image_files), 92,
            f"正在使用 {engine.label} 的完整 Product Score Matrix 执行"
            f"{'每张副图独立匹配' if independent else '分类内 Hungarian'}",
            device_label,
        )
        totals = _assign_categories(
            db,
            batch,
            mains,
            matching_files,
            product_scores,
            profiles,
            engine_output,
            category_outputs=category_outputs,
            independent_categories=set(item.category for item in matching_files) if independent else set(),
        )
        totals.update(_ignore_unused_images(ignored_files))
        db.commit()
        rename_plan = None
        if finalize:
            _set_progress(
                db, batch.id, "SAVING", len(matching_files), len(matching_files), 96,
                "正在保存诊断并生成 Rename Plan（尚不执行改名）", device_label,
            )
            rename_plan = RenamePlanService().generate(db, batch.id)
            batch.status = "COMPLETED"
            batch.finished_at = datetime.now(timezone.utc)
            _set_progress(
                db, batch.id, "COMPLETED", len(matching_files), len(matching_files), 100,
                "完整矩阵匹配完成；PSD使用合成预览，请审核后命名",
                device_label,
            )
        else:
            batch.status = "MATCHING"
        if forced_engine:
            batch.match_engine = original_engine
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
        "matched": len(matching_files),
        "psd_matched": len(psd_files),
        "psd_skipped": 0,
        "status_counts": dict(totals),
        "low_confidence": totals["LOW_CONFIDENCE"],
        "conflicts": 0,
        "rename_plan": {
            "plan_group": rename_plan["plan_group"],
            "counts": rename_plan["counts"],
        } if rename_plan else None,
        "main_body_phash": {
            "total": len(body_artifacts),
            "cached": sum(artifact.cached for artifact in body_artifacts),
        },
    }
