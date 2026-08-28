from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    BatchProgress,
    DerivedImageAsset,
    MainImage,
    MainImageFeature,
    MaskProfile,
    MatchCandidate,
    MatchedFile,
    MatchingRun,
)

from .assignment import GlobalAssignmentSolver, MatchConfidenceEvaluator
from .cropper import MaskedCropProcessor
from .features import (
    DinoFeatureProvider,
    ImageFeatureProvider,
    OpenClipFeatureProvider,
    SimilarityMatrixBuilder,
)
from .masks import require_manual_profiles
from .media import embedding_cache_path
from .psd import prepare_psd_previews_and_phash
from .rename_plan import RenamePlanService


ImageEmbedder = ImageFeatureProvider
OpenClipEmbedder = OpenClipFeatureProvider
DinoEmbedder = DinoFeatureProvider


def _feature_version(embedder: ImageEmbedder) -> str:
    return str(getattr(embedder, "feature_version", embedder.__class__.__name__))


def _model_name(embedder: ImageEmbedder) -> str:
    return str(getattr(embedder, "model_name", embedder.__class__.__name__))


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


def _encode_in_batches(embedder: ImageEmbedder, paths: list[Path],
                       on_progress=None) -> np.ndarray:
    chunks = []
    start = 0
    batch_size = max(1, settings.match_batch_size)
    while start < len(paths):
        chunk = paths[start:start + batch_size]
        try:
            chunks.append(embedder.encode(chunk))
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).casefold()
            if not is_oom or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            empty_cache = getattr(embedder, "empty_cache", None)
            if empty_cache:
                empty_cache()
            continue
        start += len(chunk)
        if on_progress:
            on_progress(start)
    vectors = np.concatenate(chunks, axis=0).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _main_embeddings(batch: Batch, mains: list[MainImage], embedder: ImageEmbedder,
                     on_progress=None) -> np.ndarray:
    model_name = _model_name(embedder)
    feature_version = _feature_version(embedder)
    paths = [Path(item.file_path) for item in mains]
    root = Path(batch.root_path)
    cache_paths = [
        embedding_cache_path(root, batch.id, source, feature_version)
        for source in paths
    ]
    cached = all(
        item.embedding_path
        and item.embedding_model == model_name
        and Path(item.embedding_path) == cache
        and cache.exists()
        for item, cache in zip(mains, cache_paths)
    )
    if cached:
        vectors = np.stack([np.load(item.embedding_path) for item in mains]).astype(np.float32)
        if on_progress:
            on_progress(len(mains))
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

    vectors = _encode_in_batches(embedder, paths, on_progress=on_progress)
    for item, vector, cache in zip(mains, vectors, cache_paths):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vector)
        item.embedding_path = str(cache)
        item.embedding_model = model_name
        item.embedding_dim = int(vector.shape[0])
        item.embedding_updated_at = datetime.now(timezone.utc)
    return vectors


def _file_embeddings(
    batch: Batch,
    files: list[MatchedFile],
    paths: list[Path],
    embedder: ImageEmbedder,
    on_progress=None,
) -> np.ndarray:
    model_name = _model_name(embedder)
    feature_version = _feature_version(embedder)
    root = Path(batch.root_path)
    cache_paths = [
        embedding_cache_path(root, batch.id, source, feature_version)
        for source in paths
    ]
    cached = all(
        item.embedding_path
        and item.embedding_model == model_name
        and Path(item.embedding_path) == cache
        and cache.exists()
        for item, cache in zip(files, cache_paths)
    )
    if cached:
        vectors = np.stack([np.load(item.embedding_path) for item in files]).astype(
            np.float32
        )
        if on_progress:
            on_progress(len(files))
        return vectors / np.clip(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None
        )

    vectors = _encode_in_batches(embedder, paths, on_progress=on_progress)
    now = datetime.now(timezone.utc)
    for item, vector, cache in zip(files, vectors, cache_paths):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vector)
        item.embedding_path = str(cache)
        item.embedding_model = model_name
        item.embedding_dim = int(vector.shape[0])
        item.embedding_updated_at = now
    return vectors


def _derived_embeddings(
    batch: Batch,
    embedder: ImageEmbedder,
    assets: list[DerivedImageAsset],
    on_progress=None,
) -> np.ndarray:
    feature_version = _feature_version(embedder)
    cache_paths = [
        embedding_cache_path(
            Path(batch.root_path),
            batch.id,
            Path(asset.asset_path),
            feature_version,
        )
        for asset in assets
    ]
    if all(path.is_file() for path in cache_paths):
        vectors = np.stack([np.load(path) for path in cache_paths]).astype(np.float32)
        if on_progress:
            on_progress(len(assets))
    else:
        vectors = _encode_in_batches(
            embedder,
            [Path(asset.asset_path) for asset in assets],
            on_progress=on_progress,
        )
        for path, vector in zip(cache_paths, vectors):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def _record_masked_main_features(
    db: Session,
    batch: Batch,
    mains: list[MainImage],
    assets: list[DerivedImageAsset],
    profile: MaskProfile,
    embedder: ImageEmbedder,
) -> None:
    feature_version = _feature_version(embedder)
    version = f"{feature_version}:mask_v{profile.version}"
    cache_paths = [
        embedding_cache_path(
            Path(batch.root_path),
            batch.id,
            Path(asset.asset_path),
            feature_version,
        )
        for asset in assets
    ]
    existing = set(db.scalars(select(MainImageFeature.main_image_id).where(
        MainImageFeature.feature_type == "MASKED_EMBEDDING",
        MainImageFeature.feature_version == version,
        MainImageFeature.mask_profile_id == profile.id,
    )).all())
    for main, asset, cache_path in zip(mains, assets, cache_paths):
        if main.id not in existing:
            db.add(MainImageFeature(
                main_image_id=main.id,
                feature_type="MASKED_EMBEDDING",
                feature_version=version,
                model_name=_model_name(embedder),
                mask_profile_id=profile.id,
                feature_path=str(cache_path),
                crop_asset_id=asset.id,
            ))


def _record_full_main_features(
    db: Session,
    mains: list[MainImage],
    embedder: ImageEmbedder,
) -> None:
    feature_version = _feature_version(embedder)
    model_name = _model_name(embedder)
    existing = set(db.scalars(select(MainImageFeature.main_image_id).where(
        MainImageFeature.feature_type == "FULL_EMBEDDING",
        MainImageFeature.feature_version == feature_version,
    )).all())
    for main in mains:
        if main.id not in existing:
            db.add(MainImageFeature(
                main_image_id=main.id,
                feature_type="FULL_EMBEDDING",
                feature_version=feature_version,
                model_name=model_name,
                feature_path=main.embedding_path,
            ))


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
    model_name: str,
    feature_version: str,
    match_min_score: float,
    match_min_margin: float,
    unmatched_score: float,
) -> Counter[str]:
    solver = GlobalAssignmentSolver(unmatched_score)
    evaluator = MatchConfidenceEvaluator(match_min_score, match_min_margin)
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
        run = MatchingRun(
            batch_id=batch.id,
            category=category,
            algorithm_version=(
                settings.psd_algorithm_version
                if is_psd else settings.masked_algorithm_version
            ),
            model_name=model_name,
            match_min_score=match_min_score,
            match_min_margin=match_min_margin,
            unmatched_score=unmatched_score,
            mask_profile_id=profile.id if profile else None,
            feature_version=(
                feature_version
                if is_psd else f"{feature_version}:mask_v{profile.version}"
            ),
            image_count=len(category_files),
            product_count=len(mains),
        )
        db.add(run)
        db.flush()

        keep = min(5, len(mains))
        for row, item in enumerate(category_files):
            item.match_method = (
                "DINO_COSINE_PSD" if is_psd else "DINO_MASKED_COSINE"
            )
            order = np.argsort(-category_matrix[row])[:keep]
            baseline_order = np.argsort(-baseline_matrix[row])[:2]
            item.top1_score = float(category_matrix[row, int(order[0])])
            item.top2_score = (
                float(category_matrix[row, int(order[1])]) if len(order) > 1 else None
            )
            item.top3_score = (
                float(category_matrix[row, int(order[2])]) if len(order) > 2 else None
            )
            item.top1_gap = (
                item.top1_score - item.top2_score
                if item.top2_score is not None else None
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
                "embedding_model": model_name,
                "feature_version": feature_version,
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


def match_batch(
    db: Session,
    batch_id: int,
    embedder: ImageEmbedder | None = None,
    *,
    match_min_score: float | None = None,
    match_min_margin: float | None = None,
    unmatched_score: float | None = None,
) -> dict:
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
    threshold_score = (
        settings.match_min_score if match_min_score is None else float(match_min_score)
    )
    threshold_margin = (
        settings.match_min_margin
        if match_min_margin is None else float(match_min_margin)
    )
    threshold_unmatched = (
        settings.unmatched_score if unmatched_score is None else float(unmatched_score)
    )
    if not all(
        0.0 <= value <= 1.0
        for value in (threshold_score, threshold_margin, threshold_unmatched)
    ):
        raise ValueError("置信度参数必须位于 0 到 1 之间")
    if threshold_unmatched > threshold_score:
        raise ValueError("自动归属最低分不能高于低置信度分数")
    profile_rows = require_manual_profiles(db, batch.id)
    profiles = {profile.image_category: profile for profile in profile_rows}
    batch.status = "MATCHING"
    batch.error_message = None
    db.commit()
    try:
        _set_progress(
            db, batch.id, "LOADING_MODEL", 0, len(mains) + len(files), 2,
            "正在下载或加载 DINO 模型，首次运行耗时较长",
        )
        extractor = embedder or DinoFeatureProvider()
        model_name = _model_name(extractor)
        feature_version = _feature_version(extractor)
        device = getattr(extractor, "device", "TEST/CPU")
        device_label = "CUDA 显卡" if str(device).startswith("cuda") else "CPU"
        load_warning = getattr(extractor, "load_warning", None)
        progress_device = f"{device_label} · {model_name}"
        _set_progress(
            db, batch.id, "MAIN_EMBEDDING", 0, len(mains), 5,
            load_warning or f"正在生成主图 DINO 特征（{device_label}）",
            progress_device,
        )
        main_vectors = _main_embeddings(
            batch, mains, extractor,
            on_progress=lambda done: _set_progress(
                db, batch.id, "MAIN_EMBEDDING", done, len(mains),
                5 + 20 * done / max(len(mains), 1),
                f"主图 DINO 特征 {done}/{len(mains)}", progress_device,
            ),
        )
        _record_full_main_features(db, mains, extractor)
        db.execute(delete(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_([item.id for item in files])
        ))
        positions: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(files):
            positions[item.category].append(index)
        psd_previews: dict[int, Path] = {}
        if "PSD" in positions:
            psd_files = [files[index] for index in positions["PSD"]]
            previews = prepare_psd_previews_and_phash(
                db, batch, psd_files, mains
            )
            psd_previews = {
                item.id: Path(asset.asset_path)
                for item, asset in zip(psd_files, previews)
            }
        query_paths = [
            psd_previews.get(item.id, Path(item.current_path))
            for item in files
        ]
        _set_progress(
            db, batch.id, "FULL_EMBEDDING", 0, len(files), 25,
            f"正在生成整图诊断与 PSD Preview 的 DINO 特征（{device_label}）",
            progress_device,
        )
        full_query_vectors = _file_embeddings(
            batch, files, query_paths, extractor,
            on_progress=lambda done: _set_progress(
                db, batch.id, "FULL_EMBEDDING", done, len(files),
                25 + 25 * done / max(len(files), 1),
                f"整图 DINO 特征 {done}/{len(files)}", progress_device,
            ),
        )
        builder = SimilarityMatrixBuilder()
        full_similarities = builder.build(full_query_vectors, main_vectors)
        baseline_matrices = {
            category: full_similarities[rows, :]
            for category, rows in positions.items()
        }
        category_matrices: dict[str, np.ndarray] = {}
        if "PSD" in positions:
            category_matrices["PSD"] = baseline_matrices["PSD"]

        processor = MaskedCropProcessor()
        visual_categories = [category for category in positions if category != "PSD"]
        for category_index, category in enumerate(visual_categories, 1):
            profile = profiles[category]
            category_files = [files[index] for index in positions[category]]
            start_percent = 50 + 38 * (category_index - 1) / max(
                len(visual_categories), 1
            )
            _set_progress(
                db,
                batch.id,
                "MASKED_DINO",
                category_index - 1,
                len(visual_categories),
                start_percent,
                f"正在裁剪并识别 {category}（Mask v{profile.version}）",
                progress_device,
            )
            file_assets = [
                processor.process_file(db, batch, item, profile).crop
                for item in category_files
            ]
            main_assets = [
                processor.process_main(db, batch, main, profile).crop
                for main in mains
            ]
            db.flush()
            query_vectors = _derived_embeddings(batch, extractor, file_assets)
            masked_main_vectors = _derived_embeddings(batch, extractor, main_assets)
            _record_masked_main_features(
                db, batch, mains, main_assets, profile, extractor
            )
            category_matrices[category] = builder.build(
                query_vectors, masked_main_vectors
            )
            db.commit()
        _set_progress(
            db, batch.id, "SIMILARITY", len(files), len(files), 90,
            "正在按分类执行遮罩 DINO 全局一对一匹配", progress_device,
        )
        totals = _assign_categories(
            db,
            batch,
            mains,
            files,
            category_matrices,
            baseline_matrices,
            profiles,
            model_name,
            feature_version,
            threshold_score,
            threshold_margin,
            threshold_unmatched,
        )
        db.commit()
        _set_progress(
            db, batch.id, "AUTO_RENAME", len(files), len(files), 94,
            "正在保存 Top5 并执行无覆盖安全改名", progress_device,
        )
        rename_plan = RenamePlanService().generate(db, batch.id)
        rename_plan = RenamePlanService().execute_ready(
            db, batch.id, rename_plan["plan_group"]
        )
        conflict_count = sum(
            1 for item in files if item.is_conflict or item.status == "CONFLICT"
        )
        batch.status = "COMPLETED"
        batch.finished_at = datetime.now(timezone.utc)
        _set_progress(
            db, batch.id, "COMPLETED", len(files), len(files), 100,
            "遮罩 DINO 匹配与安全自动命名完成，等待网页聚合审核",
            progress_device,
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
        "conflicts": conflict_count,
        "rename_plan": {
            "plan_group": rename_plan["plan_group"],
            "counts": rename_plan["counts"],
        },
    }
