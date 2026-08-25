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

from .assignment import Assignment, GlobalAssignmentSolver, MatchConfidenceEvaluator
from .cropper import MaskedCropProcessor
from .features import (
    ImageFeatureProvider,
    OpenClipFeatureProvider,
    SimilarityMatrixBuilder,
)
from .masks import ensure_default_profiles
from .media import embedding_cache_path
from .psd import build_psd_matrix
from .rename_plan import RenamePlanService


ImageEmbedder = ImageFeatureProvider
OpenClipEmbedder = OpenClipFeatureProvider


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
    for start in range(0, len(paths), settings.match_batch_size):
        chunk = paths[start:start + settings.match_batch_size]
        chunks.append(embedder.encode(chunk))
        if on_progress:
            on_progress(min(start + len(chunk), len(paths)))
    vectors = np.concatenate(chunks, axis=0).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _main_embeddings(batch: Batch, mains: list[MainImage], embedder: ImageEmbedder,
                     on_progress=None) -> np.ndarray:
    cached = all(item.embedding_path and Path(item.embedding_path).exists() for item in mains)
    if cached:
        vectors = np.stack([np.load(item.embedding_path) for item in mains]).astype(np.float32)
        if on_progress:
            on_progress(len(mains))
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

    paths = [Path(item.file_path) for item in mains]
    vectors = _encode_in_batches(embedder, paths, on_progress=on_progress)
    root = Path(batch.root_path)
    for item, vector, source in zip(mains, vectors, paths):
        cache = embedding_cache_path(root, batch.id, source)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vector)
        item.embedding_path = str(cache)
    return vectors


def _derived_embeddings(
    embedder: ImageEmbedder,
    assets: list[DerivedImageAsset],
    on_progress=None,
) -> np.ndarray:
    cache_paths = [Path(asset.asset_path).with_suffix(".openclip.npy") for asset in assets]
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
    mains: list[MainImage],
    assets: list[DerivedImageAsset],
    profile: MaskProfile,
) -> None:
    version = f"openclip_masked_v1:mask_v{profile.version}"
    existing = set(db.scalars(select(MainImageFeature.main_image_id).where(
        MainImageFeature.feature_type == "MASKED_EMBEDDING",
        MainImageFeature.feature_version == version,
        MainImageFeature.mask_profile_id == profile.id,
    )).all())
    for main, asset in zip(mains, assets):
        if main.id not in existing:
            db.add(MainImageFeature(
                main_image_id=main.id,
                feature_type="MASKED_EMBEDDING",
                feature_version=version,
                model_name=f"{settings.clip_model}:{settings.clip_pretrained}",
                mask_profile_id=profile.id,
                feature_path=str(Path(asset.asset_path).with_suffix(".openclip.npy")),
                crop_asset_id=asset.id,
            ))


def _record_full_main_features(
    db: Session,
    mains: list[MainImage],
) -> None:
    existing = set(db.scalars(select(MainImageFeature.main_image_id).where(
        MainImageFeature.feature_type == "FULL_EMBEDDING",
        MainImageFeature.feature_version == "openclip_full_v1",
    )).all())
    for main in mains:
        if main.id not in existing:
            db.add(MainImageFeature(
                main_image_id=main.id,
                feature_type="FULL_EMBEDDING",
                feature_version="openclip_full_v1",
                model_name=f"{settings.clip_model}:{settings.clip_pretrained}",
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
        run = MatchingRun(
            batch_id=batch.id,
            category=category,
            algorithm_version=(
                settings.psd_algorithm_version
                if is_psd else settings.masked_algorithm_version
            ),
            model_name=f"{settings.clip_model}:{settings.clip_pretrained}",
            match_min_score=settings.match_min_score,
            match_min_margin=settings.match_min_margin,
            unmatched_score=settings.unmatched_score,
            mask_profile_id=profile.id if profile else None,
            feature_version=(
                "openclip_full_v1"
                if is_psd else f"openclip_masked_v1:mask_v{profile.version}"
            ),
            image_count=len(category_files),
            product_count=len(mains),
        )
        db.add(run)
        db.flush()

        keep = min(5, len(mains))
        for row, item in enumerate(category_files):
            if not is_psd:
                item.match_method = "MASKED_OPENCLIP"
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
        artifact: list[dict] = []
        for assignment in assignments:
            category_row = assignment.row
            assigned_column = assignment.column
            item = category_files[category_row]
            result = evaluator.evaluate(
                category_matrix,
                Assignment(row=category_row, column=assigned_column),
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
                item.is_low_confidence = result.status == "LOW_CONFIDENCE"
                item.status = result.status
                item.review_status = "PENDING"
                if assigned_column is None:
                    item.matched_main_id = None
                    item.matched_main_basename = None
                else:
                    winner = mains[assigned_column]
                    item.matched_main_id = winner.id
                    item.matched_main_basename = winner.basename
                totals[result.status] += 1
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


def match_batch(db: Session, batch_id: int, embedder: ImageEmbedder | None = None) -> dict:
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

    batch.status = "MATCHING"
    batch.error_message = None
    db.commit()
    try:
        _set_progress(
            db, batch.id, "LOADING_MODEL", 0, len(mains) + len(files), 2,
            "正在下载或加载 OpenCLIP 模型，首次运行耗时较长",
        )
        extractor = embedder or OpenClipFeatureProvider()
        device = getattr(extractor, "device", "TEST/CPU")
        device_label = "CUDA 显卡" if str(device).startswith("cuda") else "CPU"
        _set_progress(
            db, batch.id, "MAIN_EMBEDDING", 0, len(mains), 5,
            f"正在生成主图视觉特征（{device_label}）", device_label,
        )
        main_vectors = _main_embeddings(
            batch, mains, extractor,
            on_progress=lambda done: _set_progress(
                db, batch.id, "MAIN_EMBEDDING", done, len(mains),
                5 + 20 * done / max(len(mains), 1),
                f"主图视觉特征 {done}/{len(mains)}", device_label,
            ),
        )
        _record_full_main_features(db, mains)
        db.execute(delete(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_([item.id for item in files])
        ))
        query_paths = [Path(item.current_path) for item in files]
        _set_progress(
            db, batch.id, "FULL_EMBEDDING", 0, len(files), 25,
            f"正在生成整图 baseline 与 PSD 视觉特征（{device_label}）", device_label,
        )
        full_query_vectors = _encode_in_batches(
            extractor, query_paths,
            on_progress=lambda done: _set_progress(
                db, batch.id, "FULL_EMBEDDING", done, len(files),
                25 + 30 * done / max(len(files), 1),
                f"整图视觉特征 {done}/{len(files)}", device_label,
            ),
        )
        builder = SimilarityMatrixBuilder()
        full_similarities = builder.build(full_query_vectors, main_vectors)
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
                baseline_matrices["PSD"],
            )

        profile_rows = ensure_default_profiles(db, batch.id)
        profiles = {profile.image_category: profile for profile in profile_rows}
        processor = MaskedCropProcessor()
        visual_categories = [category for category in positions if category != "PSD"]
        for category_index, category in enumerate(visual_categories, 1):
            profile = profiles[category]
            category_files = [files[index] for index in positions[category]]
            _set_progress(
                db, batch.id, "MASKED_CROP", category_index - 1,
                len(visual_categories), 55 + 35 * (category_index - 1) / max(len(visual_categories), 1),
                f"正在生成 {category} Masked Crop（Mask v{profile.version}）",
                device_label,
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
            query_vectors = _derived_embeddings(extractor, file_assets)
            masked_main_vectors = _derived_embeddings(extractor, main_assets)
            _record_masked_main_features(db, mains, main_assets, profile)
            category_matrices[category] = builder.build(
                query_vectors, masked_main_vectors
            )
            db.commit()
        _set_progress(
            db, batch.id, "SIMILARITY", len(files), len(files), 92,
            "正在按分类执行 Masked 全局一对一匹配", device_label,
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
    }
