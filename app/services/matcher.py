from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Batch,
    BatchProgress,
    MainImage,
    MatchCandidate,
    MatchedFile,
    MatchingRun,
)

from .assignment import Assignment, GlobalAssignmentSolver, MatchConfidenceEvaluator
from .media import embedding_cache_path, open_visual


class ImageEmbedder(Protocol):
    def encode(self, paths: Sequence[Path]) -> np.ndarray: ...


class OpenClipEmbedder:
    """Lazy OpenCLIP loader so scanning and review work without loading PyTorch."""

    def __init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "未安装 OpenCLIP。请执行: pip install -e .[clip]"
            ) from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained = settings.clip_pretrained
        transform_options = {}
        if pretrained == "openai":
            # open_clip normally prefers the Hugging Face mirror when both an
            # hf_hub id and the original OpenAI URL exist. Anonymous HF access
            # is often heavily throttled in this deployment region, so force
            # the official OpenAI URL and keep its preprocessing metadata.
            pretrained_cfg = dict(open_clip.get_pretrained_cfg(settings.clip_model, pretrained))
            checkpoint = open_clip.download_pretrained(
                pretrained_cfg,
                prefer_hf_hub=False,
            )
            pretrained = checkpoint
            transform_options = {
                "force_quick_gelu": bool(pretrained_cfg.get("quick_gelu", False)),
                "image_mean": pretrained_cfg.get("mean"),
                "image_std": pretrained_cfg.get("std"),
                "image_interpolation": pretrained_cfg.get("interpolation"),
                "image_resize_mode": pretrained_cfg.get("resize_mode"),
                # The checksum-verified OpenAI .pt is a TorchScript archive.
                # PyTorch 2.6+ requires opting out of weights_only for this
                # trusted legacy checkpoint format.
                "weights_only": False,
            }
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=pretrained,
            device=self.device,
            **transform_options,
        )
        self.model.eval()

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        images = [self.preprocess(open_visual(path)) for path in paths]
        tensor = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.float().cpu().numpy()


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
    similarities: np.ndarray,
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
        category_matrix = similarities[global_rows, :]
        run = MatchingRun(
            batch_id=batch.id,
            category=category,
            algorithm_version=settings.algorithm_version,
            model_name=f"{settings.clip_model}:{settings.clip_pretrained}",
            match_min_score=settings.match_min_score,
            match_min_margin=settings.match_min_margin,
            unmatched_score=settings.unmatched_score,
            image_count=len(category_files),
            product_count=len(mains),
        )
        db.add(run)
        db.flush()

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
        extractor = embedder or OpenClipEmbedder()
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
        db.execute(delete(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_([item.id for item in files])
        ))
        query_paths = [Path(item.current_path) for item in files]
        _set_progress(
            db, batch.id, "FILE_EMBEDDING", 0, len(files), 25,
            f"正在生成附图和 PSD 视觉特征（{device_label}）", device_label,
        )
        query_vectors = _encode_in_batches(
            extractor, query_paths,
            on_progress=lambda done: _set_progress(
                db, batch.id, "FILE_EMBEDDING", done, len(files),
                25 + 65 * done / max(len(files), 1),
                f"附图和 PSD 视觉特征 {done}/{len(files)}", device_label,
            ),
        )
        _set_progress(
            db, batch.id, "SIMILARITY", len(files), len(files), 92,
            "正在按分类执行全局一对一匹配", device_label,
        )
        similarities = query_vectors @ main_vectors.T
        keep = min(5, len(mains))

        for item, scores in zip(files, similarities):
            order = np.argsort(-scores)[:keep]
            candidates = [(mains[int(index)], float(scores[int(index)])) for index in order]
            item.top1_score = candidates[0][1]
            item.top2_score = candidates[1][1] if len(candidates) > 1 else None
            item.top3_score = candidates[2][1] if len(candidates) > 2 else None
            for rank, (main, score) in enumerate(candidates, 1):
                db.add(MatchCandidate(
                    matched_file_id=item.id,
                    main_image_id=main.id,
                    rank=rank,
                    similarity=score,
                ))
        totals = _assign_categories(db, batch, mains, files, similarities)
        db.commit()
        _set_progress(
            db, batch.id, "SAVING", len(files), len(files), 96,
            "正在保存匹配诊断和 Top 5 候选（本轮不改动源文件名）", device_label,
        )
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
    }
