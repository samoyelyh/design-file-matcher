from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, selectinload

from app.config import is_excluded_file_category, is_ignored_image_category, settings
from app.db import SessionLocal, get_db, init_db
from app.models import (
    Batch,
    BatchProgress,
    DerivedImageAsset,
    MainImage,
    MaskProfile,
    MatchCandidate,
    MatchedFile,
    ProductReference,
    ProductSimilarity,
)
from app.services.color_features import build_color_matrix, color_top_n_recall
from app.services.dinov3 import DINO_MODELS
from app.services.match_engines import get_match_engine
from app.services.match_engines.visionmatch_engine import VisionMatchEngine
from app.services.main_body import (
    ensure_batch_main_bodies,
    main_body_dict,
)
from app.services.media import ensure_preview, ensure_thumbnail, preview_cache_path
from app.services.masks import (
    MAIN_IMAGE_CATEGORY,
    active_manual_profile,
    mask_readiness,
    profile_dict,
    profile_history,
    profile_storage_valid,
    save_main_profile_version,
    save_profile_version,
)
from app.services.renamer import restore_original
from app.services.rename_plan import RenamePlanService
from app.services.review_scope import conflict_name_marks, ignored_review_categories
from app.services.phash_matching import masked_query_hashes
from app.services.references import (
    build_product_score_matrix,
    learn_confirmed_references,
)
from app.services.scanner import scan_batch
from app.services import qwen_lab
from app.services import visionmatch


executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="design-matcher")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="设计文件自动匹配与审核系统",
    version="0.1.0",
    lifespan=lifespan,
)

# Serialize review writes before creating their DB session. Shared-drive crop
# generation may take time; overlapping confirmations must not compete for the
# SQLite writer lock or generate a naming plan from partially saved review.
_review_write_lock = Lock()


def _review_write_guard():
    if not _review_write_lock.acquire(timeout=30):
        raise HTTPException(409, "其他审核操作正在保存，请稍后刷新核对状态再重试")
    try:
        yield
    finally:
        _review_write_lock.release()


@app.exception_handler(OperationalError)
async def database_operation_error(_request, exc: OperationalError):
    if isinstance(exc.orig, sqlite3.OperationalError) and any(
        message in str(exc.orig).lower() for message in ("locked", "busy")
    ):
        return JSONResponse(status_code=409, content={
            "detail": "数据库正在保存其他操作，请稍后刷新核对状态再重试；无需重新匹配",
        })
    raise exc


def _stored_layout(db: Session, batch: Batch) -> dict | None:
    """Build layout from indexed database paths without touching the shared drive."""
    main_path = db.scalar(select(MainImage.file_path).where(
        MainImage.batch_id == batch.id
    ).limit(1))
    if not main_path:
        return None
    root = Path(batch.root_path)
    main_dir = Path(main_path).parent
    rows = db.execute(select(
        MatchedFile.category,
        MatchedFile.original_path,
    ).where(MatchedFile.batch_id == batch.id)).all()
    category_dirs: list[Path] = []
    seen: set[Path] = set()
    psd_dirs: list[Path] = []
    for category, source_path in rows:
        directory = Path(source_path).parent
        if category == "PSD":
            if directory not in psd_dirs:
                psd_dirs.append(directory)
            continue
        if is_ignored_image_category(category):
            continue
        if directory not in seen:
            category_dirs.append(directory)
            seen.add(directory)
    return {
        "effect_root": str(main_dir.parent.relative_to(root)),
        "main_directory": str(main_dir.relative_to(root)),
        "psd_directories": [str(directory.relative_to(root)) for directory in psd_dirs],
        "category_directories": [
            str(directory.relative_to(root)) for directory in category_dirs
        ],
    }


def _required_upload_root(root_path: str, allow_root: bool = False) -> Path:
    upload_root = Path(settings.upload_root).expanduser().resolve()
    if not upload_root.is_dir():
        raise HTTPException(503, f"公共盘当前不可用: {settings.upload_root}")
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(400, f"目录不存在: {root}")
    try:
        relative = root.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(
            400,
            f"请先将文件包上传到公共盘目录: {settings.upload_root}",
        ) from exc
    if not allow_root and not relative.parts:
        raise HTTPException(400, "请选择公共盘内具体的文件包子目录")
    return root


class BatchCreate(BaseModel):
    root_path: str = Field(min_length=1)
    product_category: str = Field(default="橄榄球衣", min_length=1, max_length=128)
    match_engine: str = Field(default="VISIONMATCH", max_length=32)


class ReassignRequest(BaseModel):
    main_image_id: int
    operator: str = "LOCAL"


class ManualActionRequest(BaseModel):
    operator: str = "LOCAL"


class BulkConfirmRequest(BaseModel):
    main_image_ids: list[int] = Field(min_length=1, max_length=1000)
    operator: str = "LOCAL"


class ReviewCategoryRequest(BaseModel):
    category: str = Field(min_length=1, max_length=128)
    ignored: bool


class ConflictNameMarksRequest(BaseModel):
    file_ids: list[int] = Field(default_factory=list, max_length=1000)


class MaskProfileSaveRequest(BaseModel):
    mask_data_url: str = Field(min_length=32)
    reference_file_id: int | None = None
    reference_main_id: int | None = None
    crop_box: list[float] = Field(min_length=4, max_length=4)
    name: str | None = Field(default=None, max_length=255)


class RenamePlanExecuteRequest(BaseModel):
    plan_group: str = Field(min_length=1, max_length=36)


class MatchStartRequest(BaseModel):
    match_engine: str | None = Field(default=None, max_length=32)
    matching_model: str | None = Field(default=None, max_length=32)
    folder_engines: dict[str, dict] | None = None
    visionmatch_confirmed_budget: bool = False
    visionmatch_max_calls: int = Field(default=0, ge=0, le=200)


def _require_visionmatch(engine: str | None) -> None:
    if engine is not None and engine.strip().upper() != "VISIONMATCH":
        raise ValueError("匹配只支持 VisionMatch AI · Terra")


def _configure_batch_engine(batch: Batch, requested: str | None) -> None:
    _require_visionmatch(requested)
    batch.match_engine = batch.matching_model = "VISIONMATCH"
    batch.match_engine_version = visionmatch.VERSION
    batch.match_config_snapshot = json.dumps(VisionMatchEngine().config_snapshot(), ensure_ascii=False)


def _configure_folder_engines(
    batch: Batch,
    db: Session,
    requested: dict[str, dict] | None = None,
) -> dict:
    categories = [
        category for category in db.scalars(
            select(MatchedFile.category).where(MatchedFile.batch_id == batch.id)
            .distinct().order_by(MatchedFile.category)
        )
        if not is_excluded_file_category(category)
    ]
    if not categories:
        raise ValueError("请先扫描待匹配的图片文件夹")
    for config in (requested or {}).values():
        _require_visionmatch(config.get("engine"))
        _require_visionmatch(config.get("match_engine"))

    existing = json.loads(batch.match_config_snapshot or "{}")
    snapshot = {
        key: existing[key] for key in ("review_ignored_categories", "conflict_name_marks", "conflict_mark_originals")
        if key in existing
    }
    snapshot.update(
        version="folder_sharded_v1",
        mode="FOLDER_SHARDED",
        folder_engines={category: {"engine": "VISIONMATCH", "model": None, "independent": True}
                        for category in categories},
        all_mains=True,
        duplicate_within_folder=False,
    )
    batch.match_engine = batch.matching_model = "FOLDER_SHARDED"
    batch.match_engine_version = snapshot["version"]
    batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
    return snapshot


def _batch_dict(db: Session, batch: Batch) -> dict:
    main_count = db.scalar(select(func.count()).select_from(MainImage).where(
        MainImage.batch_id == batch.id
    )) or 0
    raw_category_rows = db.execute(select(MatchedFile.category, func.count()).where(
        MatchedFile.batch_id == batch.id
    ).group_by(MatchedFile.category)).all()
    category_rows = [
        row for row in raw_category_rows
        if not is_excluded_file_category(row[0])
    ]
    progress = db.get(BatchProgress, batch.id)
    main_body_ready_count = db.scalar(select(func.count()).select_from(MainImage).where(
        MainImage.batch_id == batch.id,
        MainImage.body_feature_version == settings.body_phash_version,
    )) or 0
    association_counts = {"exact_filename": 0, "visual": 0, "pending": 0, "conflict": 0}
    for method, count in db.execute(select(MatchedFile.match_method, func.count()).where(
        MatchedFile.batch_id == batch.id,
    ).group_by(MatchedFile.match_method)):
        key = "exact_filename" if method == "EXACT_FILENAME" else "conflict" if method == "FILENAME_MATCH_AMBIGUOUS" else "visual" if method else "pending"
        association_counts[key] += count
    try:
        snapshot = json.loads(batch.match_config_snapshot or "{}")
    except (TypeError, ValueError):
        snapshot = {}
    folder_sharded = batch.match_engine == "FOLDER_SHARDED" or snapshot.get("mode") == "FOLDER_SHARDED"
    if batch.match_engine == "QWEN" or folder_sharded:
        engine = None
        current_snapshot = (
            {"version": "folder_sharded_v1", "mode": "FOLDER_SHARDED", "folder_engines": snapshot.get("folder_engines", {})}
            if folder_sharded else
            {"version": qwen_lab.PROMPT_VERSION, "model": batch.matching_model, "concurrency": 1}
        )
    else:
        engine = get_match_engine(batch)
        current_snapshot = engine.config_snapshot()
    effective_snapshot = snapshot or current_snapshot
    available_engines = matching_models()
    return {
        "id": batch.id,
        "root_path": batch.root_path,
        "product_category": batch.product_category,
        "matching_model": batch.matching_model,
        "association_counts": association_counts,
        "match_engine": "FOLDER_SHARDED" if folder_sharded else ("QWEN" if engine is None else engine.name),
        "match_engine_version": batch.match_engine_version or current_snapshot.get("version"),
        "match_config_snapshot": effective_snapshot,
        "status": batch.status,
        "error_message": batch.error_message,
        "created_at": batch.created_at,
        "finished_at": batch.finished_at,
        "main_count": main_count,
        "main_body_readiness": {
            "ready": main_body_ready_count == main_count and main_count > 0,
            "completed": main_body_ready_count,
            "total": main_count,
        },
        "categories": dict(category_rows),
        "mask_readiness": mask_readiness(db, batch.id),
        "layout": _stored_layout(db, batch),
        "progress": {
            "stage": progress.stage,
            "current": progress.current,
            "total": progress.total,
            "percent": progress.percent,
            "message": progress.message,
            "device": progress.device,
            "updated_at": progress.updated_at,
        } if progress else {
            "stage": "READY", "current": 0, "total": 0, "percent": 0,
            "message": "等待开始", "device": None, "updated_at": None,
        },
        "low_confidence_threshold": settings.low_confidence_threshold,
        "matching_config": {
            "requires_masks": False,
            "algorithm_version": effective_snapshot.get("version"),
            "psd_algorithm_version": settings.psd_algorithm_version,
            "baseline_algorithm_version": settings.algorithm_version,
            "match_min_score": effective_snapshot.get("min_score", settings.match_min_score),
            "match_min_margin": effective_snapshot.get("min_margin", settings.match_min_margin),
            "unmatched_score": effective_snapshot.get("unmatched_score", settings.unmatched_score),
            "body_crop_padding": settings.body_crop_padding,
            "body_phash_version": settings.body_phash_version,
            "product_reference_scoring_enabled": settings.use_product_reference_scoring,
            "max_references_per_category": settings.max_references_per_category,
            "reference_weights": [
                settings.reference_weight_1,
                settings.reference_weight_2,
                settings.reference_weight_3,
            ],
            "detail_region_algorithm_version": settings.detail_region_algorithm_version,
            "available_models": available_engines,
            "available_engines": available_engines,
            "folder_sharded": folder_sharded,
            "folder_engines": snapshot.get("folder_engines", {}) if folder_sharded else {},
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/workspace-config")
def workspace_config():
    engines = matching_models()
    return {
        "upload_root": settings.upload_root,
        "matching_models": engines,
        "matching_engines": engines,
        "default_match_engine": "VISIONMATCH",
    }


@app.get("/api/matching-models")
def matching_models():
    return [visionmatch.status()]


@app.get("/api/batches/{batch_id}/color-recall")
def evaluate_color_recall(
    batch_id: int,
    top_n: int = Query(default=50, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Phase-3 diagnostic: test Color prefilter recall without changing matches."""
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch_id
    ).order_by(MainImage.id)))
    main_profile = active_manual_profile(db, batch_id, MAIN_IMAGE_CATEGORY)
    if not mains or not main_profile:
        raise HTTPException(409, "请先扫描主图并保存主图手动识别区域")
    rows = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category != "PSD",
        MatchedFile.matched_main_id.is_not(None),
    ).order_by(MatchedFile.category, MatchedFile.id)))
    ground_truth = [item for item in rows if (
        item.confirmed_at
        or item.status == "MANUAL_CONFIRMED"
        or item.review_status in {"CONFIRMED", "CORRECTED"}
    )]
    main_column = {main.id: index for index, main in enumerate(mains)}
    categories: list[dict] = []
    all_hits = all_eligible = 0
    for category in sorted({item.category for item in ground_truth}):
        files = [item for item in ground_truth if item.category == category]
        profile = active_manual_profile(db, batch_id, category)
        if not profile:
            continue
        files = [item for item in files if item.matched_main_id in main_column]
        if not files:
            continue
        matrix, cache = build_color_matrix(batch, files, mains, profile, main_profile)
        metrics = color_top_n_recall(
            matrix,
            [main_column[item.matched_main_id] for item in files],
            top_n,
        )
        categories.append({"category": category, **metrics, "cache": cache})
        all_hits += metrics["hits"]
        all_eligible += metrics["eligible"]
    return {
        "batch_id": batch_id,
        "feature_version": settings.color_feature_version,
        "top_n": min(top_n, len(mains)),
        "eligible": all_eligible,
        "hits": all_hits,
        "recall": all_hits / all_eligible if all_eligible else None,
        "categories": categories,
        "read_only": True,
    }


@app.get("/api/upload-folders")
def upload_folders(path: str | None = None):
    upload_root = Path(settings.upload_root).expanduser().resolve()
    current = _required_upload_root(path or settings.upload_root, allow_root=True)
    try:
        folders = sorted(
            (
                item for item in current.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ),
            key=lambda item: item.name.casefold(),
        )
    except OSError as exc:
        raise HTTPException(503, f"无法读取公共盘目录: {exc}") from exc
    is_root = current == upload_root
    return {
        "current_path": str(current),
        "is_root": is_root,
        "parent_path": None if is_root else str(current.parent),
        "folders": [
            {"name": folder.name, "path": str(folder)} for folder in folders
        ],
    }


@app.get("/api/batches")
def list_batches(db: Session = Depends(get_db)):
    batches = db.scalars(select(Batch).order_by(Batch.id.desc()).limit(30)).all()
    return [_batch_dict(db, batch) for batch in batches]


@app.post("/api/batches")
def create_batch(payload: BatchCreate, db: Session = Depends(get_db)):
    root = _required_upload_root(payload.root_path)
    batch = Batch(
        root_path=str(root),
        product_category=payload.product_category.strip(),
        status="CREATED",
    )
    try:
        _configure_batch_engine(batch, payload.match_engine)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    db.add(batch)
    db.commit()
    return _batch_dict(db, batch)


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: int, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    return _batch_dict(db, batch)


@app.post("/api/batches/{batch_id}/scan")
def scan(batch_id: int, db: Session = Depends(get_db)):
    try:
        return scan_batch(db, batch_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _run_match(batch_id: int) -> None:
    with SessionLocal() as db:
        try:
            _run_folder_sharded_match(db, batch_id)
        except Exception:
            # The matching worker records the failure on the batch.
            return


def _run_folder_sharded_match(db: Session, batch_id: int) -> None:
    """Run each scanned副图 folder with its own engine and independent rows."""
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    try:
        visionmatch.run_batch(db, batch_id, finalize=False)
        RenamePlanService().generate(db, batch_id)
        batch = db.get(Batch, batch_id)
        batch.status = "COMPLETED"
        batch.finished_at = datetime.now(timezone.utc)
        progress = db.get(BatchProgress, batch_id) or BatchProgress(batch_id=batch_id)
        progress.stage = "COMPLETED"
        progress.current = progress.total = sum(
            1 for item in db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch_id)).all()
            if not is_excluded_file_category(item.category)
        )
        progress.percent = 100
        progress.message = "各副图文件夹已独立匹配完成，等待人工审核"
        progress.device = "按文件夹独立执行"
        db.add(progress)
        db.commit()
    except Exception as exc:
        db.rollback()
        batch = db.get(Batch, batch_id)
        batch.status = "FAILED"
        batch.error_message = str(exc)
        db.commit()
        raise


def _validate_match_request(payload: MatchStartRequest) -> None:
    _require_visionmatch(payload.match_engine)
    _require_visionmatch(payload.matching_model)
    for config in (payload.folder_engines or {}).values():
        _require_visionmatch(config.get("engine"))
        _require_visionmatch(config.get("match_engine"))


@app.post("/api/batches/{batch_id}/visionmatch-plan")
def visionmatch_plan(batch_id: int, payload: MatchStartRequest, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    try:
        _validate_match_request(payload)
        return visionmatch.plan(db, batch_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/batches/{batch_id}/visionmatch-control")
def visionmatch_control(batch_id: int, payload: dict, db: Session = Depends(get_db)):
    from app.services.visionmatch import control
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    try:
        return control(batch_id, payload.get("action"))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/batches/{batch_id}/match", dependencies=[Depends(_review_write_guard)])
def start_match(
    batch_id: int,
    payload: MatchStartRequest | None = None,
    db: Session = Depends(get_db),
):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    if batch.status not in {"SCANNED", "COMPLETED", "FAILED"}:
        raise HTTPException(409, f"当前状态 {batch.status}，请先完成扫描")
    payload = payload or MatchStartRequest()
    try:
        _validate_match_request(payload)
        planned = visionmatch.plan(db, batch_id)
        calls = planned["calls_upper_bound"]
        if calls:
            if not payload.visionmatch_confirmed_budget or payload.visionmatch_max_calls < calls:
                raise ValueError(f"请确认 VisionMatch 调用预算：本次最多 {calls} 次")
            engine_status = visionmatch.status()
            if not engine_status["selectable"]:
                raise ValueError(engine_status["error"] or "VisionMatch AI · Terra 未就绪")
        snapshot = _configure_folder_engines(batch, db)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    snapshot["visionmatch"] = {"max_calls": calls, "confirmed": True, "planned_calls": calls}
    batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
    batch.status = "QUEUED"
    progress = db.get(BatchProgress, batch.id)
    if not progress:
        progress = BatchProgress(batch_id=batch.id)
        db.add(progress)
    progress.stage = "QUEUED"
    progress.current = 0
    progress.total = 0
    progress.percent = 0
    progress.message = "VisionMatch AI · Terra 任务已进入本地处理队列"
    progress.device = None
    db.commit()
    executor.submit(_run_match, batch_id)
    return {
        "batch_id": batch_id,
        "status": "QUEUED",
        "matching_model": batch.matching_model,
        "match_engine": batch.match_engine,
        "folder_engines": snapshot["folder_engines"],
    }


def _file_dict(item: MatchedFile, derived: dict[str, DerivedImageAsset] | None = None) -> dict:
    top1 = item.candidates[0] if item.candidates else None
    top2 = item.candidates[1] if len(item.candidates) > 1 else None
    derived = derived or {}
    return {
        "id": item.id,
        "category": item.category,
        "folder_name": Path(item.original_path).parent.name,
        "original_filename": item.original_filename,
        "current_filename": item.current_filename,
        "matched_main_id": item.matched_main_id,
        "matched_main_image_id": item.matched_main_id,
        "source_filename": item.original_filename,
        "normalized_filename": Path(item.original_filename).stem,
        "matched_main_image_path": item.matched_main.file_path if item.matched_main else None,
        "matched_main_image_filename": item.matched_main.filename if item.matched_main else None,
        "match_status": item.status,
        "matched_main_basename": item.matched_main_basename,
        "similarity": item.algorithm_score,
        "assigned_score": item.algorithm_score,
        "original_top1": {
            "main_image_id": top1.main_image_id,
            "basename": top1.main_image.basename,
            "score": top1.similarity,
        } if top1 else None,
        "original_top2": {
            "main_image_id": top2.main_image_id,
            "basename": top2.main_image.basename,
            "score": top2.similarity,
        } if top2 else None,
        "top1_top2_margin": (
            top1.similarity - top2.similarity if top1 and top2 else None
        ),
        "assigned_rank": item.assigned_rank,
        "row_margin": item.row_margin,
        "column_margin": item.column_margin,
        "algorithm_status": item.algorithm_status,
        "algorithm_product": item.algorithm_main.basename if item.algorithm_main else None,
        "algorithm_version": item.algorithm_version,
        "manual_product_id": item.manual_product_id,
        "operator": item.operator,
        "confirmed_at": item.confirmed_at,
        "baseline_top1": {
            "main_image_id": item.baseline_top1_main_id,
            "basename": item.baseline_top1_main.basename,
            "score": item.baseline_top1_score,
        } if item.baseline_top1_main else None,
        "baseline_top2_score": item.baseline_top2_score,
        "phash_distance": item.phash_distance,
        "phash_value": item.phash_value,
        "phash_best_region_type": item.phash_best_region_type,
        "match_method": item.match_method,
        "match_details": json.loads(item.match_details_json) if item.match_details_json else None,
        "derived": {
            asset_type.lower(): {
                "id": asset.id,
                "url": f"/api/derived-assets/{asset.id}",
                "feature_version": asset.feature_version,
                "mask_profile_id": asset.mask_profile_id,
            }
            for asset_type, asset in derived.items()
        },
        "status": item.status,
        "is_conflict": item.is_conflict,
        "is_low_confidence": item.is_low_confidence,
        "review_status": item.review_status,
        "thumbnail_url": f"/api/files/{item.id}/thumbnail",
        "preview_url": f"/api/files/{item.id}/preview",
        "candidates": [
            {
                "main_image_id": candidate.main_image_id,
                "basename": candidate.main_image.basename,
                "similarity": candidate.similarity,
                "distance": candidate.distance,
                "similarity_display": candidate.similarity_display,
                "reference_id": candidate.reference_id,
                "reference_category": candidate.reference_category,
                "best_distance": candidate.best_distance,
                "second_distance": candidate.second_distance,
                "third_distance": candidate.third_distance,
                "support_count": candidate.support_count,
                "aggregated_score": candidate.aggregated_score,
                "details": json.loads(candidate.details_json) if candidate.details_json else None,
                "thumbnail_url": f"/api/main-images/{candidate.main_image_id}/thumbnail",
            }
            for candidate in item.candidates
        ],
    }


def _review_data(db: Session, batch_id: int) -> tuple[list[dict], dict]:
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch_id
    ).order_by(MainImage.basename)))
    files = list(db.scalars(select(MatchedFile).options(
        selectinload(MatchedFile.candidates).selectinload(MatchCandidate.main_image)
    ).where(MatchedFile.batch_id == batch_id)))
    files = [item for item in files if not is_excluded_file_category(item.category)]
    ignored_categories = ignored_review_categories(db.get(Batch, batch_id))
    category_counts = Counter(item.category for item in files)
    review_categories = [
        {"category": category, "count": count, "ignored": category in ignored_categories}
        for category, count in sorted(category_counts.items())
    ]
    files = [item for item in files if item.category not in ignored_categories and item.status != "CONFLICT_MARKED"]
    position_counts = Counter((item.category, item.matched_main_id) for item in files if item.matched_main_id is not None)
    duplicate_ids = {item.id for item in files if item.matched_main_id is not None and position_counts[item.category, item.matched_main_id] > 1}
    assets = list(db.scalars(select(DerivedImageAsset).where(
        DerivedImageAsset.batch_id == batch_id,
        DerivedImageAsset.parent_file_id.is_not(None),
    ).order_by(DerivedImageAsset.id)))
    derived_by_file: dict[int, dict[str, DerivedImageAsset]] = defaultdict(dict)
    for asset in assets:
        derived_by_file[asset.parent_file_id][asset.asset_type] = asset
    by_main: dict[int, list[MatchedFile]] = defaultdict(list)
    category_source_counts: Counter[str] = Counter(item.category for item in files)
    category_display_names: dict[str, str] = {}
    for item in files:
        category_display_names.setdefault(
            item.category,
            Path(item.original_path).parent.name or item.category,
        )
    ignored_files = [item for item in files if item.status == "EXTRA_IGNORED"]
    unmatched_files = [
        item for item in files
        if item.matched_main_id is None and item.status != "EXTRA_IGNORED"
    ]
    # A category can be fully assigned while still missing from one or more
    # product groups when the automatic matcher gave two files in that
    # category the same main image. Those conflict files are actionable manual
    # choices too, so expose them in the same picker as truly unassigned files.
    conflict_files = [
        item for item in files
        if (item.is_conflict or item.id in duplicate_ids) and item.status != "EXTRA_IGNORED"
    ]
    manual_selection_files = list({item.id: item for item in [
        *unmatched_files,
        *conflict_files,
    ]}.values())
    manual_selection_files.sort(key=lambda item: (item.category.casefold(), item.id))
    for item in files:
        if item.matched_main_id is not None:
            by_main[item.matched_main_id].append(item)

    groups: list[dict] = []
    summary: Counter[str] = Counter()
    for main in mains:
        items = by_main.get(main.id, [])
        by_category: dict[str, list[MatchedFile]] = defaultdict(list)
        for item in items:
            by_category[item.category].append(item)
        ordered_categories = sorted(category_source_counts, key=str.casefold)
        unmatched_categories = [
            category for category in ordered_categories
            if category_source_counts[category] and not by_category.get(category)
        ]
        missing: list[str] = []
        has_conflict = any(item.is_conflict or item.id in duplicate_ids for item in items)
        has_low = any(item.status == "LOW_CONFIDENCE" for item in items)
        has_ambiguous = any(item.status == "AMBIGUOUS" for item in items)
        has_corrected = any(item.review_status == "CORRECTED" for item in items)
        confirmed = bool(items) and not has_conflict and all(item.review_status in {"CONFIRMED", "CORRECTED"} for item in items)
        pending = bool(items) and not confirmed
        flags = {
            "normal": not missing and not unmatched_categories and not has_conflict and not has_low and not has_ambiguous,
            "missing": bool(missing),
            "unmatched": bool(unmatched_categories),
            "ambiguous": has_ambiguous,
            "conflict": has_conflict,
            "low_confidence": has_low,
            "corrected": has_corrected,
            "confirmed": confirmed,
            "pending": pending,
        }
        summary["all"] += 1
        for key, value in flags.items():
            if value:
                summary[key] += 1
        score_values = [item.algorithm_score for item in items if item.algorithm_score is not None]
        groups.append({
            "main": {
                "id": main.id,
                "basename": main.basename,
                "filename": main.filename,
                "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
                "preview_url": f"/api/main-images/{main.id}/preview",
            },
            "categories": [
                {
                    "name": category_display_names.get(category, category),
                    "category": category,
                    "missing": False,
                    "unmatched": bool(category_source_counts[category]) and not bool(by_category.get(category)),
                    "files": [
                        _file_dict(item, derived_by_file.get(item.id))
                        for item in by_category.get(category, [])
                    ],
                }
                for category in ordered_categories
            ],
            "missing_categories": missing,
            "unmatched_categories": unmatched_categories,
            "minimum_similarity": min(score_values) if score_values else None,
            "flags": flags,
        })
    summary["unmatched_files"] = len(unmatched_files)
    summary["manual_selection_files"] = len(manual_selection_files)
    summary["ignored_files"] = len(ignored_files)
    summary_payload = {
        **dict(summary),
        "unmatched_by_category": dict(Counter(item.category for item in unmatched_files)),
        "manual_selection_by_category": dict(Counter(item.category for item in manual_selection_files)),
        "source_by_category": dict(category_source_counts),
    }
    return groups, {
        "review_categories": review_categories,
        "summary": summary_payload,
        "unmatched_files": [
            _file_dict(item, derived_by_file.get(item.id)) for item in unmatched_files
        ],
        "manual_selection_files": [
            _file_dict(item, derived_by_file.get(item.id)) for item in manual_selection_files
        ],
    }


@app.get("/api/batches/{batch_id}/review")
def review(batch_id: int, filter: str = Query("all"), db: Session = Depends(get_db)):
    if not db.get(Batch, batch_id):
        raise HTTPException(404, "批次不存在")
    groups, payload = _review_data(db, batch_id)
    if filter == "unmatched":
        # Show both groups with a missing category and groups containing a
        # duplicate assignment; both require the same manual picker.
        groups = [group for group in groups if group["flags"].get("unmatched", False) or group["flags"].get("conflict", False)]
    elif filter != "all":
        groups = [group for group in groups if group["flags"].get(filter, False)]
    return {"groups": groups, **payload}


@app.post("/api/batches/{batch_id}/review-categories", dependencies=[Depends(_review_write_guard)])
def set_review_category(batch_id: int, payload: ReviewCategoryRequest, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    if batch.status in {"QUEUED", "MATCHING"}:
        raise HTTPException(409, "匹配正在进行，请完成后再调整审核范围")
    exists = db.scalar(select(MatchedFile.id).where(
        MatchedFile.batch_id == batch_id, MatchedFile.category == payload.category
    ).limit(1))
    if not exists or is_excluded_file_category(payload.category):
        raise HTTPException(400, "该图片类型不属于本批次的审核范围")
    ignored = ignored_review_categories(batch)
    if payload.ignored:
        ignored.add(payload.category)
    else:
        ignored.discard(payload.category)
    snapshot = json.loads(batch.match_config_snapshot or "{}")
    snapshot["review_ignored_categories"] = sorted(ignored)
    batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
    db.commit()
    return {"ignored_categories": sorted(ignored), "rename_plan": RenamePlanService().generate(db, batch_id)}


@app.get("/api/main-images/{main_id}/references")
def product_references(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    rows = list(db.execute(select(ProductReference, MainImage.basename).join(
        MainImage, ProductReference.main_image_id == MainImage.id
    ).where(ProductReference.is_active.is_(True)).order_by(
        ProductReference.is_manual_confirmed.desc(),
        ProductReference.created_at.desc(),
    )))
    references = []
    for reference, basename in rows:
        if basename.strip().casefold() != main.basename.strip().casefold():
            continue
        source = db.get(MatchedFile, reference.source_file_id) if reference.source_file_id else None
        references.append({
            "id": reference.id,
            "main_image_id": reference.main_image_id,
            "source_file_id": reference.source_file_id,
            "source_filename": source.original_filename if source else None,
            "category": reference.category,
            "phash": reference.phash,
            "source_type": reference.source_type,
            "is_manual_confirmed": reference.is_manual_confirmed,
            "is_active": reference.is_active,
            "created_at": reference.created_at,
        })
    return {
        "product": main.basename,
        "scoring_enabled": settings.use_product_reference_scoring,
        "max_references_per_category": settings.max_references_per_category,
        "references": references,
    }


def _require_unique_positions(items: list[MatchedFile]) -> None:
    positions = defaultdict(list)
    for item in items:
        if item.matched_main_id is not None and item.status not in {"EXTRA_IGNORED", "CONFLICT_MARKED"}:
            positions[item.category, item.matched_main_id].append(item)
    duplicates = [rows for rows in positions.values() if len(rows) > 1]
    if duplicates:
        detail = "；".join(f"{rows[0].category}：{'、'.join(row.current_filename for row in rows)}" for rows in duplicates)
        raise HTTPException(409, f"同一类型中一张主图只能关联一张副图。重复位置：{detail}。请更正归属或标记为冲突图片")


@app.post("/api/main-images/{main_id}/confirm", dependencies=[Depends(_review_write_guard)])
def confirm_group(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    ignored = ignored_review_categories(db.get(Batch, main.batch_id))
    items = [
        item for item in db.scalars(select(MatchedFile).where(
            MatchedFile.matched_main_id == main.id
        )).all()
        if not is_excluded_file_category(item.category) and item.category not in ignored
    ]
    _require_unique_positions(items)
    now = datetime.now(timezone.utc)
    for item in items:
        item.review_status = "CONFIRMED"
        item.reviewed_at = now
        item.manual_product_id = main.id
        item.operator = "LOCAL"
        item.confirmed_at = now
        item.status = "MANUAL_CONFIRMED"
    references_learned = learn_confirmed_references(db, items)
    db.commit()
    plan = RenamePlanService().generate(db, main.batch_id)
    return {
        "confirmed": len(items),
        "references_learned": references_learned,
        "rename_plan": plan,
    }


@app.post("/api/batches/{batch_id}/confirm-groups", dependencies=[Depends(_review_write_guard)])
def confirm_groups(
    batch_id: int,
    payload: BulkConfirmRequest,
    db: Session = Depends(get_db),
):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    ignored = ignored_review_categories(batch)
    requested_ids = set(payload.main_image_ids)
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch_id,
        MainImage.id.in_(requested_ids),
    )))
    if len(mains) != len(requested_ids):
        raise HTTPException(400, "包含不属于当前批次的主图")
    items = [item for item in db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.matched_main_id.in_(requested_ids),
    )).all() if not is_excluded_file_category(item.category) and item.category not in ignored]
    _require_unique_positions(items)
    now = datetime.now(timezone.utc)
    for item in items:
        item.review_status = "CONFIRMED"
        item.reviewed_at = now
        item.manual_product_id = item.matched_main_id
        item.operator = payload.operator
        item.confirmed_at = now
        item.status = "MANUAL_CONFIRMED"
    references_learned = learn_confirmed_references(db, items)
    db.commit()
    plan = RenamePlanService().generate(db, batch_id)
    return {
        "confirmed_groups": len(mains),
        "confirmed_files": len(items),
        "references_learned": references_learned,
        "rename_plan": plan,
    }


@app.post("/api/files/{file_id}/reassign", dependencies=[Depends(_review_write_guard)])
def reassign(file_id: int, payload: ReassignRequest, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    main = db.get(MainImage, payload.main_image_id)
    if not item or not main:
        raise HTTPException(404, "文件或主图不存在")
    if item.batch_id != main.batch_id:
        raise HTTPException(400, "不能跨批次修改归属")
    duplicates = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == item.batch_id,
        MatchedFile.category == item.category,
        MatchedFile.matched_main_id == main.id,
        MatchedFile.id != item.id,
    )))
    if len(duplicates) > 1:
        raise HTTPException(409, "目标主图已被多张图片占用，请先标记冲突图片或选择空闲主图")
    duplicate = duplicates[0] if duplicates else None
    if duplicate:
        previous_main = db.get(MainImage, item.matched_main_id)
        if not previous_main:
            raise HTTPException(409, "目标球队已有图片，请先处理原匹配")
        occupied_previous = db.scalar(select(MatchedFile.id).where(
            MatchedFile.batch_id == item.batch_id, MatchedFile.category == item.category,
            MatchedFile.matched_main_id == previous_main.id, MatchedFile.id.not_in([item.id, duplicate.id]),
        ))
        if occupied_previous or previous_main.id == main.id:
            raise HTTPException(409, "该位置已经有重复图片，请先标记冲突图片或选择空闲主图")
        duplicate.matched_main_id = previous_main.id
        duplicate.matched_main_basename = previous_main.basename
        duplicate.manual_product_id = previous_main.id
    now = datetime.now(timezone.utc)
    if duplicate:
        duplicate.operator = payload.operator
        duplicate.confirmed_at = now
        duplicate.review_status = "CORRECTED"
        duplicate.reviewed_at = now
        duplicate.status = "MANUAL_CONFIRMED"
        duplicate.is_conflict = False
    item.matched_main_id = main.id
    item.matched_main_basename = main.basename
    item.manual_product_id = main.id
    item.operator = payload.operator
    item.confirmed_at = now
    item.review_status = "CORRECTED"
    item.reviewed_at = now
    item.status = "MANUAL_CONFIRMED"
    item.is_conflict = False
    batch = db.get(Batch, item.batch_id)
    snapshot = json.loads(batch.match_config_snapshot or "{}")
    snapshot.get("conflict_name_marks", {}).pop(str(item.id), None)
    snapshot.get("conflict_mark_originals", {}).pop(str(item.id), None)
    batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
    references_learned = learn_confirmed_references(
        db, [row for row in (duplicate, item) if row is not None]
    )
    db.commit()
    db.refresh(item)
    result = _file_dict(item)
    result["swapped_file_id"] = duplicate.id if duplicate else None
    result["references_learned"] = references_learned
    result["rename_plan"] = RenamePlanService().generate(db, item.batch_id)
    return result


@app.post("/api/files/{file_id}/confirm", dependencies=[Depends(_review_write_guard)])
def confirm_file(file_id: int, payload: ManualActionRequest, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item or item.matched_main_id is None:
        raise HTTPException(404, "没有可确认的匹配")
    if item.category in ignored_review_categories(db.get(Batch, item.batch_id)):
        raise HTTPException(409, "该图片类型已忽略，请先恢复后再确认")
    _require_unique_positions(list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == item.batch_id, MatchedFile.category == item.category,
        MatchedFile.matched_main_id == item.matched_main_id,
    ))))
    now = datetime.now(timezone.utc)
    item.manual_product_id = item.matched_main_id
    item.operator = payload.operator
    item.confirmed_at = now
    item.reviewed_at = now
    item.review_status = "CONFIRMED"
    item.status = "MANUAL_CONFIRMED"
    references_learned = learn_confirmed_references(db, [item])
    db.commit()
    result = _file_dict(item)
    result["references_learned"] = references_learned
    result["rename_plan"] = RenamePlanService().generate(db, item.batch_id)
    return result


@app.post("/api/files/{file_id}/cancel-match", dependencies=[Depends(_review_write_guard)])
def cancel_match(file_id: int, payload: ManualActionRequest, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    now = datetime.now(timezone.utc)
    item.matched_main_id = None
    item.matched_main_basename = None
    item.manual_product_id = None
    item.operator = payload.operator
    item.confirmed_at = now
    item.reviewed_at = now
    item.review_status = "CORRECTED"
    item.status = "MANUAL_CONFIRMED"
    item.is_conflict = False
    db.commit()
    result = _file_dict(item)
    result["rename_plan"] = RenamePlanService().generate(db, item.batch_id)
    return result


@app.post("/api/files/{file_id}/restore", dependencies=[Depends(_review_write_guard)])
def restore(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    try:
        restore_original(db, item)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _file_dict(item)


@app.get("/api/batches/{batch_id}/main-images/search")
def search_mains(
    batch_id: int,
    q: str = Query("", max_length=255),
    exclude_confirmed: bool = False,
    db: Session = Depends(get_db),
):
    query = select(MainImage).where(MainImage.batch_id == batch_id)
    if exclude_confirmed:
        status_rows = db.execute(select(
            MatchedFile.matched_main_id,
            MatchedFile.review_status,
        ).where(
            MatchedFile.batch_id == batch_id,
            MatchedFile.category != "PSD",
            MatchedFile.matched_main_id.is_not(None),
        )).all()
        statuses_by_main: dict[int, list[str]] = defaultdict(list)
        for main_id, review_status in status_rows:
            statuses_by_main[main_id].append(review_status)
        confirmed_ids = [
            main_id for main_id, statuses in statuses_by_main.items()
            if statuses and all(status == "CONFIRMED" for status in statuses)
        ]
        if confirmed_ids:
            query = query.where(MainImage.id.not_in(confirmed_ids))
    if q:
        query = query.where(MainImage.basename.contains(q))
    mains = db.scalars(query.order_by(MainImage.basename).limit(500)).all()
    return [{
        "id": main.id,
        "basename": main.basename,
        "filename": main.filename,
        "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
    } for main in mains]


@app.get("/api/files/{file_id}/product-candidates")
def ranked_product_candidates(
    file_id: int,
    q: str = Query("", max_length=255),
    exclude_confirmed: bool = True,
    manual_selection: bool = False,
    db: Session = Depends(get_db),
):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    if is_excluded_file_category(item.category):
        raise HTTPException(409, "该文件类型不参与图片相似度计算")
    batch = db.get(Batch, item.batch_id)
    profile = active_manual_profile(db, item.batch_id, item.category)
    if not batch:
        raise HTTPException(404, "批次不存在")
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == item.batch_id
    ).order_by(MainImage.id)))

    confirmed_ids: set[int] = set()
    if exclude_confirmed and not manual_selection:
        status_rows = db.execute(select(
            MatchedFile.matched_main_id,
            MatchedFile.review_status,
        ).where(
            MatchedFile.batch_id == item.batch_id,
            MatchedFile.category != "PSD",
            MatchedFile.matched_main_id.is_not(None),
        )).all()
        statuses_by_main: dict[int, list[str]] = defaultdict(list)
        for main_id, review_status in status_rows:
            statuses_by_main[main_id].append(review_status)
        confirmed_ids = {
            main_id for main_id, statuses in statuses_by_main.items()
            if statuses and all(status == "CONFIRMED" for status in statuses)
        }

    if item.match_method == "VISIONMATCH":
        details = json.loads(item.match_details_json or "{}")
        candidates = {row["main_image_id"]: row for row in details.get("candidates", [])}
        rows = [{"id": main.id, "main_image_id": main.id, "basename": main.basename,
                 "filename": main.filename, "aggregated_score": None, "similarity_rank": None,
                 "score_source": "VISIONMATCH_EVIDENCE", "decision": candidates.get(main.id, {}).get("decision"),
                 "evidence": candidates.get(main.id, {}).get("evidence", []),
                 "thumbnail_url": f"/api/main-images/{main.id}/thumbnail"}
                for main in mains if main.id not in confirmed_ids and (not q or q.casefold() in main.basename.casefold())]
        order = {row["main_image_id"]: index for index, row in enumerate(details.get("candidates", []))}
        rows.sort(key=lambda row: (order.get(row["id"], 999999), row["id"]))
        return rows
    if item.match_method == "EXACT_FILENAME":
        # Manual replacement is still available, without invoking a visual
        # model for an image that was associated deterministically.
        return [{
            "id": main.id, "main_image_id": main.id,
            "basename": main.basename, "filename": main.filename,
            "aggregated_score": None, "similarity_rank": None,
            "score_source": "EXACT_FILENAME",
            "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
        } for main in mains if main.id not in confirmed_ids
            and (not q or q.casefold() in main.basename.casefold())]

    # Qwen batches keep every direct pairwise verdict in match_details_json;
    # use that complete list for manual reassignment instead of falling back
    # to a legacy pHash query or showing only the persisted Top5 rows.
    if item.match_method == "QWEN_VISUAL" and item.match_details_json:
        try:
            qwen_details = json.loads(item.match_details_json)
        except (TypeError, ValueError):
            qwen_details = {}
        pairwise = qwen_details.get("pairwise", []) if isinstance(qwen_details, dict) else []
        if isinstance(pairwise, list) and pairwise:
            by_main_id = {
                int(row["main_id"]): row
                for row in pairwise
                if isinstance(row, dict) and row.get("main_id") is not None
            }
            folded_query = q.strip().casefold()
            result = []
            for main in mains:
                if main.id in confirmed_ids:
                    continue
                if folded_query and folded_query not in main.basename.casefold():
                    continue
                row = by_main_id.get(main.id, {})
                score = row.get("fusion_score")
                if score is None:
                    score = row.get("qwen_support_score")
                if score is None:
                    score = row.get("confidence", 0)
                result.append({
                    "id": main.id,
                    "main_image_id": main.id,
                    "basename": main.basename,
                    "filename": main.filename,
                    "similarity_rank": None,
                    "aggregated_score": float(score or 0),
                    "model_name": qwen_details.get("qwen_model", batch.matching_model),
                    "details": row,
                    "score_source": "QWEN_FUSION_PAIRWISE",
                    "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
                })
            result.sort(key=lambda value: (-value["aggregated_score"], value["main_image_id"]))
            for rank, value in enumerate(result, 1):
                value["similarity_rank"] = rank
            return result

    # New matching runs persist the complete matrix. Reading one sorted row is
    # fast and, critically, uses the same model selected for auto matching.
    persisted = []
    if item.matching_run_id is not None:
        persisted = list(db.execute(select(
            ProductSimilarity,
            MainImage,
        ).join(
            MainImage, ProductSimilarity.main_image_id == MainImage.id
        ).where(
            ProductSimilarity.matched_file_id == item.id,
            ProductSimilarity.matching_run_id == item.matching_run_id,
        ).order_by(
            ProductSimilarity.score.desc(),
            MainImage.id,
        )).all())
    if persisted:
        folded_query = q.strip().casefold()
        result = []
        for rank, (score, main) in enumerate(persisted, 1):
            if main.id in confirmed_ids:
                continue
            if folded_query and folded_query not in main.basename.casefold():
                continue
            result.append({
                "id": main.id,
                "main_image_id": main.id,
                "basename": main.basename,
                "filename": main.filename,
                "similarity_rank": rank,
                "aggregated_score": score.score,
                "best_distance": score.best_distance,
                "second_distance": score.second_distance,
                "third_distance": score.third_distance,
                "support_count": score.support_count,
                "model_name": score.model_name,
                "details": json.loads(score.details_json) if score.details_json else None,
                "score_source": "FULL_MATCH_MATRIX",
                "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
            })
        return result

    # Legacy pHash runs predate complete-matrix persistence. Keep their old
    # fallback, but never mislabel this path as a DINO score.
    if batch.matching_model in DINO_MODELS:
        raise HTTPException(
            409,
            "该 DINO 批次创建于完整矩阵缓存升级前，请重新匹配一次以生成全部球队排序",
        )
    if not profile:
        raise HTTPException(409, "当前分类尚未保存手动识别区域")
    try:
        query_hash = masked_query_hashes(db, batch, [item], profile)[0]
        scores = build_product_score_matrix(
            db,
            [query_hash],
            item.category,
            mains,
            max_manual_references=(
                settings.max_references_per_category
                if settings.use_product_reference_scoring
                else 0
            ),
            query_file_ids=[item.id],
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(409, f"无法计算当前图片相似度：{exc}") from exc

    folded_query = q.strip().casefold()
    ordered_columns = sorted(
        range(len(mains)),
        key=lambda column: (-float(scores.matrix[0, column]), mains[column].id),
    )
    result = []
    for rank, column in enumerate(ordered_columns, 1):
        main = mains[column]
        if main.id in confirmed_ids:
            continue
        if folded_query and folded_query not in main.basename.casefold():
            continue
        evidence = scores.evidence[0][column]
        result.append({
            "id": main.id,
            "main_image_id": main.id,
            "basename": main.basename,
            "filename": main.filename,
            "similarity_rank": rank,
            "aggregated_score": evidence.aggregated_score,
            "best_distance": evidence.best_distance,
            "second_distance": evidence.second_distance,
            "third_distance": evidence.third_distance,
            "support_count": evidence.support_count,
            "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
        })
    return result


@app.get("/api/batches/{batch_id}/mask-profiles")
def list_mask_profiles(batch_id: int, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    profiles = profile_history(db, batch_id)
    attachment_categories = db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id
    ).distinct().order_by(MatchedFile.category)).all()
    attachment_categories = [
        category for category in attachment_categories
        if not is_excluded_file_category(category)
    ]
    return {
        "product_category": batch.product_category,
        "matching_model": batch.matching_model,
        "match_engine": batch.match_engine,
        "categories": [MAIN_IMAGE_CATEGORY, *attachment_categories],
        "profiles": [
            profile_dict(profile, profile_storage_valid(db, profile))
            for profile in profiles
        ],
        "readiness": mask_readiness(db, batch_id),
    }


def _unconfirm_items(
    db: Session,
    items: list[MatchedFile],
    operator: str,
) -> dict[str, int]:
    """Withdraw approval while preserving explicit manual reassignments."""
    now = datetime.now(timezone.utc)
    reverted_ids: list[int] = []
    preserved_corrected = 0
    for item in items:
        if item.review_status != "CONFIRMED":
            continue
        was_reassigned = (
            item.algorithm_main_id is not None
            and item.matched_main_id != item.algorithm_main_id
        )
        item.reviewed_at = now
        item.operator = operator
        if was_reassigned:
            # A group confirmation must not erase a deliberate swap/reassign.
            item.review_status = "CORRECTED"
            item.status = "MANUAL_CONFIRMED"
            preserved_corrected += 1
            continue
        item.review_status = "PENDING"
        item.confirmed_at = None
        item.manual_product_id = None
        # PENDING is intentional: restoring MATCHED would let Rename Plan pass
        # immediately and make the withdrawal ineffective.
        item.status = "PENDING"
        item.is_low_confidence = item.algorithm_status in {
            "LOW_CONFIDENCE", "AMBIGUOUS"
        }
        reverted_ids.append(item.id)

    references_disabled = 0
    if reverted_ids:
        references = list(db.scalars(select(ProductReference).where(
            ProductReference.source_file_id.in_(reverted_ids),
            ProductReference.is_manual_confirmed.is_(True),
            ProductReference.is_active.is_(True),
        )))
        for reference in references:
            reference.is_active = False
        references_disabled = len(references)
    return {
        "reverted_files": len(reverted_ids),
        "preserved_corrected_files": preserved_corrected,
        "references_disabled": references_disabled,
    }


@app.post("/api/main-images/{main_id}/unconfirm", dependencies=[Depends(_review_write_guard)])
def unconfirm_group(
    main_id: int,
    payload: ManualActionRequest,
    db: Session = Depends(get_db),
):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    ignored = ignored_review_categories(db.get(Batch, main.batch_id))
    items = [
        item for item in db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == main.batch_id,
            MatchedFile.matched_main_id == main.id,
        )).all()
        if not is_excluded_file_category(item.category) and item.category not in ignored
    ]
    result = _unconfirm_items(db, items, payload.operator)
    db.commit()
    return {
        **result,
        "rename_plan": RenamePlanService().generate(db, main.batch_id),
    }


@app.post("/api/batches/{batch_id}/unconfirm-groups", dependencies=[Depends(_review_write_guard)])
def unconfirm_groups(
    batch_id: int,
    payload: BulkConfirmRequest,
    db: Session = Depends(get_db),
):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    ignored = ignored_review_categories(batch)
    requested_ids = set(payload.main_image_ids)
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch_id,
        MainImage.id.in_(requested_ids),
    )))
    if len(mains) != len(requested_ids):
        raise HTTPException(400, "包含不属于当前批次的主图")
    items = [
        item for item in db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id,
            MatchedFile.matched_main_id.in_(requested_ids),
        )).all()
        if not is_excluded_file_category(item.category) and item.category not in ignored
    ]
    result = _unconfirm_items(db, items, payload.operator)
    db.commit()
    return {
        "requested_groups": len(mains),
        **result,
        "rename_plan": RenamePlanService().generate(db, batch_id),
    }


@app.get("/api/batches/{batch_id}/main-bodies")
def list_main_bodies(batch_id: int, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    mains = list(db.scalars(select(MainImage).where(
        MainImage.batch_id == batch_id
    ).order_by(MainImage.basename)))
    profile = active_manual_profile(db, batch_id, MAIN_IMAGE_CATEGORY)
    if not profile:
        raise HTTPException(409, "请先完成并保存主图手动遮罩")
    try:
        crop_box = (
            profile.crop_left,
            profile.crop_top,
            profile.crop_right,
            profile.crop_bottom,
        )
        with Image.open(profile.mask_path) as stored_mask:
            mask = stored_mask.convert("L")
            mask.load()
        artifacts = ensure_batch_main_bodies(
            db,
            batch,
            mains,
            manual_box=tuple(float(value) for value in crop_box),
            manual_mask=mask,
            manual_mask_path=profile.mask_path,
        )
        db.commit()
    except (OSError, ValueError) as exc:
        db.rollback()
        raise HTTPException(422, f"主图主体提取失败: {exc}") from exc
    return {
        "batch_id": batch_id,
        "padding": settings.body_crop_padding,
        "items": [main_body_dict(item) for item in artifacts],
    }


@app.get("/api/main-images/{main_id}/body-mask")
def main_body_mask(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    return _file_response(main.body_mask_path)


@app.get("/api/main-images/{main_id}/body-crop")
def main_body_crop(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    return _file_response(main.body_crop_path)


@app.get("/api/batches/{batch_id}/category-files/{image_category}")
def category_files(batch_id: int, image_category: str, db: Session = Depends(get_db)):
    if image_category == MAIN_IMAGE_CATEGORY:
        mains = db.scalars(select(MainImage).where(
            MainImage.batch_id == batch_id
        ).order_by(MainImage.basename)).all()
        return [{
            "id": item.id,
            "filename": item.filename,
            "preview_url": f"/api/main-images/{item.id}/preview",
            "thumbnail_url": f"/api/main-images/{item.id}/thumbnail",
            "reference_type": "main_image",
        } for item in mains]
    items = db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category == image_category,
    ).order_by(MatchedFile.original_filename)).all()
    return [{
        "id": item.id,
        "filename": item.original_filename,
        "preview_url": f"/api/files/{item.id}/preview",
        "thumbnail_url": f"/api/files/{item.id}/thumbnail",
    } for item in items]


@app.post("/api/batches/{batch_id}/mask-profiles/{image_category}")
def save_mask_profile(
    batch_id: int,
    image_category: str,
    payload: MaskProfileSaveRequest,
    db: Session = Depends(get_db),
):
    try:
        if image_category == MAIN_IMAGE_CATEGORY:
            if payload.reference_main_id is None:
                raise ValueError("请选择一张参考主图")
            profile, _mask = save_main_profile_version(
                db,
                batch_id,
                payload.mask_data_url,
                payload.reference_main_id,
                tuple(payload.crop_box),
            )
            # Match attachment-category behavior: persist the proportional
            # rule now and apply it to every main during the matching stage,
            # where progress is already visible. This keeps Save independent
            # of the number and network location of main images.
            db.commit()
        else:
            if payload.reference_file_id is None:
                raise ValueError("请选择一张参考图片")
            profile = save_profile_version(
                db,
                batch_id,
                image_category,
                payload.mask_data_url,
                payload.reference_file_id,
                tuple(payload.crop_box),
                payload.name,
            )
    except (OSError, ValueError) as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return profile_dict(profile)


@app.get("/api/mask-profiles/{profile_id}/mask")
def mask_profile_image(profile_id: int, db: Session = Depends(get_db)):
    profile = db.get(MaskProfile, profile_id)
    if not profile:
        raise HTTPException(404, "Mask Profile不存在")
    return _file_response(profile.mask_path)


@app.get("/api/batches/{batch_id}/rename-plan")
def get_rename_plan(batch_id: int, db: Session = Depends(get_db)):
    plan = RenamePlanService().latest(db, batch_id)
    return plan or {"batch_id": batch_id, "counts": {}, "plans": []}


@app.post("/api/batches/{batch_id}/rename-plan", dependencies=[Depends(_review_write_guard)])
def generate_rename_plan(batch_id: int, db: Session = Depends(get_db)):
    try:
        return RenamePlanService().generate(db, batch_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/batches/{batch_id}/rename-conflict-marks", dependencies=[Depends(_review_write_guard)])
def set_conflict_name_marks(batch_id: int, payload: ConflictNameMarksRequest, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    if batch.status in {"QUEUED", "MATCHING"}:
        raise HTTPException(409, "匹配正在进行，请完成后再标记冲突图片")
    service = RenamePlanService()
    preview = service.latest(db, batch_id) or service.generate(db, batch_id)
    allowed = {row["source_file_id"] for row in preview["plans"] if row.get("can_mark_conflict") or row.get("conflict_marked")}
    selected = set(payload.file_ids)
    if not selected <= allowed:
        raise HTTPException(400, "只能标记本批次已确认且可用的冲突图片；缺失文件请先恢复")
    old_marks = conflict_name_marks(batch)
    snapshot = json.loads(batch.match_config_snapshot or "{}")
    history = snapshot.setdefault("conflict_mark_originals", {})
    for row in preview["plans"]:
        if row.get("can_mark_conflict"):
            item = db.get(MatchedFile, row["source_file_id"])
            history.setdefault(str(item.id), {"matched_main_id": item.matched_main_id,
                "matched_main_basename": item.matched_main_basename, "review_status": item.review_status, "status": item.status})
    next_number = max(old_marks.values(), default=0) + 1
    marks = {}
    for file_id in sorted(selected):
        number = old_marks.get(file_id)
        if number is None:
            number = next_number
            next_number += 1
        item = db.get(MatchedFile, file_id)
        if not Path(item.current_path).is_file():
            raise HTTPException(400, f"图片当前不存在，无法标记命名：{item.current_path}")
        if len(service.marked_filename(item, number)) > 255:
            raise HTTPException(400, f"标记后的文件名过长：{item.original_filename}")
        marks[str(file_id)] = number
        item.matched_main_id = None
        item.matched_main_basename = None
        item.manual_product_id = None
        item.confirmed_at = None
        item.review_status = "PENDING"
        item.status = "CONFLICT_MARKED"
        item.is_conflict = True
        for reference in db.scalars(select(ProductReference).where(ProductReference.source_file_id == file_id)):
            reference.is_active = False
    for file_id in set(old_marks) - selected:
        item = db.get(MatchedFile, file_id)
        if item and item.status == "CONFLICT_MARKED":
            item.status = "UNMATCHED"
            item.is_conflict = False
    snapshot["conflict_name_marks"] = marks
    batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
    db.commit()
    return {"conflict_name_marks": marks, "rename_plan": service.generate(db, batch_id)}


@app.post("/api/batches/{batch_id}/rename-plan/execute", dependencies=[Depends(_review_write_guard)])
def execute_rename_plan(
    batch_id: int,
    payload: RenamePlanExecuteRequest,
    db: Session = Depends(get_db),
):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    _required_upload_root(batch.root_path)
    try:
        return RenamePlanService().execute(db, batch_id, payload.plan_group)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _file_response(path: str | None) -> FileResponse:
    if not path or not Path(path).is_file():
        raise HTTPException(404, "预览文件不存在")
    return FileResponse(path)


def _thumbnail(source_path: str, thumbnail_path: str | None) -> FileResponse:
    source = Path(source_path)
    if not source.is_file() or not thumbnail_path:
        raise HTTPException(404, "源文件不存在")
    try:
        target = ensure_thumbnail(source, Path(thumbnail_path))
    except Exception as exc:
        raise HTTPException(422, f"无法生成缩略图: {exc}") from exc
    return FileResponse(target, media_type="image/jpeg")


@app.get("/api/main-images/{main_id}/thumbnail")
def main_thumbnail(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    return _thumbnail(main.file_path, main.thumbnail_path)


@app.get("/api/files/{file_id}")
def file_detail(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "图片不存在")
    return _file_dict(item)


@app.get("/api/files/{file_id}/thumbnail")
def file_thumbnail(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    if item.category == "PSD":
        from app.services.psd import PsdPreviewExtractor
        try:
            asset = PsdPreviewExtractor().extract(db, item.batch, item)
            db.commit()
        except Exception as exc:
            raise HTTPException(422, f"无法读取PSD预览：{exc}") from exc
        return _thumbnail(asset.asset_path, item.thumbnail_path)
    return _thumbnail(item.current_path, item.thumbnail_path)


@app.get("/api/derived-assets/{asset_id}")
def derived_asset(asset_id: int, db: Session = Depends(get_db)):
    asset = db.get(DerivedImageAsset, asset_id)
    if not asset:
        raise HTTPException(404, "派生识别资产不存在")
    return _file_response(asset.asset_path)


def _preview(source_path: str, batch: Batch) -> FileResponse:
    source = Path(source_path)
    if not source.is_file():
        raise HTTPException(404, "源文件不存在")
    target = preview_cache_path(Path(batch.root_path), batch.id, source)
    try:
        ensure_preview(source, target)
    except Exception as exc:
        raise HTTPException(422, f"无法生成预览: {exc}") from exc
    return FileResponse(target, media_type="image/jpeg")


@app.get("/api/main-images/{main_id}/preview")
def main_preview(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    return _preview(main.file_path, main.batch)


@app.get("/api/files/{file_id}/preview")
def file_preview(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    if item.category == "PSD":
        from app.services.psd import PsdPreviewExtractor
        try:
            asset = PsdPreviewExtractor().extract(db, item.batch, item)
            db.commit()
        except Exception as exc:
            raise HTTPException(422, f"无法读取PSD预览：{exc}") from exc
        return _preview(asset.asset_path, item.batch)
    return _preview(item.current_path, item.batch)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
