from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.config import is_ignored_image_category, settings
from app.db import SessionLocal, get_db, init_db
from app.models import (
    Batch,
    BatchProgress,
    DerivedImageAsset,
    MainImage,
    MaskProfile,
    MatchCandidate,
    MatchedFile,
)
from app.services.matcher import match_batch
from app.services.main_body import (
    ensure_batch_main_bodies,
    ensure_main_body,
    main_body_dict,
)
from app.services.media import ensure_preview, ensure_thumbnail, preview_cache_path
from app.services.masks import (
    MAIN_IMAGE_CATEGORY,
    active_manual_profile,
    mask_readiness,
    profile_dict,
    profile_history,
    save_main_profile_version,
    save_profile_version,
)
from app.services.renamer import restore_original
from app.services.rename_plan import RenamePlanService
from app.services.scanner import scan_batch


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
        if is_ignored_image_category(category):
            continue
        directory = Path(source_path).parent
        if directory not in seen:
            category_dirs.append(directory)
            seen.add(directory)
        if category == "PSD" and directory not in psd_dirs:
            psd_dirs.append(directory)
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


class ReassignRequest(BaseModel):
    main_image_id: int
    operator: str = "LOCAL"


class ManualActionRequest(BaseModel):
    operator: str = "LOCAL"


class BulkConfirmRequest(BaseModel):
    main_image_ids: list[int] = Field(min_length=1, max_length=1000)
    operator: str = "LOCAL"


class MaskProfileSaveRequest(BaseModel):
    mask_data_url: str = Field(min_length=32)
    reference_file_id: int | None = None
    reference_main_id: int | None = None
    crop_box: list[float] = Field(min_length=4, max_length=4)
    name: str | None = Field(default=None, max_length=255)


class RenamePlanExecuteRequest(BaseModel):
    plan_group: str = Field(min_length=1, max_length=36)


def _batch_dict(db: Session, batch: Batch) -> dict:
    main_count = db.scalar(select(func.count()).select_from(MainImage).where(
        MainImage.batch_id == batch.id
    )) or 0
    raw_category_rows = db.execute(select(MatchedFile.category, func.count()).where(
        MatchedFile.batch_id == batch.id
    ).group_by(MatchedFile.category)).all()
    category_rows = [
        row for row in raw_category_rows
        if not is_ignored_image_category(row[0])
    ]
    progress = db.get(BatchProgress, batch.id)
    main_body_ready_count = db.scalar(select(func.count()).select_from(MainImage).where(
        MainImage.batch_id == batch.id,
        MainImage.body_feature_version == settings.body_phash_version,
    )) or 0
    return {
        "id": batch.id,
        "root_path": batch.root_path,
        "product_category": batch.product_category,
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
            "algorithm_version": settings.masked_algorithm_version,
            "psd_algorithm_version": settings.psd_algorithm_version,
            "baseline_algorithm_version": settings.algorithm_version,
            "match_min_score": settings.match_min_score,
            "match_min_margin": settings.match_min_margin,
            "unmatched_score": settings.unmatched_score,
            "body_crop_padding": settings.body_crop_padding,
            "body_phash_version": settings.body_phash_version,
            "detail_region_algorithm_version": settings.detail_region_algorithm_version,
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/workspace-config")
def workspace_config():
    return {"upload_root": settings.upload_root}


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
            match_batch(db, batch_id)
        except Exception:
            # match_batch records the actionable error on the batch.
            return


@app.post("/api/batches/{batch_id}/match")
def start_match(batch_id: int, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    if batch.status not in {"SCANNED", "COMPLETED", "FAILED"}:
        raise HTTPException(409, f"当前状态 {batch.status}，请先完成扫描")
    readiness = mask_readiness(db, batch.id)
    if not readiness["ready"]:
        missing = "、".join(readiness["missing_categories"])
        raise HTTPException(409, f"请先在步骤 03 完成裁剪框和手绘 Mask：{missing}")
    batch.status = "QUEUED"
    progress = db.get(BatchProgress, batch.id)
    if not progress:
        progress = BatchProgress(batch_id=batch.id)
        db.add(progress)
    progress.stage = "QUEUED"
    progress.current = 0
    progress.total = 0
    progress.percent = 0
    progress.message = "任务已进入本地处理队列"
    progress.device = None
    db.commit()
    executor.submit(_run_match, batch_id)
    return {"batch_id": batch_id, "status": "QUEUED"}


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
    files = [item for item in files if not is_ignored_image_category(item.category)]
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
        has_conflict = any(item.is_conflict for item in items)
        has_low = any(item.status == "LOW_CONFIDENCE" for item in items)
        has_ambiguous = any(item.status == "AMBIGUOUS" for item in items)
        has_corrected = any(item.review_status == "CORRECTED" for item in items)
        confirmed = bool(items) and all(item.review_status == "CONFIRMED" for item in items)
        pending = not confirmed
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
    summary["ignored_files"] = len(ignored_files)
    summary_payload = {
        **dict(summary),
        "unmatched_by_category": dict(Counter(item.category for item in unmatched_files)),
        "source_by_category": dict(category_source_counts),
    }
    return groups, {
        "summary": summary_payload,
        "unmatched_files": [
            _file_dict(item, derived_by_file.get(item.id)) for item in unmatched_files
        ],
    }


@app.get("/api/batches/{batch_id}/review")
def review(batch_id: int, filter: str = Query("all"), db: Session = Depends(get_db)):
    if not db.get(Batch, batch_id):
        raise HTTPException(404, "批次不存在")
    groups, payload = _review_data(db, batch_id)
    if filter != "all":
        groups = [group for group in groups if group["flags"].get(filter, False)]
    return {"groups": groups, **payload}


@app.post("/api/main-images/{main_id}/confirm")
def confirm_group(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    items = [
        item for item in db.scalars(select(MatchedFile).where(
            MatchedFile.matched_main_id == main.id
        )).all()
        if not is_ignored_image_category(item.category)
    ]
    now = datetime.now(timezone.utc)
    for item in items:
        item.review_status = "CONFIRMED"
        item.reviewed_at = now
        item.manual_product_id = main.id
        item.operator = "LOCAL"
        item.confirmed_at = now
        item.status = "MANUAL_CONFIRMED"
    db.commit()
    plan = RenamePlanService().generate(db, main.batch_id)
    return {"confirmed": len(items), "rename_plan": plan}


@app.post("/api/batches/{batch_id}/confirm-groups")
def confirm_groups(
    batch_id: int,
    payload: BulkConfirmRequest,
    db: Session = Depends(get_db),
):
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
    )).all() if not is_ignored_image_category(item.category)]
    now = datetime.now(timezone.utc)
    for item in items:
        item.review_status = "CONFIRMED"
        item.reviewed_at = now
        item.manual_product_id = item.matched_main_id
        item.operator = payload.operator
        item.confirmed_at = now
        item.status = "MANUAL_CONFIRMED"
    db.commit()
    plan = RenamePlanService().generate(db, batch_id)
    return {
        "confirmed_groups": len(mains),
        "confirmed_files": len(items),
        "rename_plan": plan,
    }


@app.post("/api/files/{file_id}/reassign")
def reassign(file_id: int, payload: ReassignRequest, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    main = db.get(MainImage, payload.main_image_id)
    if not item or not main:
        raise HTTPException(404, "文件或主图不存在")
    if item.batch_id != main.batch_id:
        raise HTTPException(400, "不能跨批次修改归属")
    duplicate = db.scalar(select(MatchedFile).where(
        MatchedFile.batch_id == item.batch_id,
        MatchedFile.category == item.category,
        MatchedFile.matched_main_id == main.id,
        MatchedFile.id != item.id,
    ))
    if duplicate:
        previous_main = db.get(MainImage, item.matched_main_id)
        if not previous_main:
            raise HTTPException(409, "目标球队已有图片，请先处理原匹配")
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
    db.commit()
    db.refresh(item)
    result = _file_dict(item)
    result["swapped_file_id"] = duplicate.id if duplicate else None
    result["rename_plan"] = RenamePlanService().generate(db, item.batch_id)
    return result


@app.post("/api/files/{file_id}/confirm")
def confirm_file(file_id: int, payload: ManualActionRequest, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item or item.matched_main_id is None:
        raise HTTPException(404, "没有可确认的匹配")
    now = datetime.now(timezone.utc)
    item.manual_product_id = item.matched_main_id
    item.operator = payload.operator
    item.confirmed_at = now
    item.reviewed_at = now
    item.review_status = "CONFIRMED"
    item.status = "MANUAL_CONFIRMED"
    db.commit()
    result = _file_dict(item)
    result["rename_plan"] = RenamePlanService().generate(db, item.batch_id)
    return result


@app.post("/api/files/{file_id}/cancel-match")
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


@app.post("/api/files/{file_id}/restore")
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
        if not is_ignored_image_category(category)
    ]
    return {
        "product_category": batch.product_category,
        "categories": [MAIN_IMAGE_CATEGORY, *attachment_categories],
        "profiles": [profile_dict(profile) for profile in profiles],
        "readiness": mask_readiness(db, batch_id),
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
    if not profile or any(main.body_crop_method != "MANUAL_MASK" for main in mains):
        raise HTTPException(409, "请先完成并保存主图手动遮罩")
    try:
        artifacts = ensure_batch_main_bodies(db, batch, mains)
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
            profile, mask = save_main_profile_version(
                db,
                batch_id,
                payload.mask_data_url,
                payload.reference_main_id,
                tuple(payload.crop_box),
            )
            batch = db.get(Batch, batch_id)
            mains = list(db.scalars(select(MainImage).where(
                MainImage.batch_id == batch_id
            ).order_by(MainImage.id)))
            for main in mains:
                ensure_main_body(
                    db,
                    batch,
                    main,
                    manual_box=tuple(payload.crop_box),
                    manual_mask=mask,
                )
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


@app.post("/api/batches/{batch_id}/rename-plan")
def generate_rename_plan(batch_id: int, db: Session = Depends(get_db)):
    try:
        return RenamePlanService().generate(db, batch_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/batches/{batch_id}/rename-plan/execute")
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


@app.get("/api/files/{file_id}/thumbnail")
def file_thumbnail(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
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
    return _preview(item.current_path, item.batch)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
