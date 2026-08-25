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

from app.config import EXPECTED_CATEGORIES, settings
from app.db import SessionLocal, get_db, init_db
from app.models import Batch, BatchProgress, MainImage, MatchCandidate, MatchedFile
from app.services.matcher import match_batch
from app.services.media import ensure_preview, preview_cache_path
from app.services.renamer import restore_original
from app.services.scanner import inspect_layout, scan_batch


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


class BatchCreate(BaseModel):
    root_path: str = Field(min_length=1)


class ReassignRequest(BaseModel):
    main_image_id: int
    operator: str = "LOCAL"


class ManualActionRequest(BaseModel):
    operator: str = "LOCAL"


def _batch_dict(db: Session, batch: Batch) -> dict:
    main_count = db.scalar(select(func.count()).select_from(MainImage).where(
        MainImage.batch_id == batch.id
    )) or 0
    category_rows = db.execute(select(MatchedFile.category, func.count()).where(
        MatchedFile.batch_id == batch.id
    ).group_by(MatchedFile.category)).all()
    progress = db.get(BatchProgress, batch.id)
    return {
        "id": batch.id,
        "root_path": batch.root_path,
        "status": batch.status,
        "error_message": batch.error_message,
        "created_at": batch.created_at,
        "finished_at": batch.finished_at,
        "main_count": main_count,
        "categories": dict(category_rows),
        "layout": inspect_layout(Path(batch.root_path)),
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
            "algorithm_version": settings.algorithm_version,
            "match_min_score": settings.match_min_score,
            "match_min_margin": settings.match_min_margin,
            "unmatched_score": settings.unmatched_score,
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/batches")
def list_batches(db: Session = Depends(get_db)):
    batches = db.scalars(select(Batch).order_by(Batch.id.desc()).limit(30)).all()
    return [_batch_dict(db, batch) for batch in batches]


@app.post("/api/batches")
def create_batch(payload: BatchCreate, db: Session = Depends(get_db)):
    root = Path(payload.root_path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(400, f"目录不存在: {root}")
    batch = Batch(root_path=str(root), status="CREATED")
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


def _file_dict(item: MatchedFile) -> dict:
    top1 = item.candidates[0] if item.candidates else None
    top2 = item.candidates[1] if len(item.candidates) > 1 else None
    return {
        "id": item.id,
        "category": item.category,
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
    by_main: dict[int, list[MatchedFile]] = defaultdict(list)
    category_source_counts: Counter[str] = Counter(item.category for item in files)
    unmatched_files = [item for item in files if item.matched_main_id is None]
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
        ordered_categories = list(EXPECTED_CATEGORIES)
        ordered_categories.extend(sorted(set(by_category) - set(ordered_categories)))
        unmatched_categories = [
            category for category in EXPECTED_CATEGORIES
            if category_source_counts[category] and not by_category.get(category)
        ]
        missing = [
            category for category in EXPECTED_CATEGORIES
            if not category_source_counts[category]
        ]
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
                    "name": category,
                    "missing": not category_source_counts[category],
                    "unmatched": bool(category_source_counts[category]) and not bool(by_category.get(category)),
                    "files": [_file_dict(item) for item in by_category.get(category, [])],
                }
                for category in ordered_categories
            ],
            "missing_categories": missing,
            "unmatched_categories": unmatched_categories,
            "minimum_similarity": min(score_values) if score_values else None,
            "flags": flags,
        })
    summary["unmatched_files"] = len(unmatched_files)
    summary_payload = {
        **dict(summary),
        "unmatched_by_category": dict(Counter(item.category for item in unmatched_files)),
        "source_by_category": dict(category_source_counts),
    }
    return groups, {
        "summary": summary_payload,
        "unmatched_files": [_file_dict(item) for item in unmatched_files],
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
    items = db.scalars(select(MatchedFile).where(MatchedFile.matched_main_id == main.id)).all()
    now = datetime.now(timezone.utc)
    for item in items:
        item.review_status = "CONFIRMED"
        item.reviewed_at = now
        item.manual_product_id = main.id
        item.operator = "LOCAL"
        item.confirmed_at = now
        item.status = "MANUAL_CONFIRMED"
    db.commit()
    return {"confirmed": len(items)}


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
        raise HTTPException(409, "该商品在此分类中已经有一张图片，请先取消原归属")
    now = datetime.now(timezone.utc)
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
    return _file_dict(item)


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
    return _file_dict(item)


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
    return _file_dict(item)


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
def search_mains(batch_id: int, q: str = Query("", max_length=255), db: Session = Depends(get_db)):
    query = select(MainImage).where(MainImage.batch_id == batch_id)
    if q:
        query = query.where(MainImage.basename.contains(q))
    mains = db.scalars(query.order_by(MainImage.basename).limit(30)).all()
    return [{
        "id": main.id,
        "basename": main.basename,
        "filename": main.filename,
        "thumbnail_url": f"/api/main-images/{main.id}/thumbnail",
    } for main in mains]


def _file_response(path: str | None) -> FileResponse:
    if not path or not Path(path).is_file():
        raise HTTPException(404, "预览文件不存在")
    return FileResponse(path)


@app.get("/api/main-images/{main_id}/thumbnail")
def main_thumbnail(main_id: int, db: Session = Depends(get_db)):
    main = db.get(MainImage, main_id)
    if not main:
        raise HTTPException(404, "主图不存在")
    return _file_response(main.thumbnail_path)


@app.get("/api/files/{file_id}/thumbnail")
def file_thumbnail(file_id: int, db: Session = Depends(get_db)):
    item = db.get(MatchedFile, file_id)
    if not item:
        raise HTTPException(404, "文件不存在")
    return _file_response(item.thumbnail_path)


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
