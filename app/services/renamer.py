from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import MatchedFile, RenameOperation


def _now():
    return datetime.now(timezone.utc)


def _desired_path(item: MatchedFile) -> Path:
    return Path(item.current_path).parent / f"{item.matched_main_basename}{item.extension}"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
    except OSError:
        return os.path.normcase(str(left.absolute())) == os.path.normcase(str(right.absolute()))


def _log(db: Session, item: MatchedFile, operation: str, source: Path, target: Path,
         success: bool, message: str | None = None) -> None:
    db.add(RenameOperation(
        matched_file_id=item.id,
        operation=operation,
        from_path=str(source),
        to_path=str(target),
        success=success,
        message=message,
    ))


def _rename_one(db: Session, item: MatchedFile, operation: str = "AUTO_RENAME") -> bool:
    source = Path(item.current_path)
    target = _desired_path(item)
    if _same_path(source, target):
        item.current_filename = target.name
        item.current_path = str(target)
        item.status = "LOW_CONFIDENCE" if item.is_low_confidence else "NORMAL"
        return True
    if not source.exists():
        item.status = "ERROR"
        _log(db, item, operation, source, target, False, "源文件不存在")
        return False
    if target.exists():
        item.is_conflict = True
        item.status = "CONFLICT"
        _log(db, item, operation, source, target, False, "目标文件已存在，未覆盖")
        return False
    try:
        os.rename(source, target)
    except OSError as exc:
        item.status = "ERROR"
        _log(db, item, operation, source, target, False, str(exc))
        return False
    item.current_filename = target.name
    item.current_path = str(target)
    item.renamed_at = _now()
    item.status = "LOW_CONFIDENCE" if item.is_low_confidence else "NORMAL"
    _log(db, item, operation, source, target, True)
    return True


def reconcile_category(db: Session, batch_id: int, category: str,
                       operation: str = "AUTO_RENAME") -> None:
    items = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == batch_id,
        MatchedFile.category == category,
    )))
    active = [i for i in items if i.matched_main_id is not None and i.status != "RESTORED"]
    groups: dict[int, list[MatchedFile]] = defaultdict(list)
    for item in active:
        item.is_conflict = False
        groups[item.matched_main_id].append(item)

    for group in groups.values():
        if len(group) > 1:
            for item in group:
                item.is_conflict = True
                item.status = "CONFLICT"
                source = Path(item.current_path)
                _log(db, item, operation, source, _desired_path(item), False,
                     "同一主图和分类匹配到多个文件")
        else:
            _rename_one(db, group[0], operation)
    db.commit()


def reconcile_batch(db: Session, batch_id: int) -> None:
    categories = db.scalars(select(MatchedFile.category).where(
        MatchedFile.batch_id == batch_id
    ).distinct()).all()
    for category in categories:
        reconcile_category(db, batch_id, category)


def restore_original(db: Session, item: MatchedFile) -> None:
    source = Path(item.current_path)
    target = Path(item.original_path)
    if _same_path(source, target):
        item.status = "RESTORED"
        item.review_status = "CORRECTED"
        item.reviewed_at = _now()
        db.commit()
        return
    if not source.exists():
        raise FileNotFoundError("当前文件不存在")
    if target.exists():
        item.is_conflict = True
        item.status = "CONFLICT"
        _log(db, item, "RESTORE", source, target, False, "原文件名已被占用，未覆盖")
        db.commit()
        raise FileExistsError("原文件名已被占用，未覆盖")
    os.rename(source, target)
    item.current_filename = target.name
    item.current_path = str(target)
    item.is_conflict = False
    item.status = "RESTORED"
    item.review_status = "CORRECTED"
    item.reviewed_at = _now()
    _log(db, item, "RESTORE", source, target, True)
    db.commit()
