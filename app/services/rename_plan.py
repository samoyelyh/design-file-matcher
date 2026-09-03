from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import is_ignored_image_category
from app.models import Batch, MatchedFile, RenameOperation, RenamePlan


def _norm(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def _cleanup_design_matcher(batch: Batch) -> dict:
    root = Path(batch.root_path).resolve()
    matcher_root = (root / ".design_matcher").resolve()
    if matcher_root.parent != root or matcher_root.name != ".design_matcher":
        return {"removed": False, "error": "临时目录路径无效"}
    if not matcher_root.exists():
        return {"removed": True, "existed": False}
    try:
        shutil.rmtree(matcher_root)
    except OSError as exc:
        return {
            "removed": False,
            "error": f"临时文件清理失败: {exc}",
        }
    return {"removed": True, "existed": True}


class RenamePlanService:
    ready_match_statuses = {"MATCHED", "MANUAL_CONFIRMED"}

    def generate(self, db: Session, batch_id: int) -> dict:
        batch = db.get(Batch, batch_id)
        if not batch:
            raise ValueError("批次不存在")
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id
        ).order_by(MatchedFile.id)))
        plan_group = str(uuid.uuid4())
        plans: list[RenamePlan] = []
        target_groups: dict[str, list[RenamePlan]] = {}
        for item in files:
            if is_ignored_image_category(item.category):
                continue
            source = Path(item.current_path)
            target = (
                source.parent / f"{item.matched_main_basename}{item.extension}"
                if item.matched_main_id and item.matched_main_basename else None
            )
            status = "READY"
            error = None
            if item.status == "EXTRA_IGNORED":
                status, error = "IGNORED", "分类文件数量超过主图数量，保持原名"
            elif not source.is_file():
                status, error = "FILE_MISSING", "源文件当前不存在"
            elif item.status not in self.ready_match_statuses:
                status, error = "BLOCKED", f"匹配状态 {item.status} 尚未人工确认"
            elif target is None:
                status, error = "BLOCKED", "没有有效匹配商品"
            elif _norm(source) == _norm(target):
                status = "NOOP"
            elif target.exists():
                status, error = "RENAME_CONFLICT", "目标文件已经存在，禁止覆盖"
            plan = RenamePlan(
                plan_group=plan_group,
                batch_id=batch_id,
                source_file_id=item.id,
                source_path=str(source),
                original_filename=item.original_filename,
                matched_main_image_id=item.matched_main_id,
                target_filename=target.name if target else None,
                target_path=str(target) if target else None,
                category=item.category,
                match_status=item.status,
                rename_status=status,
                error_message=error,
            )
            plans.append(plan)
            if target:
                target_groups.setdefault(_norm(target), []).append(plan)

        for group in target_groups.values():
            if len(group) > 1:
                for plan in group:
                    plan.rename_status = "RENAME_CONFLICT"
                    plan.error_message = "多个源文件计划写入同一个目标路径"
        db.add_all(plans)
        db.commit()
        return {
            "batch_id": batch_id,
            "plan_group": plan_group,
            "counts": dict(Counter(plan.rename_status for plan in plans)),
            "plans": [self.to_dict(plan) for plan in plans],
        }

    def latest(self, db: Session, batch_id: int) -> dict | None:
        latest = db.scalar(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id
        ).order_by(RenamePlan.id.desc()))
        if not latest:
            return None
        plans = list(db.scalars(select(RenamePlan).where(
            RenamePlan.plan_group == latest.plan_group
        ).order_by(RenamePlan.id)))
        plans = [
            plan for plan in plans
            if not is_ignored_image_category(plan.category)
        ]
        return {
            "batch_id": batch_id,
            "plan_group": latest.plan_group,
            "counts": dict(Counter(plan.rename_status for plan in plans)),
            "plans": [self.to_dict(plan) for plan in plans],
        }

    def execute(self, db: Session, batch_id: int, plan_group: str) -> dict:
        plans = list(db.scalars(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id,
            RenamePlan.plan_group == plan_group,
        ).order_by(RenamePlan.id)))
        if not plans:
            raise ValueError("Rename Plan不存在")
        for plan in plans:
            if is_ignored_image_category(plan.category):
                plan.rename_status = "IGNORED"
                plan.error_message = "白底图不参与匹配，保持原名"
        blockers = [
            plan for plan in plans
            if (
                not is_ignored_image_category(plan.category)
                and plan.rename_status not in {"READY", "NOOP", "RENAMED", "IGNORED"}
            )
        ]
        if blockers:
            raise ValueError(f"仍有 {len(blockers)} 项未确认或冲突，禁止执行整批改名")
        validated: list[tuple[RenamePlan, MatchedFile, Path, Path | None]] = []
        for plan in plans:
            if (
                is_ignored_image_category(plan.category)
                or plan.rename_status in {"RENAMED", "IGNORED"}
            ):
                continue
            item = db.get(MatchedFile, plan.source_file_id)
            source = Path(plan.source_path)
            target = Path(plan.target_path) if plan.target_path else None
            error = None
            if not item or item.current_path != plan.source_path or not source.is_file():
                error = "源路径已变化或文件不存在"
            elif plan.rename_status != "NOOP" and (target is None or target.exists()):
                error = "执行前发现目标路径冲突"
            if error:
                plan.rename_status = "RENAME_CONFLICT"
                plan.error_message = error
                db.commit()
                raise ValueError(error)
            validated.append((plan, item, source, target))

        now = datetime.now(timezone.utc)
        for plan, item, source, target in validated:
            if plan.rename_status == "NOOP":
                plan.rename_status = "RENAMED"
                plan.executed_at = now
                continue
            assert target is not None
            os.rename(source, target)
            item.current_path = str(target)
            item.current_filename = target.name
            item.renamed_at = now
            plan.rename_status = "RENAMED"
            plan.executed_at = now
            db.add(RenameOperation(
                matched_file_id=item.id,
                operation="RENAME_PLAN",
                from_path=str(source),
                to_path=str(target),
                success=True,
                message=f"plan_group={plan_group}",
            ))
        db.commit()
        result = self.latest(db, batch_id) or {}
        result_plans = result.get("plans", [])
        if result_plans and all(
            plan["rename_status"] in {"RENAMED", "IGNORED"}
            for plan in result_plans
        ):
            batch = db.get(Batch, batch_id)
            if batch:
                result["workspace_cleanup"] = _cleanup_design_matcher(batch)
        return result

    @staticmethod
    def to_dict(plan: RenamePlan) -> dict:
        return {
            "id": plan.id,
            "plan_group": plan.plan_group,
            "source_file_id": plan.source_file_id,
            "source_path": plan.source_path,
            "original_filename": plan.original_filename,
            "matched_main_image_id": plan.matched_main_image_id,
            "target_filename": plan.target_filename,
            "target_path": plan.target_path,
            "category": plan.category,
            "match_status": plan.match_status,
            "rename_status": plan.rename_status,
            "error_message": plan.error_message,
            "created_at": plan.created_at,
            "executed_at": plan.executed_at,
        }
