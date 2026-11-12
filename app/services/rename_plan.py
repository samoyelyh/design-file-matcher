from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import is_excluded_file_category
from app.models import Batch, MatchedFile, RenameOperation, RenamePlan
from app.services.review_scope import conflict_name_marks, ignored_review_categories, is_review_confirmed


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

    @staticmethod
    def marked_filename(item: MatchedFile, number: int) -> str:
        return f"A-{Path(item.original_filename).stem}-冲突图片{number}{item.extension}"

    def describe(self, db: Session, batch: Batch, payloads: list[dict]) -> list[dict]:
        marks = conflict_name_marks(batch)
        history = json.loads(batch.match_config_snapshot or "{}").get("conflict_mark_originals", {})
        ignored = ignored_review_categories(batch)
        items = {item.id: item for item in db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch.id))}
        normal_groups: dict[str, list[dict]] = {}
        for payload in payloads:
            item = items.get(payload["source_file_id"])
            source = Path(item.current_path) if item and payload["rename_status"] == "RENAMED" else Path(payload["source_path"])
            payload["source_path"] = str(source)
            payload.update(current_filename=source.name, source_directory=str(source.parent),
                           conflict_marked=payload["source_file_id"] in marks,
                           conflict_mark_number=marks.get(payload["source_file_id"]), can_mark_conflict=False,
                           category_ignored=bool(item and item.category in ignored))
            try:
                details = json.loads(item.match_details_json or "{}") if item else {}
            except (TypeError, ValueError):
                details = {}
            if not isinstance(details, dict):
                details = {}
            payload["matching_position_conflict"] = bool(item and (item.is_conflict or details.get("duplicate_target_ids")))
            if item and is_review_confirmed(item) and item.status != "EXTRA_IGNORED":
                normal_target = source.parent / f"{item.matched_main_basename}{item.extension}"
                payload["normal_target_path"] = str(normal_target)
                normal_groups.setdefault(_norm(normal_target), []).append(payload)
        for payload in payloads:
            peers = normal_groups.get(_norm(Path(payload["normal_target_path"])), []) if payload.get("normal_target_path") else []
            related = len(peers) > 1 or payload["matching_position_conflict"] or payload["rename_status"] == "RENAME_CONFLICT" or payload["conflict_marked"] or str(payload["source_file_id"]) in history
            payload["normal_position_conflict"] = len(peers) > 1
            payload["can_mark_conflict"] = related and payload["rename_status"] in {"READY", "NOOP", "RENAMED", "RENAME_CONFLICT", "SKIPPED"}
            payload["conflict_peers"] = [{"source_file_id": row["source_file_id"], "current_filename": row["current_filename"],
                                           "source_path": row["source_path"]} for row in peers if row is not payload]
        return payloads

    def generate(self, db: Session, batch_id: int) -> dict:
        batch = db.get(Batch, batch_id)
        if not batch:
            raise ValueError("批次不存在")
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id
        ).order_by(MatchedFile.id)))
        ignored = ignored_review_categories(batch)
        marks = conflict_name_marks(batch)
        plan_group = str(uuid.uuid4())
        plans: list[RenamePlan] = []
        target_groups: dict[str, list[RenamePlan]] = {}
        recovered_sources = 0
        for item in files:
            if is_excluded_file_category(item.category):
                continue
            source = Path(item.current_path)
            original = Path(item.original_path)
            eligible = item.category not in ignored and (is_review_confirmed(item) or item.id in marks) and item.status != "EXTRA_IGNORED"
            if eligible and not source.is_file() and original.is_file():
                missing_path = item.current_path
                source = original
                item.current_path = str(original)
                item.current_filename = original.name
                item.renamed_at = None
                recovered_sources += 1
                db.add(RenameOperation(
                    matched_file_id=item.id,
                    operation="RECOVER_FILE",
                    from_path=missing_path,
                    to_path=str(original),
                    success=True,
                    message="生成命名预览时找回已放回文件；保留原匹配与审核数据",
                ))
            target = (
                source.parent / (self.marked_filename(item, marks[item.id]) if item.id in marks
                                 else f"{item.matched_main_basename}{item.extension}")
                if item.id in marks or (item.matched_main_id and item.matched_main_basename) else None
            )
            status = "READY"
            error = None
            if item.category in ignored or item.status == "EXTRA_IGNORED":
                status, error = "IGNORED", "本批次忽略该类型或多余图片，保持原名"
            elif item.id not in marks and not is_review_confirmed(item):
                status, error = "SKIPPED", "尚未确认有效归属，本次保持原名；可稍后确认再改名"
            elif not source.is_file():
                status, error = "FILE_MISSING", "源文件当前不存在"
            elif item.id not in marks and item.status not in self.ready_match_statuses:
                status, error = "BLOCKED", f"匹配状态 {item.status} 尚未人工确认"
            elif target is None:
                status, error = "BLOCKED", "没有有效匹配商品"
            elif _norm(source) == _norm(target):
                status = "NOOP"
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
            if target and status in {"READY", "NOOP"}:
                target_groups.setdefault(_norm(target), []).append(plan)

        for group in target_groups.values():
            if len(group) > 1:
                for plan in group:
                    plan.rename_status = "RENAME_CONFLICT"
                    plan.error_message = "多个源文件计划写入同一个目标路径"

        # A target that is another READY source is a safe rename cycle, not an
        # overwrite conflict. Execution stages all sources to temporary names
        # before any final target is written.
        movable_sources = {
            _norm(Path(plan.source_path))
            for plan in plans
            if plan.rename_status == "READY"
        }
        for plan in plans:
            if plan.rename_status != "READY" or not plan.target_path:
                continue
            target = Path(plan.target_path)
            if target.exists() and _norm(target) not in movable_sources:
                plan.rename_status = "RENAME_CONFLICT"
                plan.error_message = "目标位置已有本批次之外的文件，禁止覆盖"
        db.add_all(plans)
        db.commit()
        return {
            "batch_id": batch_id,
            "plan_group": plan_group,
            "counts": dict(Counter(plan.rename_status for plan in plans)),
            "plans": self.describe(db, batch, [self.to_dict(plan) for plan in plans]),
            "recovered_source_count": recovered_sources,
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
            if not is_excluded_file_category(plan.category)
        ]
        batch = db.get(Batch, batch_id)
        ignored = ignored_review_categories(batch)
        marks = conflict_name_marks(batch)
        payloads = []
        for plan in plans:
            payload = self.to_dict(plan)
            item = db.get(MatchedFile, plan.source_file_id)
            # Old previews may predate confirmation-only naming or a scope change.
            if plan.rename_status != "RENAMED" and item:
                if item.category in ignored or item.status == "EXTRA_IGNORED":
                    payload.update(rename_status="IGNORED", error_message="本批次忽略，保持原名")
                elif item.id not in marks and not is_review_confirmed(item):
                    payload.update(rename_status="SKIPPED", error_message="尚未确认有效归属，本次保持原名")
            payloads.append(payload)
        return {
            "batch_id": batch_id,
            "plan_group": latest.plan_group,
            "counts": dict(Counter(plan["rename_status"] for plan in payloads)),
            "plans": self.describe(db, batch, payloads),
        }

    def execute(self, db: Session, batch_id: int, plan_group: str) -> dict:
        plans = list(db.scalars(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id,
            RenamePlan.plan_group == plan_group,
        ).order_by(RenamePlan.id)))
        if not plans:
            raise ValueError("Rename Plan不存在")
        batch = db.get(Batch, batch_id)
        if batch.status in {"QUEUED", "MATCHING"}:
            raise ValueError("匹配正在进行，请完成后再改名")
        latest = db.scalar(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id
        ).order_by(RenamePlan.id.desc()))
        if latest.plan_group != plan_group:
            raise ValueError("命名预览已更新，请使用最新预览")
        ignored = ignored_review_categories(batch)
        marks = conflict_name_marks(batch)
        for plan in plans:
            if plan.rename_status == "RENAMED":
                continue
            item = db.get(MatchedFile, plan.source_file_id)
            if is_excluded_file_category(plan.category) or plan.category in ignored or (item and item.status == "EXTRA_IGNORED"):
                plan.rename_status = "IGNORED"
                plan.error_message = "该分类不参与处理，保持原名"
            elif item and item.id not in marks and not is_review_confirmed(item):
                plan.rename_status = "SKIPPED"
                plan.error_message = "尚未确认有效归属，本次保持原名"
        blockers = [
            plan for plan in plans
            if (
                not is_excluded_file_category(plan.category)
                and plan.rename_status not in {"READY", "NOOP", "RENAMED", "IGNORED", "SKIPPED"}
            )
        ]
        if blockers:
            details = "；".join(f"{plan.category}/{Path(plan.source_path).name}：{plan.error_message}（{plan.source_path}）" for plan in blockers)
            raise ValueError(f"已确认项目中仍有 {len(blockers)} 项文件缺失或命名冲突：{details}")
        movable_sources = {
            _norm(Path(plan.source_path))
            for plan in plans
            if plan.rename_status == "READY"
        }
        validated: list[tuple[RenamePlan, MatchedFile, Path, Path | None]] = []
        for plan in plans:
            if (
                is_excluded_file_category(plan.category)
                or plan.rename_status in {"RENAMED", "IGNORED", "SKIPPED"}
            ):
                continue
            item = db.get(MatchedFile, plan.source_file_id)
            source = Path(plan.source_path)
            target = Path(plan.target_path) if plan.target_path else None
            error = None
            if not item or item.current_path != plan.source_path or not source.is_file():
                error = "源路径已变化或文件不存在"
            elif (
                item.matched_main_id != plan.matched_main_image_id
                or target is None
                or _norm(target) != _norm(source.parent / (self.marked_filename(item, marks[item.id]) if item.id in marks
                                                          else f"{item.matched_main_basename}{item.extension}"))
            ):
                error = "审核归属已变化，请重新预览命名结果"
            elif (
                plan.rename_status != "NOOP"
                and target is not None
                and target.exists()
                and _norm(target) not in movable_sources
            ):
                error = "执行前发现目标位置已有外部文件，禁止覆盖"
            if error:
                plan.rename_status = "RENAME_CONFLICT"
                plan.error_message = error
                db.commit()
                raise ValueError(error)
            validated.append((plan, item, source, target))

        staged: list[tuple[RenamePlan, MatchedFile, Path, Path, Path]] = []
        completed: list[tuple[RenamePlan, MatchedFile, Path, Path, Path]] = []
        try:
            for plan, item, source, target in validated:
                if plan.rename_status == "NOOP":
                    continue
                assert target is not None
                temporary = source.with_name(
                    f".{source.name}.design_matcher_tmp_{uuid.uuid4().hex}"
                )
                if temporary.exists():
                    raise OSError(f"临时路径已存在: {temporary}")
                os.rename(source, temporary)
                staged.append((plan, item, source, target, temporary))

            for _plan, _item, _source, target, _temporary in staged:
                if target.exists():
                    raise OSError(f"目标位置在执行期间被占用: {target}")

            for entry in staged:
                _plan, _item, _source, target, temporary = entry
                os.rename(temporary, target)
                completed.append(entry)
        except OSError as exc:
            # First move completed targets back to their private temporary
            # paths, then restore every original source path.
            for _plan, _item, _source, target, temporary in reversed(completed):
                try:
                    if target.exists() and not temporary.exists():
                        os.rename(target, temporary)
                except OSError:
                    pass
            for _plan, _item, source, _target, temporary in reversed(staged):
                try:
                    if temporary.exists() and not source.exists():
                        os.rename(temporary, source)
                except OSError:
                    pass
            raise ValueError(f"重命名失败，已尝试恢复原文件名: {exc}") from exc

        now = datetime.now(timezone.utc)
        for plan, item, source, target in validated:
            if plan.rename_status == "NOOP":
                plan.rename_status = "RENAMED"
                plan.executed_at = now
                continue
            assert target is not None
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
        ) and not ignored:
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
