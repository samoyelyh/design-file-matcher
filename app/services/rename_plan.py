from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import delete, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import is_excluded_file_category
from app.models import (
    Batch, DesignCode, DesignCodingPlan, DesignPackageInfo, MainCodeRename, MainImage, MainImageContent,
    MatchedFile, RenameOperation, RenamePlan,
)
from app.services import design_coding, ticket_abbreviations, ticket_coding
from app.services.code_texts import publish_code_texts
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

    def describe(self, db: Session, batch: Batch, payloads: list[dict], stems=None) -> list[dict]:
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
                stem = (stems or {}).get(item.matched_main_id, item.matched_main_basename)
                normal_target = source.parent / f"{stem}{item.extension}"
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

    def generate(self, db: Session, batch_id: int, *, with_codes: bool = False) -> dict:
        if with_codes:
            with design_coding.lock:
                previous = self.coding_plan(db, batch_id)
                if previous:
                    saved = json.loads(previous.summary_json)
                    work = saved.get("ticket")
                    if work and not work.get("complete") and (work["receipts"] or work["pending"] is not None):
                        return self._ticket_preview_payload(previous, saved)
                if ticket_coding.api_url():
                    return self._preview_ticket(db, batch_id)
                if not design_coding.catalog_path():
                    raise ValueError("编码目录尚未配置。可以继续使用普通更名。")
                return self._generate(db, batch_id, with_codes=True)
        return self._generate(db, batch_id, with_codes=False)

    def _generate(self, db: Session, batch_id: int, *, with_codes: bool, ticket_plan=None) -> dict:
        batch = db.get(Batch, batch_id)
        if not batch:
            raise ValueError("批次不存在")
        if with_codes and batch.status in {"QUEUED", "MATCHING"}:
            raise ValueError("请在匹配完成后预览编码改名")
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch_id
        ).order_by(MatchedFile.id)))
        ignored = ignored_review_categories(batch)
        marks = conflict_name_marks(batch)
        plan_group = ticket_plan.plan_group if ticket_plan else str(uuid.uuid4())
        code_plans = []
        coding = None
        stems = {}
        if with_codes:
            mains = list(db.scalars(select(MainImage).where(MainImage.batch_id == batch_id).order_by(MainImage.id)))
            if ticket_plan:
                coding = json.loads(ticket_plan.summary_json)
                work = coding["ticket"]
                codes = {entry["main_image_id"]: ticket_coding.apply_receipt(db, work, entry)
                         for entry in work["mains"]}
                for row in coding["rows"]:
                    if code := codes.get(row["main_image_id"]):
                        row.update(design_coding.proposal(code))
                ticket_plan.summary_json = json.dumps(coding, ensure_ascii=False)
            else:
                codes, coding = design_coding.prepare(db, batch, mains)
            root = Path(batch.root_path).resolve()
            for main in mains:
                code = codes.get(main.id)
                if not code:
                    continue
                source = Path(main.file_path)
                target = source.with_name(code.target_stem + source.suffix)
                within_root = source.resolve().is_relative_to(root)
                code_plans.append(MainCodeRename(
                    plan_group=plan_group, batch_id=batch_id, main_image_id=main.id,
                    source_path=str(source), target_path=str(target), source_sha256=code.source_sha256,
                    rename_status=('NOOP' if _norm(source) == _norm(target) else 'READY') if within_root else 'BLOCKED',
                    error_message=None if within_root else '主图路径超出所选设计包。',
                ))
                stems[main.id] = code.target_stem
            if not ticket_plan:
                db.add(DesignCodingPlan(plan_group=plan_group, batch_id=batch_id, summary_json=json.dumps(coding, ensure_ascii=False)))
            db.add_all(code_plans)
        plans: list[RenamePlan] = []
        target_groups = {}
        for plan in code_plans:
            if plan.rename_status in {'READY', 'NOOP'}:
                target_groups.setdefault(_norm(Path(plan.target_path)), []).append(plan)
        recovered_sources = 0
        for item in files:
            if is_excluded_file_category(item.category):
                continue
            source = Path(item.current_path)
            original = Path(item.original_path)
            has_code = not with_codes or item.matched_main_id in stems or (not ticket_plan and item.id in marks)
            eligible = has_code and item.category not in ignored and (is_review_confirmed(item) or item.id in marks) and item.status != "EXTRA_IGNORED"
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
                                 else f"{stems.get(item.matched_main_id, item.matched_main_basename)}{item.extension}")
                if has_code and (item.id in marks or (item.matched_main_id and item.matched_main_basename)) else None
            )
            status = "READY"
            error = None
            if item.category in ignored or item.status == "EXTRA_IGNORED":
                status, error = "IGNORED", "本批次忽略该类型或多余图片，保持原名"
            elif item.id not in marks and not is_review_confirmed(item):
                status, error = "SKIPPED", "尚未确认有效归属，本次保持原名；可稍后确认再改名"
            elif not has_code:
                status, error = "SKIPPED", "关联主图的编码待核对"
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
            for plan in [*plans, *code_plans]
            if plan.rename_status == "READY"
        }
        for plan in [*plans, *code_plans]:
            if plan.rename_status != "READY" or not plan.target_path:
                continue
            target = Path(plan.target_path)
            if target.exists() and _norm(target) not in movable_sources:
                plan.rename_status = "RENAME_CONFLICT"
                plan.error_message = "目标位置已有本批次之外的文件，禁止覆盖"
        db.add_all(plans)
        db.commit()
        result = {
            "batch_id": batch_id,
            "plan_group": plan_group,
            "counts": dict(Counter(plan.rename_status for plan in [*plans, *code_plans])),
            "plans": self.describe(db, batch, [self.to_dict(plan) for plan in plans], stems),
            "recovered_source_count": recovered_sources,
        }
        if coding is not None:
            result.update(coding=self._public_coding(coding), main_plans=[self.main_dict(plan) for plan in code_plans])
        return result

    def _ticket_files(self, db, batch, main_ids):
        """Snapshot exactly the associated assets whose names confirmation can change."""
        ignored = ignored_review_categories(batch)
        marks = conflict_name_marks(batch)
        root = Path(batch.root_path).resolve()
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == batch.id, MatchedFile.matched_main_id.in_(main_ids)
        ).order_by(MatchedFile.id)))
        positions = set()
        selected = []
        for item in files:
            mark = marks.get(item.id)
            if (item.category in ignored or is_excluded_file_category(item.category)
                    or item.status == "EXTRA_IGNORED" or (not mark and not is_review_confirmed(item))):
                continue
            path = Path(item.current_path)
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"关联文件「{path.name}」不存在或超出设计包，请先核对。")
            if not mark and (item.status not in self.ready_match_statuses or not item.matched_main_basename):
                raise ValueError(f"关联文件「{path.name}」的审核归属需核对。")
            position = (_norm(path.parent), self.marked_filename(item, mark).casefold()
                        if mark else (item.matched_main_id, item.extension.casefold()))
            if position in positions:
                raise ValueError(f"关联文件「{path.name}」与另一文件占用相同的命名位置，请先处理冲突。")
            positions.add(position)
            selected.append([item.id, item.matched_main_id, item.current_path, item.extension, mark])
        return selected

    def _validate_ticket_sources(self, db, batch, work):
        root = Path(batch.root_path).resolve()
        paths = set()
        for item in work["mains"]:
            main = db.get(MainImage, item["main_image_id"])
            content = db.get(MainImageContent, item["main_image_id"])
            path = Path(item["source_path"])
            try:
                valid = (main is not None and main.batch_id == batch.id and main.file_path == str(path)
                         and path.resolve().is_relative_to(root) and path.is_file()
                         and _norm(path) not in paths and design_coding.file_hash(path) == item["source_sha256"]
                         and design_coding.content_signature(content) == item["content_signature"])
            except OSError:
                valid = False
            if not valid:
                raise ValueError(f"主图「{path.name}」或审核结果已变化，请核对命名预览。")
            paths.add(_norm(path))
        assets = self._ticket_files(db, batch, [item["main_image_id"] for item in work["mains"]])
        if any(_norm(Path(row[2])) in paths for row in assets) or len({_norm(Path(row[2])) for row in assets}) != len(assets):
            raise ValueError("编码涉及重复文件，请先核对图片归属。")
        return assets

    @staticmethod
    def _public_coding(coding):
        return {key: value for key, value in coding.items() if key != "ticket"}

    def _ticket_preview_payload(self, plan, coding):
        work = coding["ticket"]
        public = self._public_coding(coding)
        count = len(work["mains"])
        remaining = count - len(work["receipts"])
        public.update(ticket_quantity=remaining, ticket_group_count=count, ticket_user_id=work["user_id"],
                      ticket_file_count=count + len(work.get("assets", [])),
                      ticket_ready=bool(count and not coding.get("error") and work["pending"] is None),
                      ticket_theme_create_enabled=bool(ticket_abbreviations.enabled() and not work["receipts"]
                                                       and work["pending"] is None))
        if work["pending"] is not None:
            public["error"] = "有一笔领号结果需要核对。系统保留了已返回的号码，核对前暂停继续领号。"
        by_id = {entry["main_image_id"]: entry for entry in work["mains"]}
        public["rows"] = [dict(row) for row in coding["rows"]]
        for row in public["rows"]:
            code = work["receipts"].get(str(row["main_image_id"]))
            if code:
                item = by_id[row["main_image_id"]]
                row.update(artist_code=item["subject_name"] + code, operation_code=f'{item["abbreviation"]}-{code}',
                           decision="TICKET_ALLOCATED")
        return {"batch_id": plan.batch_id, "plan_group": plan.plan_group, "counts": {}, "plans": [], "coding": public}

    def _preview_ticket(self, db, batch_id):
        batch = db.get(Batch, batch_id)
        if batch is None:
            raise ValueError("批次不存在")
        if batch.status in {"QUEUED", "MATCHING"}:
            raise ValueError("请在匹配完成后预览编码改名")
        mains = list(db.scalars(select(MainImage).where(MainImage.batch_id == batch_id).order_by(MainImage.id)))
        coding = ticket_coding.mapping(db, batch, mains)
        work = coding["ticket"]
        if work["mains"] and not coding.get("error"):
            try:
                work["assets"] = self._validate_ticket_sources(db, batch, work)
            except ValueError as exc:
                coding["error"] = str(exc)
        plan = DesignCodingPlan(plan_group=str(uuid.uuid4()), batch_id=batch_id,
                                summary_json=json.dumps(coding, ensure_ascii=False))
        db.add(plan)
        db.commit()
        return self._ticket_preview_payload(plan, coding)

    def complete_ticket_abbreviations(self, db, batch_id, plan_group):
        with design_coding.lock:
            batch = db.get(Batch, batch_id)
            plan = self.coding_plan(db, batch_id)
            if not batch or not plan or plan.plan_group != plan_group or batch.status in {"QUEUED", "MATCHING"}:
                raise ValueError("当前编码预览已变化，请重新预览。")
            coding = json.loads(plan.summary_json)
            work = coding.get("ticket") or {}
            if (work.get("api_url") != ticket_coding.api_url() or not work.get("theme_type_id")
                    or work.get("receipts") or work.get("pending") is not None):
                raise ValueError("当前预览无法补全缩写，请先核对主题或已领取的号码。")
            tokens = design_coding.package_tokens(batch.root_path)
            profile = db.get(DesignPackageInfo, batch.id)
            manual = json.loads(profile.manual_json or "{}") if profile else {}
            product_name = manual.get("product_name") or tokens["category"]
            if tokens != coding["tokens"] or ticket_coding.name_key(product_name) != ticket_coding.name_key(
                coding["ticket_catalog"]["product_name"]
            ):
                raise ValueError("设计包分类已变化，请重新预览。")
            root = Path(batch.root_path).resolve()
            names = []
            for item in work.get("new_themes", []):
                main = db.get(MainImage, item["main_image_id"])
                recognition = db.get(MainImageContent, item["main_image_id"])
                if not main or main.batch_id != batch_id or design_coding.content_signature(recognition) != item["signature"]:
                    raise ValueError("主图识别结果已变化，请重新预览。")
                path = Path(main.file_path)
                try:
                    valid = path.resolve().is_relative_to(root) and design_coding.file_hash(path) == item["source_sha256"]
                except OSError:
                    valid = False
                if not valid:
                    raise ValueError("主图文件已变化，请重新预览。")
                names.append(item["name"])
            if not names:
                raise ValueError("这次预览没有适合自动补全缩写的小主题。")
            result = ticket_abbreviations.fill(work["api_url"], work["product_id"], work["theme_type_id"], names)
            return {"results": result, "preview": self._preview_ticket(db, batch_id)}

    @staticmethod
    def _save_ticket_state(db, plan, coding):
        latest_id = (select(DesignCodingPlan.id).where(DesignCodingPlan.batch_id == plan.batch_id)
                     .order_by(DesignCodingPlan.id.desc()).limit(1).scalar_subquery())
        result = db.execute(update(DesignCodingPlan).where(
            DesignCodingPlan.id == plan.id, DesignCodingPlan.summary_json == plan.summary_json,
            DesignCodingPlan.id == latest_id,
        ).values(summary_json=json.dumps(coding, ensure_ascii=False)).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            db.rollback()
            raise ValueError("编码预览或领号状态已更新，请刷新后核对。")
        db.commit()
        db.refresh(plan)

    def _prepare_ticket_execution(self, db, batch, plan, coding):
        work = coding["ticket"]
        if work["pending"] is not None:
            raise ValueError("上一笔领号结果未确认，请先在领号系统核对。")
        if coding.get("error") or not work["mains"]:
            raise ValueError(coding.get("error") or "没有已核对的主图编码。")
        if work["api_url"] != ticket_coding.api_url() or work["user_id"] != ticket_coding.operator_id():
            raise ValueError("领号接口或领号用户已变更，请核对后重新预览。")
        if self._validate_ticket_sources(db, batch, work) != work["assets"]:
            raise ValueError("关联图片的审核范围已变化，请核对后重新预览。")
        for item in work["mains"]:
            main_id = str(item["main_image_id"])
            if main_id in work["receipts"]:
                continue
            work["pending"] = main_id
            # Commit the intent before POST. A lost response or process restart
            # must never trigger a second claim for the same image.
            self._save_ticket_state(db, plan, coding)
            try:
                number = ticket_coding.claim_one(work, item, coding["source_code_prefix"])
            except (ValueError, httpx.HTTPError) as exc:
                raise ValueError("领号结果无法确认，请先在领号系统核对，再处理本批次。") from exc
            existing = db.scalar(select(DesignCode.main_image_id).where(
                DesignCode.source_code == number, DesignCode.main_image_id != item["main_image_id"],
            ).limit(1))
            if number in work["receipts"].values() or existing is not None:
                raise ValueError("领号系统返回的号码在本项目已有记录，请先核对。")
            work["receipts"][main_id] = number
            work["pending"] = None
            self._save_ticket_state(db, plan, coding)
        db.expire_all()
        if self._validate_ticket_sources(db, batch, work) != work["assets"]:
            raise ValueError("领号已记录，关联图片范围有变化，请先核对原预览。")
        db.execute(delete(RenamePlan).where(RenamePlan.plan_group == plan.plan_group)
                   .execution_options(synchronize_session="fetch"))
        db.execute(delete(MainCodeRename).where(MainCodeRename.plan_group == plan.plan_group)
                   .execution_options(synchronize_session="fetch"))
        self._generate(db, batch.id, with_codes=True, ticket_plan=plan)

    @staticmethod
    def coding_plan(db, batch_id, plan_group=None):
        query = select(DesignCodingPlan).where(DesignCodingPlan.batch_id == batch_id)
        if plan_group:
            query = query.where(DesignCodingPlan.plan_group == plan_group)
        return db.scalar(query.order_by(DesignCodingPlan.id.desc()).limit(1))

    def latest_plan(self, db: Session, batch_id: int, *, with_codes: bool):
        if with_codes:
            return self.coding_plan(db, batch_id)
        return db.scalar(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id,
            RenamePlan.plan_group.not_in(select(DesignCodingPlan.plan_group).where(
                DesignCodingPlan.batch_id == batch_id
            )),
        ).order_by(RenamePlan.id.desc()))

    def latest(self, db: Session, batch_id: int, *, with_codes: bool = False) -> dict | None:
        latest = self.latest_plan(db, batch_id, with_codes=with_codes)
        if not latest:
            return None
        coding_plan = latest if with_codes else None
        plan_group = latest.plan_group
        coding = json.loads(coding_plan.summary_json) if coding_plan else None
        if coding and "ticket" in coding and not coding["ticket"].get("complete"):
            return self._ticket_preview_payload(coding_plan, coding)
        main_plans = list(db.scalars(select(MainCodeRename).where(MainCodeRename.plan_group == plan_group).order_by(MainCodeRename.main_image_id))) if coding_plan else []
        stems = {plan.main_image_id: Path(plan.target_path).stem for plan in main_plans}
        plans = list(db.scalars(select(RenamePlan).where(
            RenamePlan.plan_group == plan_group
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
        result = {
            "batch_id": batch_id,
            "plan_group": plan_group,
            "counts": dict(Counter(plan['rename_status'] for plan in [*payloads, *(self.main_dict(row) for row in main_plans)])),
            "plans": self.describe(db, batch, payloads, stems),
        }
        if coding_plan:
            result.update(coding=self._public_coding(coding), main_plans=[self.main_dict(plan) for plan in main_plans])
        return result

    def execute(self, db: Session, batch_id: int, plan_group: str, *, confirm_coding: bool = False) -> dict:
        if confirm_coding:
            with design_coding.lock:
                return self._execute(db, batch_id, plan_group, confirm_coding=True)
        return self._execute(db, batch_id, plan_group, confirm_coding=False)

    def _execute(self, db: Session, batch_id: int, plan_group: str, *, confirm_coding: bool) -> dict:
        plans = list(db.scalars(select(RenamePlan).where(
            RenamePlan.batch_id == batch_id,
            RenamePlan.plan_group == plan_group,
        ).order_by(RenamePlan.id)))
        coding_plan = self.coding_plan(db, batch_id, plan_group)
        if not plans and not coding_plan:
            raise ValueError("Rename Plan不存在")
        if bool(coding_plan) != confirm_coding:
            raise ValueError("请打开对应的更名预览并确认本次操作")
        coding = json.loads(coding_plan.summary_json) if coding_plan else None
        ticket = coding.get("ticket") if coding else None
        if coding_plan and ticket is None and not design_coding.catalog_path():
            raise ValueError("编码目录未启用，请重新生成命名预览")
        batch = db.get(Batch, batch_id)
        if batch.status in {"QUEUED", "MATCHING"}:
            raise ValueError("匹配正在进行，请完成后再改名")
        latest = self.latest_plan(db, batch_id, with_codes=confirm_coding)
        if not latest or latest.plan_group != plan_group:
            raise ValueError("命名预览已更新，请使用最新预览")
        if ticket is not None and not ticket.get("complete"):
            self._prepare_ticket_execution(db, batch, coding_plan, coding)
            coding = json.loads(coding_plan.summary_json)
            ticket = coding["ticket"]
            plans = list(db.scalars(select(RenamePlan).where(
                RenamePlan.plan_group == plan_group,
            ).order_by(RenamePlan.id)))
        main_plans = list(db.scalars(select(MainCodeRename).where(
            MainCodeRename.plan_group == plan_group).order_by(MainCodeRename.main_image_id))) if coding_plan else []
        if coding_plan:
            if not main_plans:
                raise ValueError("没有可应用的主图编码，请先核对主图内容和缩写")
            prefixes = coding.get('prefixes') or {}
            if (not prefixes.keys() >= {'category', 'theme'} or
                    coding.get('source_code_prefix') != design_coding.number_prefix(prefixes)):
                raise ValueError("旧命名预览的号段顺序已更新，请重新预览命名。")
        stems = {plan.main_image_id: Path(plan.target_path).stem for plan in main_plans}
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
        for plan in main_plans:
            if plan.rename_status not in {'READY', 'NOOP', 'RENAMED'}:
                raise ValueError(f"主图 {Path(plan.source_path).name}：{plan.error_message}")
        movable_sources = {
            _norm(Path(plan.source_path))
            for plan in [*plans, *main_plans]
            if plan.rename_status == "READY"
        }
        validated = []
        code_rows = []
        if main_plans:
            root = Path(batch.root_path).resolve()
            for plan in main_plans:
                main = db.get(MainImage, plan.main_image_id)
                source = Path(plan.target_path if plan.rename_status == 'RENAMED' else plan.source_path)
                target = Path(plan.target_path)
                if (not main or main.file_path != str(source) or not source.is_file()
                        or not source.resolve().is_relative_to(root)
                        or design_coding.file_hash(source) != plan.source_sha256):
                    raise ValueError(f"主图 {source.name} 已变化，请重新预览命名。")
                code = db.get(DesignCode, main.id)
                if not code or code.target_stem != target.stem or (not code.applied and code.content_signature !=
                        design_coding.content_signature(db.get(MainImageContent, main.id))):
                    raise ValueError(f"主图 {source.name} 的内容审核结果已更新，请重新预览命名。")
                code_rows.append(code)
                if plan.rename_status == 'RENAMED':
                    continue
                if plan.rename_status == 'READY' and target.exists() and _norm(target) not in movable_sources:
                    raise ValueError(f"主图目标 {target.name} 已被占用，禁止覆盖。")
                validated.append((plan, main, source, target))
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
                                                          else f"{stems.get(item.matched_main_id, item.matched_main_basename)}{item.extension}"))
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

        staged = []
        completed = []
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
            with publish_code_texts(batch.root_path, code_rows) as code_text_files:
                self._record_renames(db, batch, plan_group, validated)
                if ticket is not None and not ticket.get("complete"):
                    ticket["complete"] = True
                    for row in coding["rows"]:
                        if str(row["main_image_id"]) in ticket["receipts"]:
                            row["applied"] = True
                    coding_plan.summary_json = json.dumps(coding, ensure_ascii=False)
                db.commit()
        except (OSError, UnicodeError, SQLAlchemyError) as exc:
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
            db.rollback()
            raise ValueError(f"命名或写入号段文件失败，已尝试恢复原文件: {exc}") from exc

        result = self.latest(db, batch_id, with_codes=confirm_coding) or {}
        if code_text_files:
            result['code_text_files'] = code_text_files
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
    def _record_renames(db, batch, plan_group, validated):
        now = datetime.now(timezone.utc)
        coded_stems = {}
        for plan, item, source, target in validated:
            if isinstance(plan, MainCodeRename):
                if plan.rename_status == 'READY':
                    design_coding.apply_main_path(batch, item, target)
                item.basename = target.stem
                coded_stems[item.id] = item.basename
                db.get(DesignCode, item.id).applied = True
                plan.rename_status = 'RENAMED'
                continue
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
        if coded_stems:
            for item in db.scalars(select(MatchedFile).where(
                MatchedFile.batch_id == batch.id, MatchedFile.matched_main_id.in_(coded_stems)
            )):
                item.matched_main_basename = coded_stems[item.matched_main_id]
    @staticmethod
    def main_dict(plan):
        return {'main_image_id': plan.main_image_id, 'category': '主图',
                'source_path': plan.source_path, 'target_path': plan.target_path,
                'current_filename': Path(plan.target_path if plan.rename_status == 'RENAMED' else plan.source_path).name,
                'target_filename': Path(plan.target_path).name, 'rename_status': plan.rename_status,
                'error_message': plan.error_message}

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
