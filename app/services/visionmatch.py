"""HTTP adapter to the existing VisionMatch V2 service, never its naming API."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import delete, select

from app.config import BASE_DIR, is_excluded_file_category
from app.models import Batch, MainImage, MatchedFile, MatchCandidate, ProductSimilarity

URL = "http://127.0.0.1:8066"
VERSION = "visionmatch_terra_group_v3_duplicate_groups"
CONCURRENCY = 4
_status_cache = (0.0, {})


def request(method, path, body=None):
    # This loopback service must not pass through the workstation HTTP proxy.
    attempts = 3 if method.upper() == "GET" else 1
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=30, trust_env=False) as client:
                response = client.request(method, URL + path, json=body)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
            raise ValueError(f"VisionMatch：{str(detail)[:1200]}") from exc
        except httpx.TransportError as exc:
            if attempt + 1 < attempts:
                time.sleep(1)
                continue
            raise ValueError(f"VisionMatch 8066 连接或响应异常：{str(exc)[:300]}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ValueError(f"VisionMatch 8066 连接或响应异常：{str(exc)[:300]}") from exc


def status():
    global _status_cache
    if time.monotonic() - _status_cache[0] < 5:
        return dict(_status_cache[1])
    result = {"id": "VISIONMATCH", "label": "VisionMatch AI · Terra",
              "installed": False, "selectable": False, "device": "Codex CLI · ChatGPT 登录",
              "concurrency": CONCURRENCY,
              "description": "GPT-6.1 Terra 拼图判断；最多4个文件夹并行，每目录1～50张，启动前确认调用预算，无需遮罩"}
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            response = client.get(URL + "/api/health")
            response.raise_for_status()
            data = response.json()
        ready = data.get("ok") is True and data.get("product") == "VisionMatch V2" and not data.get("validation_replay")
        result.update(installed=ready, selectable=ready, model=data.get("model"),
                      error=None if ready else "8066 不是正式 VisionMatch V2 后端")
    except (httpx.HTTPError, ValueError):
        result["error"] = "请先启动 VisionMatch AI 的 start-v2.cmd（本机8066）"
    _status_cache = (time.monotonic(), result)
    return dict(result)


def is_locked(item):
    return bool(item.confirmed_at or item.status in {"MANUAL_CONFIRMED", "CONFLICT_MARKED"}
                or item.review_status in {"CONFIRMED", "CORRECTED"}
                or item.match_method in {"EXACT_FILENAME", "FILENAME_EXACT"})


def inputs(db, batch_id, categories=None):
    mains = list(db.scalars(select(MainImage).where(MainImage.batch_id == batch_id).order_by(MainImage.id)))
    files = list(db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch_id).order_by(MatchedFile.id)))
    groups = defaultdict(list)
    for item in files:
        if not is_excluded_file_category(item.category) and (categories is None or item.category in categories):
            groups[item.category].append(item)
    return [(category, [m for m in mains if m.id not in {
        f.matched_main_id for f in rows if is_locked(f) and f.matched_main_id
    }], [f for f in rows if not is_locked(f)]) for category, rows in sorted(groups.items())]


def plan(db, batch_id, categories=None):
    folders = []
    for category, mains, files in inputs(db, batch_id, categories):
        if not files:
            continue
        if not 1 <= len(mains) <= 50 or len(files) > 50:
            raise ValueError(f"{category}：VisionMatch 拼图模式要求可用主图和待识别图各1～50张；"
                             f"当前为 {len(mains)} / {len(files)}，不会自动截断候选")
        folders.append({"category": category, "mains": len(mains), "files": len(files)})
    return {"folders": folders, "calls_upper_bound": len(folders), "model": "gpt-6.1-terra", "concurrency": CONCURRENCY,
            "message": "每分类一次拼图调用，无自动重试；不确定项保留人工复核，不生成虚构相似度"}


def workspace(batch_id):
    base = Path(os.environ.get("DESIGN_MATCHER_VISIONMATCH_WORKSPACE", BASE_DIR / "data" / "visionmatch"))
    root = base / str(int(batch_id))
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_link(batch_id, link):
    path = workspace(batch_id) / "latest.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(link, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def get_link(batch_id):
    path = workspace(batch_id) / "latest.json"
    if not path.is_file():
        path = BASE_DIR / "data" / "visionmatch" / str(int(batch_id)) / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def control(batch_id, action):
    if action not in {"pause", "resume", "cancel"}:
        raise ValueError("无效操作")
    link = get_link(batch_id)
    links = link.get("runs", [link] if link.get("run_id") else [])
    if not links and not link.get("in_progress"):
        raise ValueError("当前没有可控制的 VisionMatch 运行")
    path = workspace(batch_id) / "control.json"
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"paused": action == "pause", "cancelled": action == "cancel"}), encoding="utf-8")
    temp.replace(path)
    controlled = 0
    errors = []
    for item in links:
        try:
            state = request("GET", f"/api/batches/{item['remote_batch_id']}")
            run = next((r for r in state["runs"] if r["id"] == item.get("run_id")), None)
            if run is None and item.get("run_id") is None and len(state["runs"]) == 1:
                run = state["runs"][0]  # A timed-out submission may have succeeded.
            if run is None or run["state"] in {"COMPLETE", "COMPLETE_WITH_ERRORS", "CANCELLED", "QUOTA_OR_AUTH_PAUSED", "PAUSED_UNKNOWN"}:
                continue
            if action == "resume" and run.get("used_calls", 0) >= run.get("max_calls", 0):
                continue
            request("POST", f"/api/runs/{run['id']}/{action}", {})
            controlled += 1
        except ValueError as exc:
            errors.append(f"{item.get('category', '')}：{exc}")
    if errors:
        raise ValueError("本地控制已生效；部分后台操作未确认：" + "；".join(errors))
    return {"ok": True, "controlled_runs": controlled}


def control_state(batch_id):
    path = workspace(batch_id) / "control.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(objects, directory, kind, db=None, batch=None):
    from .media import open_visual
    directory.mkdir(parents=True)
    mapping = {}
    hashes = {}
    for obj in objects:
        source = Path(obj.file_path if kind == "main" else obj.current_path)
        if not source.is_file():
            raise ValueError(f"图片文件已改名、移动或不可访问：{source.name}；请先核对文件路径")
        before = file_hash(source)
        extension = source.suffix.lower()
        destination = directory / f"{kind}_{obj.id}{extension if extension in {'.png', '.jpg', '.jpeg', '.webp'} else '.png'}"
        if extension in {".png", ".jpg", ".jpeg", ".webp"}:
            shutil.copyfile(source, destination)
        else:
            if source.suffix.casefold() == ".psd" and db is not None:
                from .psd import PsdPreviewExtractor
                asset = PsdPreviewExtractor().extract(db, batch, obj)
                shutil.copyfile(asset.asset_path, destination)
            else:
                open_visual(source).save(destination, "PNG")
        if file_hash(source) != before:
            raise ValueError("源图片在导入时发生变化，请重新检查文件")
        digest = file_hash(destination)
        record = {"id": obj.id, "hash": digest, "source": str(source), "source_hash": before, "kind": kind}
        if digest in hashes:
            # Only remove the temporary copy created above. Every original is
            # retained and verified again before its individual result is saved.
            destination.unlink()
            hashes[digest].setdefault("duplicates", []).append(record)
        else:
            hashes[digest] = record
            mapping[str(destination.resolve()).casefold()] = record
    return mapping


def validate_result(state, mapping):
    """Check identity AND content before changing any 8088 associations."""
    remote = {}
    seen_paths = set()
    for image in state.get("images", []):
        key = str(Path(image["path"]).resolve()).casefold()
        local = mapping.get(key)
        if local is None or local["hash"] != image.get("sha256"):
            raise ValueError("VisionMatch 返回图片身份不符，已拒绝写回")
        if key in seen_paths or image["id"] in remote:
            raise ValueError("VisionMatch 返回图片重复，已拒绝写回")
        seen_paths.add(key)
        if local.get("kind") and not image["local_id"].startswith("A" if local["kind"] == "main" else "B"):
            raise ValueError("VisionMatch 返回图片角色不符，已拒绝写回")
        for original in [local, *local.get("duplicates", [])]:
            if file_hash(original["source"]) != original["source_hash"]:
                raise ValueError("识别期间源图片已变化，已拒绝写回")
        remote[image["id"]] = local["id"]
    if len(remote) != len(mapping):
        raise ValueError("VisionMatch 返回图片不完整，已拒绝写回")
    targets = {row["target_id"] for row in state.get("matches", [])}
    expected = {image["id"] for image in state["images"] if image["local_id"].startswith("B")}
    if targets != expected or len(targets) != len(state.get("matches", [])):
        raise ValueError("VisionMatch 返回归属缺失或重复")
    for row in state["matches"]:
        source_id = row.get("assigned_source_id")
        if source_id is not None and source_id not in {
            image["id"] for image in state["images"] if image["local_id"].startswith("A")
        }:
            raise ValueError("VisionMatch 返回无效主图")
        sources = {image["id"] for image in state["images"] if image["local_id"].startswith("A")}
        if row.get("proposed_source_id") is not None and row["proposed_source_id"] not in sources:
            raise ValueError("VisionMatch 返回无效主图候选")
        if any(candidate.get("source_id") not in sources for candidate in row.get("candidates", [])):
            raise ValueError("VisionMatch 返回无效主图候选")
    return remote


def fill_vacant_slots(updates, mains, occupied):
    """Globally complete remaining category slots without inventing AI scores.

    Existing assignments are fixed. Model candidate order is only a tie-break
    preference for the reduced problem; every vacant main remains eligible.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    by_category = defaultdict(list)
    used = set(occupied)
    for item, main_id, row, candidates in updates:
        if main_id is None:
            by_category[item.category].append((item, candidates))
        else:
            used.add((item.category, main_id))
    filled = {}
    for category, pending in by_category.items():
        pending.sort(key=lambda entry: entry[0].id)
        available = sorted(main_id for main_id in mains if (category, main_id) not in used)
        if not available:
            continue
        columns = {main_id: column for column, main_id in enumerate(available)}
        max_rank = max((len(candidates) for _, candidates in pending), default=0)
        candidate_bonus = (len(pending) + 1) * (max_rank + 1)
        preference = np.zeros((len(pending), len(available)), dtype=np.float64)
        for row, (_, candidates) in enumerate(pending):
            for rank, candidate in enumerate(candidates):
                column = columns.get(candidate["main_image_id"])
                if column is not None and candidate.get("decision") in {"MATCH", "UNCERTAIN"}:
                    preference[row, column] = candidate_bonus + max_rank - rank
        rows, cols = linear_sum_assignment(-preference)
        for row, column in zip(rows, cols):
            filled[pending[int(row)][0].id] = available[int(column)]
    return filled


def apply_result(db, batch_id, state, mapping, link):
    remote = validate_result(state, mapping)
    batch = db.get(Batch, batch_id)
    independent = json.loads(batch.match_config_snapshot or "{}").get("mode") == "FOLDER_SHARDED"
    records = {image["id"]: mapping[str(Path(image["path"]).resolve()).casefold()] for image in state["images"]}
    mains = {m.id: m for m in db.scalars(select(MainImage).where(MainImage.batch_id == batch_id))}
    files = {f.id: f for f in db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch_id))}
    target_ids = {record["id"] for row in state["matches"]
                  for record in [records[row["target_id"]], *records[row["target_id"]].get("duplicates", [])]}
    reserved = {(f.category, f.matched_main_id) for f in files.values()
                if f.matched_main_id and (is_locked(f) or f.id not in target_ids)}
    used = set()
    updates = []
    for row in state["matches"]:
        source_id = row.get("assigned_source_id")
        main_id = remote.get(source_id)
        source_record = records.get(source_id, {})
        duplicate_mains = [source_record["id"], *[record["id"] for record in source_record.get("duplicates", [])]] if source_record.get("duplicates") else []
        candidates = []
        for candidate in row.get("candidates", []):
            record = records[candidate["source_id"]]
            for original in [record, *record.get("duplicates", [])]:
                candidates.append({**candidate, "main_image_id": original["id"], "basename": mains[original["id"]].basename})
        if duplicate_mains:
            main_id = None  # Identical main images cannot identify one product.
        target_record = records[row["target_id"]]
        originals = [target_record, *target_record.get("duplicates", [])]
        for index, original in enumerate(originals):
            item = files[original["id"]]
            if is_locked(item):
                continue
            selected_main = main_id
            if index:
                selected_main = None
            elif selected_main is not None and (item.category, selected_main) in used:
                raise ValueError("VisionMatch 返回同分类重复归属，已拒绝写回")
            if (item.category, selected_main) in reserved:
                selected_main = None
            if selected_main:
                used.add((item.category, selected_main))
            annotated = {**row, "duplicate_target_ids": [record["id"] for record in originals] if len(originals) > 1 else [],
                         "duplicate_main_ids": duplicate_mains, "independent": independent}
            updates.append((item, selected_main, annotated, candidates))
    fillable = [update for update in updates if not update[2].get("duplicate_target_ids") and not update[2].get("duplicate_main_ids")]
    filled = {} if independent else fill_vacant_slots(fillable, mains, reserved | used)
    for item, main_id, row, candidates in updates:
        model_main_id = main_id
        auto_filled = item.id in filled
        if auto_filled:
            main_id = filled[item.id]
        db.execute(delete(MatchCandidate).where(MatchCandidate.matched_file_id == item.id))
        db.execute(delete(ProductSimilarity).where(ProductSimilarity.matched_file_id == item.id))
        for field in ("top1_score", "top2_score", "top3_score", "algorithm_score", "image_best_score",
                      "image_second_best_score", "assigned_rank", "row_competitor_score", "row_margin",
                      "column_competitor_score", "column_margin", "phash_distance", "phash_value",
                      "phash_best_region_type", "baseline_top1_main_id", "baseline_top1_score",
                      "baseline_top2_score", "matching_run_id", "image_best_main_id"):
            setattr(item, field, None)
        item.matched_main_id = item.algorithm_main_id = main_id
        item.matched_main_basename = mains[main_id].basename if main_id else None
        item.match_method = "VISIONMATCH"
        item.algorithm_version = VERSION
        suggestion = row.get("suggestion") or {}
        supported = (not auto_filled and not row.get("duplicate_target_ids") and not row.get("duplicate_main_ids")
                     and row["status"] == "MATCHED" and suggestion.get("decision") == "MATCH"
                     and suggestion.get("evidence") and not suggestion.get("conflicts")
                     and not suggestion.get("needs_review"))
        item.status = "MATCHED" if main_id and supported else "AMBIGUOUS" if main_id or row.get("duplicate_main_ids") else "UNMATCHED"
        item.algorithm_status = item.status
        item.review_status = "PENDING"
        item.is_conflict = False
        item.is_low_confidence = item.status != "MATCHED"
        item.match_details_json = json.dumps({
            "model": "gpt-6.1-terra", "mode": "CONTACT_SHEET", "matrix_complete": False,
            "remote_batch_id": link["remote_batch_id"], "run_id": link["run_id"],
            "backend_status": row["status"], "suggestion": row.get("suggestion", {}),
            "auto_filled": auto_filled,
            "assignment_method": "AUTO_FILL_VACANT" if auto_filled else "VISIONMATCH",
            "model_assigned_main_id": model_main_id,
            "fill_reason": "补入同分类剩余空位，需人工复核" if auto_filled else None,
            "candidates": candidates, "notes": row.get("notes", []),
            "duplicate_target_ids": row.get("duplicate_target_ids", []),
            "duplicate_main_ids": row.get("duplicate_main_ids", []),
            "duplicate_note": "副图内容重复，共用一次识别结果；请复核商品归属" if row.get("duplicate_target_ids") else "主图内容相同，无法仅凭图片确定商品" if row.get("duplicate_main_ids") else None,
        }, ensure_ascii=False)
    db.flush()
    return len(updates)


def _parallel_manifest(batch_id, root, links, in_progress=True):
    latest = dict(links[-1]) if links else {}
    latest.update(
        attempt_id=root.name, concurrency=CONCURRENCY, in_progress=in_progress,
        runs=[{key: link.get(key) for key in ("remote_batch_id", "run_id", "category", "state")}
              for link in links],
    )
    save_link(batch_id, latest)


def _run_parallel_folders(db, batch_id, root, targets, mappings, independent, progress_offset, progress_total):
    """Keep up to four remote runs in flight; write results on this DB thread."""
    from .matcher import _set_progress

    pending = deque(targets)
    active = {}
    links = []
    results = []
    failures = []
    completed = 0
    halt_submissions = False
    progress_count = progress_total or len(targets)

    def checkpoint(link):
        (Path(link["work_dir"]) / "link.json").write_text(json.dumps(link, ensure_ascii=False, indent=2), encoding="utf-8")
        _parallel_manifest(batch_id, root, links)

    def stopped():
        flags = control_state(batch_id)
        if flags.get("cancelled"):
            raise ValueError("VisionMatch 已停止后续调用；在途请求可能继续消耗额度，已完成结果保留")
        return flags.get("paused", False)

    _parallel_manifest(batch_id, root, links)
    try:
        while pending or active:
            paused = stopped()
            while pending and len(active) < CONCURRENCY and not paused and not halt_submissions:
                category, source, target = pending.popleft()
                link = None
                try:
                    remote = request("POST", "/api/batches", {"name": f"8088批次{batch_id} · {category}"})
                    remote_id = remote["id"]
                    link = {"remote_batch_id": remote_id, "category": category, "run_id": None,
                            "mapping": mappings[category], "work_dir": str(source.parent), "state": "IMPORTING"}
                    links.append(link)
                    checkpoint(link)
                    request("POST", f"/api/batches/{remote_id}/import", {
                        "source": str(source), "targets": [{"path": str(target), "label": category}],
                    })
                    while stopped():
                        _set_progress(db, batch_id, "VISIONMATCH", progress_offset + completed, progress_count,
                                      5 + 85 * (progress_offset + completed) / progress_count,
                                      f"已暂停领取新调用；已完成 {completed}/{len(targets)} 个文件夹", f"最多{CONCURRENCY}路并发")
                        time.sleep(1)
                    link["state"] = "SUBMITTING"
                    checkpoint(link)
                    # A submission is attempted only once. Each folder keeps
                    # its own one-call cap inside the original total budget.
                    job = request("POST", f"/api/batches/{remote_id}/match", {
                        "confirmed_budget": True, "max_calls": 1, "concurrency": CONCURRENCY,
                    })
                    link.update(run_id=job["run_id"], state="RUNNING")
                    checkpoint(link)
                    active[remote_id] = {"link": link, "deadline": time.monotonic() + 3600}
                except ValueError as exc:
                    if link is not None:
                        link["state"] = "SUBMISSION_UNKNOWN" if link["state"] == "SUBMITTING" else "FAILED"
                        checkpoint(link)
                    failures.append(f"{category}：{exc}")
                    halt_submissions = True
                paused = stopped()

            made_progress = False
            for remote_id, entry in list(active.items()):
                paused = stopped()
                link = entry["link"]
                category = link["category"]
                try:
                    state = entry.get("ready") or request("GET", f"/api/batches/{remote_id}")
                    run = next(r for r in state["runs"] if r["id"] == link["run_id"])
                    jobs = [j for j in state.get("jobs", []) if j["run_id"] == link["run_id"]]
                    if run["state"] in {"CANCELLED", "COMPLETE_WITH_ERRORS", "QUOTA_OR_AUTH_PAUSED", "PAUSED_UNKNOWN"} or any(j["state"] == "FAILED" for j in jobs):
                        if run["state"] in {"QUOTA_OR_AUTH_PAUSED", "PAUSED_UNKNOWN"}:
                            halt_submissions = True
                        error = next((j.get("error") for j in jobs if j.get("error")), run["state"])
                        raise ValueError(f"VisionMatch 停止或失败：{error}；没有自动重试")
                    if run["state"] == "COMPLETE":
                        if paused:
                            entry["ready"] = state
                            continue
                        if any(f["state"] != "COMPLETE" for f in state["folders"] if f["role"] == "TARGET"):
                            raise ValueError("VisionMatch 未覆盖全部目标，已保留原关联")
                        (Path(link["work_dir"]) / "result.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                        if independent:
                            db.expire_all()
                            apply_result(db, batch_id, state, mappings[category], link)
                            db.commit()
                        else:
                            results.append((state, mappings[category], link))
                        link["state"] = "COMPLETE"
                        completed += 1
                        active.pop(remote_id)
                        checkpoint(link)
                        made_progress = True
                        break  # Refill this slot before collecting other results.
                    elif time.monotonic() > entry["deadline"]:
                        halt_submissions = True
                        raise ValueError("VisionMatch 等待超时；请检查8066记录，未自动重试")
                except ValueError as exc:
                    db.rollback()
                    if "连接或响应异常" in str(exc):
                        halt_submissions = True
                    link["state"] = "FAILED"
                    failures.append(f"{category}：{exc}")
                    active.pop(remote_id)
                    checkpoint(link)
                    made_progress = True
                    break

            names = "、".join(entry["link"]["category"] for entry in active.values())
            message = ("已暂停领取新调用；" if control_state(batch_id).get("paused") else "")
            message += f"已完成 {completed}/{len(targets)} 个文件夹，运行中 {len(active)}/{CONCURRENCY}"
            if names:
                message += f"：{names}"
            _set_progress(db, batch_id, "VISIONMATCH", progress_offset + completed, progress_count,
                          5 + 85 * (progress_offset + completed) / progress_count, message, f"最多{CONCURRENCY}路并发")
            if halt_submissions and not active:
                break
            if active or pending:
                # Fill a newly available slot before waiting on the next poll.
                if made_progress or (pending and len(active) < CONCURRENCY and not halt_submissions and not control_state(batch_id).get("paused")):
                    continue
                time.sleep(2)
        if failures:
            raise ValueError("；".join(failures))
        return results
    finally:
        _parallel_manifest(batch_id, root, links, in_progress=False)


def run_batch(db, batch_id, categories=None, finalize=True, progress_offset=0, progress_total=None):
    from .matcher import _set_progress
    from .filename_matching import associate_filenames
    from .rename_plan import RenamePlanService
    batch = db.get(Batch, batch_id)
    try:
        associate_filenames(db, batch_id)
        info = plan(db, batch_id, categories)
        config = json.loads(batch.match_config_snapshot or "{}").get("visionmatch", {})
        if info["calls_upper_bound"] > config.get("max_calls", 0):
            raise ValueError("VisionMatch 调用预算不足，请重新确认预计调用次数")
        batch.status = "MATCHING"
        batch.error_message = None
        snapshot = json.loads(batch.match_config_snapshot or "{}")
        snapshot.setdefault("visionmatch", {})["concurrency"] = CONCURRENCY
        batch.match_config_snapshot = json.dumps(snapshot, ensure_ascii=False)
        db.commit()
        (workspace(batch_id) / "control.json").write_text("{}", encoding="utf-8")
        if info["calls_upper_bound"]:
            root = workspace(batch_id) / uuid.uuid4().hex
            mappings = {}
            targets = []
            _parallel_manifest(batch_id, root, [])
            # Each category has its own source directory: manual/exact locks
            # reserve different main images in different categories.
            for index, (category, mains, files) in enumerate(inputs(db, batch_id, categories)):
                if not files:
                    continue
                _set_progress(db, batch_id, "PREPARING", progress_offset, progress_total or info["calls_upper_bound"],
                              1, f"正在准备 {category}：主图 {len(mains)} 张，副图 {len(files)} 张", f"最多{CONCURRENCY}路并发")
                part = root / str(index)
                source = part / "main"
                target = part / "target"
                mappings[category] = {**stage(mains, source, "main"), **stage(files, target, "file", db=db, batch=batch)}
                db.commit()
                targets.append((category, source, target))
            all_results = _run_parallel_folders(db, batch_id, root, targets, mappings,
                snapshot.get("mode") == "FOLDER_SHARDED", progress_offset, progress_total)
            for state, mapping, link in all_results:
                validate_result(state, mapping)
            db.expire_all()
            for state, mapping, link in all_results:
                apply_result(db, batch_id, state, mapping, link)
            db.commit()
        batch = db.get(Batch, batch_id)
        if finalize:
            RenamePlanService().generate(db, batch_id)
            batch.status = "COMPLETED"
            batch.finished_at = datetime.now(timezone.utc)
            _set_progress(db, batch_id, "COMPLETED", 1, 1, 100, "VisionMatch 完成，请审核归属和判断依据", "GPT-6.1 Terra")
        db.commit()
    except Exception as exc:
        db.rollback()
        batch = db.get(Batch, batch_id)
        batch.status = "FAILED"
        batch.error_message = str(exc)[:3000]
        _set_progress(db, batch_id, "FAILED", 0, 0, 0, batch.error_message)
        raise
