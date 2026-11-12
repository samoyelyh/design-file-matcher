from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Batch, BatchProgress, MainImage, MatchCandidate, MatchedFile
from app.services.qwen_lab import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    compare,
    final_cache_key,
    judge_top2,
    pair_cache_key,
    preload_images,
    rank_pairwise,
    status as qwen_status,
)
from app.services.qwen_fusion import QwenFusionScorer


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(db, batch_id, stage, current, total, message):
    row = db.get(BatchProgress, batch_id) or BatchProgress(batch_id=batch_id)
    row.stage, row.current, row.total = stage, current, total
    row.percent = round(current / total * 100, 2) if total else 0
    row.message = message
    row.device = "GPU (Ollama concurrency=1)"
    db.add(row)
    db.commit()


def _selected_model(batch: Batch, state: dict, category: str | None = None) -> str:
    try:
        snapshot = json.loads(batch.match_config_snapshot or "{}")
    except (TypeError, ValueError):
        snapshot = {}
    folder_engines = snapshot.get("folder_engines", {}) if isinstance(snapshot, dict) else {}
    folder_config = folder_engines.get(category, {}) if category else {}
    model = (
        folder_config.get("model")
        or snapshot.get("model")
        or batch.matching_model
        or state.get("selected_model")
    )
    if not state.get("reachable") or model not in set(state.get("vision_models", [])):
        raise RuntimeError(f"所选 Qwen-VL 未就绪或不支持图片输入：{model}")
    return model


def run_qwen_batch(
    db: Session,
    batch_id: int,
    *,
    categories: set[str] | None = None,
    finalize: bool = True,
    allow_duplicates: bool = False,
) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("批次不存在")
    from .filename_matching import EXACT, associate_filenames
    associate_filenames(db, batch_id)
    mains = list(db.scalars(select(MainImage).where(MainImage.batch_id == batch_id).order_by(MainImage.id)))
    files = list(db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch_id).order_by(MatchedFile.id)))
    files = [
        f for f in files
        if f.category not in {"PSD", "主图"}
        and f.match_method != EXACT
        and f.status != "CONFLICT_MARKED"
        and (not categories or f.category in categories)
    ]
    if mains and not files:
        if finalize:
            batch.status = "COMPLETED"
            batch.finished_at = datetime.now(timezone.utc)
        db.commit()
        return {"batch_id": batch_id, "counts": {"EXACT_FILENAME_MATCH": 0}, "model": None}
    if not mains or not files:
        raise ValueError("没有可匹配的主图或副图")
    state = qwen_status()
    # Every image receives all-main Qwen comparisons, all-main fusion scores,
    # and one final Top2 decision. LightGlue work is included in each fusion
    # score unit because it refines only the Color TopN within that step.
    selected_models = {
        _selected_model(batch, state, category)
        for category in {f.category for f in files}
    }
    if len(selected_models) > 1:
        raise RuntimeError("同一次 Qwen 分片任务不能混用多个模型，请按模型分组执行")
    selected_model = next(iter(selected_models))
    state = {**state, "selected_model": selected_model}
    total = len(files) * (2 * len(mains) + 1)
    main_hashes = {main.id: _sha(main.file_path) for main in mains}
    mains_sha = hashlib.sha256("|".join(f"{main.id}:{main_hashes[main.id]}" for main in mains).encode()).hexdigest()
    # Prepare the reusable main-image payloads in parallel while keeping the
    # actual Ollama worker single-flight on the GPU.
    preload_images([main.file_path for main in mains])
    done = 0
    batch.status = "MATCHING"
    batch.error_message = None
    batch.match_engine = "QWEN"
    batch.matching_model = selected_model
    db.commit()
    totals = Counter()
    all_mains = mains
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-query-prefetch") as prefetch_pool:
            for category in sorted({f.category for f in files}):
                category_files = [f for f in files if f.category == category]
                reserved = set(db.scalars(select(MatchedFile.matched_main_id).where(
                    MatchedFile.batch_id == batch_id, MatchedFile.category == category,
                    MatchedFile.match_method == EXACT,
                )))
                mains = [main for main in all_mains if main.id not in reserved]
                if not mains:
                    for item in category_files:
                        item.status = "EXTRA_IGNORED"
                        item.review_status = "IGNORED"
                    db.commit()
                    continue
                try:
                    snapshot = json.loads(batch.match_config_snapshot or "{}")
                except (TypeError, ValueError):
                    snapshot = {}
                fusion_snapshot = snapshot.get("fusion", {}) if isinstance(snapshot, dict) else {}
                fusion_scorer = QwenFusionScorer(
                    db,
                    batch,
                    category,
                    mains,
                    qwen_weight=fusion_snapshot.get("qwen_weight"),
                    lightglue_color_weight=fusion_snapshot.get("lightglue_color_weight"),
                )
                # Read this category's Query images while the single Ollama
                # worker is processing the first prepared items.
                query_prefetch = prefetch_pool.submit(preload_images, [item.current_path for item in category_files])
                chosen_by_file = {}
                for item in category_files:
                    query_sha = _sha(item.current_path)
                    target = f"只比较 {category} 图片中的主要商品，忽略模特和背景。"
                    cache_key = f"{PROMPT_VERSION}:{SCHEMA_VERSION}:{query_sha}:{mains_sha}:{category}:{state['selected_model']}"
                    pair_rows = []
                    previous = {}
                    try:
                        previous = json.loads(item.match_details_json or "")
                        if not isinstance(previous, dict):
                            previous = {}
                    except Exception:
                        previous = {}
                    pair_cache = previous.get("pair_cache", {})
                    if not isinstance(pair_cache, dict):
                        pair_cache = {}
                    current_pair_cache = {}
                    for index, main in enumerate(mains, 1):
                        pair_key = pair_cache_key(query_sha, main_hashes[main.id], category, state["selected_model"], target)
                        cached_pair = pair_cache.get(pair_key)
                        if isinstance(cached_pair, dict) and {"match", "confidence", "evidence", "conflicts"}.issubset(cached_pair):
                            pair_row = {**cached_pair, "candidate_index": index, "main_id": main.id}
                            pair_rows.append(pair_row)
                            current_pair_cache[pair_key] = {k: v for k, v in cached_pair.items() if k != "candidate_index"}
                            done += 1
                            _progress(db, batch_id, "QWEN_PAIRWISE", done, total, f"分类 {category}：{item.original_filename} 已复用判断 {index}/{len(mains)}")
                            continue
                        try:
                            try:
                                error = None
                                for attempt in range(2):
                                    try:
                                        result = compare(item.current_path, main.file_path, target, index, state["selected_model"])
                                        error = None
                                        break
                                    except Exception as exc:
                                        error = exc
                                if error:
                                    raise error
                                pair_row = {"candidate_index": index, "main_id": main.id, **result["result"], "elapsed_seconds": result.get("elapsed_seconds")}
                                pair_rows.append(pair_row)
                                current_pair_cache[pair_key] = {k: v for k, v in pair_row.items() if k not in {"candidate_index", "main_id"}}
                            except Exception as exc:
                                pair_rows.append({"candidate_index": index, "main_id": main.id, "match": False, "confidence": 0, "evidence": [], "conflicts": [str(exc)], "error": str(exc)})
                        finally:
                            done += 1
                            _progress(db, batch_id, "QWEN_PAIRWISE", done, total, f"分类 {category}：{item.original_filename} 两图判断 {index}/{len(mains)}")
                    def fusion_progress(index: int, message: str) -> None:
                        nonlocal done
                        done += 1
                        _progress(db, batch_id, "LIGHTGLUE_COLOR", done, total,
                                  f"分类 {category}：{message}")

                    pair_rows, fusion_metadata = fusion_scorer.score_query(
                        item, pair_rows, on_candidate_done=fusion_progress
                    )
                    valid = [r for r in pair_rows if not r.get("error")]
                    ranked = rank_pairwise(valid)
                    top2 = ranked[:2]
                    top2_scoring = [{
                        "candidate_index": row["candidate_index"],
                        "qwen_support_score": row.get("qwen_support_score"),
                        "color_score": row.get("color_score"),
                        "local_score": row.get("local_score"),
                        "lightglue_color_score": row.get("lightglue_color_score"),
                        "fusion_score": row.get("fusion_score"),
                    } for row in top2]
                    final_scoring_context = {
                        "version": fusion_metadata.get("version"),
                        "weights": fusion_metadata.get("weights"),
                        "candidates": top2_scoring,
                    }
                    final_result = None
                    final_cache = previous.get("final_cache", {})
                    if not isinstance(final_cache, dict):
                        final_cache = {}
                    if len(top2) == 2:
                        top2_shas = [main_hashes[mains[r["candidate_index"] - 1].id] for r in top2]
                        final_key = final_cache_key(
                            query_sha, top2_shas, category, state["selected_model"], target,
                            scoring_context=final_scoring_context,
                        )
                        cached_final = final_cache.get(final_key)
                        if isinstance(cached_final, dict) and isinstance(cached_final.get("result"), dict):
                            final_data = cached_final["result"]
                            selected_main_id = cached_final.get("selected_main_id")
                            selected_index = next((r["candidate_index"] for r in top2 if r["main_id"] == selected_main_id), None)
                            if selected_index is not None:
                                final_result = {"result": final_data, "selected_candidate_index": selected_index, "top2_candidate_indices": [r["candidate_index"] for r in top2], "cached": True}
                        else:
                            try:
                                final = judge_top2(
                                    item.current_path,
                                    [m.file_path for m in mains],
                                    [{"candidate_index": r["candidate_index"], "result": r} for r in top2],
                                    target,
                                    state["selected_model"],
                                    scoring_context=top2_scoring,
                                )
                                final_result = final
                                final_data = final["result"]
                                selected_index = final.get("selected_candidate_index")
                                selected_main_id = mains[selected_index - 1].id if selected_index and 1 <= selected_index <= len(mains) else None
                                final_cache = {final_key: {"result": final_data, "selected_main_id": selected_main_id}}
                            except Exception as exc:
                                # A failed final call is a technical failure, not an AI abstention.
                                # Keep the best pairwise candidate so the item still reaches review.
                                selected_index = ranked[0]["candidate_index"]
                                selected_main_id = ranked[0]["main_id"]
                                final_data = {
                                    "selected": "PAIRWISE_FALLBACK",
                                    "confidence": float(ranked[0].get("confidence", 0)),
                                    "evidence": list(ranked[0].get("evidence", [])),
                                    "conflicts": list(ranked[0].get("conflicts", [])),
                                    "decision_mode": "PAIRWISE_FALLBACK",
                                    "decision_note": f"Top2裁决调用失败，已强制采用两图排序第一名并送人工复核：{exc}",
                                }
                                final_result = {"result": final_data, "selected_candidate_index": selected_index, "top2_candidate_indices": [r["candidate_index"] for r in top2]}
                                final_cache = {final_key: {"result": final_data, "selected_main_id": selected_main_id}}
                        if final_result is None:
                            # Ignore a stale/incomplete cached decision and make a fresh visual choice.
                            try:
                                final = judge_top2(
                                    item.current_path,
                                    [m.file_path for m in mains],
                                    [{"candidate_index": r["candidate_index"], "result": r} for r in top2],
                                    target,
                                    state["selected_model"],
                                    scoring_context=top2_scoring,
                                )
                                final_result = final
                                final_data = final["result"]
                                selected_index = final["selected_candidate_index"]
                                selected_main_id = mains[selected_index - 1].id
                                final_cache = {final_key: {"result": final_data, "selected_main_id": selected_main_id}}
                            except Exception as exc:
                                selected_index = ranked[0]["candidate_index"]
                                selected_main_id = ranked[0]["main_id"]
                                final_data = {
                                    "selected": "PAIRWISE_FALLBACK",
                                    "confidence": float(ranked[0].get("confidence", 0)),
                                    "evidence": list(ranked[0].get("evidence", [])),
                                    "conflicts": list(ranked[0].get("conflicts", [])),
                                    "decision_mode": "PAIRWISE_FALLBACK",
                                    "decision_note": f"Top2裁决调用失败，已强制采用两图排序第一名并送人工复核：{exc}",
                                }
                                final_result = {"result": final_data, "selected_candidate_index": selected_index, "top2_candidate_indices": [r["candidate_index"] for r in top2]}
                                final_cache = {final_key: {"result": final_data, "selected_main_id": selected_main_id}}
                    elif len(top2) == 1:
                        # If only one comparison succeeded, it is the only available choice.
                        only = top2[0]
                        final_data = {
                            "selected": "ONLY_CANDIDATE",
                            "confidence": float(only.get("confidence", 0)),
                            "evidence": list(only.get("evidence", [])),
                            "conflicts": list(only.get("conflicts", [])),
                            "decision_mode": "ONLY_CANDIDATE",
                            "decision_note": "只有一个候选完成视觉判断，系统已强制选择该候选并送人工复核。",
                        }
                        final_result = {"result": final_data, "selected_candidate_index": only["candidate_index"], "top2_candidate_indices": [only["candidate_index"]]}
                    else:
                        # No model returned a usable visual verdict; record a technical failure,
                        # not a successful "no match" decision.
                        pair_errors = [r.get("error") for r in pair_rows if r.get("error")]
                        final_data = {
                            "decision_status": "INFERENCE_ERROR",
                            "confidence": 0,
                            "evidence": [],
                            "conflicts": pair_errors[:5] or ["没有成功完成视觉比较的候选"],
                            "decision_note": "本图没有可用的模型判断结果，需重试或人工选择主图。",
                        }
                        final_result = {"selected_candidate_index": None, "top2_candidate_indices": []}
                    done += 1
                    _progress(db, batch_id, "QWEN_TOP2", done, total, f"分类 {category}：{item.original_filename} Top2裁决")
                    selected_index = final_result.get("selected_candidate_index") if final_result else None
                    selected_main = mains[selected_index - 1] if selected_index and 1 <= selected_index <= len(mains) else None
                    selected_pair = next((r for r in ranked if r["candidate_index"] == selected_index), None)
                    if selected_main and not final_data.get("decision_note"):
                        if not any(r.get("match") for r in valid):
                            final_data["decision_mode"] = "FORCED_BEST_CANDIDATE"
                            final_data["decision_note"] = "所有候选的两图判断都未确认同款；系统仍按最不确定的否定结果强制选出最优候选，请人工重点复核。"
                        elif selected_pair and not selected_pair.get("match"):
                            final_data["decision_mode"] = "TOP2_OVERRIDE"
                            final_data["decision_note"] = "Top2裁决选中了两图初判为不匹配的候选，请人工复核。"
                        else:
                            final_data["decision_mode"] = "TOP2_VISION"
                    details = {
                        "qwen_model": state["selected_model"],
                        "qwen_cache_key": cache_key,
                        "pair_cache": current_pair_cache,
                        "pairwise": pair_rows,
                        "top2": [r["main_id"] for r in top2],
                        "final": final_data,
                        "final_scoring_context": final_scoring_context,
                        "fusion": fusion_metadata,
                        "decision_note": final_data.get("decision_note"),
                        "final_cache": final_cache,
                        "match_method": "QWEN_VISUAL",
                    }
                    item.match_details_json = json.dumps(details, ensure_ascii=False)
                    item.match_method = "QWEN_VISUAL"
                    item.matched_main_id = selected_main.id if selected_main else None
                    item.matched_main_basename = selected_main.basename if selected_main else None
                    item.algorithm_main_id = item.matched_main_id
                    item.status = "PENDING" if selected_main else "UNMATCHED"
                    item.review_status = "PENDING"
                    item.is_low_confidence = bool(selected_main and selected_pair and not selected_pair.get("match"))
                    if not selected_main:
                        totals["UNMATCHED"] += 1
                    db.execute(delete(MatchCandidate).where(MatchCandidate.matched_file_id == item.id))
                    for rank, row in enumerate(ranked[:5], 1):
                        rank_score = row.get("fusion_score")
                        if rank_score is None:
                            rank_score = row.get("qwen_support_score", row.get("confidence", 0))
                        db.add(MatchCandidate(
                            matched_file_id=item.id,
                            main_image_id=row["main_id"],
                            rank=rank,
                            similarity=float(rank_score),
                            similarity_display=float(rank_score),
                            details_json=json.dumps(row, ensure_ascii=False),
                        ))
                    db.commit()
                query_prefetch.result()
                # Folder-sharded mode allows many source images in this
                # category to choose the same main image. Legacy Qwen runs
                # retain the old conflict marker unless explicitly enabled.
                by_main = defaultdict(list)
                for item in category_files:
                    if item.matched_main_id:
                        by_main[item.matched_main_id].append(item)
                if allow_duplicates:
                    for item in category_files:
                        item.is_conflict = False
                        totals[item.status] += 1
                else:
                    for main_id, rows in by_main.items():
                        if len(rows) > 1:
                            for item in rows:
                                item.is_conflict = True
                                item.matched_main_id = None
                                item.matched_main_basename = None
                                item.status = "UNMATCHED"
                                item.review_status = "PENDING"
                                totals["MATCH_CONFLICT"] += 1
                        else:
                            totals[rows[0].status] += 1
                db.commit()
        if finalize:
            batch.status = "COMPLETED"
            batch.finished_at = datetime.now(timezone.utc)
            _progress(db, batch_id, "DONE", total, total, "Qwen视觉匹配完成，等待人工审核")
        else:
            batch.status = "MATCHING"
        db.commit()
        return {"batch_id": batch_id, "counts": dict(totals), "model": state["selected_model"]}
    except Exception as exc:
        batch.status = "FAILED"
        batch.error_message = str(exc)
        db.commit()
        raise
