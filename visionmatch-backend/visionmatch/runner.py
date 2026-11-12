from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import provider
from .matching import reviews_needed
from .scan import cache_key, cli_version, engine_hash, validate_sources
from .schemas import PAIR_PROMPT, PAIR_SCHEMA, REVIEW_PROMPT, REVIEW_SCHEMA, validate_output
from .store import RunLock, cache_read, cache_write, connect, read_plan


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state(db, key: str) -> str | None:
    row = db.execute("SELECT value FROM run_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def plan_reviews(db, plan: dict) -> None:
    if state(db, "review_planned") == "1":
        return
    pending = db.execute("SELECT COUNT(*) FROM pairs WHERE status='PENDING'").fetchone()[0]
    if pending:
        return
    pairs = db.execute("SELECT * FROM pairs ORDER BY a,b").fetchall()
    needed = reviews_needed(pairs, len(plan["images_a"]), len(plan["images_b"]), plan["settings"])
    db.executemany("INSERT OR IGNORE INTO reviews(a,b1,b2) VALUES (?,?,?)", needed)
    db.execute("UPDATE run_state SET value='1' WHERE key='review_planned'")
    db.commit()


def item_info(row, kind: str, plan: dict) -> tuple[str, list[dict], str, dict]:
    a = row["a"]
    if kind == "PAIR":
        return (f'{a}:{row["b"]}', [plan["images_a"][a], plan["images_b"][row["b"]]], PAIR_PROMPT, PAIR_SCHEMA)
    return (str(a), [plan["images_a"][a], plan["images_b"][row["b1"]],
                     plan["images_b"][row["b2"]]], REVIEW_PROMPT, REVIEW_SCHEMA)


def update_item(db, kind: str, row, *, status: str, result: dict | None = None,
                metadata: dict | None = None, error: str | None = None) -> None:
    table = "pairs" if kind == "PAIR" else "reviews"
    where = "a=? AND b=?" if kind == "PAIR" else "a=?"
    args = (row["a"], row["b"]) if kind == "PAIR" else (row["a"],)
    db.execute(f"UPDATE {table} SET status=?,result=?,metadata=?,error=? WHERE {where}",
               (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
                json.dumps(metadata, ensure_ascii=False) if metadata is not None else None, error, *args))


def increment_attempt(db, kind: str, row) -> None:
    if kind == "PAIR":
        db.execute("UPDATE pairs SET attempts=attempts+1 WHERE a=? AND b=?", (row["a"], row["b"]))
    else:
        db.execute("UPDATE reviews SET attempts=attempts+1 WHERE a=?", (row["a"],))


def resume_errors(db, max_attempts: int) -> None:
    # Manual opt-in only; attempts with unknown remote outcome can consume quota again.
    db.execute("UPDATE pairs SET status='PENDING', error=NULL WHERE status='ERROR' AND attempts<?", (max_attempts,))
    db.execute("UPDATE reviews SET status='PENDING', error=NULL WHERE status='ERROR' AND attempts<?", (max_attempts,))
    # A previously failed pair may change both Top2 and one-to-one conflicts. Discard old reviews.
    if db.execute("SELECT COUNT(*) FROM pairs WHERE status='PENDING'").fetchone()[0]:
        db.execute("DELETE FROM reviews")
        db.execute("UPDATE run_state SET value='0' WHERE key='review_planned'")
    db.commit()


def run(run_dir: Path, max_calls: int, *, approval: str | None = None, retry_errors: bool = False,
        progress=print, check_environment: bool = True) -> dict:
    if max_calls < 1:
        raise ValueError("最大调用次数必须大于 0。")
    plan = read_plan(run_dir)
    if max_calls > 4 and approval != plan["plan_id"]:
        raise ValueError("超过 4 次调用需确认此计划，并传入完整 plan_id。")
    if engine_hash() != plan["engine_hash"]:
        raise ValueError("匹配程序已变化，请创建新任务，避免混合不同版本的结果。")
    config = plan["settings"]
    with RunLock(run_dir):
        validate_sources(plan)
        if check_environment:
            provider.login_status()
            if cli_version() != plan["cli_version"]:
                raise ValueError("Codex CLI 版本已改变，请重新生成计划，避免混合基准结果。")
        with connect(run_dir / "run.sqlite3") as db:
            # A process crash can leave a launched request with an unknown remote result.
            db.execute("UPDATE attempts SET status='UNKNOWN',finished_at=?,error_kind='INTERRUPTED_UNKNOWN' WHERE status='STARTED'", (utcnow(),))
            db.execute("UPDATE pairs SET status='ERROR',error='INTERRUPTED_UNKNOWN' WHERE status='RUNNING'")
            db.execute("UPDATE reviews SET status='ERROR',error='INTERRUPTED_UNKNOWN' WHERE status='RUNNING'")
            db.commit()
            if retry_errors:
                resume_errors(db, config["max_attempts_per_item"])
            stop_reason = None
            while True:
                plan_reviews(db, plan)
                row = db.execute("SELECT * FROM pairs WHERE status='PENDING' ORDER BY a,b LIMIT 1").fetchone()
                kind = "PAIR"
                if row is None:
                    row = db.execute("SELECT * FROM reviews WHERE status='PENDING' ORDER BY a LIMIT 1").fetchone()
                    kind = "REVIEW"
                if row is None:
                    break
                item_key, images, prompt, schema = item_info(row, kind, plan)
                key = cache_key(plan, kind, images)
                cached = None if plan["fresh"] else cache_read(key)
                if cached:
                    try:
                        validate_output(cached["result"], schema)
                    except (ValueError, KeyError, TypeError):
                        cached = None
                if cached:
                    update_item(db, kind, row, status="DONE", result=cached["result"],
                                metadata={"cached": True, "source_request_dir": cached.get("request_dir")})
                    db.commit()
                    progress(f"缓存 {kind} {item_key}")
                    continue
                used = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                if used >= max_calls:
                    stop_reason = "CALL_LIMIT"
                    break
                increment_attempt(db, kind, row)
                update_item(db, kind, row, status="RUNNING")
                cur = db.execute("INSERT INTO attempts(kind,item_key,started_at,status) VALUES (?,?,?,'STARTED')",
                                 (kind, item_key, utcnow()))
                attempt_id = cur.lastrowid
                db.commit()
                progress(f"调用 {used + 1}/{max_calls} {kind} {item_key}")
                started = time.monotonic()
                try:
                    answer = provider.invoke(run_dir / "requests", images, prompt, kind, config["model"],
                                             config["reasoning_effort"], config["timeout_seconds"])
                except provider.ProviderError as exc:
                    error = provider.scrub(str(exc))
                    db.execute("UPDATE attempts SET status='ERROR',finished_at=?,error_kind=?,details=?,elapsed_seconds=? WHERE id=?",
                               (utcnow(), exc.kind, error, round(time.monotonic() - started, 3), attempt_id))
                    attempts = row["attempts"] + 1
                    can_retry = not exc.stop and attempts < config["max_attempts_per_item"]
                    update_item(db, kind, row, status="PENDING" if can_retry else "ERROR", error=exc.kind + ": " + error)
                    db.commit()
                    progress(f"调用失败 {exc.kind}，{'稍后重试' if can_retry else '已记录'}")
                    if exc.stop:
                        stop_reason = exc.kind
                        break
                    continue
                except KeyboardInterrupt:
                    db.execute("UPDATE attempts SET status='UNKNOWN',finished_at=?,error_kind='INTERRUPTED_UNKNOWN' WHERE id=?", (utcnow(), attempt_id))
                    update_item(db, kind, row, status="ERROR", error="INTERRUPTED_UNKNOWN")
                    db.commit()
                    stop_reason = "INTERRUPTED_UNKNOWN"
                    break
                metadata = {k: v for k, v in answer.items() if k != "result"}
                db.execute("UPDATE attempts SET status='SUCCESS',finished_at=?,elapsed_seconds=?,usage=?,request_dir=? WHERE id=?",
                           (utcnow(), answer["elapsed_seconds"], json.dumps(answer["usage"]), answer["request_dir"], attempt_id))
                update_item(db, kind, row, status="DONE", result=answer["result"], metadata=metadata)
                db.commit()
                if not plan["fresh"]:
                    cache_write(key, answer)
            plan_reviews(db, plan)
            counts = {"pairs_done": db.execute("SELECT COUNT(*) FROM pairs WHERE status='DONE'").fetchone()[0],
                      "pairs_error": db.execute("SELECT COUNT(*) FROM pairs WHERE status='ERROR'").fetchone()[0],
                      "pairs_pending": db.execute("SELECT COUNT(*) FROM pairs WHERE status='PENDING'").fetchone()[0],
                      "reviews_done": db.execute("SELECT COUNT(*) FROM reviews WHERE status='DONE'").fetchone()[0],
                      "reviews_error": db.execute("SELECT COUNT(*) FROM reviews WHERE status='ERROR'").fetchone()[0],
                      "reviews_pending": db.execute("SELECT COUNT(*) FROM reviews WHERE status='PENDING'").fetchone()[0],
                      "cli_attempts": db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                      "stop_reason": stop_reason}
            db.execute("INSERT OR REPLACE INTO run_state VALUES ('last_status', ?)", (json.dumps(counts),))
            db.commit()
    return counts
