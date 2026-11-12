from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from . import provider, store
from .execution import ExecutionOptions
from .provider_async import RequestError, invoke_async
from .queue_store import CacheStore, QueueStore
from .runner import item_info, plan_reviews
from .scan import cache_key, cli_version, engine_hash, validate_sources
from .schemas import PAIR_SCHEMA, REVIEW_SCHEMA, validate_output


def preview(run_dir: Path, options: ExecutionOptions | None = None) -> dict:
    """Read-only preflight: no model processes and no queue/cache mutations."""
    plan = store.read_plan(run_dir)
    options = options or ExecutionOptions.from_settings(plan["settings"])
    validate_sources(plan)
    db = store.connect(run_dir / "run.sqlite3")
    cache_db = sqlite3.connect(store.CACHE_DB) if not plan["fresh"] and store.CACHE_DB.exists() else None
    unique, completed, hits, failed = {}, 0, 0, 0
    try:
        known_success=set()
        for kind,table in (('PAIR','pairs'),('REVIEW','reviews')):
            for row in db.execute(f"SELECT * FROM {table} WHERE status='DONE'").fetchall():
                _,images,_,schema=item_info(row,kind,plan)
                validate_output(json.loads(row['result']),schema)
                known_success.add(cache_key(plan,kind,images))
        for kind, table in (("PAIR", "pairs"), ("REVIEW", "reviews")):
            for row in db.execute(f"SELECT * FROM {table} ORDER BY a").fetchall():
                _, images, _, schema = item_info(row, kind, plan)
                key = cache_key(plan, kind, images)
                if row["status"] == "DONE":
                    validate_output(json.loads(row["result"]), schema)
                    completed += 1
                    continue
                if key in known_success:
                    continue
                if key in unique:
                    continue
                cached = None
                if cache_db:
                    value = cache_db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
                    if value:
                        try:
                            cached = json.loads(value[0])
                            validate_output(cached["result"], schema)
                        except (ValueError, KeyError, TypeError):
                            cached = None
                if cached:
                    hits += 1
                elif row["status"] == "ERROR":
                    failed += 1
                unique[key] = {"cached": bool(cached), "attempts": row["attempts"], "failed": row["status"] == "ERROR"}
        used = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        review_planned = db.execute("SELECT value FROM run_state WHERE key='review_planned'").fetchone()[0] == "1"
        future_reviews = 0 if review_planned else plan["review_max"]
        pending = [entry for entry in unique.values() if not entry["cached"] and not entry["failed"]]
        estimated = sum(max(0, options.max_retries + 1 - entry["attempts"]) for entry in pending) + future_reviews * (options.max_retries + 1)
        return {"plan_id": plan["plan_id"], "completed_logical_pairs_or_reviews": completed,
                "pending_unique_tasks": len(pending), "cache_hits_available": hits, "failed_tasks": failed,
                "future_top2_max": future_reviews, "maximum_additional_calls_before_budget": estimated,
                "budget_already_reserved": used, "budget_limit": options.max_model_calls,
                "budget_remaining": max(0, options.max_model_calls - used),
                "maximum_additional_calls": min(estimated, max(0, options.max_model_calls - used)),
                "concurrency": options.concurrency, "fresh": plan["fresh"]}
    finally:
        db.close()
        if cache_db:
            cache_db.close()


def format_progress(value: dict) -> str:
    eta = f"{value['eta_seconds']:.1f}s" if value["eta_seconds"] is not None else "计算中"
    average = f"{value['mean_request_seconds']:.2f}s" if value["mean_request_seconds"] is not None else "计算中"
    summary = (f"完成 {value['completed']}/{value['total_tasks']} | 运行 {value['running']}/{value['configured_concurrency']} | "
            f"等待 {value['pending']} | 重试等待 {value['retry_wait']} | 失败 {value['failed']} | 取消 {value['cancelled']} | "
            f"缓存 {value['cache_hits']} | 调用已启动 {value['model_calls_started']} / 预算预留 {value['budget_reserved']} | "
            f"平均 {average} | 吞吐 {value['throughput_per_second']:.2f}/s | ETA {eta}")
    workers=' | '.join(f"Worker {row['worker']}：{row['kind']} A{row['a']+1}-B{row['b1']+1} 运行中" for row in value['workers'])
    return summary + ('\n'+workers if workers else '')


async def run_async(run_dir: Path, options: ExecutionOptions, *, approval: str | None = None,
                    resume: bool = False, retry_errors: bool = False, progress=print,
                    progress_callback=None, check_environment: bool = True, invoker=None,
                    global_lock_dir: Path | None = None) -> dict:
    options.validate()
    if not check_environment and invoker is None:
        raise ValueError('离线模式必须显式提供模拟 invoker，防止意外调用真实模型。')
    plan = store.read_plan(run_dir)
    if engine_hash() != plan["engine_hash"]:
        raise ValueError("视觉引擎版本已变化，请创建新计划。")
    if check_environment and approval != plan["plan_id"]:
        raise ValueError("开始真实请求前需要确认计划，并提供 --approve-plan；请先使用 queue-plan 查看待执行量。")
    invoke = invoker or invoke_async
    project_lock = global_lock_dir or store.ROOT / "runtime" / "dispatcher"
    project_lock.mkdir(parents=True, exist_ok=True)
    started, session = time.monotonic(), uuid.uuid4().hex
    outcome, active, peak, last_progress = "COMPLETE", {}, 0, 0.0
    with ExitStack() as locks:
        locks.enter_context(store.RunLock(project_lock))
        locks.enter_context(store.RunLock(run_dir))
        validate_sources(plan)
        if check_environment:
            provider.login_status()
            if cli_version() != plan["cli_version"]:
                raise ValueError("Codex CLI 版本已变化，请创建新计划。")
        ledger = QueueStore(run_dir, plan, options, session)
        try:
            cache = CacheStore()
        except BaseException:
            ledger.close('CACHE_INIT_ERROR')
            raise
        semaphore = asyncio.Semaphore(options.concurrency)

        async def worker(job: dict):
            async with semaphore:
                tick = time.monotonic()
                try:
                    # item_info uses immutable stable image indices, never completion order.
                    source = {"a": job["a"], "b": job["b1"], "b1": job["b1"], "b2": job["b2"]}
                    _, images, prompt, schema = item_info(source, job["kind"], plan)
                    answer = await invoke(run_dir / "requests", images, prompt, job["kind"], plan["settings"]["model"],
                        plan["settings"]["reasoning_effort"], options.request_timeout,
                        on_started=lambda pid: ledger.launched(job, pid))
                    validate_output(answer["result"], schema)
                    return job, answer, None, time.monotonic() - tick, False
                except asyncio.CancelledError:
                    return job, None, RequestError("CANCELLED_REMOTE_UNKNOWN", "已停止本地请求；远端执行和用量可能已发生。"), time.monotonic() - tick, True
                except RequestError as exc:
                    return job, None, exc, time.monotonic() - tick, False
                except provider.ProviderError as exc:
                    wrapped = RequestError(exc.kind, str(exc), retryable=not exc.stop, stop=exc.stop)
                    return job, None, wrapped, time.monotonic() - tick, False
                except Exception as exc:
                    wrapped = RequestError("WORKER_ERROR", provider.scrub(str(exc)))
                    return job, None, wrapped, time.monotonic() - tick, False

        def finish(value) -> None:
            job, answer, error, elapsed, cancelled = value
            if answer is not None:
                ledger.succeed(job, answer)
                if not plan["fresh"]:
                    try:
                        cache.write(job["task_key"], job["kind"], answer)
                    except (sqlite3.Error, OSError):
                        with ledger.db:
                            ledger.pause("CACHE_ERROR")
            else:
                base = min(options.retry_backoff_max, options.retry_backoff * 2 ** (job["attempts"] - 1))
                delay = min(options.retry_backoff_max, base * (1 + random.uniform(0, options.retry_jitter)))
                delay = max(delay, getattr(error, "retry_after", None) or 0)
                ledger.fail(job, error, elapsed, delay, cancelled=cancelled)

        def emit(force: bool = False) -> None:
            nonlocal last_progress
            if force or time.monotonic() - last_progress >= options.progress_interval:
                value = ledger.snapshot(started)
                if progress:
                    progress(format_progress(value))
                if progress_callback:
                    progress_callback(value)
                with ledger.db:
                    ledger.db.execute("INSERT OR REPLACE INTO run_state VALUES ('live_queue_status',?)", (json.dumps(value, ensure_ascii=False),))
                last_progress = time.monotonic()

        try:
            ledger.sync("PAIR")
            ledger.sync("REVIEW")
            if retry_errors and ledger.db.execute("SELECT COUNT(*) FROM pairs WHERE status='ERROR'").fetchone()[0] and ledger.db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]:
                raise ValueError('旧任务已有 Top2 记录且基础评分需要重算；请创建新任务，避免沿用过期候选排序。历史结果不会被覆盖。')
            ledger.recover(resume, retry_errors)
            ledger.cached(cache)
            emit(True)
            while True:
                command, reason = ledger.get_control()
                if command == "STOP":
                    for task in active:
                        task.cancel()
                # Do not enqueue reviews until every base pair is a valid SUCCESS.
                base = ledger.counts("PAIR")
                base_ready = base["SUCCESS"] == sum(base.values())
                if base_ready:
                    plan_reviews(ledger.db, plan)
                    ledger.sync("REVIEW")
                    ledger.cached(cache)
                kind = "REVIEW" if base_ready else "PAIR"
                if command == "RUN":
                    occupied = set(active.values())
                    for worker_id in range(1, options.concurrency + 1):
                        if worker_id in occupied:
                            continue
                        job = ledger.claim(kind, worker_id)
                        if not job:
                            break
                        task = asyncio.create_task(worker(job))
                        active[task] = worker_id
                        peak = max(peak, len(active))
                emit()
                if active:
                    done, _ = await asyncio.wait(active, timeout=options.poll_interval, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        active.pop(task)
                        finish(task.result())
                    continue
                counts = ledger.counts()
                if command != "RUN":
                    outcome = "STOPPED" if command == "STOP" else "PAUSED" if reason in (None,'USER') else reason
                    break
                phase_counts = ledger.counts(kind)
                if not base_ready and not (phase_counts['PENDING'] + phase_counts['RETRY_WAIT']):
                    outcome = "FAILED" if phase_counts['FAILED'] else "CANCELLED"
                    break
                if counts['PENDING'] + counts['RETRY_WAIT'] and ledger.used_budget() >= options.max_model_calls:
                    outcome = "CALL_LIMIT"
                    break
                if counts['PENDING'] + counts['RETRY_WAIT']:
                    next_row = ledger.db.execute("SELECT MIN(next_at) FROM queue_jobs WHERE kind=? AND state='RETRY_WAIT'", (kind,)).fetchone()
                    if next_row[0] and next_row[0] - time.time() > options.retry_backoff_max:
                        outcome = "RETRY_DEFERRED"
                        break
                    await asyncio.sleep(options.poll_interval)
                    continue
                outcome = "FAILED" if counts['FAILED'] else "CANCELLED" if counts['CANCELLED'] else "COMPLETE"
                break
        except asyncio.CancelledError:
            outcome = "STOPPED"
            for task in active:
                task.cancel()
            if active:
                for value in await asyncio.gather(*active):
                    finish(value)
            active.clear()
        except BaseException:
            outcome = "DISPATCH_ERROR"
            for task in active:
                task.cancel()
            if active:
                for value in await asyncio.gather(*active):
                    finish(value)
            raise
        finally:
            emit(True)
            snapshot = ledger.snapshot(started)
            snapshot.update(outcome=outcome, peak_concurrency=peak, elapsed_seconds=round(time.monotonic() - started, 3),
                            session_id=session, complete=outcome == "COMPLETE")
            with ledger.db:
                ledger.db.execute("INSERT OR REPLACE INTO run_state VALUES ('last_queue_status',?)", (json.dumps(snapshot, ensure_ascii=False),))
            cache.close()
            ledger.close(outcome)
    return snapshot


def run(run_dir: Path, max_calls: int | None = None, *, concurrency: int | None = None,
        approval: str | None = None, execution_overrides: dict | None = None, **kwargs) -> dict:
    plan = store.read_plan(run_dir)
    options = ExecutionOptions.from_settings(plan["settings"], **{**(execution_overrides or {}),
        "concurrency": concurrency, "max_model_calls": max_calls})
    return asyncio.run(run_async(run_dir, options, approval=approval, **kwargs))
