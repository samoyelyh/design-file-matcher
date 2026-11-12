"""Single-writer queue ledger; workers never write matching rows or cache files."""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from . import store
from .execution import ExecutionOptions, execution_fingerprint
from .provider import scrub
from .runner import item_info
from .scan import cache_key
from .schemas import PAIR_SCHEMA, REVIEW_SCHEMA, validate_output

STATES = {"PENDING", "RUNNING", "SUCCESS", "FAILED", "RETRY_WAIT", "CANCELLED"}


class CacheStore:
    """Preserve the existing cache table/key/value format, serialize each write."""
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else store.CACHE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=15)
        self.db.execute("PRAGMA busy_timeout=15000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def read(self, key: str, kind: str) -> dict | None:
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
            validate_output(value["result"], PAIR_SCHEMA if kind == "PAIR" else REVIEW_SCHEMA)
            return value
        except (ValueError, KeyError, TypeError):
            return None

    def write(self, key: str, kind: str, value: dict) -> None:
        validate_output(value["result"], PAIR_SCHEMA if kind == "PAIR" else REVIEW_SCHEMA)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # Existing valid entries remain valid. Replace only a malformed entry.
            old = self.read(key, kind)
            if old is None:
                self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def close(self) -> None:
        self.db.close()


def control(run_dir: Path, command: str) -> None:
    if command not in {"PAUSE", "STOP"}:
        raise ValueError("只能请求 PAUSE 或 STOP。")
    db = store.connect(run_dir / "run.sqlite3")
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS queue_control (id INTEGER PRIMARY KEY CHECK(id=1), command TEXT NOT NULL, reason TEXT, updated REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS queue_control_events (id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, reason TEXT, at REAL NOT NULL)")
            db.execute("INSERT INTO queue_control VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET command=excluded.command,reason=excluded.reason,updated=excluded.updated",
                       (command, "USER", time.time()))
            db.execute("INSERT INTO queue_control_events(command,reason,at) VALUES (?,?,?)", (command,'USER',time.time()))
    finally:
        db.close()


class QueueStore:
    def __init__(self, run_dir: Path, plan: dict, options: ExecutionOptions, session_id: str):
        self.run_dir, self.plan, self.options, self.session_id = run_dir, plan, options, session_id
        self.db = store.connect(run_dir / "run.sqlite3")
        self.db.execute("PRAGMA busy_timeout=15000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS queue_jobs (
            task_key TEXT PRIMARY KEY, kind TEXT NOT NULL, a INTEGER NOT NULL, b1 INTEGER NOT NULL, b2 INTEGER,
            state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','SUCCESS','FAILED','RETRY_WAIT','CANCELLED')),
            attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0, worker INTEGER,
            result TEXT, metadata TEXT, error TEXT, cache_hit INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS queue_targets (
            kind TEXT NOT NULL, a INTEGER NOT NULL, b INTEGER NOT NULL, task_key TEXT NOT NULL,
            PRIMARY KEY(kind,a,b), FOREIGN KEY(task_key) REFERENCES queue_jobs(task_key));
        CREATE TABLE IF NOT EXISTS queue_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_key TEXT NOT NULL, old_state TEXT, new_state TEXT NOT NULL,
            at REAL NOT NULL, worker INTEGER, attempt_id INTEGER, detail TEXT);
        CREATE TABLE IF NOT EXISTS queue_control (
            id INTEGER PRIMARY KEY CHECK(id=1), command TEXT NOT NULL, reason TEXT, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS queue_sessions (
            id TEXT PRIMARY KEY, started REAL NOT NULL, finished REAL, options TEXT NOT NULL, outcome TEXT);
        CREATE TABLE IF NOT EXISTS queue_control_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, reason TEXT, at REAL NOT NULL);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(attempts)")}
        for name, typ in (("session_id", "TEXT"), ("task_key", "TEXT"), ("worker", "INTEGER"), ("pid", "INTEGER"), ("launched", "INTEGER")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE attempts ADD COLUMN {name} {typ}")
        self.db.execute("UPDATE attempts SET launched=1 WHERE launched IS NULL AND status='SUCCESS'")
        self.db.execute("INSERT OR IGNORE INTO queue_control VALUES (1,'RUN',NULL,?)", (time.time(),))
        self.db.execute("INSERT INTO queue_sessions(id,started,options) VALUES (?,?,?)",
                        (session_id, time.time(), json.dumps({**options.__dict__,'execution_fingerprint':execution_fingerprint()})))
        self.db.commit()

    def event(self, key: str, old: str | None, new: str, *, worker=None, attempt=None, detail=None) -> None:
        self.db.execute("INSERT INTO queue_events(task_key,old_state,new_state,at,worker,attempt_id,detail) VALUES (?,?,?,?,?,?,?)",
                        (key, old, new, time.time(), worker, attempt, detail))

    def publish(self, job, status: str, result=None, metadata=None, error=None) -> None:
        legacy = {"SUCCESS": "DONE", "FAILED": "ERROR", "RUNNING": "RUNNING"}.get(status, "PENDING")
        targets = self.db.execute("SELECT * FROM queue_targets WHERE task_key=?", (job["task_key"],)).fetchall()
        for target in targets:
            table, clause = ("pairs", "a=? AND b=?") if target["kind"] == "PAIR" else ("reviews", "a=?")
            ids = (target["a"], target["b"]) if table == "pairs" else (target["a"],)
            self.db.execute(f"UPDATE {table} SET status=?,result=?,metadata=?,error=?,attempts=? WHERE {clause} AND status<>'DONE'",
                (legacy, json.dumps(result, ensure_ascii=False) if result is not None else None,
                 json.dumps(metadata, ensure_ascii=False) if metadata is not None else None, error,
                 job["attempts"], *ids))

    def sync(self, kind: str) -> None:
        table = "pairs" if kind == "PAIR" else "reviews"
        rows = self.db.execute(f"SELECT * FROM {table} ORDER BY a" + (",b" if kind == "PAIR" else "")).fetchall()
        for row in rows:
            _, images, _, schema = item_info(row, kind, self.plan)
            key = cache_key(self.plan, kind, images)
            b1, b2 = (row["b"], None) if kind == "PAIR" else (row["b1"], row["b2"])
            state = {"DONE": "SUCCESS", "ERROR": "FAILED", "RUNNING": "RUNNING"}.get(row["status"], "PENDING")
            if state == "SUCCESS":
                try:
                    validate_output(json.loads(row["result"]), schema)
                except (ValueError, TypeError):
                    raise ValueError("已有成功结果未通过当前 Schema；请检查原任务，不自动覆盖。")
            cur = self.db.execute("INSERT OR IGNORE INTO queue_jobs(task_key,kind,a,b1,b2,state,attempts,result,metadata,error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                  (key, kind, row["a"], b1, b2, state, row["attempts"], row["result"], row["metadata"], row["error"]))
            if cur.rowcount:
                self.event(key, None, state, detail="IMPORTED" if row["status"] != "PENDING" else "CREATED")
            elif state=='SUCCESS':
                existing=self.db.execute('SELECT * FROM queue_jobs WHERE task_key=?',(key,)).fetchone()
                if existing['state']!='SUCCESS':
                    self.db.execute("UPDATE queue_jobs SET state='SUCCESS',result=?,metadata=?,error=NULL,worker=NULL WHERE task_key=?",
                        (row['result'],row['metadata'],key))
                    self.event(key,existing['state'],'SUCCESS',detail='REUSED_EXISTING_SUCCESS')
            self.db.execute("INSERT OR IGNORE INTO queue_targets VALUES (?,?,?,?)", (kind, row["a"], row["b"] if kind == "PAIR" else -1, key))
        # Identical image content can map several logical pairs to one cache key.
        for job in self.db.execute("SELECT * FROM queue_jobs WHERE kind=? AND state='SUCCESS'", (kind,)).fetchall():
            self.publish(job, "SUCCESS", json.loads(job["result"]), json.loads(job["metadata"]) if job["metadata"] else {})
        self.db.commit()

    def recover(self, resume: bool, retry_errors: bool) -> None:
        with self.db:
            self.db.execute("UPDATE attempts SET status='UNKNOWN',finished_at=?,error_kind='INTERRUPTED_UNKNOWN' WHERE status='STARTED'",
                            (self.timestamp(),))
            for job in self.db.execute("SELECT * FROM queue_jobs WHERE state='RUNNING'").fetchall():
                new = "PENDING" if job["attempts"] < self.options.max_retries + 1 else "FAILED"
                self.db.execute("UPDATE queue_jobs SET state=?,worker=NULL,error='INTERRUPTED_UNKNOWN',next_at=0 WHERE task_key=?", (new, job["task_key"]))
                self.publish(job, new, error="INTERRUPTED_UNKNOWN")
                self.event(job["task_key"], "RUNNING", new, detail="RECOVERED_REMOTE_OUTCOME_UNKNOWN")
            if resume:
                self.db.execute("UPDATE queue_control SET command='RUN',reason=NULL,updated=? WHERE id=1", (time.time(),))
                self.db.execute("INSERT INTO queue_control_events(command,reason,at) VALUES ('RUN','EXPLICIT_RESUME',?)",(time.time(),))
            if resume or retry_errors:
                wanted = ("CANCELLED", "FAILED") if retry_errors else ("CANCELLED",)
                for job in self.db.execute("SELECT * FROM queue_jobs WHERE state IN (" + ",".join("?" for _ in wanted) + ")", wanted).fetchall():
                    if job["attempts"] < self.options.max_retries + 1:
                        self.db.execute("UPDATE queue_jobs SET state='PENDING',next_at=0,worker=NULL,error=NULL WHERE task_key=?", (job["task_key"],))
                        self.publish(job, "PENDING")
                        self.event(job["task_key"], job["state"], "PENDING", detail="EXPLICIT_RESUME")

    @staticmethod
    def timestamp() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def get_control(self) -> tuple[str, str | None]:
        row = self.db.execute("SELECT command,reason FROM queue_control WHERE id=1").fetchone()
        return row[0], row[1]

    def pause(self, reason: str) -> None:
        self.db.execute("UPDATE queue_control SET command='PAUSE',reason=?,updated=? WHERE id=1", (reason, time.time()))
        self.db.execute("INSERT INTO queue_control_events(command,reason,at) VALUES ('PAUSE',?,?)",(reason,time.time()))

    def cached(self, cache: CacheStore) -> int:
        if self.plan["fresh"]:
            return 0
        hits = 0
        with self.db:
            for job in self.db.execute("SELECT * FROM queue_jobs WHERE state IN ('PENDING','RETRY_WAIT')").fetchall():
                answer = cache.read(job["task_key"], job["kind"])
                if not answer:
                    continue
                metadata = {"cached": True, "source_request_dir": answer.get("request_dir")}
                self.db.execute("UPDATE queue_jobs SET state='SUCCESS',result=?,metadata=?,cache_hit=1 WHERE task_key=?",
                                (json.dumps(answer["result"], ensure_ascii=False), json.dumps(metadata), job["task_key"]))
                self.publish(job, "SUCCESS", answer["result"], metadata)
                self.event(job["task_key"], job["state"], "SUCCESS", detail="CACHE_HIT")
                hits += 1
        return hits

    def used_budget(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    def claim(self, kind: str, worker: int) -> dict | None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.get_control()[0] != "RUN" or self.used_budget() >= self.options.max_model_calls:
                self.db.rollback()
                return None
            while True:
                job = self.db.execute("SELECT * FROM queue_jobs WHERE kind=? AND state IN ('PENDING','RETRY_WAIT') AND next_at<=? ORDER BY a,b1,task_key LIMIT 1",
                                      (kind, time.time())).fetchone()
                if not job:
                    self.db.rollback()
                    return None
                if job["attempts"] >= self.options.max_retries + 1:
                    self.db.execute("UPDATE queue_jobs SET state='FAILED',error='RETRY_LIMIT' WHERE task_key=?", (job["task_key"],))
                    self.publish(job, "FAILED", error="RETRY_LIMIT")
                    self.event(job["task_key"], job["state"], "FAILED", detail="RETRY_LIMIT")
                    self.db.commit()
                    self.db.execute("BEGIN IMMEDIATE")
                    continue
                current = dict(job)
                current["attempts"] += 1
                self.db.execute("UPDATE queue_jobs SET state='RUNNING',worker=?,attempts=?,next_at=0 WHERE task_key=?", (worker, current["attempts"], job["task_key"]))
                item_key = f"{job['a']}:{job['b1']}" if kind == "PAIR" else str(job["a"])
                attempt = self.db.execute("INSERT INTO attempts(kind,item_key,started_at,status,session_id,task_key,worker,launched) VALUES (?,?,?,'STARTED',?,?,?,0)",
                    (kind, item_key, self.timestamp(), self.session_id, job["task_key"], worker)).lastrowid
                current.update(worker=worker, attempt_id=attempt)
                self.publish(current, "RUNNING")
                self.event(job["task_key"], job["state"], "RUNNING", worker=worker, attempt=attempt, detail="BUDGET_RESERVED")
                self.db.commit()
                return current
        except BaseException:
            self.db.rollback()
            raise

    def launched(self, job: dict, pid: int) -> None:
        with self.db:
            self.db.execute("UPDATE attempts SET launched=1,pid=? WHERE id=?", (pid, job["attempt_id"]))

    def succeed(self, job: dict, answer: dict) -> None:
        metadata = {key: value for key, value in answer.items() if key != "result"}
        with self.db:
            self.db.execute("UPDATE attempts SET status='SUCCESS',finished_at=?,elapsed_seconds=?,usage=?,request_dir=?,launched=1 WHERE id=?",
                (self.timestamp(), answer["elapsed_seconds"], json.dumps(answer.get("usage")), answer["request_dir"], job["attempt_id"]))
            self.db.execute("UPDATE queue_jobs SET state='SUCCESS',worker=NULL,result=?,metadata=?,error=NULL WHERE task_key=?",
                (json.dumps(answer["result"], ensure_ascii=False), json.dumps(metadata, ensure_ascii=False), job["task_key"]))
            self.publish(job, "SUCCESS", answer["result"], metadata)
            self.event(job["task_key"], "RUNNING", "SUCCESS", worker=job["worker"], attempt=job["attempt_id"])

    def fail(self, job: dict, error, elapsed: float, delay: float = 0, *, cancelled: bool = False) -> None:
        detail = scrub(str(error))
        stop = getattr(error, "stop", False)
        retry = (getattr(error, "retryable", False) or (stop and error.kind == "QUOTA")) and job["attempts"] < self.options.max_retries + 1
        new = "CANCELLED" if cancelled else "RETRY_WAIT" if retry else "FAILED"
        with self.db:
            self.db.execute("UPDATE attempts SET status=?,finished_at=?,error_kind=?,details=?,elapsed_seconds=?,request_dir=? WHERE id=?",
                ("CANCELLED" if cancelled else "ERROR", self.timestamp(), error.kind, detail, round(elapsed, 3), getattr(error, "request_dir", None), job["attempt_id"]))
            self.db.execute("UPDATE queue_jobs SET state=?,worker=NULL,error=?,next_at=? WHERE task_key=?",
                (new, error.kind + ": " + detail, time.time() + delay, job["task_key"]))
            self.publish(job, new, error=error.kind + ": " + detail)
            self.event(job["task_key"], "RUNNING", new, worker=job["worker"], attempt=job["attempt_id"], detail=error.kind)
            if stop:
                self.pause(error.kind)

    def counts(self, kind: str | None = None) -> Counter:
        rows = self.db.execute("SELECT state,COUNT(*) FROM queue_jobs" + (" WHERE kind=?" if kind else "") + " GROUP BY state", (kind,) if kind else ()).fetchall()
        return Counter({row[0]: row[1] for row in rows})

    def snapshot(self, started: float) -> dict:
        counts = self.counts()
        duration = max(time.monotonic() - started, 0.001)
        session = self.db.execute("SELECT COUNT(*) FROM attempts WHERE session_id=? AND status='SUCCESS'", (self.session_id,)).fetchone()
        average = self.db.execute('SELECT AVG(elapsed_seconds) FROM attempts WHERE session_id=? AND launched=1',(self.session_id,)).fetchone()[0]
        throughput = session[0] / duration
        remaining = counts['PENDING'] + counts['RETRY_WAIT'] + counts['RUNNING']
        workers = [dict(row) for row in self.db.execute("SELECT worker,kind,a,b1,state FROM queue_jobs WHERE state='RUNNING' ORDER BY worker")]
        return {"total_tasks": sum(counts.values()), "logical_tasks": self.db.execute("SELECT COUNT(*) FROM queue_targets").fetchone()[0],
                "completed": counts['SUCCESS'], "running": counts['RUNNING'], "pending": counts['PENDING'],
                "failed": counts['FAILED'], "retry_wait": counts['RETRY_WAIT'], "cancelled": counts['CANCELLED'],
                "cache_hits": self.db.execute("SELECT COUNT(*) FROM queue_jobs WHERE cache_hit=1").fetchone()[0],
                "budget_reserved": self.used_budget(), "model_calls_started": self.db.execute("SELECT COUNT(*) FROM attempts WHERE launched=1").fetchone()[0],
                "configured_concurrency": self.options.concurrency, "mean_request_seconds": average,
                "throughput_per_second": throughput, "eta_seconds": remaining / throughput if throughput else None,
                "workers": workers, "recent": [dict(row) for row in self.db.execute("SELECT * FROM queue_events ORDER BY id DESC LIMIT 5")]}

    def close(self, outcome: str) -> None:
        with self.db:
            self.db.execute("UPDATE queue_sessions SET finished=?,outcome=? WHERE id=?", (time.time(), outcome, self.session_id))
        self.db.close()
