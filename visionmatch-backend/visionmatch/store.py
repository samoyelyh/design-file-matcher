from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .scan import canonical_hash

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "runtime" / "cache.sqlite3"


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def create_run(plan: dict) -> Path:
    root = Path(plan["settings"]["output_folder"])
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / (datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    with connect(run_dir / "run.sqlite3") as db:
        db.executescript("""
        CREATE TABLE pairs (a INTEGER NOT NULL, b INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING',
            attempts INTEGER NOT NULL DEFAULT 0, result TEXT, metadata TEXT, error TEXT, PRIMARY KEY(a,b));
        CREATE TABLE reviews (a INTEGER PRIMARY KEY, b1 INTEGER NOT NULL, b2 INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING', attempts INTEGER NOT NULL DEFAULT 0,
            result TEXT, metadata TEXT, error TEXT);
        CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, item_key TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, error_kind TEXT, details TEXT,
            elapsed_seconds REAL, usage TEXT, request_dir TEXT);
        CREATE TABLE run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE human_decisions (a INTEGER PRIMARY KEY, b INTEGER, decision TEXT NOT NULL,
            note TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE UNIQUE INDEX human_unique_b ON human_decisions(b) WHERE decision='MATCH';
        CREATE TABLE human_history (id INTEGER PRIMARY KEY AUTOINCREMENT, a INTEGER NOT NULL,
            b INTEGER, decision TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL);
        """)
        db.executemany("INSERT INTO pairs(a,b) VALUES (?,?)",
                       ((a, b) for a in range(len(plan["images_a"])) for b in range(len(plan["images_b"]))))
        db.execute("INSERT INTO run_state VALUES ('plan_id', ?)", (plan["plan_id"],))
        db.execute("INSERT INTO run_state VALUES ('review_planned', '0')")
    return run_dir


def read_plan(run_dir: Path) -> dict:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    core_keys = ("settings", "images_a", "images_b", "cli_version", "pair_schema_hash", "review_schema_hash", "prompt_hash", "engine_hash", "fresh", "limit_a", "limit_b")
    if canonical_hash({key: plan[key] for key in core_keys}) != plan["plan_id"]:
        raise ValueError("计划已改变，请重新创建任务。")
    with connect(run_dir / "run.sqlite3") as db:
        row = db.execute("SELECT value FROM run_state WHERE key='plan_id'").fetchone()
        if row is None or row[0] != plan["plan_id"]:
            raise ValueError("计划与任务数据库不一致。")
    return plan


def cache_read(key: str) -> dict | None:
    if not CACHE_DB.exists():
        return None
    with connect(CACHE_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None


def cache_write(key: str, value: dict) -> None:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with connect(CACHE_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))


class RunLock:
    """One dispatcher per run; OS releases the lock when a process exits."""
    def __init__(self, run_dir: Path):
        self.path = run_dir / ".run.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("任务已有运行中的调度器。") from exc
        return self

    def __exit__(self, *_):
        self.handle.seek(0)
        if __import__("os").name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
