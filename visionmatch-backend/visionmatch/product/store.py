from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager,closing
from datetime import datetime
from pathlib import Path

def now():return datetime.now().astimezone().isoformat()
def encode(value):return json.dumps(value,ensure_ascii=False)
def decode(value,default=None):return json.loads(value) if value else default

class InstanceLock:
    """One live product scheduler per data root, including before recovery."""
    def __init__(self,root):self.path=Path(root)/'instance.lock';self.handle=None
    def acquire(self):
        try:
            self.handle=self.path.open('a+b');self.handle.seek(0)
            if not self.handle.read(1):self.handle.write(b'0');self.handle.flush()
            self.handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            if self.handle:self.handle.close();self.handle=None
            raise RuntimeError('该数据目录已有运行中的 V2 服务，禁止重复启动或恢复其任务')
    def release(self):
        if self.handle:self.handle.close();self.handle=None

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'product.sqlite';self.lock=threading.RLock()
        with closing(self.connect()) as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 0,created TEXT NOT NULL,settings TEXT NOT NULL,started TEXT,finished TEXT);
            CREATE TABLE IF NOT EXISTS folders(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL REFERENCES batches(id),label TEXT NOT NULL,path TEXT NOT NULL,role TEXT NOT NULL,ordinal INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'SCANNED',error TEXT,metrics TEXT,UNIQUE(batch_id,label));
            CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY,batch_id TEXT NOT NULL REFERENCES batches(id),folder_id TEXT NOT NULL REFERENCES folders(id),local_id TEXT NOT NULL,path TEXT NOT NULL,original_path TEXT NOT NULL,name TEXT NOT NULL,sha256 TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,extension TEXT NOT NULL,roi TEXT,psd_path TEXT,UNIQUE(folder_id,local_id));
            CREATE TABLE IF NOT EXISTS matches(target_id INTEGER PRIMARY KEY REFERENCES images(id),batch_id TEXT NOT NULL,folder_id TEXT NOT NULL,proposed_source_id INTEGER,assigned_source_id INTEGER,status TEXT NOT NULL DEFAULT 'PENDING',confirmed INTEGER NOT NULL DEFAULT 0,locked INTEGER NOT NULL DEFAULT 0,manual INTEGER NOT NULL DEFAULT 0,candidates TEXT NOT NULL DEFAULT '[]',suggestion TEXT,notes TEXT NOT NULL DEFAULT '[]',review TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,batch_id TEXT,kind TEXT NOT NULL,created TEXT NOT NULL,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL,max_calls INTEGER NOT NULL,used_calls INTEGER NOT NULL DEFAULT 0,state TEXT NOT NULL,concurrency INTEGER NOT NULL,started TEXT NOT NULL,finished TEXT);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL,folder_id TEXT NOT NULL,run_id TEXT NOT NULL,kind TEXT NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL,started TEXT,finished TEXT,error TEXT,metadata TEXT);
            CREATE TABLE IF NOT EXISTS model_cache(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS naming_plans(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL,revision INTEGER NOT NULL,mode TEXT NOT NULL,digest TEXT NOT NULL,payload TEXT NOT NULL,status TEXT NOT NULL,created TEXT NOT NULL,result TEXT);
            CREATE TABLE IF NOT EXISTS review_commands(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL,created TEXT NOT NULL,action TEXT NOT NULL,before_state TEXT NOT NULL,after_state TEXT NOT NULL,undone INTEGER NOT NULL DEFAULT 0);
            ''')
    def connect(self):
        db=sqlite3.connect(self.path,timeout=20);db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA busy_timeout=20000');return db
    @contextmanager
    def transaction(self):
        with self.lock:
            db=self.connect()
            try:db.execute('BEGIN IMMEDIATE');yield db;db.commit()
            except BaseException:db.rollback();raise
            finally:db.close()
    def rows(self,sql,args=()):
        with closing(self.connect()) as db:return [dict(x) for x in db.execute(sql,args)]
    def one(self,sql,args=()):
        rows=self.rows(sql,args)
        return rows[0] if rows else None
    def event(self,db,batch,kind,payload):db.execute('INSERT INTO events(batch_id,kind,created,payload) VALUES (?,?,?,?)',(batch,kind,now(),encode(payload)))
    def batch_root(self,batch):
        if not self.one('SELECT id FROM batches WHERE id=?',(batch,)):raise ValueError('批次不存在')
        path=self.root/'batches'/batch;path.mkdir(parents=True,exist_ok=True);return path
