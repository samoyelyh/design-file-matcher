"""Budgeted CLI transport for the experimental mode. No implicit real-call budget."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator

from .. import provider
from ..schemas import schema_hash
from .sheets import digest,save_json

class Client:
    def __init__(self, root, *, approved_plan=None, max_calls=0):
        self.root=Path(root);self.approved_plan=approved_plan;self.max_calls=max_calls
        self.db=sqlite3.connect(self.root/'ultra_requests.sqlite',timeout=15)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, cache_key TEXT, status TEXT, metadata TEXT); CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT);')
        self.db.commit()

    def close(self):self.db.close()

    def call(self, images, prompt, schema, plan_id, *, timeout=120, kind='GROUP', validator=None):
        binary=Path(provider.executable())
        cli_identity=[str(binary),binary.stat().st_size,binary.stat().st_mtime_ns]
        key=digest({'hashes':[provider.file_hash(Path(p)) for p in images],'model':'gpt-6.1-terra','effort':'medium','prompt':prompt,'schema':schema_hash(schema),'kind':kind,'cli_identity':cli_identity})
        cached=self.db.execute('SELECT value FROM cache WHERE key=?',(key,)).fetchone()
        if cached:
            value=json.loads(cached[0]);Draft202012Validator(schema).validate(value['result'])
            if validator:validator(value['result'])
            return {**value,'cache_hit':True}
        if self.approved_plan!=plan_id or self.max_calls<1:raise ValueError('No approved real-call budget for this plan')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]>=self.max_calls:raise ValueError('call budget exhausted')
            ident=uuid.uuid4().hex
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?)',(ident,key,'RESERVED','{}'));self.db.commit()
        except BaseException:self.db.rollback();raise
        folder=self.root/'requests'/ident;folder.mkdir(parents=True,exist_ok=False)
        start=time.perf_counter();meta={'kind':kind,'model':'gpt-6.1-terra','cache_hit':False,'request_dir':str(folder),'status':'FAILED','usage':None}
        try:
            provider.login_status();staged=[]
            for i,path in enumerate(images,1):
                src=Path(path);dest=folder/f'image_{i}{src.suffix.lower()}';shutil.copyfile(src,dest)
                if provider.file_hash(src)!=provider.file_hash(dest):raise ValueError('staging hash mismatch')
                staged.append(dest)
            save_json(folder/'schema.json',schema);(folder/'prompt.txt').write_text(prompt,encoding='utf-8')
            args=provider.codex_args(workdir=folder,images=staged,schema_path=folder/'schema.json',output_path=folder/'result.json',model='gpt-6.1-terra',effort='medium')
            process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',env=provider.safe_environment(),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:stdout,stderr=process.communicate(prompt,timeout=timeout)
            except subprocess.TimeoutExpired:
                provider.terminate_process(process);stdout,stderr=process.communicate();raise RuntimeError('TIMEOUT_REMOTE_UNKNOWN')
            events=[];turns=0;tools=False
            for line in stdout.splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                typ=event.get('type');item=event.get('item') or {}
                if typ=='turn.completed':meta['usage']=event.get('usage');turns+=1
                if typ=='thread.started':meta['thread_id']=event.get('thread_id')
                if typ in ('item.started','item.completed') and item.get('type') not in ('agent_message','reasoning',None):tools=True
                if typ in ('error','turn.failed'):meta['service_error']=provider.scrub(json.dumps(event,ensure_ascii=False))
                events.append({'type':typ,'item_type':item.get('type'),'usage':event.get('usage'),'error':meta.get('service_error') if typ in ('error','turn.failed') else None})
            save_json(folder/'events.json',events)
            if process.returncode or tools or turns!=1:raise RuntimeError(meta.get('service_error') or provider.scrub(stderr+stdout))
            result=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'));Draft202012Validator(schema).validate(result)
            if validator:validator(result)
            meta.update(status='SUCCESS',result=result)
            return meta
        except Exception as exc:
            meta['error']=provider.scrub(str(exc));raise
        finally:
            meta['elapsed_seconds']=round(time.perf_counter()-start,3);save_json(folder/'attempt.json',meta)
            if meta['status']=='SUCCESS':self.db.execute('INSERT OR REPLACE INTO cache VALUES (?,?)',(key,json.dumps(meta,ensure_ascii=False)))
            self.db.execute('UPDATE calls SET status=?,metadata=? WHERE id=?',(meta['status'],json.dumps(meta,ensure_ascii=False),ident));self.db.commit()
