from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..provider import executable,file_hash,scrub
from ..schemas import PAIR_SCHEMA,PAIR_PROMPT,REVIEW_SCHEMA,REVIEW_PROMPT,validate_output
from ..matching import auto_eligible
from ..ultra.sheets import make_sheet,digest,save_json
from ..ultra.workflow import group_schema
from ..ultra.requests import Client
from ..ultra.__main__ import SHARED_PROMPT
from .store import now,encode,decode

logger=logging.getLogger(__name__)

class Queue:
    def __init__(self,service,invoker=None):
        self.service=service;self.store=service.store;self.invoker=invoker
        self.stop_event=threading.Event();self.sheet_lock=threading.Lock();self.pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='visionmatch-v2');self.active={};self.limit=2;self.thread=None
        self.loop_error=None;self.last_tick=None
    def recover(self):
        with self.store.transaction() as db:
            affected=db.execute("SELECT DISTINCT run_id FROM jobs WHERE state='RUNNING'").fetchall()
            db.execute("UPDATE jobs SET state='INTERRUPTED',error='进程中断，远端用量未知，需显式恢复' WHERE state='RUNNING'")
            for r in affected:db.execute("UPDATE runs SET state='PAUSED_UNKNOWN' WHERE id=?",(r['run_id'],))
    def start(self):
        self.recover();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=2)
        self.pool.shutdown(wait=False,cancel_futures=True)
    def enqueue(self,batch,max_calls,concurrency,folders=None,kind='GROUP',body=None):
        if self.thread is not None and (not self.thread.is_alive() or self.loop_error):
            raise ValueError('匹配调度器未就绪，请先检查后端状态；未创建新的模型任务')
        if not 1<=max_calls<=200 or concurrency not in (1,2,3,4):raise ValueError('预算1～200，并发1～4')
        if self.active and concurrency!=self.limit:raise ValueError('当前有任务执行，请在空闲后调整全局并发')
        state=self.service.detail(batch);selected=[f for f in state['folders'] if f['role']=='TARGET' and (not folders or f['id'] in folders)]
        if not selected:raise ValueError('没有目标文件夹')
        ident=uuid.uuid4().hex
        with self.store.transaction() as db:
            self.service._mutable(db,batch)
            if db.execute("SELECT 1 FROM jobs WHERE batch_id=? AND state IN ('RUNNING','PENDING','INTERRUPTED')",(batch,)).fetchone():raise ValueError('该批次已有未完成任务，请先恢复或停止领取')
            db.execute('INSERT INTO runs(id,batch_id,max_calls,state,concurrency,started) VALUES (?,?,?,?,?,?)',(ident,batch,max_calls,'RUNNING',concurrency,now()))
            for f in selected:db.execute('INSERT INTO jobs(id,batch_id,folder_id,run_id,kind,body,state) VALUES (?,?,?,?,?,?,?)',(uuid.uuid4().hex,batch,f['id'],ident,kind,encode(body or {}),'PENDING'))
            db.execute("UPDATE batches SET status='MATCHING',started=?,finished=NULL WHERE id=?",(now(),batch));self.store.event(db,batch,'MODEL_BUDGET_AUTHORIZED',{'run_id':ident,'max_calls':max_calls,'concurrency':concurrency,'kind':kind})
        self.limit=concurrency;return ident
    def claim(self):
        with self.store.transaction() as db:
            job=db.execute("SELECT j.*,r.max_calls,r.used_calls FROM jobs j JOIN runs r ON r.id=j.run_id WHERE j.state='PENDING' AND r.state='RUNNING' ORDER BY r.started,j.folder_id LIMIT 1").fetchone()
            if not job:return None
            if job['used_calls']>=job['max_calls']:
                db.execute("UPDATE runs SET state='BUDGET_PAUSED' WHERE id=?",(job['run_id'],));return None
            db.execute("UPDATE jobs SET state='RUNNING',started=? WHERE id=?",(now(),job['id']))
            db.execute('UPDATE runs SET used_calls=used_calls+1 WHERE id=?',(job['run_id'],))
            db.execute("UPDATE folders SET state='RUNNING',error=NULL WHERE id=?",(job['folder_id'],))
            return dict(job)
    def perform(self,job):
        start=time.perf_counter();batch=job['batch_id'];folder=job['folder_id'];root=self.store.batch_root(batch)/'requests'/job['id'];root.mkdir(parents=True,exist_ok=True)
        settings=self.service.detail(batch)['settings'];sources,targets=self.service.records(batch,folder)
        for image in sources+targets:
            if file_hash(Path(image['path']))!=image['sha256']:raise ValueError('源图已变化，请重新导入批次')
        body=decode(job['body'],{})
        if job['kind']=='GROUP':
            with self.sheet_lock:
                a=make_sheet(sources,self.store.root/'sheet_cache',cell=settings['cell']);b=make_sheet(targets,self.store.root/'sheet_cache',cell=settings['cell'])
            images=[a['path'],b['path']];schema=group_schema(sources,targets);prompt='第一张为A主图索引，第二张为B目标图索引。处理全部B编号，给出候选与证据。\n'+SHARED_PROMPT
        else:
            target=self.store.one('SELECT * FROM images WHERE id=? AND batch_id=?',(body['target_id'],batch))
            candidates=[self.store.one('SELECT * FROM images WHERE id=? AND batch_id=?',(n,batch)) for n in body['source_ids']]
            if not target or not candidates or any(not x for x in candidates):raise ValueError('无效原图复核编号')
            if len(candidates)>1:images=[target['path'],candidates[0]['path'],candidates[1]['path']];schema=REVIEW_SCHEMA;prompt=REVIEW_PROMPT
            else:images=[candidates[0]['path'],target['path']];schema=PAIR_SCHEMA;prompt=PAIR_PROMPT
        binary=Path(executable());cache_key=digest({'images':[file_hash(Path(x)) for x in images],'schema':schema,'prompt':prompt,'model':'gpt-6.1-terra','effort':'medium','cli':[str(binary),binary.stat().st_mtime_ns]})
        cached=self.store.one('SELECT value FROM model_cache WHERE key=?',(cache_key,))
        if cached:
            result=decode(cached['value']);answer={**result,'cache_hit':True,'usage':None,'historical_usage':result.get('usage')}
        elif self.invoker:answer=self.invoker(images,prompt,schema,job)
        else:
            client=Client(root,approved_plan=job['id'],max_calls=1)
            try:answer=client.call(images,prompt,schema,job['id'],kind=job['kind'],timeout=settings['request_timeout'])
            finally:client.close()
        metadata={k:answer.get(k) for k in ('usage','elapsed_seconds','cache_hit','thread_id')};metadata.update(wall_seconds=round(time.perf_counter()-start,3),real_cli_calls=0 if answer.get('cache_hit') else 1)
        save_json(root/'response.json',answer)
        valid_cache=True
        if job['kind']=='GROUP':
            coverage=self.service.ingest(batch,folder,answer['result'],metadata,'CACHE' if answer.get('cache_hit') else 'REAL_MODEL')
            valid_cache=not coverage['missing']
        else:
            result=answer['result'];validate_output(result,schema)
            if schema is REVIEW_SCHEMA:
                winner=result['winner'];sid=body['source_ids'][0 if winner=='CANDIDATE_1' else 1] if winner in ('CANDIDATE_1','CANDIDATE_2') else None
                detail=result['candidate_1' if winner=='CANDIDATE_1' else 'candidate_2'] if sid else None
            else:sid=body['source_ids'][0] if result['verdict']=='MATCH' else None;detail=result
            with self.store.transaction() as db:
                current=db.execute('SELECT * FROM matches WHERE target_id=?',(body['target_id'],)).fetchone()
                db.execute('UPDATE matches SET review=? WHERE target_id=?',(encode({'before':dict(current),'result':result,'metadata':metadata}),body['target_id']))
                if not current['manual']:
                    cs=[{'source_id':sid,'decision':'MATCH','needs_review':not auto_eligible(detail),'evidence':detail['matched_features']}] if sid else []
                    db.execute('UPDATE matches SET candidates=? WHERE target_id=?',(encode(cs),body['target_id']));self.service.reconcile(db,batch,folder)
                db.execute("UPDATE folders SET state='COMPLETE',metrics=? WHERE id=?",(encode(metadata),folder))
                db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(batch,));self.store.event(db,batch,'ORIGINAL_REVIEW_COMPLETED',{'target_id':body['target_id'],'source_id':sid,'metadata':metadata})
        with self.store.transaction() as db:
            if answer.get('cache_hit'):db.execute('UPDATE runs SET used_calls=used_calls-1 WHERE id=?',(job['run_id'],))
            elif valid_cache:db.execute('INSERT OR REPLACE INTO model_cache VALUES (?,?)',(cache_key,encode(answer)))
            db.execute("UPDATE jobs SET state='SUCCESS',finished=?,metadata=? WHERE id=?",(now(),encode(metadata),job['id']))
        return metadata
    def complete_runs(self):
        with self.store.transaction() as db:
            runs=db.execute("SELECT * FROM runs WHERE state IN ('RUNNING','BUDGET_PAUSED','PAUSED','CANCELLED') AND finished IS NULL").fetchall()
            for r in runs:
                states=[x[0] for x in db.execute('SELECT state FROM jobs WHERE run_id=?',(r['id'],))]
                if states and not any(x in ('PENDING','RUNNING') for x in states):
                    status='CANCELLED' if r['state']=='CANCELLED' else 'COMPLETE_WITH_ERRORS' if any(x!='SUCCESS' for x in states) else 'COMPLETE'
                    db.execute('UPDATE runs SET state=?,finished=? WHERE id=?',(status,now(),r['id']));db.execute("UPDATE batches SET status='REVIEW',finished=? WHERE id=?",(now(),r['batch_id']))
                elif r['state']=='BUDGET_PAUSED' and r['used_calls']<r['max_calls']:db.execute("UPDATE runs SET state='RUNNING' WHERE id=?",(r['id'],))
    def tick(self):
        for future,job in list(self.active.items()):
            if not future.done():continue
            try:future.result()
            except Exception as exc:
                from ..provider_async import classify_error
                error=classify_error(str(exc))
                with self.store.transaction() as db:
                    db.execute("UPDATE jobs SET state='FAILED',finished=?,error=? WHERE id=?",(now(),str(exc)[:3000],job['id']));db.execute("UPDATE folders SET state='ERROR',error=? WHERE id=?",(str(exc)[:1500],job['folder_id']))
                    db.execute("UPDATE matches SET status='ERROR',notes=? WHERE folder_id=? AND manual=0",(encode([str(exc)[:1500]]),job['folder_id']))
                    if error.stop:db.execute("UPDATE runs SET state='QUOTA_OR_AUTH_PAUSED' WHERE id=?",(job['run_id'],))
                    self.store.event(db,job['batch_id'],'MODEL_JOB_FAILED',{'job_id':job['id'],'error':str(exc)[:3000]})
            # Keep finished futures until their DB bookkeeping commits. Retrying
            # a DB write must never submit the model request a second time.
            self.active.pop(future)
        self.complete_runs()
        while len(self.active)<self.limit:
            job=self.claim()
            if not job:break
            self.active[self.pool.submit(self.perform,job)]=job

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.tick();self.loop_error=None;self.last_tick=now()
            except Exception as exc:
                self.loop_error=scrub(str(exc))[:1000]
                try:logger.exception('VisionMatch scheduler error; retaining queued jobs and call budgets')
                except OSError:pass  # A full log disk must not kill the worker.
                self.stop_event.wait(2)
                continue
            self.stop_event.wait(.15)
