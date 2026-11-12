from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .store import Store,InstanceLock,encode,decode
from .service import Service
from .naming import Naming
from .tasks import Queue
from .reviewing import Reviews,eligible
from ..ultra.sheets import digest
from ..provider import executable,ProviderError

STATIC=Path(__file__).parent/'static'
EXTS={'.png','.jpg','.jpeg','.webp'}

def create_app(data_root=None,start_worker=True,invoker=None):
    store=Store(data_root or os.getenv('VISIONMATCH_V2_DATA') or Path(__file__).resolve().parents[2]/'runtime'/'v2')
    service=Service(store);naming=Naming(service);queue=Queue(service,invoker);reviews=Reviews(service)
    @asynccontextmanager
    async def lifespan(app):
        guard=InstanceLock(store.root)
        try:
            if start_worker:guard.acquire();naming.recover();queue.start()
            yield
        finally:queue.close();guard.release()
    app=FastAPI(title='美工图片智能匹配与批量命名 V2',lifespan=lifespan)
    app.state.store=store;app.state.service=service;app.state.naming=naming;app.state.queue=queue

    @app.exception_handler(ValueError)
    async def bad_request(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)
    @app.exception_handler(OSError)
    async def file_error(request,exc):return JSONResponse({'detail':str(exc)},status_code=400)

    @app.middleware('http')
    async def local_only(request,call_next):
        # Local single-user app. Reject cross-origin writes and DNS rebinding.
        host=request.headers.get('host','').split(':')[0]
        if host not in ('localhost','127.0.0.1','testserver','::1'):return JSONResponse({'detail':'仅限本机访问'},status_code=403)
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'拒绝跨来源写入'},status_code=403)
        return await call_next(request)

    @app.get('/api/health')
    def health():
        try:cli_path=executable();cli_ready=True;cli_error=None
        except ProviderError as exc:cli_path=None;cli_ready=False;cli_error=str(exc)
        worker_alive=bool(queue.thread and queue.thread.is_alive())
        worker_ready=not start_worker or (worker_alive and queue.loop_error is None)
        return {'ok':cli_ready and worker_ready,'product':'VisionMatch V2','model':'gpt-6.1-terra','auth':'现有 Codex CLI ChatGPT 登录','data_root':str(store.root),'validation_replay':invoker is not None,'cli_ready':cli_ready,'cli_path':cli_path,'cli_error':cli_error,'worker_alive':worker_alive,'worker_error':queue.loop_error,'worker_last_tick':queue.last_tick}
    @app.get('/api/batches')
    def batches():return store.rows('SELECT id,name,status,created FROM batches ORDER BY created DESC')
    @app.post('/api/batches')
    def create(body:dict):return service.create(str(body.get('name','新批次'))[:160])
    @app.get('/api/batches/{bid}')
    def detail(bid:str):
        state=service.view(bid);state['jobs']=store.rows('SELECT * FROM jobs WHERE batch_id=? ORDER BY started DESC',(bid,))
        for j in state['jobs']:j['metadata']=decode(j['metadata'],{})
        for row in state['matches']:row['can_bulk_confirm'],row['bulk_confirm_reason']=eligible(row)
        state['undo']=reviews.latest(bid)
        return state
    @app.post('/api/batches/{bid}/import')
    def local_import(bid:str,body:dict):return service.import_local(bid,body['source'],body['targets'])

    @app.post('/api/batches/{bid}/upload')
    async def upload(bid:str,files:list[UploadFile]=File(...),source_folder:str=Form('主图')):
        root=store.batch_root(bid)/'uploads'/uuid.uuid4().hex;root.mkdir(parents=True)
        total=0;count=0
        def write(name,stream):
            nonlocal total,count
            parts=PurePosixPath(name.replace('\\','/'))
            if parts.is_absolute() or '..' in parts.parts or any(':' in p for p in parts.parts):raise ValueError('上传路径不安全')
            if parts.suffix.lower() not in EXTS:return
            dest=root.joinpath(*parts.parts)
            if not dest.resolve().is_relative_to(root.resolve()):raise ValueError('非法路径')
            if any(p.name.casefold()==dest.name.casefold() for p in dest.parent.iterdir()) if dest.parent.exists() else False:raise ValueError('上传文件名重复')
            dest.parent.mkdir(parents=True,exist_ok=True);count+=1
            if count>1000:raise ValueError('单批最多1000张')
            with dest.open('xb') as out:
                while chunk:=stream.read(1024*1024):
                    total+=len(chunk)
                    if total>512*1024*1024:raise ValueError('单批上传超过512MB')
                    out.write(chunk)
        for file in files:
            if Path(file.filename or '').suffix.lower()=='.zip':
                with zipfile.ZipFile(file.file) as arc:
                    for info in arc.infolist():
                        if info.is_dir():continue
                        if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP不允许符号链接')
                        if info.file_size>64*1024*1024:raise ValueError('单图超过64MB')
                        with arc.open(info) as stream:write(info.filename,stream)
            else:write(file.filename or '',file.file)
        groups=sorted({p.parent for p in root.rglob('*') if p.suffix.lower() in EXTS and p.is_file()})
        selected=[p for p in groups if p.name==source_folder]
        if len(selected)!=1:raise ValueError('需有且仅有一个主图目录，请核对主图目录名')
        return service.import_local(bid,str(selected[0]),[{'path':str(p),'label':p.name} for p in groups if p!=selected[0]])

    @app.get('/api/images/{iid}')
    def image(iid:int,full:bool=False,psd:bool=False):
        record=store.one('SELECT * FROM images WHERE id=?',(iid,))
        if not record:raise HTTPException(404)
        path=Path(record['psd_path'] if psd else record['path']) if not psd or record['psd_path'] else None
        if not path:raise ValueError('尚未关联PSD')
        if full and not psd:return FileResponse(path)
        if not psd:
            stamp=path.stat();cache=store.root/'thumbnails';cache.mkdir(exist_ok=True)
            dest=cache/(digest([record['sha256'],stamp.st_mtime_ns,stamp.st_size,560])+'.jpg')
            if not dest.exists():
                temporary=cache/(uuid.uuid4().hex+'.partial')
                try:
                    with Image.open(path) as picture:
                        picture=ImageOps.exif_transpose(picture);picture.thumbnail((560,560));picture.convert('RGB').save(temporary,format='JPEG',quality=88)
                    os.replace(temporary,dest)
                finally:
                    if temporary.exists():temporary.unlink()
            return FileResponse(dest,media_type='image/jpeg',headers={'Cache-Control':'private, max-age=86400'})
        output=io.BytesIO()
        if psd:
            from psd_tools import PSDImage
            picture=PSDImage.open(path).composite()
        else:picture=ImageOps.exif_transpose(Image.open(path))
        picture.thumbnail((1000,1000));picture.convert('RGB').save(output,format='JPEG',quality=88)
        return Response(output.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'no-cache'})
    @app.post('/api/images/{iid}/psd')
    def psd(iid:int,body:dict):
        record=store.one('SELECT * FROM images WHERE id=?',(iid,));path=Path(body['path']).resolve()
        if not record or path.suffix.lower()!='.psd' or not path.is_file():raise ValueError('无效PSD路径')
        with store.transaction() as db:
            service._mutable(db,record['batch_id']);db.execute('UPDATE images SET psd_path=? WHERE id=?',(str(path),iid));store.event(db,record['batch_id'],'PSD_LINKED',{'image_id':iid,'path':str(path)})
        return {'ok':True}
    @app.post('/api/batches/{bid}/review/{tid}')
    def review(bid:str,tid:int,body:dict):
        result=reviews.apply(bid,body['action'],[tid],[{'target_id':tid,'source_id':body['source_id']}] if body['action']=='assign' else None,allow_locked=True,strict_confirm=False)
        return {**service.detail(bid),'operation':result}
    @app.post('/api/batches/{bid}/confirm')
    def confirm(bid:str,body:dict):return reviews.apply(bid,'confirm',body['target_ids'])
    @app.post('/api/batches/{bid}/reviews')
    def batch_review(bid:str,body:dict):return reviews.apply(bid,body['action'],body['target_ids'],body.get('mappings'),body.get('allow_locked',False))
    @app.post('/api/batches/{bid}/undo')
    def undo(bid:str,body:dict):return reviews.undo(bid,body.get('command_id'))
    @app.post('/api/batches/{bid}/selection-export')
    def selection_export(bid:str,body:dict):
        state=service.view(bid);ids={int(x) for x in body['target_ids']};selected=[m for m in state['matches'] if m['target_id'] in ids]
        if len(selected)!=len(ids):raise ValueError('导出范围含无效图片')
        return {'batch_id':bid,'selected_count':len(selected),'matches':selected}
    @app.post('/api/batches/{bid}/recording')
    def replay(bid:str,body:dict):return service.import_recording(bid,body['folder_id'],body['plan_path'],body['result_path'])

    @app.get('/api/batches/{bid}/match-plan')
    def match_plan(bid:str):
        state=service.detail(bid);folders=[f for f in state['folders'] if f['role']=='TARGET']
        return {'target_folders':len(folders),'planned_group_jobs':len(folders),'calls_upper_bound_without_retries':len(folders),'message':'无自动重试。原图复核另行授权。缓存命中不计为新模型调用。'}
    @app.post('/api/batches/{bid}/match')
    def match(bid:str,body:dict):
        if body.get('confirmed_budget') is not True:raise ValueError('启动前须确认实际模型调用预算')
        state=service.detail(bid);allowed={f['id'] for f in state['folders'] if f['role']=='TARGET'};selected=body.get('folder_ids')
        if selected is not None and (not selected or not set(selected)<=allowed):raise ValueError('请选择有效目标目录，空选择不会扩大为全部')
        if body.get('scope')=='failed':
            selected=[f['id'] for f in state['folders'] if f['id'] in allowed and f['state'] in ('ERROR','PARTIAL') and (selected is None or f['id'] in selected)]
            if not selected:raise ValueError('当前范围没有失败或结果不完整的目录')
        return {'run_id':queue.enqueue(bid,int(body['max_calls']),int(body.get('concurrency',2)),selected)}
    @app.post('/api/batches/{bid}/original-review')
    def original_review(bid:str,body:dict):
        if body.get('confirmed_budget') is not True:raise ValueError('原图复核需确认1次调用预算')
        row=store.one('SELECT * FROM matches WHERE batch_id=? AND target_id=?',(bid,body['target_id']))
        if not row or not 1<=len(body.get('source_ids',[]))<=2:raise ValueError('原图复核需1～2张主图候选')
        return {'run_id':queue.enqueue(bid,1,queue.limit,[row['folder_id']],'REVIEW',body)}
    @app.post('/api/runs/{rid}/{action}')
    def run_control(rid:str,action:str,body:dict):
        run=store.one('SELECT * FROM runs WHERE id=?',(rid,))
        if not run:raise ValueError('运行不存在')
        if action not in ('pause','resume','cancel'):raise ValueError('无效运行操作')
        with store.transaction() as db:
            if action=='pause':db.execute("UPDATE runs SET state='PAUSED' WHERE id=?",(rid,))
            elif action=='cancel':db.execute("UPDATE runs SET state='CANCELLED' WHERE id=?",(rid,));db.execute("UPDATE jobs SET state='CANCELLED' WHERE run_id=? AND state IN ('PENDING','INTERRUPTED')",(rid,))
            else:
                if run['used_calls']>=run['max_calls']:raise ValueError('预算已用尽，不能自动扩大；请建立明确授权的新运行')
                if queue.active and queue.limit!=run['concurrency']:raise ValueError('其他任务运行中，需等待后恢复本并发设置')
                queue.limit=run['concurrency']
                db.execute("UPDATE jobs SET state='PENDING' WHERE run_id=? AND state IN ('FAILED','INTERRUPTED')",(rid,));db.execute("UPDATE runs SET state='RUNNING' WHERE id=?",(rid,))
            store.event(db,run['batch_id'],'RUN_'+action.upper(),{'run_id':rid,'budget_unchanged':True})
        return {'ok':True}
    @app.post('/api/batches/{bid}/naming')
    def naming_plan(bid:str,body:dict):return naming.generate(bid,body.get('mode','COPY_AND_RENAME'),body.get('target_ids'))
    @app.get('/api/plans/{pid}')
    def get_plan(pid:str):return naming.get(pid)
    @app.post('/api/plans/{pid}/execute')
    def execute(pid:str,body:dict):return naming.execute(pid,body.get('confirmed_plan_id'),body.get('allow_in_place') is True)
    @app.post('/api/plans/{pid}/rollback')
    def rollback(pid:str,body:dict):return naming.rollback(pid,body.get('confirmed_plan_id'))

    @app.get('/api/batches/{bid}/export/{kind}')
    def export(bid:str,kind:str):
        state=service.view(bid);events=store.rows('SELECT * FROM events WHERE batch_id=? ORDER BY id',(bid,));root=store.batch_root(bid)/'downloads';root.mkdir(exist_ok=True)
        raw=json.dumps({**state,'events':events},ensure_ascii=False,indent=2).encode('utf-8')
        table=io.StringIO(newline='');fields=['folder','target','source','new_name','status','confirmed','locked'];writer=csv.DictWriter(table,fieldnames=fields);writer.writeheader()
        labels={f['id']:f['label'] for f in state['folders']}
        for m in state['matches']:writer.writerow({'folder':labels[m['folder_id']],'target':m['target']['path'],'source':m['source']['path'] if m['source'] else '', 'new_name':m['new_name'],'status':m['status'],'confirmed':m['confirmed'],'locked':m['locked']})
        if kind=='json':return Response(raw,media_type='application/json',headers={'Content-Disposition':'attachment; filename=results.json'})
        if kind=='csv':return Response(table.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=rename_mapping.csv'})
        if kind not in ('zip','html'):raise HTTPException(404)
        def thumb(record):
            if not record:return ''
            with Image.open(record['path']) as img:
                img=ImageOps.exif_transpose(img);img.thumbnail((220,220));out=io.BytesIO();img.convert('RGB').save(out,'JPEG',quality=75)
            return '<img width="160" src="data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()+'">'
        rows=['<tr><td>'+html.escape(labels[m['folder_id']])+'</td><td>'+thumb(m['source'])+'</td><td>'+thumb(m['target'])+'</td><td>'+html.escape(m['target']['name'])+'</td><td>'+html.escape(m['new_name'] or '')+'</td><td>'+m['status']+'</td></tr>' for m in state['matches']]
        report=('<!doctype html><meta charset="utf-8"><title>命名审核报告</title><style>body{font:15px sans-serif}td{padding:10px;border-bottom:1px solid #ddd}img{object-fit:contain}</style><h1>'+html.escape(state['name'])+'</h1><p>此报告记录审核状态；人工确认前不执行命名。识别来源与用量见 results.json。</p><table><tr><th>文件夹</th><th>主图</th><th>目标</th><th>原名</th><th>新名</th><th>状态</th></tr>'+''.join(rows)+'</table>').encode('utf-8')
        if kind=='html':return Response(report,media_type='text/html',headers={'Content-Disposition':'attachment; filename=index.html'})
        if any(p['status']=='EXECUTING' for p in state['naming_plans']):raise ValueError('命名执行中，等待完成后再导出完整ZIP')
        dest=root/('batch_export_'+uuid.uuid4().hex+'.zip');seen_outputs=set()
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as arc:
            arc.writestr('results.json',raw);arc.writestr('rename_mapping.csv',table.getvalue().encode('utf-8-sig'));arc.writestr('events.json',encode(events));arc.writestr('index.html',report)
            for plan in state['naming_plans']:
                p=naming.get(plan['id']);output=Path(p['payload']['output_root'])
                if str(output) in seen_outputs:continue
                seen_outputs.add(str(output))
                if output.exists():
                    for path in output.rglob('*'):
                        if path.is_file():arc.write(path,'naming/'+output.name+'/'+path.relative_to(output).as_posix())
        return FileResponse(dest,filename='VisionMatch_V2_export.zip')
    app.mount('/',StaticFiles(directory=STATIC,html=True),name='web')
    return app

app=create_app()
