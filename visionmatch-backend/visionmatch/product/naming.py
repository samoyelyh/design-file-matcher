from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path

from ..provider import file_hash
from ..ultra.naming import preview,execute_copy,valid_name
from ..ultra.sheets import digest,save_json
from .store import now,encode,decode

OLD_PROJECT=Path(r'C:\Users\Woda\Documents\ChatGPT\美工命名系统').resolve()

def fold(path):return str(Path(path).resolve()).casefold()

def checked_move_plan(rows,allowed_roots):
    roots={fold(p) for p in allowed_roots};eligible=[r for r in rows if r['status']=='READY']
    source_set={fold(r['original_path']) for r in eligible};destinations=[]
    for row in eligible:
        before=Path(row['original_path']).resolve();after=Path(row['output_path']).resolve()
        if fold(before.parent) not in roots or before.parent!=after.parent:raise ValueError('原地命名路径不属于本批次目标目录')
        if before.is_relative_to(OLD_PROJECT) or after.is_relative_to(OLD_PROJECT):raise ValueError('禁止修改旧项目目录')
        if not valid_name(after.name):raise ValueError('非法目标文件名')
        if file_hash(before)!=row['original_sha256']:raise ValueError('原文件内容发生变化')
        destinations.append(fold(after))
        existing=next((p for p in after.parent.iterdir() if p.name.casefold()==after.name.casefold()),None)
        if existing and fold(existing) not in source_set:raise ValueError('目标位置已有外部文件，禁止覆盖')
    if len(destinations)!=len(set(destinations)):raise ValueError('目标文件名大小写冲突')
    return eligible

def execute_in_place(rows,allowed_roots,journal_root):
    """Two-phase same-directory hard-link staging; no overwrite or image conversion."""
    root=Path(journal_root);root.mkdir(parents=True,exist_ok=True)
    eligible=checked_move_plan(rows,allowed_roots)
    rollback=[]
    for row in eligible:
        before=Path(row['original_path']);after=Path(row['output_path'])
        rollback.append({'target_id':row['target_id'],'before':str(before),'after':str(after),'sha256':row['original_sha256'],'temporary':str(before.parent/('.vmv2-'+uuid.uuid4().hex+'.tmp')),'phase':'PLANNED'})
    save_json(root/'rollback_manifest.json',{'mode':'RENAME_IN_PLACE','created':now(),'items':rollback,'requires_explicit_confirmation':True})
    def persist():save_json(root/'in_place_journal.json',{'updated':now(),'items':rollback})
    try:
        for r in rollback:
            os.link(r['before'],r['temporary']);r['phase']='STAGED';persist()
        for r in rollback:
            if file_hash(Path(r['before']))!=r['sha256']:raise ValueError('暂存后源图变化，停止')
            Path(r['before']).unlink();r['phase']='SOURCE_REMOVED';persist()
        for r in rollback:
            os.link(r['temporary'],r['after']);r['phase']='PUBLISHED';persist()
            if file_hash(Path(r['after']))!=r['sha256']:raise ValueError('命名后哈希不一致')
        for r in rollback:Path(r['temporary']).unlink();r['phase']='VERIFIED';persist()
    except Exception:
        # Preserve all staging links and the recovery journal; never remove an
        # unexpected external file while attempting an automatic rollback.
        persist();raise
    outcomes=[];by_id={r['target_id']:r for r in rollback}
    for row in rows:
        item={**row,'mode':'RENAME_IN_PLACE','executed_at':now()}
        if row['target_id'] in by_id:item.update(status='RENAMED_VERIFIED',output_sha256=row['original_sha256'])
        outcomes.append(item)
    return outcomes

class Naming:
    def __init__(self,service):self.service=service;self.store=service.store
    def recover(self):
        with self.store.transaction() as db:
            for plan in db.execute("SELECT id,batch_id FROM naming_plans WHERE status='EXECUTING'").fetchall():
                db.execute("UPDATE naming_plans SET status='INTERRUPTED' WHERE id=?",(plan['id'],));self.store.event(db,plan['batch_id'],'NAMING_INTERRUPTED',{'plan_id':plan['id'],'automatic_file_mutation':False})
    def generate(self,batch,mode='COPY_AND_RENAME',target_ids=None):
        if mode not in ('COPY_AND_RENAME','RENAME_IN_PLACE'):raise ValueError('无效命名模式')
        with self.store.transaction() as db:
            b=self.service._mutable(db,batch);revision=b['revision']
            state=self.service.view(batch)
            root=self.store.batch_root(batch)/'exports'/f'{mode.lower()}_revision_{revision}'
            if target_ids is not None:
                selected={int(x) for x in target_ids}
                if not selected or not selected<={m['target_id'] for m in state['matches']}:raise ValueError('命名范围为空或含无效目标图片')
                state['matches']=[m for m in state['matches'] if m['target_id'] in selected]
                root=root/('scope_'+digest(sorted(selected))[:12])
            all_rows=[]
            for folder in (f for f in state['folders'] if f['role']=='TARGET'):
                sources,targets=self.service.records(batch,folder['id'])
                by_db={x['id']:x['local_id'] for x in state['images']}
                decisions=[]
                for row in (r for r in state['matches'] if r['folder_id']==folder['id']):
                    decisions.append({'target_id':by_db[row['target_id']],'source_id':by_db.get(row['assigned_source_id']),'decision':'MATCH' if row['assigned_source_id'] else 'UNMATCHED','confirmed':bool(row['confirmed'] and row['status']=='CONFIRMED'),'needs_review':not bool(row['confirmed']),'conflicts':[],'evidence':row['notes'] or row['suggestion'].get('evidence',[]),'model':'gpt-6.1-terra' if row['suggestion'] else 'HUMAN_CONFIRMED'})
                rows=preview({'folder_id':folder['label'],'sources':sources,'targets':targets},decisions,root)
                targets_by_local={x['local_id']:x for x in state['images'] if x['folder_id']==folder['id']}
                for row in rows:
                    record=targets_by_local[row['target_id']];row['local_target_id']=row['target_id'];row['target_id']=record['id'];row['folder_uuid']=folder['id']
                    if mode=='RENAME_IN_PLACE' and row['new_name']:
                        after=Path(row['original_path']).with_name(row['new_name']);row['output_path']=str(after)
                        if Path(row['original_path']).name==after.name and row['status']=='READY':row['status']='NOOP';row['reason']='已使用主图名称，无需改名'
                        if Path(row['original_path']).resolve().is_relative_to(OLD_PROJECT):row.update(status='BLOCKED',reason='禁止修改旧项目')
                all_rows.extend(rows)
            for row in all_rows:
                if row['status'] not in ('READY','NOOP'):continue
                try:
                    if file_hash(Path(row['original_path']))!=row['original_sha256'] or file_hash(Path(row['source_path']))!=row['source_sha256']:raise ValueError('原图或主图内容变化，请重新导入')
                except (OSError,ValueError) as exc:row.update(status='BLOCKED',reason=str(exc))
            if mode=='COPY_AND_RENAME':
                owned=set();audit=root/'naming_audit.jsonl'
                if audit.exists():
                    for line in audit.read_text(encoding='utf-8').splitlines():
                        try:owned.add(__import__('json').loads(line)['operation_id'])
                        except (ValueError,KeyError):continue
                for row in all_rows:
                    if row['status']!='READY':continue
                    target=Path(row['output_path'])
                    existing=next((p for p in target.parent.iterdir() if p.name.casefold()==target.name.casefold()),None) if target.parent.exists() else None
                    if existing and not (row['operation_id'] in owned and existing.is_file() and file_hash(existing)==row['original_sha256']):row.update(status='BLOCKED',reason='输出位置存在未验证的同名文件，禁止覆盖')
            # In-place cycles are allowed only when every occupied target is also
            # an eligible source in this immutable plan.
            if mode=='RENAME_IN_PLACE':
                changed=True
                while changed:
                    changed=False;movable={fold(r['original_path']) for r in all_rows if r['status']=='READY'}
                    for row in all_rows:
                        if row['status']!='READY':continue
                        target=Path(row['output_path']);existing=next((p for p in target.parent.iterdir() if p.name.casefold()==target.name.casefold()),None)
                        if existing and fold(existing) not in movable:row.update(status='BLOCKED',reason='目标位置存在非本计划可移动文件');changed=True
            payload={'batch_id':batch,'revision':revision,'mode':mode,'output_root':str(root),'rows':all_rows}
            fingerprint=digest(payload)
            existing=db.execute('SELECT id FROM naming_plans WHERE batch_id=? AND digest=?',(batch,fingerprint)).fetchone()
            if existing:return self.get(existing['id'])
            ident=uuid.uuid4().hex
            db.execute('INSERT INTO naming_plans VALUES (?,?,?,?,?,?,?,?,?)',(ident,batch,revision,mode,fingerprint,encode(payload),'PREVIEW',now(),None))
            self.store.event(db,batch,'NAMING_PREVIEW',{'plan_id':ident,'mode':mode,'revision':revision})
        return self.get(ident)
    def get(self,ident):
        plan=self.store.one('SELECT * FROM naming_plans WHERE id=?',(ident,))
        if not plan:raise ValueError('命名计划不存在')
        plan['payload']=decode(plan['payload']);plan['result']=decode(plan['result'])
        plan['counts']=dict(__import__('collections').Counter(x['status'] for x in plan['payload']['rows']))
        return plan
    def execute(self,ident,confirmed_plan_id,allow_in_place=False):
        if ident!=confirmed_plan_id:raise ValueError('必须确认当前命名预览ID')
        plan=self.get(ident);payload=plan['payload'];batch=plan['batch_id']
        if plan['mode']=='RENAME_IN_PLACE' and not allow_in_place:raise ValueError('原地改名必须明确选择并确认')
        with self.store.transaction() as db:
            b=self.service._mutable(db,batch)
            if b['revision']!=plan['revision']:raise ValueError('审核或图片状态已变化，请重新生成命名预览')
            if db.execute("SELECT 1 FROM jobs WHERE batch_id=? AND state='RUNNING'",(batch,)).fetchone():raise ValueError('模型任务仍在运行，请等待结束或暂停后再命名')
            if plan['status']=='EXECUTED':
                for row in plan['result']['rows']:
                    if row['status'] in ('COPIED_VERIFIED','SKIPPED_VALID','RENAMED_VERIFIED') and file_hash(Path(row['output_path']))!=row['original_sha256']:raise ValueError('已执行输出发生变化')
                return plan
            if plan['status']=='EXECUTING':raise ValueError('命名正在执行')
            if plan['status']=='ROLLED_BACK':raise ValueError('计划已回退，请重新生成预览')
            if plan['mode']=='RENAME_IN_PLACE' and plan['status'] in ('FAILED','INTERRUPTED'):raise ValueError('原地命名中断，请先根据回退清单恢复；禁止盲目重复执行')
            db.execute("UPDATE naming_plans SET status='EXECUTING' WHERE id=?",(ident,))
        root=Path(payload['output_root']);root.mkdir(parents=True,exist_ok=True)
        save_json(root/'naming_preview.json',payload)
        try:
            if plan['mode']=='COPY_AND_RENAME':outcomes=execute_copy(payload['rows'],root)
            else:
                for r in payload['rows']:
                    if r['status'] in ('READY','NOOP') and (file_hash(Path(r['original_path']))!=r['original_sha256'] or file_hash(Path(r['source_path']))!=r['source_sha256']):raise ValueError('主图或目标图内容变化，停止原地命名')
                roots=[f['path'] for f in self.service.detail(batch)['folders'] if f['role']=='TARGET']
                outcomes=execute_in_place(payload['rows'],roots,root)
            result={'rows':outcomes,'output_root':str(root),'mode':plan['mode']}
            save_json(root/'results.json',result)
            with (root/'rename_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                fields=['target_id','folder_id','original_path','original_name','original_sha256','source_path','source_name','new_name','output_path','decision','model','status','reason']
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(outcomes)
            with self.store.transaction() as db:
                if plan['mode']=='RENAME_IN_PLACE':
                    for r in outcomes:
                        if r['status']=='RENAMED_VERIFIED':db.execute('UPDATE images SET path=?,name=? WHERE id=?',(r['output_path'],r['new_name'],r['target_id']))
                intended={r['target_id'] for r in payload['rows'] if r['status']=='READY'}
                failed=any(r['target_id'] in intended and r['status'] not in ('COPIED_VERIFIED','SKIPPED_VALID','RENAMED_VERIFIED') for r in outcomes)
                db.execute('UPDATE naming_plans SET status=?,result=? WHERE id=?',('EXECUTED_WITH_ERRORS' if failed else 'EXECUTED',encode(result),ident));self.store.event(db,batch,'NAMING_EXECUTED',{'plan_id':ident,'mode':plan['mode'],'results':[{k:r.get(k) for k in ('target_id','status','reason','output_path')} for r in outcomes]})
            return self.get(ident)
        except Exception as exc:
            with self.store.transaction() as db:db.execute("UPDATE naming_plans SET status='FAILED',result=? WHERE id=?",(encode({'error':str(exc),'output_root':str(root)}),ident));self.store.event(db,batch,'NAMING_FAILED',{'plan_id':ident,'error':str(exc)})
            raise
    def rollback(self,ident,confirmed_plan_id):
        if ident!=confirmed_plan_id:raise ValueError('必须明确确认回退计划')
        plan=self.get(ident)
        if plan['mode']!='RENAME_IN_PLACE' or plan['status']!='EXECUTED':raise ValueError('只有已完成的原地改名可使用该回退入口')
        with self.store.transaction() as db:
            self.service._mutable(db,plan['batch_id']);db.execute("UPDATE naming_plans SET status='EXECUTING' WHERE id=?",(ident,))
        reverse=[]
        for r in plan['result']['rows']:
            if r['status']=='RENAMED_VERIFIED':reverse.append({**r,'original_path':r['output_path'],'output_path':r['original_path'],'status':'READY'})
        roots=[f['path'] for f in self.service.detail(plan['batch_id'])['folders'] if f['role']=='TARGET']
        try:output=execute_in_place(reverse,roots,Path(plan['payload']['output_root'])/'rollback_execution')
        except Exception as exc:
            with self.store.transaction() as db:
                db.execute("UPDATE naming_plans SET status='FAILED' WHERE id=?",(ident,));self.store.event(db,plan['batch_id'],'ROLLBACK_FAILED',{'plan_id':ident,'error':str(exc)})
            raise
        with self.store.transaction() as db:
            for r in output:
                if r['status']=='RENAMED_VERIFIED':db.execute('UPDATE images SET path=?,name=? WHERE id=?',(r['output_path'],Path(r['output_path']).name,r['target_id']))
            db.execute("UPDATE naming_plans SET status='ROLLED_BACK' WHERE id=?",(ident,));db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(plan['batch_id'],));self.store.event(db,plan['batch_id'],'NAMING_ROLLED_BACK',{'plan_id':ident})
        return self.get(ident)
