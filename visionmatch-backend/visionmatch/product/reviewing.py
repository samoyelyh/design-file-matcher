"""Atomic batch edits and persistent, conflict-checked review undo."""
from __future__ import annotations

import uuid
from collections import Counter
from .store import now,encode,decode

def eligible(row):
    if row['status']=='CONFIRMED':return False,'已经确认'
    if row['status']!='MATCHED' or not row['assigned_source_id']:return False,'待复核、未匹配或错误项需单独处理'
    candidates=decode(row['candidates'],[]) if isinstance(row['candidates'],str) else row['candidates']
    candidate=next((c for c in candidates if c['source_id']==row['assigned_source_id']),{})
    if candidate.get('decision')!='MATCH' or candidate.get('needs_review') or not candidate.get('evidence'):return False,'缺少充分的一致视觉证据'
    return True,''

class Reviews:
    def __init__(self,service):self.service=service;self.store=service.store
    def apply(self,batch,action,target_ids,mappings=None,allow_locked=False,strict_confirm=True):
        ids=list(dict.fromkeys(int(x) for x in target_ids))
        if not ids or len(ids)>1000:raise ValueError('请选择1～1000张目标图')
        if action not in ('confirm','assign','review','unmatched','unconfirm','unlock','restore'):raise ValueError('无效批量操作')
        mapping={int(m['target_id']):int(m['source_id']) for m in mappings or []}
        if action=='assign' and set(mapping)!=set(ids):raise ValueError('归属映射必须恰好覆盖所选图片')
        with self.store.transaction() as db:
            self.service._mutable(db,batch)
            rows=[db.execute('SELECT * FROM matches WHERE batch_id=? AND target_id=?',(batch,t)).fetchone() for t in ids]
            if any(r is None for r in rows):raise ValueError('所选图片不属于当前批次')
            folders=sorted({r['folder_id'] for r in rows})
            def snapshot():return [dict(r) for f in folders for r in db.execute('SELECT * FROM matches WHERE folder_id=? ORDER BY target_id',(f,))]
            before=snapshot();skipped=[];active=[]
            for r in rows:
                if action=='confirm' and strict_confirm:
                    valid,reason=eligible(r)
                    if not valid:skipped.append({'target_id':r['target_id'],'reason':reason});continue
                if r['locked'] and not allow_locked and action!='unlock':
                    skipped.append({'target_id':r['target_id'],'reason':'人工锁定已保留'});continue
                active.append(r)
            if action=='assign':
                if skipped:raise ValueError('批量改归属包含人工锁定项，请明确允许修改所选锁定项，或取消选择')
                source_ids={r['id'] for r in db.execute("SELECT i.id FROM images i JOIN folders f ON f.id=i.folder_id WHERE i.batch_id=? AND f.role='SOURCE'",(batch,))}
                if not set(mapping.values())<=source_ids:raise ValueError('主图不属于当前批次')
                keys=[(r['folder_id'],mapping[r['target_id']]) for r in active]
                if len(keys)!=len(set(keys)):raise ValueError('同一目标文件夹内，多张图片不能分配给同一主图；请调整逐行映射')
                for r in active:
                    other=db.execute('SELECT * FROM matches WHERE folder_id=? AND assigned_source_id=? AND target_id<>?',(r['folder_id'],mapping[r['target_id']],r['target_id'])).fetchone()
                    if other and other['target_id'] not in ids and other['locked']:raise ValueError('主图已被未选中的人工锁定项占用，未修改任何结果')
                # Release every selected row first, so swaps are order independent.
                for r in active:db.execute('UPDATE matches SET assigned_source_id=NULL WHERE target_id=?',(r['target_id'],))
                for r in active:
                    db.execute("UPDATE matches SET assigned_source_id=NULL,manual=0,confirmed=0,locked=0,status='REVIEW' WHERE folder_id=? AND assigned_source_id=? AND target_id<>?",(r['folder_id'],mapping[r['target_id']],r['target_id']))
            for r in active:
                tid=r['target_id']
                if action in ('assign','confirm'):
                    sid=mapping[tid] if action=='assign' else r['assigned_source_id']
                    if not sid:raise ValueError('没有可确认的主图')
                    db.execute("UPDATE matches SET assigned_source_id=?,status='CONFIRMED',confirmed=1,manual=1,locked=1,notes=? WHERE target_id=?",(sid,encode(['人工改归属' if action=='assign' else '人工确认']),tid))
                elif action=='unmatched':db.execute("UPDATE matches SET assigned_source_id=NULL,status='UNMATCHED',confirmed=1,manual=1,locked=1,notes=? WHERE target_id=?",(encode(['人工标记未匹配']),tid))
                elif action in ('review','unconfirm'):db.execute("UPDATE matches SET status='REVIEW',confirmed=0,manual=1,locked=0,notes=? WHERE target_id=?",(encode(['人工标记待复核' if action=='review' else '人工撤销确认']),tid))
                elif action=='unlock':db.execute('UPDATE matches SET locked=0 WHERE target_id=?',(tid,))
                elif action=='restore':db.execute('UPDATE matches SET manual=0,confirmed=0,locked=0 WHERE target_id=?',(tid,))
            for f in folders:self.service.reconcile(db,batch,f)
            after=snapshot();previous={r['target_id']:r for r in before};changed=[r['target_id'] for r in after if r!=previous[r['target_id']]]
            command=None
            if changed:
                command=uuid.uuid4().hex;db.execute('INSERT INTO review_commands VALUES (?,?,?,?,?,?,0)',(command,batch,now(),action,encode(before),encode(after)))
                db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(batch,))
                self.store.event(db,batch,'BATCH_REVIEW',{'command_id':command,'action':action,'selected':ids,'applied':len(active),'skipped':skipped,'affected':changed})
        return {'command_id':command,'applied':len(active),'skipped':skipped,'affected_target_ids':changed}
    def latest(self,batch):return self.store.one('SELECT id,action,created FROM review_commands WHERE batch_id=? AND undone=0 ORDER BY rowid DESC LIMIT 1',(batch,))
    def undo(self,batch,command_id=None):
        with self.store.transaction() as db:
            self.service._mutable(db,batch)
            command=db.execute('SELECT * FROM review_commands WHERE batch_id=? AND undone=0 ORDER BY rowid DESC LIMIT 1',(batch,)).fetchone()
            if not command:raise ValueError('没有可撤销的审核操作')
            if command_id and command_id!=command['id']:raise ValueError('待撤销操作已变化，请刷新后重试')
            before=decode(command['before_state']);after=decode(command['after_state'])
            for saved in after:
                current=db.execute('SELECT * FROM matches WHERE target_id=?',(saved['target_id'],)).fetchone()
                if not current or dict(current)!=saved:raise ValueError('受影响结果已被后续审核或模型更新，拒绝覆盖；请重新核对')
            for r in before:
                keys=[k for k in r if k not in ('target_id','batch_id','folder_id')]
                db.execute('UPDATE matches SET '+','.join(k+'=?' for k in keys)+' WHERE target_id=?',tuple(r[k] for k in keys)+(r['target_id'],))
            db.execute('UPDATE review_commands SET undone=1 WHERE id=?',(command['id'],));db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(batch,))
            self.store.event(db,batch,'REVIEW_UNDONE',{'command_id':command['id'],'affected':[r['target_id'] for r in before]})
        return {'undone':command['id'],'affected_target_ids':[r['target_id'] for r in before]}
