from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from pathlib import Path

from PIL import Image
from jsonschema import Draft202012Validator

from ..provider import file_hash
from ..matching import solve_assignment
from ..ultra.sheets import scan,digest
from ..ultra.workflow import group_schema
from .store import Store,now,encode,decode

DEFAULTS={'model':'gpt-6.1-terra','cell':700,'concurrency':2,'max_calls':0,'request_timeout':300}

class Service:
    def __init__(self,store:Store):self.store=store
    def create(self,name):
        batch=uuid.uuid4().hex
        with self.store.transaction() as db:
            db.execute('INSERT INTO batches(id,name,status,created,settings) VALUES (?,?,?,?,?)',(batch,name.strip() or '未命名批次','DRAFT',now(),encode(DEFAULTS)))
            self.store.event(db,batch,'BATCH_CREATED',{'name':name})
        self.store.batch_root(batch);return self.detail(batch)
    def detail(self,batch):
        row=self.store.one('SELECT * FROM batches WHERE id=?',(batch,))
        if not row:raise ValueError('批次不存在')
        row['settings']=decode(row['settings']);row['folders']=self.store.rows('SELECT * FROM folders WHERE batch_id=? ORDER BY ordinal',(batch,))
        for f in row['folders']:
            f['count']=self.store.one('SELECT COUNT(*) n FROM images WHERE folder_id=?',(f['id'],))['n'];f['metrics']=decode(f['metrics'],{})
        row['counts']={x['status']:x['n'] for x in self.store.rows('SELECT status,COUNT(*) n FROM matches WHERE batch_id=? GROUP BY status',(batch,))}
        row['runs']=self.store.rows('SELECT * FROM runs WHERE batch_id=? ORDER BY started DESC',(batch,))
        return row
    def _mutable(self,db,batch):
        b=db.execute('SELECT * FROM batches WHERE id=?',(batch,)).fetchone()
        if not b:raise ValueError('批次不存在')
        if db.execute("SELECT 1 FROM naming_plans WHERE batch_id=? AND status='EXECUTING'",(batch,)).fetchone():raise ValueError('命名执行中，请等待完成')
        return b
    def import_local(self,batch,source,targets):
        if not targets:raise ValueError('至少选择一个目标文件夹')
        paths=[Path(source).resolve()]+[Path(x['path']).resolve() for x in targets]
        if len(set(str(p).casefold() for p in paths))!=len(paths):raise ValueError('主图与目标目录不可重复')
        if any(a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):raise ValueError('输入目录不可互相包含')
        labels=['主图']+[x.get('label') or Path(x['path']).name for x in targets]
        if len({x.casefold() for x in labels})!=len(labels):raise ValueError('文件夹名称重复，请为目标设置不同显示名称')
        from ..ultra.naming import valid_name
        if not all(valid_name(x) for x in labels):raise ValueError('文件夹显示名称不能包含Windows非法字符')
        records=[scan(paths[0],'A')]+[scan(p,'B') for p in paths[1:]]
        warnings=[]
        for label,images in zip(labels,records):
            stems=Counter(Path(x['name']).stem.casefold() for x in images)
            if any(n>1 for n in stems.values()):warnings.append({'folder':label,'kind':'DUPLICATE_STEM','message':'存在同主体名不同扩展名，命名预览将检查冲突'})
        with self.store.transaction() as db:
            self._mutable(db,batch)
            if db.execute('SELECT 1 FROM folders WHERE batch_id=?',(batch,)).fetchone():raise ValueError('批次已导入图片；请创建新批次，避免覆盖审核记录')
            for n,(path,label,images) in enumerate(zip(paths,labels,records)):
                fid=uuid.uuid4().hex
                db.execute('INSERT INTO folders(id,batch_id,label,path,role,ordinal) VALUES (?,?,?,?,?,?)',(fid,batch,label,str(path),'SOURCE' if n==0 else 'TARGET',n))
                for item in images:
                    cur=db.execute('INSERT INTO images(batch_id,folder_id,local_id,path,original_path,name,sha256,width,height,extension) VALUES (?,?,?,?,?,?,?,?,?,?)',(batch,fid,item['id'],item['path'],item['path'],item['name'],item['sha256'],*item['size'],Path(item['name']).suffix))
                    if n:db.execute('INSERT INTO matches(target_id,batch_id,folder_id) VALUES (?,?,?)',(cur.lastrowid,batch,fid))
            db.execute("UPDATE batches SET status='SCANNED',revision=revision+1 WHERE id=?",(batch,));self.store.event(db,batch,'IMAGES_IMPORTED',{'counts':[len(x) for x in records],'warnings':warnings})
        return {**self.detail(batch),'warnings':warnings}
    def records(self,batch,folder):
        def item(x):return {**x,'id':x['local_id'],'size':[x['width'],x['height']]}
        source=self.store.one("SELECT id FROM folders WHERE batch_id=? AND role='SOURCE'",(batch,))
        if not source:raise ValueError('尚未导入主图')
        return [item(x) for x in self.store.rows('SELECT * FROM images WHERE folder_id=? ORDER BY local_id',(source['id'],))],[item(x) for x in self.store.rows('SELECT * FROM images WHERE folder_id=? ORDER BY local_id',(folder,))]
    def reconcile(self,db,batch,folder):
        """Resolve all unlocked rows together; hard manual assignments reserve columns."""
        rows=[dict(x) for x in db.execute('SELECT * FROM matches WHERE folder_id=? ORDER BY target_id',(folder,))]
        fixed={r['assigned_source_id'] for r in rows if r['manual'] and r['assigned_source_id'] is not None}
        free=[r for r in rows if not r['manual']]
        sources=[r['id'] for r in db.execute("SELECT id FROM images WHERE batch_id=? AND local_id LIKE 'A%' ORDER BY id",(batch,)) if r['id'] not in fixed]
        indexes={sid:i for i,sid in enumerate(sources)};edges={};candidates={}
        for a,row in enumerate(free):
            if row['status']=='ERROR':continue
            for rank,c in enumerate(decode(row['candidates'],[])):
                sid=c['source_id']
                if sid in indexes and c.get('decision') in ('MATCH','UNCERTAIN'):
                    edges[a,indexes[sid]]=max(1,100-rank);candidates[(row['target_id'],sid)]=c
        chosen=solve_assignment(len(free),len(sources),edges)
        for a,row in enumerate(free):
            sid=sources[chosen[a]] if a in chosen else None
            suggestion=decode(row['suggestion'],{})
            if row['status']=='ERROR':status='ERROR'
            elif sid is not None:
                c=candidates[(row['target_id'],sid)]
                status='MATCHED' if c.get('decision')=='MATCH' and not c.get('needs_review') else 'REVIEW'
            elif suggestion.get('decision')=='UNMATCHED':status='UNMATCHED'
            else:status='REVIEW'
            db.execute('UPDATE matches SET assigned_source_id=?,status=?,confirmed=0 WHERE target_id=?',(sid,status,row['target_id']))
    def ingest(self,batch,folder,result,metadata,provenance='REAL_MODEL'):
        sources,targets=self.records(batch,folder);schema=group_schema(sources,targets)
        # Preserve partial coverage explicitly. Structurally invalid records never
        # become assignments; raw output remains available in the job ledger.
        try:Draft202012Validator(schema).validate(result);schema_error=None
        except Exception as exc:schema_error=str(exc)[:1000]
        ai={x['local_id']:x['id'] for x in self.store.rows("SELECT * FROM images WHERE batch_id=? AND local_id LIKE 'A%'",(batch,))}
        bi={x['local_id']:x['id'] for x in self.store.rows('SELECT * FROM images WHERE folder_id=?',(folder,))}
        raw=result.get('targets',[]) if isinstance(result,dict) else []
        counts=Counter(r.get('target_id') for r in raw if isinstance(r,dict));mapped={r.get('target_id'):r for r in raw if isinstance(r,dict)}
        proposed=Counter(r.get('source_id') for r in raw if isinstance(r,dict) and r.get('source_id') and r.get('decision')=='MATCH')
        collisions=[aid for aid,n in proposed.items() if n>1]
        missing=[]
        with self.store.transaction() as db:
            self._mutable(db,batch)
            for bid,tid in bi.items():
                row=mapped.get(bid)
                error=schema_error or ('MODEL_OMITTED' if row is None else 'DUPLICATE_TARGET' if counts[bid]>1 else None)
                current=db.execute('SELECT * FROM matches WHERE target_id=?',(tid,)).fetchone()
                if error:
                    missing.append(bid)
                    if not current['manual']:db.execute("UPDATE matches SET status='ERROR',suggestion=NULL,candidates='[]',assigned_source_id=NULL,notes=? WHERE target_id=?",(encode([error]),tid))
                    continue
                sid=ai.get(row.get('source_id'));primary=row['decision']
                conflict=primary=='MATCH' and (not sid or not row['evidence'] or row['conflicts'])
                if conflict:missing.append(bid)
                candidates=[]
                if sid and primary in ('MATCH','UNCERTAIN'):
                    candidates.append({'source_id':sid,'decision':'UNCERTAIN' if conflict else primary,'needs_review':row['needs_review'] or bool(conflict) or row.get('source_id') in collisions,'evidence':row['evidence']})
                for aid in row['alternative_source_ids']:
                    if ai[aid]!=sid:candidates.append({'source_id':ai[aid],'decision':'UNCERTAIN','needs_review':True,'evidence':row['review_reasons']})
                db.execute("UPDATE matches SET proposed_source_id=?,suggestion=?,candidates=?,status=CASE WHEN manual=0 THEN 'PENDING' ELSE status END WHERE target_id=?",(sid,encode({**row,'provenance':provenance}),encode(candidates),tid))
            self.reconcile(db,batch,folder)
            db.execute('UPDATE folders SET state=?,error=?,metrics=? WHERE id=?',('PARTIAL' if missing else 'COMPLETE',encode(missing) if missing else None,encode({**metadata,'one_to_one_conflicts':len(collisions)}),folder))
            db.execute("UPDATE batches SET status='REVIEW',revision=revision+1 WHERE id=?",(batch,));self.store.event(db,batch,'SUGGESTIONS_IMPORTED',{'folder_id':folder,'provenance':provenance,'missing_or_invalid':missing,'metadata':metadata})
        return {'missing':missing,'covered':len(bi)-len(missing)}
    def review(self,batch,target,action,source=None,locked=True,note=''):
        with self.store.transaction() as db:
            self._mutable(db,batch)
            row=db.execute('SELECT * FROM matches WHERE batch_id=? AND target_id=?',(batch,target)).fetchone()
            if not row:raise ValueError('目标图不存在')
            before=dict(row)
            if action in ('assign','confirm'):
                source=source if action=='assign' else row['assigned_source_id']
                if not source:raise ValueError('没有可确认的主图')
                main=db.execute("SELECT i.id FROM images i JOIN folders f ON f.id=i.folder_id WHERE i.id=? AND i.batch_id=? AND f.role='SOURCE'",(source,batch)).fetchone()
                if not main:raise ValueError('主图不属于该批次')
                other=db.execute('SELECT * FROM matches WHERE folder_id=? AND assigned_source_id=? AND target_id<>?',(row['folder_id'],source,target)).fetchone()
                if other and other['locked']:raise ValueError('该主图已被人工锁定的目标图占用；请先解锁或修改该结果')
                if other:db.execute("UPDATE matches SET manual=0,confirmed=0,locked=0,assigned_source_id=NULL,status='REVIEW' WHERE target_id=?",(other['target_id'],))
                db.execute("UPDATE matches SET assigned_source_id=?,status='CONFIRMED',confirmed=1,locked=?,manual=1,notes=? WHERE target_id=?",(source,int(locked),encode([note or '人工确认']),target))
            elif action=='unmatched':db.execute("UPDATE matches SET assigned_source_id=NULL,status='UNMATCHED',confirmed=1,manual=1,locked=?,notes=? WHERE target_id=?",(int(locked),encode([note or '人工确认未匹配']),target))
            elif action=='unlock':db.execute('UPDATE matches SET locked=0 WHERE target_id=?',(target,))
            elif action=='restore':db.execute('UPDATE matches SET manual=0,confirmed=0,locked=0 WHERE target_id=?',(target,))
            else:raise ValueError('无效审核操作')
            self.reconcile(db,batch,row['folder_id'])
            db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(batch,));self.store.event(db,batch,'HUMAN_REVIEW',{'target_id':target,'action':action,'source_id':source,'before':before,'note':note})
        return self.detail(batch)
    def bulk_confirm(self,batch,targets,locked=True):
        # Validate the entire selection before making the first modification.
        with self.store.transaction() as db:
            self._mutable(db,batch)
            rows=[db.execute('SELECT * FROM matches WHERE batch_id=? AND target_id=?',(batch,t)).fetchone() for t in targets]
            if any(not r or not r['assigned_source_id'] or r['status']=='ERROR' for r in rows):raise ValueError('所选结果含缺失主图或错误项')
            keys=[(r['folder_id'],r['assigned_source_id']) for r in rows]
            if len(keys)!=len(set(keys)):raise ValueError('所选结果存在一对一冲突')
            for r in rows:
                db.execute("UPDATE matches SET confirmed=1,manual=1,locked=?,status='CONFIRMED' WHERE target_id=?",(int(locked),r['target_id']))
            db.execute('UPDATE batches SET revision=revision+1 WHERE id=?',(batch,));self.store.event(db,batch,'BULK_CONFIRMED',{'targets':targets,'locked':locked})
        return self.detail(batch)
    def view(self,batch):
        state=self.detail(batch);images=self.store.rows('SELECT * FROM images WHERE batch_id=? ORDER BY local_id',(batch,));image_map={x['id']:x for x in images}
        matches=self.store.rows('SELECT * FROM matches WHERE batch_id=? ORDER BY folder_id,target_id',(batch,))
        for row in matches:
            row['target']=image_map[row['target_id']];row['source']=image_map.get(row['assigned_source_id'])
            row['suggestion']=decode(row['suggestion'],{});row['candidates']=decode(row['candidates'],[]);row['notes']=decode(row['notes'],[]);row['review']=decode(row['review'])
            row['new_name']=Path(row['source']['name']).stem+row['target']['extension'] if row['source'] else None
        sources=[x for x in images if x['local_id'].startswith('A')]
        state.update(images=images,matches=matches,groups=[{'source':s,'matches':[m for m in matches if m['assigned_source_id']==s['id']]} for s in sources],unassigned=[m for m in matches if m['assigned_source_id'] is None])
        state['naming_plans']=self.store.rows('SELECT id,mode,status,created,revision FROM naming_plans WHERE batch_id=? ORDER BY created DESC',(batch,))
        return state
    def import_recording(self,batch,folder,plan_path,result_path):
        """Explicit replay of a recorded real model response, bound by content hashes."""
        old=json.loads(Path(plan_path).read_text(encoding='utf-8'));saved=json.loads(Path(result_path).read_text(encoding='utf-8'));result=saved['result']
        sources,targets=self.records(batch,folder)
        current_a={x['sha256']:x['id'] for x in sources};current_b={x['sha256']:x['id'] for x in targets}
        if set(current_a)!={x['sha256'] for x in old['sources']} or set(current_b)!={x['sha256'] for x in old['targets']}:raise ValueError('录制结果与本批次图片哈希集合不一致')
        a={x['id']:current_a[x['sha256']] for x in old['sources']};b={x['id']:current_b[x['sha256']] for x in old['targets']}
        # Older exported reports add workflow metadata. Ignore only those known
        # fields; they must never import a human confirmation or defeat Schema.
        remapped={'targets':[{**{k:v for k,v in r.items() if k not in ('confirmed','model')},'target_id':b[r['target_id']],'source_id':a[r['source_id']] if r['source_id'] else None,'alternative_source_ids':[a[x] for x in r['alternative_source_ids']]} for r in result['targets']]}
        return self.ingest(batch,folder,remapped,{'mode':'RECORDED_REAL_RESULT_REPLAY','new_cli_calls':0,'historical_usage':saved.get('usage'),'historical_seconds':saved.get('elapsed_seconds'),'recording':str(result_path)},'RECORDED_REAL_RESULT_REPLAY')
