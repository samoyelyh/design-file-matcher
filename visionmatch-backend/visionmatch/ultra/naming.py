from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from ..provider import file_hash
from .sheets import digest, save_json

def valid_name(name):
    reserved={'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
    return bool(name and len(name)<=240 and name==name.rstrip(' .') and not re.search(r'[<>:"/\\|?*\x00-\x1f]',name) and name.split('.')[0].upper() not in reserved)

def preview(plan, decisions, export_root):
    """No disk mutation. All conflicts are found BEFORE any file is copied."""
    sources={x['id']:x for x in plan['sources']}; targets={x['id']:x for x in plan['targets']}
    folder=plan['folder_id']; root=Path(export_root).resolve()
    rows=[]
    for row in decisions:
        source=sources.get(row.get('source_id')); target=targets.get(row.get('target_id'))
        eligible=(row.get('decision')=='MATCH' and row.get('confirmed') is True and not row.get('needs_review') and not row.get('conflicts'))
        if source and target:
            name=Path(source['name']).stem+Path(target['name']).suffix
            destination=root/folder/name
        else: name='';destination=None
        reason='' if eligible and source and target else 'not confirmed MATCH, unresolved review/conflict, or invalid image ID'
        if name and not valid_name(name):reason='invalid Windows filename'
        entry={'operation_id':digest([folder,row.get('target_id'),row.get('source_id'),target['sha256'] if target else None,name]),'folder_id':folder,'target_id':row.get('target_id'),'source_id':row.get('source_id'),'original_path':target['path'] if target else '', 'original_name':target['name'] if target else '', 'original_sha256':target['sha256'] if target else '', 'source_path':source['path'] if source else '', 'source_name':source['name'] if source else '', 'source_sha256':source['sha256'] if source else '', 'new_name':name,'output_path':str(destination) if destination else '', 'decision':row.get('decision'),'confirmed':row.get('confirmed',False),'needs_review':row.get('needs_review',False),'conflicts':row.get('conflicts',[]),'evidence':row.get('evidence',[]),'model':row.get('model','NOT_RECORDED'),'status':'READY' if not reason else 'BLOCKED','reason':reason}
        rows.append(entry)
    active=[r for r in rows if r['status']=='READY']
    a_counts=Counter(r['source_id'] for r in active);b_counts=Counter(r['target_id'] for r in active)
    paths=Counter(r['output_path'].casefold() for r in active)
    for row in active:
        if a_counts[row['source_id']]>1 or b_counts[row['target_id']]>1 or paths[row['output_path'].casefold()]>1:
            row.update(status='BLOCKED',reason='one-to-one or Windows case-insensitive name collision')
    return rows

def execute_copy(rows, export_root):
    root=Path(export_root).resolve();root.mkdir(parents=True,exist_ok=True)
    audit=root/'naming_audit.jsonl'; old={}
    if audit.exists():
        for line in audit.read_text(encoding='utf-8').splitlines():
            try:
                value=json.loads(line);old[value['operation_id']]=value
            except (ValueError,KeyError):continue
    outcomes=[]
    def log(value):
        with audit.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    for original in rows:
        row={**original,'executed_at':datetime.now().astimezone().isoformat(),'mode':'COPY_AND_RENAME'}
        if row['decision']!='MATCH' or not row['confirmed'] or row.get('needs_review') or row.get('conflicts'):
            row.update(status='BLOCKED',reason='not an eligible confirmed MATCH')
        if row['status']!='READY':outcomes.append(row);continue
        tmp=None
        try:
            source=Path(row['original_path']); main=Path(row['source_path']);dest=Path(row['output_path']).resolve()
            if not dest.is_relative_to(root) or source.resolve().is_relative_to(root):raise ValueError('invalid output containment')
            if file_hash(source)!=row['original_sha256'] or file_hash(main)!=row['source_sha256']:raise ValueError('source/main content changed')
            dest.parent.mkdir(parents=True,exist_ok=True)
            existing=next((p for p in dest.parent.iterdir() if p.name.casefold()==dest.name.casefold()),None)
            if existing:
                if row['operation_id'] in old and file_hash(existing)==row['original_sha256']:
                    row.update(status='SKIPPED_VALID',reason='previous operation verified');outcomes.append(row);log(row);continue
                raise FileExistsError('existing destination, overwrite forbidden')
            log({**row,'status':'STARTED'})
            tmp=dest.parent/('.'+uuid.uuid4().hex+'.partial')
            with source.open('rb') as src,tmp.open('xb') as dst:shutil.copyfileobj(src,dst)
            if file_hash(tmp)!=row['original_sha256']:raise ValueError('copied bytes hash mismatch')
            # Atomic publish without overwrite, also safe across interruptions.
            os.link(tmp,dest)
            row.update(status='COPIED_VERIFIED',reason='',output_sha256=file_hash(dest))
        except (OSError,ValueError) as error:row.update(status='BLOCKED',reason=str(error))
        finally:
            if tmp and tmp.exists():tmp.unlink()
        outcomes.append(row);log(row)
    save_json(root/'rename_manifest.json',outcomes)
    with (root/'rename_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['folder_id','target_id','source_id','original_path','original_name','original_sha256','source_path','source_name','new_name','output_path','decision','evidence','model','executed_at','mode','status','reason']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in outcomes:writer.writerow({**row,'evidence':json.dumps(row['evidence'],ensure_ascii=False)})
    return outcomes

def in_place_preview(rows):
    """Optional mode is preview-only. Requires a separate explicit execution approval."""
    changes=[]
    for row in rows:
        if row['status']=='READY':
            before=Path(row['original_path']);after=before.with_name(row['new_name'])
            existing=next((p for p in before.parent.iterdir() if p.name.casefold()==after.name.casefold() and p.resolve()!=before.resolve()),None)
            changes.append({'before':str(before),'after':str(after),'sha256':row['original_sha256'],'blocked':bool(existing),'reason':'destination already occupied' if existing else '', 'rollback':{'before':str(after),'after':str(before)}})
    return {'mode':'RENAME_IN_PLACE','enabled':False,'requires_explicit_confirmation':True,'changes':changes,'plan_id':digest(changes)}
