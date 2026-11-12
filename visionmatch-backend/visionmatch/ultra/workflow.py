from __future__ import annotations

import html
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from ..schemas import SHARED_RULES
from ..matching import solve_assignment
from .sheets import scan,make_sheet,digest,save_json
from .naming import preview,execute_copy,in_place_preview

def group_schema(sources,targets):
    aid={'type':['string','null'],'enum':[None]+[x['id'] for x in sources]}
    props={'target_id':{'type':'string','enum':[x['id'] for x in targets]},'source_id':aid,'decision':{'type':'string','enum':['MATCH','DIFFERENT','UNCERTAIN','UNMATCHED']},'evidence':{'type':'array','items':{'type':'string'}},'conflicts':{'type':'array','items':{'type':'string'}},'needs_review':{'type':'boolean'},'alternative_source_ids':{'type':'array','items':{'type':'string','enum':[x['id'] for x in sources]}},'review_reasons':{'type':'array','items':{'type':'string'}}}
    return {'type':'object','properties':{'targets':{'type':'array','items':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}},'required':['targets'],'additionalProperties':False}

def validate_group(result,plan):
    Draft202012Validator(group_schema(plan['sources'],plan['targets'])).validate(result)
    rows=result['targets']; ids=[r['target_id'] for r in rows]
    if Counter(ids)!=Counter(x['id'] for x in plan['targets']):raise ValueError('missing or duplicate target IDs')
    for row in rows:
        if row['decision']=='MATCH' and (row['source_id'] is None or not row['evidence'] or row['conflicts']):raise ValueError('MATCH lacks consistent evidence')
        if row['decision']=='UNMATCHED' and row['source_id'] is not None:raise ValueError('unmatched cannot assign a source')
        if row['decision']=='DIFFERENT' and (not row['conflicts'] or row['source_id'] is None):raise ValueError('DIFFERENT needs observed pair conflict')
    return rows

def review_tasks(plan,rows):
    counts=Counter(r['source_id'] for r in rows if r['decision']=='MATCH')
    def small_ids(sheet):
        encoded=sheet.get('cli_encoded_size')
        d=sheet.get('dimensions',{})
        scale=min(encoded[0]/d['width'],encoded[1]/d['height']) if encoded and d else 1
        return {t['id'] for t in sheet['tiles'] if t['needs_original_review'] or min(t['bounds'][2:])*scale<240}
    small=small_ids(plan['target_sheet'])
    source_small=small_ids(plan['source_sheet'])
    tasks=[]
    for row in rows:
        reasons=list(row.get('review_reasons',[]))
        if row['decision']=='UNCERTAIN' or row['needs_review']:reasons.append('uncertain or model requested review')
        if len(row['alternative_source_ids'])>0:reasons.append('multiple plausible main images')
        if row['decision']=='MATCH' and counts[row['source_id']]>1:reasons.append('one-to-one conflict')
        if row['target_id'] in small or row['source_id'] in source_small:reasons.append('garment rendering small')
        if reasons:tasks.append({'target_id':row['target_id'],'source_ids':list(dict.fromkeys(([row['source_id']] if row['source_id'] else [])+row['alternative_source_ids'])),'reasons':reasons,'status':'PENDING','before':row})
    return tasks

def prepare(root, sources, target_sets, cache_dir, *, cell=700):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    source_sheet=make_sheet(sources,cache_dir,cell=cell)
    shutil.copyfile(source_sheet['path'],root/'A_contact_sheet.png')
    plans=[]
    for n,(name,targets) in enumerate(target_sets,1):
        folder_id=f'T{n:02}_{name}'
        for invalid in '<>:"/\\|?*':folder_id=folder_id.replace(invalid,'_')
        b=make_sheet(targets,cache_dir,cell=cell)
        folder=root/folder_id;folder.mkdir()
        shutil.copyfile(b['path'],folder/'B_contact_sheet.png')
        plan={'mode':'ULTRA_CONTACT_SHEET_MATCH','folder_id':folder_id,'sources':sources,'targets':targets,'source_sheet':source_sheet,'target_sheet':b,'model':'gpt-6.1-terra','max_calls':0}
        plan['plan_id']=digest(plan);save_json(folder/'plan.json',plan);save_json(folder/'schema.json',group_schema(sources,targets));plans.append(plan)
    save_json(root/'plans.json',plans)
    return plans

def corrected_rows(plan,corrections):
    """An explicit human correction file is authoritative, never inferred from names."""
    valid_a={x['id'] for x in plan['sources']};valid_b={x['id'] for x in plan['targets']}
    by_b={x['target_id']:x for x in corrections}
    if len(by_b)!=len(corrections) or set(by_b)!=valid_b:raise ValueError('human correction must cover each target exactly once')
    used=[];rows=[]
    for bid,item in by_b.items():
        aid=item.get('source_id');decision=item.get('decision')
        if aid is not None and aid not in valid_a:raise ValueError('invalid source ID')
        if decision not in ('MATCH','UNCERTAIN','UNMATCHED','DIFFERENT'):raise ValueError('invalid decision')
        if decision=='MATCH':
            if not aid:raise ValueError('MATCH requires source');
            used.append(aid)
        elif aid is not None:raise ValueError('non-match human correction must not assign a source')
        rows.append({'target_id':bid,'source_id':aid,'decision':decision,'confirmed':decision=='MATCH','needs_review':decision=='UNCERTAIN','conflicts':[],'evidence':[item.get('note','人工确认')],'model':'HUMAN_CONFIRMED'})
    if len(used)!=len(set(used)):raise ValueError('human correction has one-to-one conflict')
    return rows

def assign_supported(plan,rows,tasks):
    """Only evidence-supported completed edges; unresolved comparisons are never DIFFERENT."""
    pending={r['target_id'] for r in rows if r.get('needs_review')}
    blocked={t['target_id'] for t in tasks if t['status']!='SUCCESS' or t['target_id'] in pending}
    a={x['id']:n for n,x in enumerate(plan['sources'])};b={x['id']:n for n,x in enumerate(plan['targets'])}
    edges={(a[r['source_id']],b[r['target_id']]):100 for r in rows if r['decision']=='MATCH' and not r.get('needs_review') and r['target_id'] not in blocked and not r['conflicts']}
    chosen=solve_assignment(len(a),len(b),edges)
    back_a={v:k for k,v in a.items()};back_b={v:k for k,v in b.items()}
    return {'assignment':{back_b[target]:back_a[source] for source,target in chosen.items()},'pending_review_ids':sorted(blocked),'matrix_complete':False,'group_target_coverage_complete':True,'note':'Group target coverage is distinct from a full Cartesian comparison matrix; incomplete evidence is not DIFFERENT.'}

def apply_reviews(rows,tasks):
    by_b={r['target_id']:r for r in rows}
    for task in tasks:
        row=by_b[task['target_id']];row['needs_review']=True
        if task['status']!='SUCCESS':continue
        after=task['after'];row['original_review']=after;row['group_decision_before_review']={'source_id':row['source_id'],'decision':row['decision']}
        if 'winner' in after:
            if after['winner'] not in ('CANDIDATE_1','CANDIDATE_2'):continue
            n=0 if after['winner']=='CANDIDATE_1' else 1
            aid=task['source_ids'][n];detail=after['candidate_1' if n==0 else 'candidate_2']
        else:aid=task['source_ids'][0];detail=after
        if detail['verdict']=='MATCH' and not detail['needs_review']:
            row.update(source_id=aid,decision='MATCH',needs_review=False,evidence=detail['matched_features'],conflicts=[])
    counts=Counter(r['source_id'] for r in rows if r['decision']=='MATCH')
    for row in rows:
        if row['decision']=='MATCH' and counts[row['source_id']]>1:row['needs_review']=True
    return rows

def report(root,plan,rows,manifest=None, *, provenance='NO_MODEL_CALL'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    assets=root/'assets';assets.mkdir(exist_ok=True)
    sources={x['id']:x for x in plan['sources']};targets={x['id']:x for x in plan['targets']}
    for item in [*sources.values(),*targets.values()]:
        dst=assets/(item['id']+'.jpg')
        if not dst.exists():
            with Image.open(item['path']) as im:
                im=im.convert('RGB');im.thumbnail((320,320));im.save(dst,quality=90)
    operations={r['target_id']:r for r in manifest or []};esc=html.escape
    source_gallery='<details><summary>查看全部主图参考</summary>'+''.join(f'<span style="display:inline-block;text-align:center"><img src="assets/{aid}.jpg"><br>{aid}<br>{esc(item["name"])}</span>' for aid,item in sources.items())+'</details>'
    body=[];choices='<option value="">未匹配</option><option value="UNCERTAIN">待复核</option>'+''.join(f'<option value="{a}">{a} {esc(v["name"])}</option>' for a,v in sources.items())
    for row in rows:
        aid=row.get('source_id');bid=row['target_id'];op=operations.get(bid,{})
        visual=f'<img src="assets/{aid}.jpg">' if aid else '无对应主图'
        selected=aid or ('UNCERTAIN' if row['decision']=='UNCERTAIN' else '')
        select=choices.replace(f'value="{selected}"',f'value="{selected}" selected')
        after=row.get('original_review',{})
        review_text=after.get('winner',after.get('verdict','未运行'))
        display_status={'COPIED_VERIFIED':'已复制并校验','SKIPPED_VALID':'已有有效结果，未重复复制','BLOCKED':'已跳过或存在冲突'}.get(op.get('status'),'未生成')
        review_required='待原图复核，尚未确认' if row.get('needs_review') else '无需额外复核或已复核'
        body.append(f'<tr><td>{visual}<br>{esc(sources[aid]["name"]) if aid else ""}</td><td><img src="assets/{bid}.jpg"><br>{bid} {esc(targets[bid]["name"])}</td><td>{esc(row["decision"])}<br>{review_required}<br>原图复核：{esc(review_text)}</td><td>{esc("；".join(row.get("evidence",[])))}</td><td>{esc(op.get("new_name",""))}<br>{esc(display_status)}</td><td><select data-b="{bid}">{select}</select></td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>超高清归属与命名</title><style>body{font:15px/1.5 Microsoft YaHei,Arial;margin:24px;color:#173049}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd6e2;padding:9px}img{width:150px;height:150px;object-fit:contain}select{max-width:270px;padding:8px}button{padding:12px}</style><h1>服装归属与复制命名</h1><p>'''+esc(plan['folder_id'])+' · '+esc(provenance)+'''</p><p>人工修改后导出 corrections.json，再使用 apply-corrections 命令生成新的输出版本。页面不会直接改动原始图片。</p><table><tr><th>主图</th><th>目标图与原文件名</th><th>判断</th><th>证据</th><th>新文件名与生成状态</th><th>人工纠正</th></tr>'''+''.join(body)+'''</table><button id="save">导出人工纠正 JSON</button><p id="error"></p><script>document.querySelector('#save').onclick=()=>{const rows=[...document.querySelectorAll('select[data-b]')].map(s=>({target_id:s.dataset.b,source_id:s.value.startsWith('A')?s.value:null,decision:s.value.startsWith('A')?'MATCH':s.value==='UNCERTAIN'?'UNCERTAIN':'UNMATCHED',note:'人工页面确认'}));const a=rows.filter(x=>x.source_id).map(x=>x.source_id);if(a.length!==new Set(a).size){document.querySelector('#error').textContent='同一主图被重复选择，请修正';return}const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));link.download='corrections.json';link.click();URL.revokeObjectURL(link.href)};</script></html>'''
    page=page.replace('<table><tr><th>主图</th>',source_gallery+'<table><tr><th>主图</th>')
    (root/'index.html').write_text(page,encoding='utf-8')
    save_json(root/'results.json',{'provenance':provenance,'folder_id':plan['folder_id'],'rows':rows,'rename_operations':manifest or []})

def copy_confirmed(plan,rows,destination):
    destination=Path(destination)
    rename_preview=preview(plan,rows,destination)
    save_json(destination/'rename_preview.json',rename_preview)
    save_json(destination/'rename_in_place_preview_DISABLED.json',in_place_preview(rename_preview))
    operations=execute_copy(rename_preview,destination)
    report(destination,plan,rows,operations,provenance='HUMAN_CONFIRMED_ORIGINAL_BYTES_COPY')
    return operations
