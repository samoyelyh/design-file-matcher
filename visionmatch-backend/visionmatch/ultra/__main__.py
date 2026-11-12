from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..schemas import PAIR_PROMPT,PAIR_SCHEMA,REVIEW_PROMPT,REVIEW_SCHEMA,validate_output
from .sheets import scan,LEVELS,save_json
from .workflow import prepare,group_schema,validate_group,review_tasks,corrected_rows,copy_confirmed,report,assign_supported,apply_reviews
from .requests import Client

def required_milestones(count):
    if count<=16:return []
    return [16]+([30] if count>=50 else [])

def main():
    parser=argparse.ArgumentParser(description='独立超高清拼图实验：prepare 不调用模型，copy 只复制已确认结果')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--target',action='append',required=True);p.add_argument('--output',required=True);p.add_argument('--cache',required=True);p.add_argument('--level',choices=LEVELS,default='ultra')
    c=sub.add_parser('apply-corrections');c.add_argument('--plan',required=True);c.add_argument('--corrections',required=True);c.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--approve-plan',required=True);r.add_argument('--max-calls',type=int,required=True);r.add_argument('--milestones');r.add_argument('--max-reviews',type=int,default=2)
    args=parser.parse_args()
    if args.command=='prepare':
        if not 1<=len(args.target)<=8:raise ValueError('requires 1..8 independent target folders')
        plans=prepare(args.output,scan(args.source,'A'),[(Path(path).name,scan(path,'B')) for path in args.target],args.cache,cell=LEVELS[args.level])
        print(json.dumps({'plans':[{'plan_id':p['plan_id'],'folder_id':p['folder_id']} for p in plans],'real_calls':0},ensure_ascii=False));return
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    from .sheets import digest
    from ..provider import file_hash
    if digest({k:v for k,v in plan.items() if k!='plan_id'})!=plan['plan_id']:raise ValueError('plan modified; prepare a new immutable plan')
    for image in plan['sources']+plan['targets']:
        if file_hash(Path(image['path']))!=image['sha256']:raise ValueError('source modified; prepare new plan')
    if args.command=='apply-corrections':
        rows=corrected_rows(plan,json.loads(Path(args.corrections).read_text(encoding='utf-8')))
        # A new output revision avoids overwriting files from an earlier correction.
        revision=digest([plan['plan_id'],sorted(rows,key=lambda row:row['target_id'])])
        destination=Path(args.output)/('revision_'+revision[:16])
        operations=copy_confirmed(plan,rows,destination)
        print(json.dumps({'output':str(destination),'counts':{s:sum(x['status']==s for x in operations) for s in {x['status'] for x in operations}}},ensure_ascii=False));return
    if args.max_calls<1 or args.max_reviews<0 or args.max_reviews>2:raise ValueError('explicit positive call budget and 0..2 original reviews required')
    for sheet in (plan['source_sheet'],plan['target_sheet']):
        if file_hash(Path(sheet['path']))!=sheet['png_sha256']:raise ValueError('sheet modified; prepare new plan')
    count=max(len(plan['sources']),len(plan['targets']))
    if count>16:
        milestones=json.loads(Path(args.milestones).read_text(encoding='utf-8')) if args.milestones else {}
        # 17–49 images are the intermediate stage; only the 50-image stage
        # requires an independently accepted intermediate run as well.
        required=required_milestones(count)
        if not all(str(n) in milestones.get('human_confirmed_stages',{}) for n in required):raise ValueError('先完成16、再30、最后50的真实测试及人工验收；当前仅允许离线准备')
    start=time.perf_counter();client=Client(plan_path.parent,approved_plan=args.approve_plan,max_calls=args.max_calls)
    try:
        # Read-only prompt construction audit costs no model request. Its decoded
        # dimensions conservatively trigger original reviews after local shrinking.
        from scripts.audit_codex_image_input import audit
        audits=[]
        for sheet in (plan['source_sheet'],plan['target_sheet']):
            observation=audit(sheet['path']);audits.append(observation)
            if len(observation['encoded_images'])!=1:raise ValueError('cannot establish local encoded image size')
            sheet['cli_encoded_size']=observation['encoded_images'][0]['encoded_size']
        save_json(plan_path.parent/'current_cli_input_audit.json',audits)
        prompt='第一张为全部A主图，第二张为全部B目标图。逐个处理每个B编号，列出所有可行A候选；没有对应允许UNMATCHED，不能看清必须UNCERTAIN。\n'+SHARED_PROMPT
        answer=client.call([plan['source_sheet']['path'],plan['target_sheet']['path']],prompt,group_schema(plan['sources'],plan['targets']),plan['plan_id'],validator=lambda result:validate_group(result,plan))
        rows=validate_group(answer['result'],plan);tasks=review_tasks(plan,rows)
        source={x['id']:x for x in plan['sources']};target={x['id']:x for x in plan['targets']}
        for task in tasks[:args.max_reviews]:
            candidates=task['source_ids'][:2]
            if not candidates:continue
            if len(candidates)==2:
                schema,prompt=REVIEW_SCHEMA,REVIEW_PROMPT
                images=[target[task['target_id']]['path']]+[source[x]['path'] for x in candidates]
            else:
                schema,prompt=PAIR_SCHEMA,PAIR_PROMPT;images=[source[candidates[0]]['path'],target[task['target_id']]['path']]
            review=client.call(images,prompt,schema,plan['plan_id'],kind='REVIEW',validator=lambda value:validate_output(value,schema))
            task.update(status='SUCCESS',after=review['result'])
        save_json(plan_path.parent/'review_tasks.json',tasks)
        rows=apply_reviews(rows,tasks)
        save_json(plan_path.parent/'supported_global_assignment.json',assign_supported(plan,rows,tasks))
        for row in rows:
            row['confirmed']=False # A model MATCH does not authorize naming as human-confirmed.
            row['model']='gpt-6.1-terra'
        save_json(plan_path.parent/'model_result.json',answer)
        report(plan_path.parent/'report',plan,rows,provenance='REAL_MODEL_SUGGESTIONS_REQUIRES_NAMING_CONFIRMATION')
        save_json(plan_path.parent/'performance.json',{'seconds_since_run':round(time.perf_counter()-start,3),'includes_scan_and_sheet':False,'automatic_naming':False,'accuracy':None})
    finally:client.close()

SHARED_PROMPT='''Compare the clothing product identity, not the image identity.
比较同一服装设计及配色：衣身、领袖、正背面印花、数字文字及局部。模特、姿势、背景、合理光照、普通褶皱或遮挡不单独构成不同款。
不同球队主题或同图案不同颜色均需逐项核实。MATCH必须有充分可见一致证据且无冲突；DIFFERENT必须有明确冲突；不确定和无法看清时请求原图复核。
每个目标编号出现一次，MATCH中主图最多分配一次。只读取附带的两张图片，不允许根据文件名或路径推测，不运行任何工具。'''

if __name__=='__main__':main()
