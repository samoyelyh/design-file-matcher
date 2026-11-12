"""Offline Qwen benchmark. JSON manifest only; never writes production data."""
from __future__ import annotations
import argparse, hashlib, json, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import Field, model_validator
from typing import Literal
from app.services import qwen_lab as q

class Case(q.StrictOutput):
    id: str
    query: str
    candidates: list[str] = Field(min_length=5, max_length=5)
    correct_index: int | None = Field(ge=1, le=5)
    target: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    confirmed_at: str = Field(min_length=1)
    annotation_source: str = Field(min_length=1)

class Manifest(q.StrictOutput):
    model: str
    cases: list[Case] = Field(min_length=21, max_length=50)

class Candidate(q.PairVerdict):
    candidate_index: Literal[1,2,3,4,5]

class Multi(q.StrictOutput):
    results: list[Candidate] = Field(min_length=5, max_length=5)
    @model_validator(mode='after')
    def unique(self):
        if {r.candidate_index for r in self.results} != {1,2,3,4,5}:
            raise ValueError('Candidate IDs must be unique and complete')
        return self

def rank(rows):
    return sorted(rows, key=lambda r:(not r['match'], -r['confidence'], r['candidate_index']))

def metrics(records):
    positive=[r for r in records if r['truth'] is not None]
    valid=[r for r in records if not r['error']]
    times=sorted(r['seconds'] for r in records)
    return {'n':len(records),'positive_cases':len(positive),
        'top1_accuracy':sum(r['selected']==r['truth'] for r in positive)/len(positive) if positive else None,
        'top3_accuracy':sum(r['truth'] in r['top3'] for r in positive)/len(positive) if positive else None,
        'wrong_matches':sum(r['selected'] is not None and r['selected']!=r['truth'] for r in records),
        'unable_to_judge':sum(r['selected'] is None for r in valid),
        'errors':len(records)-len(valid),
        'false_positive_pairs':sum(r['false_positive_pairs'] for r in valid),
        'negative_pairs':sum(r['negative_pairs'] for r in valid),
        'negative_query_false_positives':sum(r['truth'] is None and r['selected'] is not None for r in valid),
        'negative_query_count':sum(r['truth'] is None for r in records),
        'mean_seconds':sum(times)/len(times), 'p95_seconds':times[math.ceil(.95*len(times))-1],
        'mean_model_calls':sum(r['calls'] for r in records)/len(records)}

def preflight(manifest):
    ids=set(); queries=set(); hashes={}
    for c in manifest.cases:
        if c.id in ids:raise ValueError('Duplicate case ID')
        ids.add(c.id)
        for value in [c.query,*c.candidates]:
            p=Path(value)
            if not p.is_absolute() or not p.is_file():raise ValueError(f'Unreadable absolute image path: {p}')
            hashes[value]=hashlib.sha256(p.read_bytes()).hexdigest()
        if hashes[c.query] in queries:raise ValueError('Duplicate query bytes: do not inflate sample count')
        queries.add(hashes[c.query])
        if len({hashes[p] for p in c.candidates})!=5:raise ValueError('Duplicate candidates')
    return hashes

def execute(c, scheme, model, out):
    calls=[]; started=time.perf_counter(); rows=[]; chosen=None; top3=[]; error=None
    def call(fn,*args,**kwargs):
        try:
            result=fn(*args,**kwargs);calls.append(result);return result
        except Exception as exc:
            calls.append({'error':str(exc),'raw':getattr(exc,'raw',None)});raise
    try:
        if scheme=='A':
            labels=' IMAGE_1 = QUERY；'+'；'.join(f'IMAGE_{i+1} = CANDIDATE_{i}' for i in range(1,6))
            prompt=q.RULES+labels+f'。禁止猜测或交换角色。目标：{c.target}。逐一直接看图片比较QUERY目标与各候选是否同款，返回results，每项candidate_index/match/confidence/evidence/conflicts。'
            result=call(q.generate,prompt,[c.query,*c.candidates],model,Multi)
            rows=result['result']['results']
        else:
            pairs=[call(q.compare,c.query,image,c.target,i,model) for i,image in enumerate(c.candidates,1)]
            rows=[dict(p['result'],candidate_index=p['candidate_index']) for p in pairs]
        ordered=rank(rows); top3=[r['candidate_index'] for r in ordered[:3]]
        chosen=ordered[0]['candidate_index'] if ordered[0]['match'] else None
        if scheme=='C':
            final=call(q.judge_top2,c.query,c.candidates,pairs,c.target,model)
            chosen=final['selected_candidate_index']
            # Top2 rerank cannot increase the original Top3 recall.
            if chosen is not None: top3=[chosen]+[i for i in top3 if i!=chosen]
    except Exception as exc:
        error=str(exc);chosen=None;top3=[]
    record={'case_id':c.id,'scheme':scheme,'truth':c.correct_index,'selected':chosen,'top3':top3,
        'seconds':time.perf_counter()-started,'calls':len(calls),'error':error,
        'false_positive_pairs':sum(r['match'] and r['candidate_index']!=c.correct_index for r in rows),
        'negative_pairs':5-(c.correct_index is not None),'candidate_results':rows,'raw_calls':calls}
    (out/f'{c.id}_{scheme}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    m=Manifest.model_validate_json(args.manifest.read_text(encoding='utf-8')); hashes=preflight(m)
    if any(not c.id.replace('-','').replace('_','').isalnum() for c in m.cases):raise ValueError('Case IDs must be filename-safe')
    if args.validate_only:print(f'Validated {len(m.cases)} unique annotated cases');return
    args.out.mkdir(parents=True,exist_ok=False)
    show=q._request('/api/show',{'model':m.model});tags=q.list_models()
    if 'vision' not in show.get('capabilities',[]):raise ValueError('Vision required')
    metadata={'manifest':m.model_dump(),'sha256':hashes,'options':q.OPTIONS,'model_show':show,'tags':tags,'rules':q.RULES}
    (args.out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    # Fixed warmup excluded from timed samples. Every scheme still executes its own calls.
    q.compare(m.cases[0].query,m.cases[0].candidates[0],m.cases[0].target,1,m.model)
    records=[]
    for number,c in enumerate(m.cases):
        for scheme in [('A','B','C'),('B','C','A'),('C','A','B')][number%3]:
            r=execute(c,scheme,m.model,args.out);records.append(r)
            print(c.id,scheme,r['selected'],round(r['seconds'],2),r['error'],flush=True)
    summary={s:metrics([r for r in records if r['scheme']==s]) for s in 'ABC'}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Qwen benchmark','', '|方案|Top1|Top3|错误匹配|无法判断|错误|FP候选对|平均秒|P95秒|平均调用|','|---|---|---|---|---|---|---|---|---|---|']
    for s,v in summary.items():lines.append('|'+ '|'.join(str(x) for x in [s,v['top1_accuracy'],v['top3_accuracy'],v['wrong_matches'],v['unable_to_judge'],v['errors'],v['false_positive_pairs'],round(v['mean_seconds'],2),round(v['p95_seconds'],2),v['mean_model_calls']])+'|')
    lines+=['','Top1/Top3分母为有正确候选的全部样本，技术失败计未命中。Top3为排序召回，不表示自动通过；C只重排B的Top2，因此Top3沿用同次Pair的Top3集合。','FP候选对为错误候选被match=true；C此项统计裁决前Pair，最终错误另见错误匹配数。无匹配Query的FP单列于summary.json；若无负例，不估计负例FP率。','无法判断为成功推理但未选出候选；技术错误另列。耗时含图片读取、Vision检查及校验，失败也计时。P95使用nearest-rank。','A/B/C按样本轮换执行顺序，B/C各自执行5次比较，不复用结果。排除一次预热，Ollama内部缓存仍可能影响耗时。','confidence仅作现有Judge排序次序和记录，无自动通过阈值。所有最终判断直接输入图片；Top2不接收文字描述或前次理由。','','## 逐条结果','|样本|方案|真值|选择|Top3|秒|错误|','|---|---|---|---|---|---|---|']
    for r in records:lines.append(f"|{r['case_id']}|{r['scheme']}|{r['truth']}|{r['selected']}|{r['top3']}|{r['seconds']:.2f}|{r['error']}|")
    (args.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
