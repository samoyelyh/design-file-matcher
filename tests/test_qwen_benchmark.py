import json
from pathlib import Path
from scripts.benchmark_qwen_judge import metrics,Manifest,preflight
import pytest

def test_metrics_failures_abstention_false_positive():
    rows=[]
    for truth,selected,error,seconds,fp in [(3,3,None,1,0),(3,2,None,2,1),(3,None,None,3,0),(3,None,'invalid',4,0)]:
        rows.append(dict(truth=truth,selected=selected,top3=[3,2,1] if selected else [],error=error,seconds=seconds,calls=5,false_positive_pairs=fp,negative_pairs=4))
    r=metrics(rows)
    assert r['top1_accuracy']==.25 and r['top3_accuracy']==.5
    assert r['wrong_matches']==1 and r['unable_to_judge']==1 and r['errors']==1
    assert r['false_positive_pairs']==1 and r['mean_seconds']==2.5 and r['p95_seconds']==4

def test_manifest_rejects_insufficient_data():
    with pytest.raises(ValueError):Manifest.model_validate({'model':'qwen','cases':[]})

def test_duplicate_queries_rejected(tmp_path):
    files=[]
    for i in range(6):
        p=tmp_path/f'{i}.png';p.write_bytes(bytes([i]));files.append(str(p))
    case=dict(id='x',query=files[0],candidates=files[1:],correct_index=1,target='pants',confirmed_by='human',confirmed_at='2026-09-29',annotation_source='review')
    m=Manifest.model_validate({'model':'qwen','cases':[dict(case,id=str(i)) for i in range(30)]})
    with pytest.raises(ValueError,match='Duplicate query'):preflight(m)
