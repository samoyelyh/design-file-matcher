import json
import pytest
from app.services import qwen_lab as q


def test_schema_sent_and_invalid_output_rejected(monkeypatch):
    seen=[]
    def request(path,payload=None,timeout=8):
        if path=='/api/tags': return {'models':[{'name':'test'}]}
        if path=='/api/show': return {'capabilities':['vision']}
        seen.append(payload)
        return {'done':True,'response':json.dumps({'match':True,'confidence':2,'evidence':[],'conflicts':[]})}
    monkeypatch.setattr(q,'_request',request)
    with pytest.raises(q.StructuredOutputError): q.compare('abc','def','左侧裤子',model='test')
    assert seen[0]['format']==q.PairVerdict.model_json_schema()
    assert 'IMAGE_1=QUERY' in seen[0]['prompt']
    assert 'IMAGE_2=CANDIDATE_1' in seen[0]['prompt']


def test_final_identity_mapping_and_forced_choice(monkeypatch):
    pairs=[{'candidate_index':i,'result':{'match':i==3,'confidence':0.8 if i==3 else 0.1,'evidence':['PRIVATE_DESCRIPTION']}} for i in range(1,6)]
    seen=[]
    def generate(prompt,images,model,schema,*args):
        assert 'PRIVATE_DESCRIPTION' not in prompt
        seen.append(images)
        return {'result':{'selected':'TOP1'}}
    monkeypatch.setattr(q,'generate',generate)
    r=q.judge_top2('Q',['a','b','c','d','e'],pairs,'左侧裤子')
    assert seen==[['Q','c','a']]
    assert r['selected_candidate_index']==3
    assert 'NONE' not in q.FinalVerdict.model_json_schema()['properties']['selected']['enum']


def test_pairwise_ranking_prefers_positive_then_least_certain_rejection():
    rows=[
        {'candidate_index':1,'match':False,'confidence':0.98},
        {'candidate_index':2,'match':False,'confidence':0.15},
        {'candidate_index':3,'match':True,'confidence':0.61},
        {'candidate_index':4,'match':False,'confidence':0.42},
        {'candidate_index':5,'match':True,'confidence':0.92},
    ]
    assert [r['candidate_index'] for r in q.rank_pairwise(rows)] == [5,3,2,4,1]
    wrapped=[{'candidate_index':2,'result':{'match':False,'confidence':0.9}},
             {'candidate_index':1,'result':{'match':True,'confidence':0.2}}]
    assert [r['candidate_index'] for r in q.rank_pairwise(wrapped)] == [1,2]


def test_pairwise_ranking_uses_fusion_score_when_present():
    rows = [
        {'candidate_index': 1, 'match': True, 'confidence': .99, 'fusion_score': .55},
        {'candidate_index': 2, 'match': False, 'confidence': .91, 'fusion_score': .88},
        {'candidate_index': 3, 'match': True, 'confidence': .75, 'fusion_score': .70},
    ]
    assert [r['candidate_index'] for r in q.rank_pairwise(rows)] == [2, 3, 1]


def test_top2_forces_choice_when_all_pairwise_verdicts_are_negative(monkeypatch):
    pairs=[{'candidate_index':i,'result':{'match':False,'confidence':c,'evidence':[],'conflicts':[]}} for i,c in enumerate((0.95,0.15,0.55,0.35,0.75),1)]
    def generate(prompt,images,model,schema,*args):
        assert '不可弃权' in prompt
        return {'result':{'selected':'TOP1','confidence':0.15,'evidence':['最接近'],'conflicts':['两者都有差异']}}
    monkeypatch.setattr(q,'generate',generate)
    result=q.judge_top2('Q',['a','b','c','d','e'],pairs,'左侧裤子')
    assert result['top2_candidate_indices']==[2,4]
    assert result['selected_candidate_index']==2


def test_top2_final_uses_images_and_includes_only_numeric_fusion_context(monkeypatch):
    pairs = [
        {'candidate_index': 1, 'fusion_score': .91, 'result': {'match': True, 'confidence': .8}},
        {'candidate_index': 2, 'fusion_score': .87, 'result': {'match': True, 'confidence': .7}},
        {'candidate_index': 3, 'fusion_score': .1, 'result': {'match': False, 'confidence': .99}},
    ]
    seen = {}

    def generate(prompt, images, model, schema, *args):
        seen['prompt'] = prompt
        seen['images'] = images
        return {'result': {'selected': 'TOP1', 'confidence': .8, 'evidence': [], 'conflicts': []}}

    monkeypatch.setattr(q, 'generate', generate)
    context = [
        {'candidate_index': 1, 'qwen_support_score': .8, 'lightglue_color_score': .95, 'fusion_score': .91},
        {'candidate_index': 2, 'qwen_support_score': .7, 'lightglue_color_score': .98, 'fusion_score': .87},
    ]
    result = q.judge_top2('QUERY_ORIGINAL', ['MAIN_1', 'MAIN_2', 'MAIN_3'], pairs,
                          '只比较左侧裤子', 'qwen3-vl:4b-instruct', scoring_context=context)
    assert seen['images'] == ['QUERY_ORIGINAL', 'MAIN_1', 'MAIN_2']
    assert '融合排序分=0.91' in seen['prompt']
    assert '不是概率' in seen['prompt']
    assert '候选描述' not in seen['prompt']
    assert result['selected_candidate_index'] == 1


def test_final_cache_key_invalidates_when_fusion_weights_change():
    baseline = q.final_cache_key('q', ['a', 'b'], '白底', 'qwen', '服装')
    weighted = q.final_cache_key('q', ['a', 'b'], '白底', 'qwen', '服装',
                                 scoring_context={'weights': {'qwen': .7, 'lightglue_color': .3}})
    assert baseline != weighted


def test_reject_schema_drift():
    with pytest.raises(ValueError):q.PairVerdict.model_validate({'match':True,'confidence':0.9,'evidence':'wrong type','conflicts':[]})
    with pytest.raises(ValueError):q.FinalVerdict.model_validate({'selected':'CANDIDATE_9','confidence':0.9,'evidence':[],'conflicts':[]})
    with pytest.raises(ValueError):q.FinalVerdict.model_validate({'selected':'NONE','confidence':0.9,'evidence':[],'conflicts':[]})
    with pytest.raises(ValueError):q.compare('q','c',' ')


def test_request_updates_timeout_on_reused_socket(monkeypatch):
    class FakeSocket:
        timeout = None
        def settimeout(self, value): self.timeout = value

    class FakeResponse:
        status = 200
        def read(self): return b'{}'

    class FakeConnection:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get('timeout')
            self.sock = FakeSocket()
        def request(self, *args, **kwargs): pass
        def getresponse(self): return FakeResponse()
        def close(self): pass

    monkeypatch.setattr(q, 'OLLAMA_URL', 'http://localhost:11434')
    monkeypatch.setattr(q, '_http_connections', {})
    monkeypatch.setattr(q.http.client, 'HTTPConnection', FakeConnection)
    q._request('/api/tags', timeout=8)
    q._request('/api/generate', {'model': 'test'}, timeout=900)
    conn = q._http_connections['http://localhost:11434']
    assert conn.timeout == 900
    assert conn.sock.timeout == 900
