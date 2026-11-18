from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionmatch.product.app import create_app
from visionmatch.product.tasks import BASE_ARTWORK_RULE, BLADE_SHOE_BLACK_SOLE_RULE, GROUP_PROMPT, group_prompt


@pytest.mark.parametrize(('folder', 'extra_rules'), [
    ('黑底主图', BLADE_SHOE_BLACK_SOLE_RULE),
    ('黑鞋底主图', BLADE_SHOE_BLACK_SOLE_RULE),
    ('黑底基码图', BASE_ARTWORK_RULE + BLADE_SHOE_BLACK_SOLE_RULE),
    ('基码图 （黑底）', BASE_ARTWORK_RULE + BLADE_SHOE_BLACK_SOLE_RULE),
    ('黑底定制预览图', BLADE_SHOE_BLACK_SOLE_RULE),
    ('定制预览图（黑底）', BLADE_SHOE_BLACK_SOLE_RULE),
    ('白底基码图', BASE_ARTWORK_RULE),
    ('换底图', ''),
    ('其他黑底主图', ''),
])
def test_blade_shoe_rule_applies_only_to_designated_black_sole_folders(folder, extra_rules):
    previous_rules = BASE_ARTWORK_RULE if '基码图' in folder else ''
    assert group_prompt(folder) == GROUP_PROMPT + previous_rules
    assert group_prompt(folder, 'other_shoe') == GROUP_PROMPT + previous_rules
    assert group_prompt(folder, 'blade_shoe') == GROUP_PROMPT + extra_rules


def test_api_carries_profile_to_terra_without_confirming_result(tmp_path, monkeypatch):
    monkeypatch.setattr('visionmatch.product.tasks.executable', lambda: Path(__file__))
    received = []

    def invoke(images, prompt, schema, job):
        received.append(prompt)
        sources, targets = app.state.service.records(job['batch_id'], job['folder_id'])
        return {'result': {'targets': [{
            'source_id': sources[0]['id'], 'target_id': targets[0]['id'],
            'decision': 'MATCH', 'evidence': ['队标和鞋面固定图案一致'], 'conflicts': [],
            'needs_review': False, 'alternative_source_ids': [], 'review_reasons': [],
        }]}}

    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    Image.new('RGB', (32, 32), 'white').save(source / 'main.png')
    Image.new('RGB', (32, 32), 'black').save(target / 'detail.png')
    app = create_app(tmp_path / 'app', start_worker=False, invoker=invoke)
    with TestClient(app) as client:
        batch = app.state.service.create('测试批次')['id']
        app.state.service.import_local(batch, str(source), [{'path': str(target), 'label': '黑底主图'}])
        response = client.post(f'/api/batches/{batch}/match', json={
            'confirmed_budget': True, 'max_calls': 1, 'matching_profile': 'blade_shoe',
        })
        assert response.status_code == 200, response.text
        queue = app.state.queue
        queue.perform(queue.claim())

        assert received == [GROUP_PROMPT + BLADE_SHOE_BLACK_SOLE_RULE]
        match = app.state.service.view(batch)['matches'][0]
        assert match['status'] == 'MATCHED'
        assert not match['confirmed']
