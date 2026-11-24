from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionmatch.product.app import create_app
from visionmatch.product.tasks import BASE_ARTWORK_RULE, BLADE_SHOE_BLACK_SOLE_RULE, GROUP_PROMPT, group_prompt


@pytest.mark.parametrize('folder', [
    '黑底主图', '黑鞋底主图', '黑底基码图', '基码图 （黑底）', '白底基码图',
    '黑底定制预览图', '定制预览图（黑底）', '产品定制图', '产品细节图',
    '室内场景图', '户外场景图', '其他附图',
])
def test_shoe_rule_follows_product_across_target_folders(folder):
    previous_rules = BASE_ARTWORK_RULE if '基码图' in folder else ''
    assert group_prompt(folder) == GROUP_PROMPT + previous_rules
    assert group_prompt(folder, 'other_shoe') == GROUP_PROMPT + previous_rules
    assert group_prompt(folder, 'blade_shoe') == GROUP_PROMPT + previous_rules + BLADE_SHOE_BLACK_SOLE_RULE


def test_shoe_profile_has_separate_cache_and_does_not_confirm_result(tmp_path, monkeypatch):
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
        app.state.service.import_local(batch, str(source), [{'path': str(target), 'label': '产品定制图'}])
        queue = app.state.queue
        for profile, expected_cache_hit in [(None, False), ('blade_shoe', False), ('blade_shoe', True), (None, True)]:
            response = client.post(f'/api/batches/{batch}/match', json={
                'confirmed_budget': True, 'max_calls': 1, 'matching_profile': profile,
            })
            assert response.status_code == 200, response.text
            metadata = queue.perform(queue.claim())
            assert bool(metadata['cache_hit']) == expected_cache_hit
            queue.complete_runs()

        assert received == [GROUP_PROMPT, GROUP_PROMPT + BLADE_SHOE_BLACK_SOLE_RULE]
        match = app.state.service.view(batch)['matches'][0]
        assert match['status'] == 'MATCHED'
        assert not match['confirmed']
