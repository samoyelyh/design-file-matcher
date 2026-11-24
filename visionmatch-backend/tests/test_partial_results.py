import json
import sqlite3
import pytest
from jsonschema import ValidationError
from PIL import Image

from visionmatch.product.app import create_app
from visionmatch.ultra import requests as transport


def test_product_group_keeps_valid_rows_from_an_incomplete_cli_receipt(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    Image.new('RGB', (24, 24), 'white').save(source / 'main.png')
    for index in range(5):
        Image.new('RGB', (24, 24), (index * 30, 0, 0)).save(target / f'{index}.png')
    app = create_app(tmp_path / 'runtime', start_worker=False)
    service, queue = app.state.service, app.state.queue
    batch = service.create('测试')['id']
    folder = service.import_local(batch, str(source), [{'path': str(target), 'label': '定制图'}])['folders'][1]['id']
    row = {'source_id': 'A001', 'decision': 'MATCH', 'evidence': ['印花一致'], 'conflicts': [],
           'needs_review': False, 'alternative_source_ids': [], 'review_reasons': []}
    valid = {**row, 'target_id': 'B001'}
    malformed = {**row, 'target_id': 'B002', 'source_id': []}
    duplicate = {**row, 'target_id': 'B003'}
    wrong_source = {**row, 'target_id': 'B005', 'source_id': 'A999'}
    receipt = {'targets': [valid, malformed, duplicate, duplicate, wrong_source]}

    monkeypatch.setattr('visionmatch.product.tasks.executable', lambda: __file__)
    monkeypatch.setattr(transport.provider, 'executable', lambda: __file__)
    monkeypatch.setattr(transport.provider, 'login_status', lambda: None)

    def args(**kwargs):
        cli_schema = json.loads(kwargs['schema_path'].read_text(encoding='utf-8'))
        assert 'evidence' in cli_schema['properties']['targets']['items']['required']
        kwargs['output_path'].write_text(json.dumps(receipt), encoding='utf-8')
        return ['offline-test']

    class FakeProcess:
        returncode = 0

        def __init__(self, args, **kwargs):
            assert args == ['offline-test']

        def communicate(self, prompt, timeout):
            return json.dumps({'type': 'turn.completed'}), ''

    monkeypatch.setattr(transport.provider, 'codex_args', args)
    monkeypatch.setattr(transport.subprocess, 'Popen', FakeProcess)
    try:
        queue.enqueue(batch, 1, 1)
        job = queue.claim()
        queue.perform(job)
        state = service.view(batch)
        assert state['folders'][1]['state'] == 'PARTIAL'
        assert set(json.loads(state['folders'][1]['error'])) == {'B002', 'B003', 'B004', 'B005'}
        matches = {m['target']['local_id']: m for m in state['matches']}
        assert matches['B001']['status'] == 'MATCHED'
        for label in ('B002', 'B003', 'B004', 'B005'):
            assert matches[label]['status'] == 'ERROR' and matches[label]['assigned_source_id'] is None
        assert matches['B003']['notes'] == ['DUPLICATE_TARGET']
        assert matches['B004']['notes'] == ['MODEL_OMITTED']
        assert queue.store.one('SELECT COUNT(*) AS n FROM model_cache')['n'] == 0
        job_dir = queue.store.batch_root(batch) / 'requests' / job['id']
        with sqlite3.connect(job_dir / 'ultra_requests.sqlite') as db:
            assert db.execute('SELECT COUNT(*) FROM cache').fetchone()[0] == 0
        assert json.loads((job_dir / 'response.json').read_text(encoding='utf-8'))['result'] == receipt

        for foreign in ({**valid, 'target_id': 'B999'}, {'decision': 'MATCH'}):
            with pytest.raises(ValidationError):
                service.ingest(batch, folder, {'targets': [valid, foreign]}, {})
            assert service.view(batch)['matches'] == state['matches']

        # The original standalone transport and other callers keep strict validation.
        strict_dir = tmp_path / 'strict'
        strict_dir.mkdir()
        strict = transport.Client(strict_dir, approved_plan='plan', max_calls=1)
        try:
            schema_file = next((job_dir / 'requests').glob('*/schema.json'))
            schema = json.loads(schema_file.read_text(encoding='utf-8'))
            with pytest.raises(ValidationError):
                strict.call([source / 'main.png'], 'offline', schema, 'plan')
        finally:
            strict.close()
    finally:
        queue.close()
