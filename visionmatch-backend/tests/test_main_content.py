from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from visionmatch import provider
from visionmatch.product import content
from visionmatch.product.app import create_app


def test_content_only_sends_a_labeled_sheet_and_enforces_budget(tmp_path, monkeypatch):
    im = tmp_path / 'main.png'
    Image.new('RGB', (36, 36), 'blue').save(im)
    monkeypatch.setattr(provider, 'executable', lambda: str(Path(__file__)))
    calls = []

    class FakeClient:
        def __init__(self, root, *, approved_plan, max_calls):
            import sqlite3

            assert max_calls == 1
            self.db = sqlite3.connect(':memory:')
            self.db.execute('CREATE TABLE calls (status, metadata)')

        def call(self, images, prompt, schema, plan, **options):
            assert len(images) == 1
            assert Path(images[0]).name == 'contact_sheet.png'
            assert options['kind'] == 'MAIN_CONTENT'
            assert schema['properties']['images']['items']['properties']['image_id']['enum'] == ['M11', 'M52']
            answer = {'images': [{'image_id': ident, 'confidence': 0.9, **dict.fromkeys(content.LIST_FIELDS, [])}
                                 for ident in ['M52', 'M11']]}
            options['validator'](answer)
            calls.append(answer)
            return {'result': answer, 'cache_hit': True}

        def close(self):
            self.db.close()

    monkeypatch.setattr(content, 'Client', FakeClient)
    app = create_app(tmp_path / 'backend', start_worker=False)
    body = {'request_id': str(uuid4()), 'confirmed_budget': False, 'sources': [
        {'main_image_id': ident, 'path': str(im), 'sha256': provider.file_hash(im)} for ident in (11, 52)]}
    with TestClient(app) as client:
        assert client.post('/api/main-content/recognize', json=body).status_code == 400
        assert calls == []
        queue = app.state.queue
        assert queue.claim_content()
        assert client.post('/api/main-content/recognize', json={**body, 'confirmed_budget': True}).status_code == 409
        queue.release_content()
        answer = client.post('/api/main-content/recognize', json={**body, 'confirmed_budget': True})
        assert answer.status_code == 200, answer.text
        assert [row['main_image_id'] for row in answer.json()['images']] == [52, 11]
        assert len(calls) == 1


def test_duplicate_and_omitted_labels_are_rejected():
    import pytest

    with pytest.raises(ValueError, match='遗漏或重复'):
        content.require_coverage({'images': [{'image_id': 'M11'}, {'image_id': 'M11'}]}, ['M11', 'M52'])


def test_queued_matching_has_priority_over_content(tmp_path):
    source = tmp_path / 'main'
    target = tmp_path / 'detail'
    for folder, color in ((source, 'white'), (target, 'black')):
        folder.mkdir()
        Image.new('RGB', (32, 32), color).save(folder / 'sample.png')
    app = create_app(tmp_path / 'backend', start_worker=False)
    service, queue = app.state.service, app.state.queue
    with TestClient(app):
        batch = service.create('legacy match')['id']
        service.import_local(batch, str(source), [{'path': str(target)}])
        run = queue.enqueue(batch, 1, 1)
        assert not queue.claim_content()
        assert queue.claim()['run_id'] == run
