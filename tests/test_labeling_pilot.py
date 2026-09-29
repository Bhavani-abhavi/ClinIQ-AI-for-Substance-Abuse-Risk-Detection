"""Assisted-vs-manual pilot: counterbalanced conditions, server-side condition, idempotent saves, report."""
import pytest
from fastapi.testclient import TestClient

from labeling.pilot import PilotStudy, demo_setup
from labeling.server import create_pilot_app
from labeling.workbench import ConflictError, ValidationError


@pytest.fixture
def study(tmp_path):
    db = str(tmp_path / 'pilot.db')
    demo_setup(db)
    return PilotStudy(db)


def test_conditions_alternate_and_are_counterbalanced_between_annotators(study):
    seen = {}
    for name in ('ana', 'ben'):
        conds = []
        for _ in range(6):
            t = study.next_task(name)
            conds.append((t['item_id'], t['condition']))
            assert (t['suggestion'] is None) == (t['condition'] == 'manual')
            assert (t['highlights'] == []) or t['condition'] == 'assisted'
            study.submit(name, t['item_id'], 1)
        seen[name] = conds
    assert [c for _, c in seen['ana']] == ['assisted', 'manual'] * 3
    assert [c for _, c in seen['ben']] == ['manual', 'assisted'] * 3
    assert [i for i, _ in seen['ana']] == [i for i, _ in seen['ben']]     # same items, opposite conditions


def test_the_server_decides_the_condition_and_saves_are_idempotent(study):
    t = study.next_task('ana')
    assert study.submit('ana', t['item_id'], 0, 4.0) == {'saved': True}
    assert study.submit('ana', t['item_id'], 0, 4.0) == {'saved': True, 'duplicate': True}
    with pytest.raises(ConflictError):
        study.submit('ana', t['item_id'], 1)
    with pytest.raises(ValidationError):
        study.submit('ana', t['item_id'], 'skip')
    cond = study.db.execute('SELECT condition FROM pilot_labels WHERE item_id = ?', (t['item_id'],)).fetchone()[0]
    assert cond == 'assisted'


def test_report_measures_time_agreement_and_followed_wrong_suggestions(study):
    items = study.db.execute('SELECT item_id, proxy, suggestion FROM pilot_items ORDER BY ord').fetchall()
    for name, follow in (('ana', True), ('ben', False)):
        for item_id, proxy, suggestion in items:
            t = study.next_task(name)
            assert t['item_id'] == item_id
            assisted = t['condition'] == 'assisted'
            label = suggestion if (assisted and follow) else proxy          # ana trusts suggestions, ben checks
            study.submit(name, item_id, label, 3.0 if assisted else 6.0)
    r = study.report()
    assert r['labels'] == 2 * len(items) and r['inter_annotator']['items_with_two_labels'] == len(items)
    a = r['by_condition']['assisted']
    assert a['suggestion_wrong_vs_proxy'] >= 1 and 0 < a['followed_a_wrong_suggestion'] < a['suggestion_wrong_vs_proxy'] + 1
    assert r['by_condition']['manual']['agreement_with_proxy'] == 1.0
    assert r['per_annotator']['ana']['time_ratio_assisted_to_manual'] == 0.5
    assert study.next_task('ana') is None


def test_pilot_http_api(study):
    client = TestClient(create_pilot_app(study))
    assert client.get('/api/mode').json() == {'mode': 'pilot'}
    t = client.get('/api/task', params={'annotator': 'ana'}).json()
    assert t['progress'] == {'done': 0, 'total': t['progress']['total']} and t['condition'] == 'assisted'
    assert client.post('/api/labels', json={'annotator': 'ana', 'item_id': t['item_id'], 'label': 1, 'seconds': 2}).json() == {'saved': True}
    assert client.post('/api/labels', json={'annotator': 'ana', 'item_id': t['item_id'], 'label': 0}).status_code == 409
    assert client.get('/api/pilot/report').json()['labels'] == 1
