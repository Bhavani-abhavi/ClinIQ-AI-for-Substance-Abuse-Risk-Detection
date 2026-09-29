"""Labeling workbench: queue order, quality checks, adjudication, export and the HTTP API."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeling.active_learning import labels_to_reach, pick_batch, uncertainty_order
from labeling.demo_data import demo_rows
from labeling.server import create_app
from labeling.workbench import Config, ValidationError, Workbench, highlight_spans


@pytest.fixture
def data():
    rows, gold = demo_rows()
    return rows, gold, {r['id']: r['label'] for r in rows}


def bench(rows, gold, **cfg):
    b = Workbench(config=Config(**cfg))
    b.import_items(rows, gold_ids=gold)
    return b


def test_import_validates_dedupes_and_reports():
    rows = [{'id': 'a', 'text': 'one two three four five six seven eight nine ten'},
            {'id': 'b', 'text': 'One two three four five six seven eight nine ten'},     # duplicate, case-insensitive
            {'id': 'c', 'text': 'too short'},
            {'id': '', 'text': 'no id but long enough to count as a review of the drug'},
            {'id': 'g', 'text': 'gold without a label is rejected because it cannot score anyone', 'label': None}]
    report = Workbench().import_items(rows, gold_ids={'g'})
    assert report == {'accepted': 1, 'duplicates': 1, 'too_short': 1, 'invalid': 2, 'gold': 0}


def test_queue_asks_for_the_most_uncertain_item_once_a_model_exists(data):
    rows, gold, truth = data
    b = bench(rows, gold, gold_every=1000, retrain_every=6, overlap_rate=0)
    for _ in range(6):
        t = b.next_task('ana')
        b.submit('ana', t['item_id'], truth[t['item_id']], 3)
    assert b.model is not None
    t = b.next_task('ana')
    labeled = {i for i in b.resolved()}
    open_ids = [i for i in b._ids if i not in labeled and i not in gold]
    probs = {i: b._prob[b._index[i]] for i in open_ids}
    assert abs(probs[t['item_id']] - 0.5) == pytest.approx(min(abs(p - 0.5) for p in probs.values()))
    assert t['suggestion']['label'] in (0, 1) and 0.5 <= t['suggestion']['confidence'] <= 1


def test_cold_start_alternates_keyword_rich_and_random_order_items(data):
    rows, gold, truth = data
    b = bench(rows, gold, gold_every=1000, retrain_every=1000, overlap_rate=0)
    t1 = b.next_task('ana')
    b.submit('ana', t1['item_id'], truth[t1['item_id']])
    t2 = b.next_task('ana')
    open_plain = [i for i in b._ids if i not in gold and i != t1['item_id']]
    assert len(t1['highlights']) == max(len(b._task(i)['highlights']) for i in b._ids if i not in gold)
    assert t2['item_id'] == open_plain[0]


def test_gold_items_are_mixed_in_and_score_annotators(data):
    rows, gold, truth = data
    b = bench(rows, gold, gold_every=3, retrain_every=1000, overlap_rate=0)
    served = []
    for k in range(9):
        t = b.next_task('ana')
        served.append(t['item_id'])
        wrong = t['item_id'] in gold and k == 2                # miss the first gold item on purpose
        b.submit('ana', t['item_id'], 1 - truth[t['item_id']] if wrong else truth[t['item_id']], 4)
    assert [i in gold for i in served] == [False, False, True] * 3
    ana = b.stats()['annotators'][0]
    assert (ana['gold_seen'], ana['gold_accuracy']) == (3, round(2 / 3, 3))
    assert 'low_gold_accuracy' in ana['flags']


def test_overlap_gives_agreement_and_conflicts_wait_for_adjudication(data):
    rows, gold, truth = data
    b = bench(rows, gold, gold_every=1000, retrain_every=1000, overlap_rate=1.0)
    first = []
    for _ in range(6):
        t = b.next_task('ana')
        first.append(t['item_id'])
        b.submit('ana', t['item_id'], truth[t['item_id']], 5)
    for k in range(6):
        t = b.next_task('ben')
        assert t['item_id'] == first[k]                         # second opinions come first
        b.submit('ben', t['item_id'], 1 - truth[t['item_id']] if k == 0 else truth[t['item_id']], 5)
    s = b.stats()
    assert s['agreement']['double_labeled'] == 6 and s['agreement']['cohen_kappa'] is not None
    conflicts = b.conflicts()
    assert [c['item_id'] for c in conflicts] == [first[0]] and first[0] not in b.resolved()
    b.adjudicate(first[0], truth[first[0]], 'lead')
    assert b.conflicts() == [] and b.resolved()[first[0]] == truth[first[0]]


def test_submit_rejects_bad_input_and_double_labels(data):
    rows, gold, truth = data
    b = bench(rows, gold)
    t = b.next_task('ana')
    with pytest.raises(ValidationError):
        b.submit('ana', t['item_id'], 2)
    with pytest.raises(ValidationError):
        b.submit('', t['item_id'], 1)
    with pytest.raises(ValidationError):
        b.submit('ana', 'nope', 1)
    b.submit('ana', t['item_id'], 1)
    with pytest.raises(ValidationError):
        b.submit('ana', t['item_id'], 0)
    b.submit('ana', b.next_task('ana')['item_id'], 'skip')
    assert b.stats()['annotators'][0]['labels'] == 1


def test_export_has_only_resolved_non_gold_labels(data):
    rows, gold, truth = data
    b = bench(rows, gold, gold_every=2, overlap_rate=0)
    for _ in range(8):
        t = b.next_task('ana')
        b.submit('ana', t['item_id'], truth[t['item_id']], 3)
    out = [json.loads(line) for line in b.export_jsonl().splitlines()]
    assert len(out) == 4 and not any(r['id'] in gold for r in out)
    assert all(r['label'] == truth[r['id']] and r['annotations'] == 1 for r in out)


def test_highlights_mark_substance_terms():
    text = 'Suboxone helped my withdrawal; no cravings since detox.'
    assert [text[a:b] for a, b in highlight_spans(text)] == ['Suboxone', 'withdrawal', 'cravings', 'detox']


def test_active_learning_helpers():
    assert list(uncertainty_order(np.array([0.9, 0.52, 0.1, 0.48]))) == [1, 3, 0, 2]
    rng = np.random.default_rng(0)
    assert list(pick_batch('uncertainty', np.array([0.9, 0.5, 0.2, 0.45]), np.array([0, 2, 3]), 2, rng)) == [3, 2]
    assert labels_to_reach([100, 200, 300], [0.5, 0.8, 0.9], 0.8) == 200
    assert labels_to_reach([100], [0.5], 0.8) is None


def test_http_api_round_trip(data):
    rows, gold, truth = data
    client = TestClient(create_app(bench(rows, gold, gold_every=1000, overlap_rate=1.0)))
    assert client.get('/api/health').json() == {'ok': True}
    assert client.get('/api/task', params={'annotator': ' '}).status_code == 422
    t = client.get('/api/task', params={'annotator': 'ana'}).json()
    assert client.post('/api/labels', json={'annotator': 'ana', 'item_id': t['item_id'], 'label': 7}).status_code == 422
    assert client.post('/api/labels', json={'annotator': 'ana', 'item_id': t['item_id'],
                                            'label': truth[t['item_id']], 'seconds': 3.2}).json() == {'saved': True}
    t2 = client.get('/api/task', params={'annotator': 'ben'}).json()
    assert t2['item_id'] == t['item_id']
    client.post('/api/labels', json={'annotator': 'ben', 'item_id': t['item_id'], 'label': 1 - truth[t['item_id']]})
    assert [c['item_id'] for c in client.get('/api/conflicts').json()] == [t['item_id']]
    assert client.post('/api/adjudicate', json={'item_id': t['item_id'], 'label': truth[t['item_id']],
                                                'by': 'lead'}).json() == {'saved': True}
    lines = client.get('/api/export').text.splitlines()
    assert json.loads(lines[0])['adjudicated'] is True
    assert client.get('/api/stats').json()['conflicts'] == 0
