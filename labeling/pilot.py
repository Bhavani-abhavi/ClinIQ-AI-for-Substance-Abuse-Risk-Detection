"""
Assisted-vs-manual labeling pilot: does the model's suggestion make human labeling faster without making it worse?

    python -m labeling.pilot setup --csv drugsComTest_raw.csv --db outputs/pilot/labeling_pilot.db --items 60
    python -m labeling.pilot serve --db outputs/pilot/labeling_pilot.db          # annotators open http://127.0.0.1:8765
    python -m labeling.pilot report --db outputs/pilot/labeling_pilot.db         # -> outputs/labeling_pilot.json

Design
- A fixed set of reviews, half SUD-relevant by the proxy label so both answers come up, in one shuffled order.
- Suggestions come from a TF-IDF + logistic regression model trained beforehand on other reviews (never the
  pilot items), so every assisted item has a real suggestion from the start.
- Conditions alternate item by item (assisted: suggestion and highlighted terms; manual: text only). Practice
  and fatigue therefore hit both conditions equally. The first annotator starts assisted, the second manual,
  and so on, so with two or more people every item is seen in both conditions.
- The server decides the condition; the client cannot choose it. Times are measured from when the item is shown
  to when it is labeled.

What the report can and cannot say: labels are compared with the proxy label (a keyword rule on the review's
condition field, not clinical truth) and with each other. With few annotators and items the differences are
noisy, so the report gives counts and medians, not significance claims.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from labeling.workbench import LABELS, ConflictError, ValidationError, clean_text, highlight_spans

SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_items (item_id TEXT PRIMARY KEY, ord INTEGER NOT NULL, text TEXT NOT NULL, drug TEXT,
                                        proxy INTEGER NOT NULL, suggestion INTEGER NOT NULL, confidence REAL NOT NULL);
CREATE TABLE IF NOT EXISTS pilot_annotators (name TEXT PRIMARY KEY, start TEXT NOT NULL, joined REAL NOT NULL);
CREATE TABLE IF NOT EXISTS pilot_labels (item_id TEXT NOT NULL, annotator TEXT NOT NULL, condition TEXT NOT NULL,
                                         label INTEGER NOT NULL, seconds REAL, created REAL NOT NULL,
                                         PRIMARY KEY (item_id, annotator));
CREATE TABLE IF NOT EXISTS pilot_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _locked(method):
    """Every read and write shares one SQLite connection across server threads; hold the lock for all of them."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class PilotStudy:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ── setup ────────────────────────────────────────────────────────────
    @_locked
    def load(self, items: list[dict], meta: dict) -> int:
        """items: item_id, text, drug, proxy (0/1), suggestion (0/1), confidence; in presentation order."""
        with self.lock, self.db:
            if self.db.execute('SELECT COUNT(*) FROM pilot_items').fetchone()[0]:
                raise ValidationError('pilot already set up; use a new database')
            for k, it in enumerate(items):
                self.db.execute('INSERT INTO pilot_items VALUES (?,?,?,?,?,?,?)',
                                (it['item_id'], k, clean_text(it['text']), it.get('drug'), int(it['proxy']),
                                 int(it['suggestion']), float(it['confidence'])))
            self.db.execute('INSERT INTO pilot_meta VALUES (?,?)', ('meta', json.dumps(meta)))
        return len(items)

    # ── annotators and conditions ────────────────────────────────────────
    def _start(self, annotator: str) -> str:
        row = self.db.execute('SELECT start FROM pilot_annotators WHERE name = ?', (annotator,)).fetchone()
        if row:
            return row[0]
        n = self.db.execute('SELECT COUNT(*) FROM pilot_annotators').fetchone()[0]
        start = 'assisted' if n % 2 == 0 else 'manual'
        with self.db:
            self.db.execute('INSERT INTO pilot_annotators VALUES (?,?,?)', (annotator, start, time.time()))
        return start

    @staticmethod
    def condition(ord_: int, start: str) -> str:
        first, second = ('assisted', 'manual') if start == 'assisted' else ('manual', 'assisted')
        return first if ord_ % 2 == 0 else second

    def next_task(self, annotator: str) -> dict | None:
        annotator = (annotator or '').strip()
        if not annotator:
            raise ValidationError('annotator is required')
        with self.lock:
            start = self._start(annotator)
            row = self.db.execute('SELECT item_id, ord, text, drug, suggestion, confidence FROM pilot_items '
                                  'WHERE item_id NOT IN (SELECT item_id FROM pilot_labels WHERE annotator = ?) '
                                  'ORDER BY ord LIMIT 1', (annotator,)).fetchone()
            done = self.db.execute('SELECT COUNT(*) FROM pilot_labels WHERE annotator = ?', (annotator,)).fetchone()[0]
            total = self.db.execute('SELECT COUNT(*) FROM pilot_items').fetchone()[0]
            if row is None:
                return None
            item_id, ord_, text, drug, sug, conf = row
            cond = self.condition(ord_, start)
            assisted = cond == 'assisted'
            return {'item_id': item_id, 'text': text, 'drug': drug, 'condition': cond,
                    'suggestion': {'label': sug, 'confidence': round(conf, 3)} if assisted else None,
                    'highlights': highlight_spans(text) if assisted else [],
                    'progress': {'done': done, 'total': total}}

    def submit(self, annotator: str, item_id: str, label, seconds: float | None = None) -> dict:
        annotator = (annotator or '').strip()
        if not annotator:
            raise ValidationError('annotator is required')
        if label not in LABELS:
            raise ValidationError('label must be 0 or 1 (no skipping in the pilot)')
        with self.lock:
            row = self.db.execute('SELECT ord FROM pilot_items WHERE item_id = ?', (item_id,)).fetchone()
            if row is None:
                raise ValidationError(f'unknown pilot item {item_id}')
            prior = self.db.execute('SELECT label FROM pilot_labels WHERE item_id = ? AND annotator = ?',
                                    (item_id, annotator)).fetchone()
            if prior and prior[0] == int(label):
                return {'saved': True, 'duplicate': True}
            if prior:
                raise ConflictError('already labeled in the pilot', {'item_id': item_id, 'label': prior[0]})
            cond = self.condition(row[0], self._start(annotator))       # decided by the server, never the client
            with self.db:
                self.db.execute('INSERT INTO pilot_labels VALUES (?,?,?,?,?,?)',
                                (item_id, annotator, cond, int(label), seconds, time.time()))
            return {'saved': True}

    # ── report ───────────────────────────────────────────────────────────
    @_locked
    def report(self) -> dict:
        items = {r[0]: {'proxy': r[1], 'suggestion': r[2]} for r in
                 self.db.execute('SELECT item_id, proxy, suggestion FROM pilot_items')}
        rows = self.db.execute('SELECT item_id, annotator, condition, label, seconds FROM pilot_labels').fetchall()
        meta = json.loads((self.db.execute("SELECT value FROM pilot_meta WHERE key = 'meta'").fetchone() or ['{}'])[0])
        by_cond: dict[str, dict] = {}
        for cond in ('assisted', 'manual'):
            rs = [r for r in rows if r[2] == cond]
            secs = [r[4] for r in rs if r[4] is not None]
            agree_proxy = [r[3] == items[r[0]]['proxy'] for r in rs]
            entry = {'labels': len(rs),
                     'median_seconds': round(median(secs), 1) if secs else None,
                     'agreement_with_proxy': round(float(np.mean(agree_proxy)), 3) if rs else None}
            if cond == 'assisted':
                wrong = [r for r in rs if items[r[0]]['suggestion'] != items[r[0]]['proxy']]
                entry['followed_suggestion'] = round(float(np.mean([r[3] == items[r[0]]['suggestion'] for r in rs])), 3) if rs else None
                entry['suggestion_wrong_vs_proxy'] = len(wrong)
                entry['followed_a_wrong_suggestion'] = sum(r[3] == items[r[0]]['suggestion'] for r in wrong)
            by_cond[cond] = entry

        per_annotator = {}
        for name in sorted({r[1] for r in rows}):
            mine = [r for r in rows if r[1] == name]
            med = {c: (round(median([r[4] for r in mine if r[2] == c and r[4] is not None]), 1)
                       if any(r[2] == c and r[4] is not None for r in mine) else None) for c in ('assisted', 'manual')}
            per_annotator[name] = {'labels': len(mine), 'median_seconds': med,
                                   'time_ratio_assisted_to_manual': round(med['assisted'] / med['manual'], 2)
                                   if med['assisted'] and med['manual'] else None}

        pairs = {}
        for item_id, annotator, cond, label, _ in rows:
            pairs.setdefault(item_id, []).append(label)
        both = [v[:2] for v in pairs.values() if len(v) >= 2]
        kappa = None
        if len(both) >= 5:
            from sklearn.metrics import cohen_kappa_score
            a, b = zip(*both)
            kappa = round(float(cohen_kappa_score(a, b)), 3) if len(set(a) | set(b)) > 1 else 1.0
        return {'generated': time.strftime('%Y-%m-%d %H:%M'), 'setup': meta,
                'items': len(items), 'annotators': len(per_annotator), 'labels': len(rows),
                'by_condition': by_cond, 'per_annotator': per_annotator,
                'inter_annotator': {'items_with_two_labels': len(both), 'cohen_kappa': kappa},
                'reference': 'proxy label (keyword rule on the review condition), not clinical truth'}


def setup_from_csv(csv: str, db: str, n_items: int = 60, train_size: int = 5000, seed: int = 11) -> dict:
    from analysis.bert_classifier import build_splits
    from analysis.sud_labels import load_reviews
    from labeling.active_learning import fit_model, make_vectorizer, positive_proba

    df = load_reviews(csv).drop_duplicates('review_text').reset_index(drop=True)
    _, _, test = build_splits(load_reviews(csv))
    pool = df[~df.review_text.isin(set(test.review_text)) & (df.review_text.str.split().str.len() <= 120)]
    pos = pool[pool.is_sud_relevant].sample(n_items // 2, random_state=seed)
    neg = pool[~pool.is_sud_relevant].sample(n_items - n_items // 2, random_state=seed)
    pilot = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    rest = pool[~pool.review_text.isin(set(pilot.review_text))]
    # Balanced training set: the pilot is 50% relevant, and a model trained on the pool's ~6% would
    # under-predict "relevant" there (the same prior shift as in active_learning.py).
    half = min(train_size // 2, int(rest.is_sud_relevant.sum()))
    rest = pd.concat([rest[rest.is_sud_relevant].sample(half, random_state=seed),
                      rest[~rest.is_sud_relevant].sample(half, random_state=seed)])
    vec = make_vectorizer().fit(rest.review_text)
    model = fit_model(vec.transform(rest.review_text), rest.is_sud_relevant.astype(int).values)
    prob = positive_proba(model, vec.transform(pilot.review_text))
    threshold = 0.5
    items = [{'item_id': f'p{u}', 'text': t, 'drug': d, 'proxy': int(y), 'suggestion': int(p >= threshold),
              'confidence': max(p, 1 - p)}
             for u, t, d, y, p in zip(pilot.uniqueID, pilot.review_text, pilot.drugName, pilot.is_sud_relevant, prob)]
    suggestion_acc = float(np.mean([it['suggestion'] == it['proxy'] for it in items]))
    meta = {'items': len(items), 'positive_share': 0.5, 'suggestion_model': 'TF-IDF + logistic regression',
            'suggestion_training_reviews': int(len(rest)), 'suggestion_training_balance': '50/50, disjoint from pilot items', 'suggestion_accuracy_vs_proxy_on_pilot_items': round(suggestion_acc, 3),
            'seed': seed, 'max_words': 120}
    PilotStudy(db).load(items, meta)
    return meta


def demo_setup(db: str) -> dict:
    """A small pilot on the synthetic demo reviews (for tests and CI); two suggestions are deliberately wrong."""
    from labeling.demo_data import demo_rows

    rows, gold = demo_rows(per_template=2)
    items = []
    for k, r in enumerate(r for r in rows if r['id'] not in gold):
        wrong = k in (3, 8)
        items.append({'item_id': r['id'], 'text': r['text'], 'drug': r['drug'], 'proxy': r['label'],
                      'suggestion': 1 - r['label'] if wrong else r['label'], 'confidence': 0.6 if wrong else 0.9})
    meta = {'items': len(items), 'demo': True, 'wrong_suggestions': 2}
    PilotStudy(db).load(items, meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('setup'); s1.add_argument('--csv', required=True); s1.add_argument('--db', required=True)
    s1.add_argument('--items', type=int, default=60)
    s2 = sub.add_parser('serve'); s2.add_argument('--db', required=True); s2.add_argument('--port', type=int, default=8765)
    s2.add_argument('--demo', action='store_true', help='create a synthetic pilot in --db first if it is empty')
    s3 = sub.add_parser('report'); s3.add_argument('--db', required=True); s3.add_argument('--out', default='outputs/labeling_pilot.json')
    args = ap.parse_args()
    if args.cmd == 'setup':
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(setup_from_csv(args.csv, args.db, args.items), indent=1))
    elif args.cmd == 'serve':
        import uvicorn

        if args.demo and not PilotStudy(args.db).db.execute('SELECT COUNT(*) FROM pilot_items').fetchone()[0]:
            demo_setup(args.db)
        from labeling.server import create_pilot_app
        uvicorn.run(create_pilot_app(PilotStudy(args.db)), host='127.0.0.1', port=args.port, log_level='warning')
    else:
        report = PilotStudy(args.db).report()
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
