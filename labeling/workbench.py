"""
Labeling workbench core: ML-assisted queue, annotator quality checks, adjudication and export.

Storage is SQLite. The queue is ranked by the active learner in labeling/active_learning.py (most
uncertain first); the current model's guess is shown as a pre-label. Quality control:

- gold items with known labels are mixed in (one task in `gold_every`) to score each annotator;
- a share of items (`overlap_rate`) goes to a second annotator, giving Cohen's kappa;
- disagreements wait in a conflict queue until someone adjudicates them;
- labels faster than `fast_seconds` are flagged.

Cold start: until the model has seen both classes, tasks alternate between the review with the most
substance-use terms and the next review in random order, so the rare positive class appears early.

Only resolved labels (adjudicated, agreed, or single) train the model and leave in the export.
"""
from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score

from labeling.active_learning import fit_model, make_vectorizer, positive_proba, uncertainty_order

LABELS = {0: 'not_relevant', 1: 'sud_relevant'}
HIGHLIGHT_TERMS = sorted({
    'opioid', 'opiate', 'heroin', 'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'suboxone', 'methadone',
    'buprenorphine', 'naltrexone', 'naloxone', 'tramadol', 'withdrawal', 'withdrawals', 'detox', 'addiction',
    'addicted', 'addict', 'relapse', 'relapsed', 'sober', 'sobriety', 'rehab', 'recovery', 'alcohol', 'drinking',
    'cravings', 'craving', 'dependence', 'dependent', 'cocaine', 'meth', 'xanax', 'benzo', 'clean'}, key=len, reverse=True)
_TERM_RE = re.compile(r'\b(' + '|'.join(map(re.escape, HIGHLIGHT_TERMS)) + r')\b', re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, text TEXT NOT NULL, drug TEXT, gold INTEGER,
                                  overlap INTEGER NOT NULL DEFAULT 0, pos INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, annotator TEXT NOT NULL,
                                   label INTEGER NOT NULL, seconds REAL, created REAL NOT NULL,
                                   UNIQUE(item_id, annotator));
CREATE TABLE IF NOT EXISTS skips (item_id TEXT NOT NULL, annotator TEXT NOT NULL, PRIMARY KEY(item_id, annotator));
CREATE TABLE IF NOT EXISTS label_events (id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL,
                                         annotator TEXT NOT NULL, label INTEGER NOT NULL, action TEXT NOT NULL,
                                         at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS adjudications (item_id TEXT PRIMARY KEY, label INTEGER NOT NULL, by TEXT NOT NULL,
                                          created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS model_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, labels INTEGER NOT NULL,
                                       positives INTEGER NOT NULL, test_ap REAL, created REAL NOT NULL);
"""


def _locked(method):
    """Every read and write shares one SQLite connection across server threads; hold the lock for all of them."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class ValidationError(ValueError):
    pass


class ConflictError(ValueError):
    """The request disagrees with what is already stored (HTTP 409); `current` describes the stored state."""
    def __init__(self, message: str, current: dict):
        super().__init__(message)
        self.current = current


def highlight_spans(text: str) -> list[list[int]]:
    """[start, end] character spans of substance-use terms, for the annotator's eye (not a label)."""
    return [[m.start(), m.end()] for m in _TERM_RE.finditer(text)]


def clean_text(raw) -> str:
    return re.sub(r'\s+', ' ', html.unescape(str(raw or ''))).strip().strip('"').strip()


@dataclass
class Config:
    gold_every: int = 10
    overlap_rate: float = 0.1
    retrain_every: int = 25
    fast_seconds: float = 2.0
    min_words: int = 10
    seed: int = 7


class Workbench:
    def __init__(self, db_path: str = ':memory:', config: Config | None = None, test_set=None):
        self.cfg = config or Config()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.test_set = test_set          # optional (texts, labels) held out for model AP
        self.vectorizer = None
        self.model = None
        self._X = None
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._prob = None
        self._terms = None
        self._since_retrain = 0

    # ── import ────────────────────────────────────────────────────────────
    @_locked
    def import_items(self, rows, gold_ids=()) -> dict:
        """Validate and load items. rows: dicts with id, text, optional drug and label (used only for gold)."""
        gold_ids = set(gold_ids)
        seen = {r[0] for r in self.db.execute('SELECT lower(text) FROM items')}
        report = {'accepted': 0, 'duplicates': 0, 'too_short': 0, 'invalid': 0, 'gold': 0}
        rng = np.random.default_rng(self.cfg.seed)
        with self.lock, self.db:
            pos = self.db.execute('SELECT COALESCE(MAX(pos), -1) + 1 FROM items').fetchone()[0]
            for r in rows:
                item_id, text = str(r.get('id') or '').strip(), clean_text(r.get('text'))
                if not item_id or not text:
                    report['invalid'] += 1
                    continue
                if len(text.split()) < self.cfg.min_words:
                    report['too_short'] += 1
                    continue
                if text.lower() in seen:
                    report['duplicates'] += 1
                    continue
                gold = None
                if item_id in gold_ids:
                    if r.get('label') not in (0, 1):
                        report['invalid'] += 1
                        continue
                    gold = int(r['label'])
                    report['gold'] += 1
                overlap = int(gold is None and rng.random() < self.cfg.overlap_rate)
                self.db.execute('INSERT INTO items VALUES (?,?,?,?,?,?)',
                                (item_id, text, clean_text(r.get('drug')) or None, gold, overlap, pos))
                seen.add(text.lower())
                pos += 1
                report['accepted'] += 1
        self._fit_vectorizer()
        self.retrain()
        return report

    def _fit_vectorizer(self):
        rows = self.db.execute('SELECT id, text FROM items ORDER BY pos').fetchall()
        self._ids = [r[0] for r in rows]
        self._index = {i: k for k, i in enumerate(self._ids)}
        vec = make_vectorizer()
        vec.min_df = 1 if len(rows) < 200 else 2
        self.vectorizer = vec.fit([r[1] for r in rows])
        self._X = self.vectorizer.transform([r[1] for r in rows])
        self._terms = np.array([len(highlight_spans(r[1])) for r in rows])

    # ── labels and resolution ─────────────────────────────────────────────
    @_locked
    def _labels_by_item(self) -> dict[str, list[tuple[str, int]]]:
        out = defaultdict(list)
        for item_id, annotator, label in self.db.execute(
                'SELECT l.item_id, l.annotator, l.label FROM labels l JOIN items i ON i.id = l.item_id ORDER BY l.id'):
            out[item_id].append((annotator, label))
        return out

    @_locked
    def resolved(self, include_gold: bool = False) -> dict[str, int]:
        """item -> final label: adjudicated, else unanimous. Conflicts stay out until adjudicated."""
        adj = dict(self.db.execute('SELECT item_id, label FROM adjudications'))
        gold = {i for (i,) in self.db.execute('SELECT id FROM items WHERE gold IS NOT NULL')}
        out = {}
        for item_id, votes in self._labels_by_item().items():
            if item_id in gold and not include_gold:
                continue
            if item_id in adj:
                out[item_id] = adj[item_id]
            elif len({v for _, v in votes}) == 1:
                out[item_id] = votes[0][1]
        for item_id, label in adj.items():
            out.setdefault(item_id, label)
        return out

    @_locked
    def conflicts(self) -> list[dict]:
        adj = {i for (i,) in self.db.execute('SELECT item_id FROM adjudications')}
        texts = dict(self.db.execute('SELECT id, text FROM items'))
        return [{'item_id': i, 'text': texts[i], 'votes': [{'annotator': a, 'label': v} for a, v in votes]}
                for i, votes in self._labels_by_item().items()
                if len({v for _, v in votes}) > 1 and i not in adj]

    def adjudicate(self, item_id: str, label: int, by: str, overwrite: bool = False) -> dict:
        """Record a reviewer's decision. Repeating the same decision is a no-op; a different decision on an item
        someone already adjudicated is a conflict unless `overwrite` is set."""
        if label not in LABELS or not by.strip():
            raise ValidationError('label must be 0 or 1 and a reviewer name is required')
        with self.lock:
            if not self.db.execute('SELECT 1 FROM items WHERE id = ?', (item_id,)).fetchone():
                raise ValidationError(f'unknown item {item_id}')
            prior = self.db.execute('SELECT label, by FROM adjudications WHERE item_id = ?', (item_id,)).fetchone()
            if prior and prior[0] == label:
                return {'saved': True, 'duplicate': True}
            if prior and not overwrite:
                raise ConflictError(f'already resolved as {LABELS[prior[0]]} by {prior[1]}',
                                    {'item_id': item_id, 'label': prior[0], 'by': prior[1]})
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO adjudications VALUES (?,?,?,?)', (item_id, label, by.strip(), time.time()))
        self.retrain()
        return {'saved': True}

    # ── model ─────────────────────────────────────────────────────────────
    def retrain(self) -> dict | None:
        with self.lock:
            labels = self.resolved()
            ids = [i for i in labels if i in self._index]
            y = np.array([labels[i] for i in ids])
            self.model = fit_model(self._X[[self._index[i] for i in ids]], y) if len(ids) else None
            self._prob = positive_proba(self.model, self._X) if self._X is not None else None
            self._since_retrain = 0
            if self.model is None:
                return None
            test_ap = None
            if self.test_set is not None:
                texts, y_test = self.test_set
                test_ap = round(float(average_precision_score(
                    y_test, positive_proba(self.model, self.vectorizer.transform(texts)))), 4)
            with self.db:
                self.db.execute('INSERT INTO model_runs (labels, positives, test_ap, created) VALUES (?,?,?,?)',
                                (len(ids), int(y.sum()), test_ap, time.time()))
            return {'labels': len(ids), 'test_ap': test_ap}

    # ── tasks ─────────────────────────────────────────────────────────────
    def _task(self, item_id: str) -> dict:
        text, drug = self.db.execute('SELECT text, drug FROM items WHERE id = ?', (item_id,)).fetchone()
        p = float(self._prob[self._index[item_id]]) if self.model is not None else None
        suggestion = None if p is None else {'label': int(p >= 0.5), 'confidence': round(max(p, 1 - p), 3)}
        return {'item_id': item_id, 'text': text, 'drug': drug, 'suggestion': suggestion,
                'highlights': highlight_spans(text)}

    def next_task(self, annotator: str) -> dict | None:
        annotator = annotator.strip()
        if not annotator:
            raise ValidationError('annotator is required')
        with self.lock:
            done = {i for (i,) in self.db.execute('SELECT item_id FROM labels WHERE annotator = ?', (annotator,))}
            done |= {i for (i,) in self.db.execute('SELECT item_id FROM skips WHERE annotator = ?', (annotator,))}
            count = self.db.execute('SELECT COUNT(*) FROM labels WHERE annotator = ?', (annotator,)).fetchone()[0]
            if (count + 1) % self.cfg.gold_every == 0:
                for (i,) in self.db.execute('SELECT id FROM items WHERE gold IS NOT NULL ORDER BY pos'):
                    if i not in done:
                        return self._task(i)
            labeled_by = self._labels_by_item()
            gold = {i for (i,) in self.db.execute('SELECT id FROM items WHERE gold IS NOT NULL')}
            for (i,) in self.db.execute(
                    'SELECT l.item_id FROM labels l JOIN items i ON i.id = l.item_id '
                    'WHERE i.overlap = 1 AND i.gold IS NULL GROUP BY l.item_id HAVING COUNT(*) = 1 ORDER BY MIN(l.id)'):
                if i not in done:                     # second opinions, in the order items were first labeled
                    return self._task(i)
            candidates = [k for k, i in enumerate(self._ids)
                          if i not in done and i not in labeled_by and i not in gold]
            if not candidates:
                return None
            cand = np.array(candidates)
            if self.model is None:                    # cold start: alternate keyword-rich and random-order items
                if count % 2 == 0 and self._terms[cand].max() > 0:
                    return self._task(self._ids[cand[np.argmax(self._terms[cand])]])
                return self._task(self._ids[cand[0]])
            return self._task(self._ids[cand[uncertainty_order(self._prob[cand])[0]]])

    def submit(self, annotator: str, item_id: str, label, seconds: float | None = None, revise: bool = False) -> dict:
        """Save a label. Retrying the same label is idempotent (a network retry must not double-count); a different
        label for an item this annotator already labeled is a conflict unless `revise` is set, and revisions are
        kept in label_events."""
        annotator = (annotator or '').strip()
        if not annotator:
            raise ValidationError('annotator is required')
        if label == 'skip':
            with self.lock, self.db:
                self.db.execute('INSERT OR IGNORE INTO skips VALUES (?,?)', (item_id, annotator))
            return {'skipped': True}
        if label not in LABELS:
            raise ValidationError('label must be 0, 1 or "skip"')
        with self.lock:
            row = self.db.execute('SELECT gold FROM items WHERE id = ?', (item_id,)).fetchone()
            if row is None:
                raise ValidationError(f'unknown item {item_id}')
            prior = self.db.execute('SELECT label FROM labels WHERE item_id = ? AND annotator = ?',
                                    (item_id, annotator)).fetchone()
            if prior and prior[0] == int(label):
                return {'saved': True, 'duplicate': True}
            if prior and not revise:
                raise ConflictError(f'already labeled {LABELS[prior[0]]} by {annotator}',
                                    {'item_id': item_id, 'annotator': annotator, 'label': prior[0]})
            now = time.time()
            with self.db:
                if prior:
                    self.db.execute('UPDATE labels SET label = ?, created = ? WHERE item_id = ? AND annotator = ?',
                                    (int(label), now, item_id, annotator))
                else:
                    self.db.execute('INSERT INTO labels (item_id, annotator, label, seconds, created) VALUES (?,?,?,?,?)',
                                    (item_id, annotator, int(label), seconds, now))
                self.db.execute('INSERT INTO label_events (item_id, annotator, label, action, at) VALUES (?,?,?,?,?)',
                                (item_id, annotator, int(label), 'revise' if prior else 'label', now))
            if prior:
                self.retrain()
                return {'saved': True, 'revised': True}
            out = {'saved': True}
            if row[0] is None:
                self._since_retrain += 1
                if self._since_retrain >= self.cfg.retrain_every:
                    out['retrained'] = self.retrain()
            return out

    # ── quality ───────────────────────────────────────────────────────────
    @_locked
    def stats(self) -> dict:
        gold = dict(self.db.execute('SELECT id, gold FROM items WHERE gold IS NOT NULL'))
        per = defaultdict(lambda: {'labels': 0, 'gold_seen': 0, 'gold_correct': 0, 'seconds': [], 'fast': 0})
        for item_id, annotator, label, seconds in self.db.execute(
                'SELECT item_id, annotator, label, seconds FROM labels ORDER BY id'):
            a = per[annotator]
            a['labels'] += 1
            if seconds is not None:
                a['seconds'].append(seconds)
                a['fast'] += int(seconds < self.cfg.fast_seconds)
            if item_id in gold:
                a['gold_seen'] += 1
                a['gold_correct'] += int(label == gold[item_id])
        annotators = []
        for name, a in sorted(per.items()):
            acc = round(a['gold_correct'] / a['gold_seen'], 3) if a['gold_seen'] else None
            annotators.append({'annotator': name, 'labels': a['labels'], 'gold_seen': a['gold_seen'],
                               'gold_accuracy': acc,
                               'median_seconds': round(median(a['seconds']), 1) if a['seconds'] else None,
                               'fast_labels': a['fast'],
                               'flags': [f for f, bad in (('low_gold_accuracy', acc is not None and a['gold_seen'] >= 3
                                                           and acc < 0.8),
                                                          ('too_fast', a['fast'] > 0.2 * a['labels'])) if bad]})
        pairs = [(votes[0][1], votes[1][1]) for i, votes in self._labels_by_item().items()
                 if len(votes) >= 2 and i not in gold]
        kappa = None
        if len(pairs) >= 5:
            a, b = zip(*pairs)
            kappa = round(float(cohen_kappa_score(a, b)), 3) if len(set(a) | set(b)) > 1 else 1.0
        resolved = self.resolved()
        total = self.db.execute('SELECT COUNT(*) FROM items WHERE gold IS NULL').fetchone()[0]
        runs = [{'labels': n, 'positives': p, 'test_ap': ap} for n, p, ap in
                self.db.execute('SELECT labels, positives, test_ap FROM model_runs ORDER BY id')]
        return {'items': total, 'resolved': len(resolved),
                'label_counts': {LABELS[k]: sum(1 for v in resolved.values() if v == k) for k in LABELS},
                'annotators': annotators, 'agreement': {'double_labeled': len(pairs), 'cohen_kappa': kappa},
                'conflicts': len(self.conflicts()), 'model_runs': runs}

    @_locked
    def export(self) -> list[dict]:
        texts = dict(self.db.execute('SELECT id, text FROM items'))
        adj = {i for (i,) in self.db.execute('SELECT item_id FROM adjudications')}
        votes = self._labels_by_item()
        return [{'id': i, 'text': texts[i], 'label': v, 'label_name': LABELS[v], 'annotations': len(votes.get(i, [])),
                 'adjudicated': i in adj} for i, v in sorted(self.resolved().items())]

    @_locked
    def export_jsonl(self) -> str:
        return ''.join(json.dumps(r) + '\n' for r in self.export())
