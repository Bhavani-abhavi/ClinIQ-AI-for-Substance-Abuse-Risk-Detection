"""
Active learning for SUD-relevance labels: how many labels does uncertainty sampling save over random?

    python -m labeling.active_learning --csv /path/to/drugsComTest_raw.csv

Simulation: the proxy label stands in for the annotator. Every strategy starts from the same seed
labels, asks for a batch of reviews, "receives" their proxy labels, retrains a TF-IDF + logistic
regression model on review text only, and is scored on the fixed 600-review test set used by the
DistilBERT benchmark (analysis/bert_classifier.py). Test texts never enter the pool. Repeated over
several seeds; results go to outputs/active_learning.json.

The headline metric is average precision (area under the precision-recall curve), which does not
depend on a decision threshold. The pool is about 6% positive while the test set is 50% positive, so
small models trained on the pool put almost every test review below 0.5 and F1 at a fixed 0.5
threshold would mostly measure that prior shift. The second metric is how many truly relevant
reviews each strategy puts in front of annotators, which is what saves labeling time on a rare class.

The same learner ranks the labeling workbench's queue (labeling/server.py).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

STRATEGIES = ('random', 'uncertainty')


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)


def fit_model(X, y) -> LogisticRegression | None:
    """Balanced logistic regression; None until both classes have been labeled."""
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=4.0, class_weight='balanced', max_iter=2000).fit(X, y)


def positive_proba(model, X) -> np.ndarray:
    if model is None:
        return np.full(X.shape[0], 0.5)
    return model.predict_proba(X)[:, 1]


def uncertainty_order(prob: np.ndarray) -> np.ndarray:
    """Indices from most to least uncertain (closest to 0.5 first); ties broken by index."""
    return np.lexsort((np.arange(len(prob)), np.abs(prob - 0.5)))


def pick_batch(strategy: str, prob: np.ndarray, unlabeled: np.ndarray, k: int, rng) -> np.ndarray:
    if strategy == 'random':
        return rng.choice(unlabeled, size=min(k, len(unlabeled)), replace=False)
    if strategy == 'uncertainty':
        return unlabeled[uncertainty_order(prob[unlabeled])[:k]]
    raise ValueError(f'unknown strategy {strategy!r}')


@dataclass
class Curve:
    labels: list[int] = field(default_factory=list)
    ap: list[float] = field(default_factory=list)
    auroc: list[float] = field(default_factory=list)
    positives_found: list[int] = field(default_factory=list)


def simulate(X_pool, y_pool, X_test, y_test, strategy: str, seed: int, seed_size: int, batch: int,
             budget: int) -> Curve:
    rng = np.random.default_rng(seed)
    labeled = rng.choice(len(y_pool), size=seed_size, replace=False)
    mask = np.zeros(len(y_pool), bool)
    mask[labeled] = True
    curve = Curve()
    while True:
        idx = np.flatnonzero(mask)
        model = fit_model(X_pool[idx], y_pool[idx])
        score = positive_proba(model, X_test)
        curve.labels.append(int(mask.sum()))
        curve.ap.append(round(float(average_precision_score(y_test, score)), 4))
        curve.auroc.append(round(float(roc_auc_score(y_test, score)), 4))
        curve.positives_found.append(int(y_pool[idx].sum()))
        if mask.sum() >= budget:
            return curve
        unlabeled = np.flatnonzero(~mask)
        prob = positive_proba(model, X_pool) if strategy != 'random' else None
        mask[pick_batch(strategy, prob, unlabeled, batch, rng)] = True


def random_sizes_to_reach(X_pool, y_pool, X_test, y_test, target: float, seeds: int,
                          sizes=(3000, 4000, 6000, 8000, 9000, 10000, 11000, 12000, 16000, 24000, 32000)) -> dict:
    """Mean test AP of models trained on random subsets of growing size (beyond the simulation budget)."""
    out = {}
    for n in sizes:
        if n > len(y_pool):
            break
        aps = []
        for seed in range(seeds):
            idx = np.random.default_rng(seed).choice(len(y_pool), size=n, replace=False)
            aps.append(average_precision_score(y_test, positive_proba(fit_model(X_pool[idx], y_pool[idx]), X_test)))
        out[n] = round(float(np.mean(aps)), 4)
        if out[n] >= target:
            break
    return out


def labels_to_reach(curve_labels: list[int], curve_metric: list[float], target: float) -> int | None:
    for n, f in zip(curve_labels, curve_metric):
        if f >= target:
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument('--seed-size', type=int, default=100)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--budget', type=int, default=3000)
    ap.add_argument('--out', default='outputs/active_learning.json')
    args = ap.parse_args()

    from analysis.bert_classifier import build_splits
    from analysis.sud_labels import load_reviews

    t0 = time.time()
    df = load_reviews(args.csv).drop_duplicates('review_text').reset_index(drop=True)
    _, _, test = build_splits(load_reviews(args.csv))
    pool = df[~df.review_text.isin(set(test.review_text))].reset_index(drop=True)
    vec = make_vectorizer().fit(pd.concat([pool.review_text, test.review_text]))
    X_pool, X_test = vec.transform(pool.review_text), vec.transform(test.review_text)
    y_pool = pool.is_sud_relevant.astype(int).values
    y_test = test.is_sud_relevant.astype(int).values

    full = fit_model(X_pool, y_pool)
    full_score = positive_proba(full, X_test)
    full_ap = round(float(average_precision_score(y_test, full_score)), 4)
    full_f1_at_05 = round(float(f1_score(y_test, full_score >= 0.5)), 4)

    runs = {s: [simulate(X_pool, y_pool, X_test, y_test, s, seed, args.seed_size, args.batch, args.budget)
                for seed in range(args.seeds)] for s in STRATEGIES}
    summary = {}
    for s, curves in runs.items():
        ap = np.array([c.ap for c in curves])
        found = np.array([c.positives_found for c in curves])
        summary[s] = {'labels': curves[0].labels, 'ap_mean': ap.mean(0).round(4).tolist(),
                      'ap_std': ap.std(0).round(4).tolist(),
                      'auroc_mean': np.array([c.auroc for c in curves]).mean(0).round(4).tolist(),
                      'positives_found_mean': found.mean(0).round(1).tolist()}
    targets = {}
    for frac in (0.95, 0.98):
        target = round(frac * full_ap, 4)
        targets[f'{int(frac * 100)}pct_of_full_pool_ap'] = {
            'target_ap': target,
            **{s: labels_to_reach(summary[s]['labels'], summary[s]['ap_mean'], target) for s in STRATEGIES}}
    target95 = targets['95pct_of_full_pool_ap']['target_ap']
    random_extended = random_sizes_to_reach(X_pool, y_pool, X_test, y_test, target95, args.seeds)
    targets['95pct_of_full_pool_ap']['random_extended'] = {
        'ap_by_labels': random_extended,
        'labels': next((n for n, a in random_extended.items() if a >= target95), None)}
    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'task': 'SUD-relevant drug review (keyword proxy label), review text only',
        'model': 'TF-IDF (1-2 grams) + balanced logistic regression',
        'pool': {'reviews': int(len(pool)), 'positives': int(y_pool.sum())},
        'test': {'reviews': int(len(test)), 'positives': int(y_test.sum()),
                 'note': 'same 600-review set as analysis/bert_classifier.py'},
        'setup': {'seeds': args.seeds, 'seed_labels': args.seed_size, 'batch': args.batch, 'budget': args.budget},
        'full_pool_supervised': {'labels': int(len(y_pool)), 'ap': full_ap, 'f1_at_0.5': full_f1_at_05},
        'labels_to_reach': targets,
        'curves': summary,
        'label_source': 'proxy labels stand in for annotators; no human labeling time is measured',
        'runtime_s': round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('pool', 'full_pool_supervised', 'labels_to_reach')}, indent=1))
    for s in STRATEGIES:
        keep = (200, 500, 1000, 2000, 3000)
        print(s, 'AP', {n: f for n, f in zip(summary[s]['labels'], summary[s]['ap_mean']) if n in keep},
              'positives', {n: f for n, f in zip(summary[s]['labels'], summary[s]['positives_found_mean']) if n in keep})


if __name__ == '__main__':
    main()
