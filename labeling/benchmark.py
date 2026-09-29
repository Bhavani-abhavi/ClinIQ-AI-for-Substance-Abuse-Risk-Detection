"""
Large-queue benchmark for the labeling workbench.

    python -m labeling.benchmark --csv /path/to/drugsComTest_raw.csv --labels 600

Loads every review outside the 600-review test set (about 46,000 items) into the workbench, then plays an
annotator who takes the next task and submits its proxy label, with the model retraining every 25 labels.
Uses a SQLite file, as real sessions do. Reports latency per call (median and 95th percentile) for fetching a task, saving a label, retraining,
quality stats and export. Writes outputs/labeling_benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def pct(values: list[float]) -> dict:
    a = np.array(values) * 1000
    return {'calls': len(values), 'median_ms': round(float(np.median(a)), 2), 'p95_ms': round(float(np.percentile(a, 95)), 2),
            'max_ms': round(float(a.max()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--labels', type=int, default=600)
    ap.add_argument('--out', default='outputs/labeling_benchmark.json')
    ap.add_argument('--db', default=None, help='SQLite file (default: a fresh temporary file, as in real use)')
    args = ap.parse_args()

    from analysis.bert_classifier import build_splits
    from analysis.sud_labels import load_reviews
    from labeling.workbench import Config, Workbench

    df = load_reviews(args.csv).drop_duplicates('review_text').reset_index(drop=True)
    _, _, test = build_splits(load_reviews(args.csv))
    pool = df[~df.review_text.isin(set(test.review_text))]
    rows = [{'id': f'r{u}', 'text': t, 'drug': d, 'label': int(y)}
            for u, t, d, y in zip(pool.uniqueID, pool.review_text, pool.drugName, pool.is_sud_relevant)]
    truth = {r['id']: r['label'] for r in rows}
    gold = {r['id'] for r in rows[:100]}

    import tempfile
    db = args.db or str(Path(tempfile.mkdtemp()) / 'bench.db')
    t0 = time.perf_counter()
    bench = Workbench(db, Config(overlap_rate=0.1),
                      test_set=(test.review_text.tolist(), test.is_sud_relevant.astype(int).values))
    report_import = bench.import_items(rows, gold_ids=gold)
    import_s = time.perf_counter() - t0

    fetch, save, retrain = [], [], []
    for _ in range(args.labels):
        t = time.perf_counter()
        task = bench.next_task('bench')
        fetch.append(time.perf_counter() - t)
        if task is None:
            break
        t = time.perf_counter()
        out = bench.submit('bench', task['item_id'], truth[task['item_id']], 3.0)
        (retrain if 'retrained' in out else save).append(time.perf_counter() - t)

    t = time.perf_counter()
    stats = bench.stats()
    stats_s = time.perf_counter() - t
    t = time.perf_counter()
    exported = len(bench.export())
    export_s = time.perf_counter() - t

    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'machine': f'{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}',
        'storage': 'SQLite file' if db != ':memory:' else 'SQLite in memory',
        'items': report_import, 'import_and_first_fit_s': round(import_s, 2),
        'annotator_labels': len(save) + len(retrain),
        'next_task': pct(fetch), 'submit': pct(save), 'submit_with_retrain_every_25': pct(retrain),
        'stats_ms': round(stats_s * 1000, 1), 'export_ms': round(export_s * 1000, 1), 'exported_labels': exported,
        'model_test_ap_after_last_retrain': stats['model_runs'][-1]['test_ap'] if stats['model_runs'] else None,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
