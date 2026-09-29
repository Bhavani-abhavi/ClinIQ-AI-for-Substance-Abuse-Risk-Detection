"""
HTTP API and UI host for the labeling workbench.

    python -m labeling.server --demo                          # synthetic reviews, no dataset needed
    python -m labeling.server --csv drugsComTest_raw.csv --items 3000 --gold 60

With --csv, items are drawn from the review pool outside the 600-review test set; gold items carry the
proxy label, and every retrain reports average precision on the test set. The React UI in
labeling/ui is served from labeling/ui/dist after `npm run build`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from labeling.demo_data import demo_rows
from labeling.workbench import Config, ConflictError, ValidationError, Workbench

UI_DIST = Path(__file__).resolve().parent / 'ui' / 'dist'


class LabelIn(BaseModel):
    annotator: str
    item_id: str
    label: int | str
    seconds: float | None = None
    revise: bool = False


class AdjudicationIn(BaseModel):
    item_id: str
    label: int
    by: str
    overwrite: bool = False


def create_app(bench: Workbench) -> FastAPI:
    app = FastAPI(title='ClinIQ labeling workbench')

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except ConflictError as e:
            raise HTTPException(status_code=409, detail={'message': str(e), 'current': e.current}) from e

    @app.get('/api/health')
    def health():
        return {'ok': True}

    @app.get('/api/task')
    def task(annotator: str):
        t = guard(bench.next_task, annotator)
        return t if t is not None else Response(status_code=204)

    @app.post('/api/labels')
    def label(body: LabelIn):
        value = body.label if body.label == 'skip' else guard(int, body.label) if str(body.label).isdigit() else body.label
        return guard(bench.submit, body.annotator, body.item_id, value, body.seconds, body.revise)

    @app.get('/api/stats')
    def stats():
        return bench.stats()

    @app.get('/api/conflicts')
    def conflicts():
        return bench.conflicts()

    @app.post('/api/adjudicate')
    def adjudicate(body: AdjudicationIn):
        return guard(bench.adjudicate, body.item_id, body.label, body.by, body.overwrite)

    @app.get('/api/export')
    def export():
        return PlainTextResponse(bench.export_jsonl(), media_type='application/x-ndjson')

    if UI_DIST.exists():
        app.mount('/assets', StaticFiles(directory=UI_DIST / 'assets'), name='assets')

        @app.get('/')
        def index():
            return FileResponse(UI_DIST / 'index.html')

    return app


def bench_from_csv(csv: str, items: int, gold: int, db: str, cfg: Config) -> Workbench:
    from analysis.bert_classifier import build_splits
    from analysis.sud_labels import load_reviews

    df = load_reviews(csv).drop_duplicates('review_text').reset_index(drop=True)
    _, _, test = build_splits(load_reviews(csv))
    pool = df[~df.review_text.isin(set(test.review_text))].sample(items + gold, random_state=cfg.seed)
    rows = [{'id': f'r{u}', 'text': t, 'drug': d, 'label': int(y)}
            for u, t, d, y in zip(pool.uniqueID, pool.review_text, pool.drugName, pool.is_sud_relevant)]
    gold_ids = {r['id'] for r in rows[:gold]}
    bench = Workbench(db, cfg, test_set=(test.review_text.tolist(), test.is_sud_relevant.astype(int).values))
    print(bench.import_items(rows, gold_ids=gold_ids))
    return bench


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument('--csv')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--items', type=int, default=3000)
    ap.add_argument('--gold', type=int, default=60)
    ap.add_argument('--db', default=':memory:')
    ap.add_argument('--gold-every', type=int, default=10)
    ap.add_argument('--retrain-every', type=int, default=25)
    ap.add_argument('--overlap-rate', type=float, default=0.1)
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    cfg = Config(gold_every=args.gold_every, retrain_every=args.retrain_every, overlap_rate=args.overlap_rate)
    if args.csv:
        bench = bench_from_csv(args.csv, args.items, args.gold, args.db, cfg)
    elif args.demo:
        rows, gold_ids = demo_rows()
        bench = Workbench(args.db, cfg)
        print(bench.import_items(rows, gold_ids=gold_ids))
    else:
        ap.error('pass --csv or --demo')
    uvicorn.run(create_app(bench), host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
