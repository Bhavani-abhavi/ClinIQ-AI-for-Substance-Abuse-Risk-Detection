"""Recompute the 2008–2017 trend numbers from the raw CSV with the corrected proxy label (whole-word "meth").

    python -m analysis.temporal_check --csv /path/to/drugsComTest_raw.csv

The database pipeline (analysis/task2_temporal_behavioral.py) produced outputs/temporal_trends.csv before the
label fix; this script needs no database and writes outputs/temporal_corrected.json.
"""
import argparse
import html
import json

import pandas as pd

from analysis.sud_labels import contains_keyword, signal_category


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='outputs/temporal_corrected.json')
    args = ap.parse_args()
    df = pd.read_csv(args.csv, encoding='utf-8', encoding_errors='replace')
    df['review_text'] = df['review'].apply(html.unescape).str.strip()
    df = df[df['review_text'].str.split().str.len() >= 10].copy()
    df['year'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce').dt.year
    sud = df[(df['condition'].apply(contains_keyword) | df['drugName'].apply(contains_keyword))
             & df['year'].between(2008, 2017)].copy()
    sud['category'] = [signal_category(c, d, t) for c, d, t in zip(sud['condition'], sud['drugName'], sud['review_text'])]
    by_year = sud.groupby('year').agg(sud_reviews=('rating', 'size'),
                                      distress_share=('rating', lambda r: round(float((r <= 3).mean()), 4)),
                                      opioid_share=('category', lambda c: round(float((c == 'opioid').mean()), 4)))
    out = {'label': 'corrected proxy (whole-word "meth")', 'sud_reviews': int(len(sud)),
           'distress_definition': 'rating <= 3 of 10',
           'by_year': {int(y): {k: (int(v) if k == 'sud_reviews' else float(v)) for k, v in row.items()}
                       for y, row in by_year.iterrows()}}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
