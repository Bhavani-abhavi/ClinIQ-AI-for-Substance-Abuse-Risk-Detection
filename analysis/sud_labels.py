"""Proxy SUD labels and signal categories, as defined in data/load_reviews.py (import-safe copy).

The label is derived from the review's `condition` (falling back to the drug name), so it is a
keyword proxy for substance-use relevance, not a clinician-adjudicated diagnosis.
"""
from __future__ import annotations

import html

import pandas as pd

SUD_KEYWORDS = [
    'opioid', 'opiate', 'heroin', 'fentanyl', 'morphine',
    'hydrocodone', 'oxycodone', 'suboxone', 'methadone',
    'buprenorphine', 'naloxone', 'naltrexone', 'tramadol',
    'alcohol dependence', 'alcoholism', 'alcohol abuse',
    'withdrawal', 'benzodiazepine', 'xanax', 'valium',
    'cocaine', 'amphetamine', 'methamphetamine', 'meth',
    'substance abuse', 'drug abuse', 'drug dependence',
    'drug addiction', 'addiction treatment',
]
GROUPS = [
    ('opioid', {'opioid', 'opiate', 'heroin', 'fentanyl', 'suboxone', 'methadone', 'buprenorphine',
                'hydrocodone', 'oxycodone', 'morphine'}),
    ('alcohol', {'alcohol dependence', 'alcoholism', 'alcohol abuse'}),
    ('withdrawal', {'withdrawal'}),
    ('smoking_cessation', {'smoking', 'nicotine', 'varenicline', 'chantix'}),
    ('other_sud', {'cocaine', 'amphetamine', 'methamphetamine', 'meth', 'benzodiazepine', 'xanax', 'valium',
                   'naloxone', 'naltrexone', 'tramadol', 'substance abuse', 'drug abuse', 'drug dependence',
                   'drug addiction', 'addiction treatment'}),
]


def contains_keyword(text) -> bool:
    return isinstance(text, str) and any(kw in text.lower() for kw in SUD_KEYWORDS)


def signal_category(condition, drug, text) -> str:
    combined = ' '.join(str(x or '') for x in (condition, drug, text)).lower()
    hits = {g for g, kws in GROUPS for kw in kws if kw in combined}
    if len(hits - {'withdrawal'}) >= 2:
        return 'polysubstance'
    for g in ['opioid', 'alcohol', 'withdrawal', 'smoking_cessation', 'other_sud']:
        if g in hits:
            return g
    return 'other_sud'


def load_reviews(csv_path: str) -> pd.DataFrame:
    """Same cleaning as data/load_reviews.py: unescape HTML, keep reviews with 10+ words."""
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    df['review_text'] = df['review'].apply(html.unescape).str.strip()
    df = df[df['review_text'].str.split().str.len() >= 10].copy()
    df['is_sud_relevant'] = df['condition'].apply(contains_keyword) | df['drugName'].apply(contains_keyword)
    df['signal_category'] = [signal_category(c, d, t) if s else 'non_sud' for c, d, t, s in
                             zip(df['condition'], df['drugName'], df['review_text'], df['is_sud_relevant'])]
    return df
