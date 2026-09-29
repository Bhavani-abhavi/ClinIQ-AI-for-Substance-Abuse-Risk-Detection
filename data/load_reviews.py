import html
import os
import sys
import json
import re

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── STEP 1: Load ─────────────────────────────────────────────────────────────

SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), 'drugsComTest_raw.csv'),
    './drugsComTest_raw.csv',
    'drugsComTest_raw_csv/drugsComTest_raw.csv',
]

csv_path = None
for p in SEARCH_PATHS:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("ERROR: drugsComTest_raw.csv not found in any expected location.")
    sys.exit(1)

print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ── STEP 2: Clean ─────────────────────────────────────────────────────────────

df['review_text'] = df['review'].apply(html.unescape).str.strip()
df = df[df['review_text'].str.split().str.len() >= 10].copy()

df['review_date'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
df['review_year'] = df['review_date'].dt.year

print(f"After removing <10-word reviews: {len(df)} rows\n")

# ── STEP 3: SUD Classification ────────────────────────────────────────────────

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

def _keyword_in(kw, t):
    # "meth" must be a whole word: as a substring it matched methylphenidate, sulfamethoxazole,
    # dextromethorphan and other unrelated drug names (803 false positives; see README).
    return bool(re.search(r'\bmeth\b', t)) if kw == 'meth' else kw in t


def contains_keyword(text):
    if pd.isna(text):
        return False
    t = text.lower()
    return any(_keyword_in(kw, t) for kw in SUD_KEYWORDS)

null_condition_mask = df['condition'].isna()
null_condition_count = null_condition_mask.sum()

# classify by condition first, fall back to drug_name for nulls
df['is_sud_relevant'] = (
    df['condition'].apply(contains_keyword) |
    df['drugName'].apply(contains_keyword)
)

# track which rows were inferred from drug_name (null condition but SUD via drug)
df['_inferred_from_drug'] = (
    null_condition_mask &
    df['drugName'].apply(contains_keyword) &
    ~df['condition'].apply(contains_keyword)
)
inferred_count = df['_inferred_from_drug'].sum()

# ── STEP 4: Signal Category ───────────────────────────────────────────────────

OPIOID_KW     = {'opioid','opiate','heroin','fentanyl','suboxone','methadone',
                 'buprenorphine','hydrocodone','oxycodone','morphine'}
ALCOHOL_KW    = {'alcohol dependence','alcoholism','alcohol abuse'}
WITHDRAWAL_KW = {'withdrawal'}
SMOKING_KW    = {'smoking','nicotine','varenicline','chantix'}
OTHER_SUD_KW  = {'cocaine','amphetamine','methamphetamine','meth',
                 'benzodiazepine','xanax','valium','naloxone','naltrexone',
                 'naltrexone','tramadol','substance abuse','drug abuse',
                 'drug dependence','drug addiction','addiction treatment'}

def classify_signal(row):
    if not row['is_sud_relevant']:
        return 'non_sud'

    combined = ' '.join([
        str(row.get('condition', '') or ''),
        str(row.get('drugName', '') or ''),
        str(row.get('review_text', '') or ''),
    ]).lower()

    hits = {grp for grp, kws in [
        ('opioid',          OPIOID_KW),
        ('alcohol',         ALCOHOL_KW),
        ('withdrawal',      WITHDRAWAL_KW),
        ('smoking_cessation', SMOKING_KW),
        ('other_sud',       OTHER_SUD_KW),
    ] for kw in kws if _keyword_in(kw, combined)}

    # polysubstance: multiple distinct substance groups present
    substance_groups = hits - {'withdrawal'}
    if len(substance_groups) >= 2:
        return 'polysubstance'

    priority = ['opioid', 'alcohol', 'withdrawal', 'smoking_cessation', 'other_sud']
    for cat in priority:
        if cat in hits:
            return cat

    return 'other_sud'

df['signal_category'] = df.apply(classify_signal, axis=1)

# ── STEP 5: Summary ───────────────────────────────────────────────────────────

sud_df    = df[df['is_sud_relevant']]
non_sud_df = df[~df['is_sud_relevant']]

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total reviews:       {len(df):,}")
print(f"SUD-relevant:        {len(sud_df):,}  ({len(sud_df)/len(df)*100:.1f}%)")
print(f"Non-SUD:             {len(non_sud_df):,}")

print("\n── Signal Category Breakdown ──")
cat_counts = df['signal_category'].value_counts()
for cat, cnt in cat_counts.items():
    print(f"  {cat:<22} {cnt:>6,}")

print(f"\n── Date Range ──")
print(f"  {df['review_date'].min().date()} → {df['review_date'].max().date()}")

print("\n── Year Distribution ──")
year_counts = df['review_year'].value_counts().sort_index()
for yr, cnt in year_counts.items():
    print(f"  {int(yr) if pd.notna(yr) else 'N/A'}: {cnt:,}")

print("\n── Rating Distribution ──")
print(f"  {'Category':<12} {'Mean Rating':>12}  {'Median':>8}")
print(f"  {'SUD':<12} {sud_df['rating'].mean():>12.2f}  {sud_df['rating'].median():>8.1f}")
print(f"  {'Non-SUD':<12} {non_sud_df['rating'].mean():>12.2f}  {non_sud_df['rating'].median():>8.1f}")

print("\n── Top 10 Drugs in SUD Reviews ──")
top_drugs = sud_df['drugName'].value_counts().head(10)
for drug, cnt in top_drugs.items():
    print(f"  {drug:<30} {cnt:>5,}")

print("\n── 3 Sample Distress Reviews (rating ≤ 3, SUD-relevant) ──")
distress = sud_df[sud_df['rating'] <= 3].head(3)
for _, row in distress.iterrows():
    print(f"\n  Drug: {row['drugName']}  |  Condition: {row['condition']}  |  Rating: {row['rating']}")
    print(f"  {row['review_text'][:300]}...")

print(f"\n── Null Conditions ──")
print(f"  Total null conditions:        {null_condition_count:,}")
print(f"  Inferred SUD from drug name:  {inferred_count:,}")
print("=" * 60)

# ── STEP 6: Load to PostgreSQL ────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("\nERROR: DATABASE_URL not set in .env — skipping database load.")
    sys.exit(1)

print("\nConnecting to PostgreSQL...")
conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()

INSERT_SQL = """
INSERT INTO drug_reviews (
    review_id, drug_name, condition, review_text,
    rating, review_date, useful_count,
    is_sud_relevant, signal_category,
    review_year
) VALUES %s
ON CONFLICT (review_id) DO NOTHING;
"""

BATCH_SIZE = 500
records = []
for _, row in df.iterrows():
    records.append((
        int(row['uniqueID']),
        row['drugName'] if pd.notna(row['drugName']) else None,
        row['condition'] if pd.notna(row['condition']) else None,
        row['review_text'],
        int(row['rating']) if pd.notna(row['rating']) else None,
        row['review_date'].date() if pd.notna(row['review_date']) else None,
        int(row['usefulCount']) if pd.notna(row['usefulCount']) else None,
        bool(row['is_sud_relevant']),
        row['signal_category'],
        int(row['review_year']) if pd.notna(row['review_year']) else None,
    ))

total_inserted = 0
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    execute_values(cur, INSERT_SQL, batch)
    total_inserted += len(batch)
    print(f"  Inserted rows {i+1}–{min(i+BATCH_SIZE, len(records)):,} / {len(records):,}")

conn.commit()
cur.close()
conn.close()

print(f"\nDone. {total_inserted:,} rows loaded into drug_reviews.")
