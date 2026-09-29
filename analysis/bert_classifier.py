"""
Fine-tune a BERT-family classifier (DistilBERT) for SUD-relevant drug reviews, tracked in Weights & Biases.

    python -m analysis.bert_classifier --csv /path/to/drugsComTest_raw.csv
    WANDB_MODE=online python -m analysis.bert_classifier ...   # sync to a W&B account (default: offline)

Labels are the project's keyword proxy (the review's `condition`, falling back to the drug name), not
clinician adjudication. By default the model sees the drug name and review text, the same inputs as the
rule baseline; the `condition` field that defines the label is never an input. The drug name can itself
set the label (for example Suboxone), so `--text-only` trains on the review text alone, and the rules are
also scored on review text alone, to show how much comes from reading the review.

Evaluation set, rebuilt with the original recipe: 300 SUD-relevant reviews stratified across signal
categories by usefulness (60 per category) plus 300 non-SUD reviews drawn with a fixed seed. Review texts
in the evaluation set are removed from training (the dataset repeats reviews across brand and generic
names). The rule baseline is re-run on the same set with its built-in keyword dictionary; the original
also added ICD-10 terms from the database, which is not available offline.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from analysis.detection import rule_classify
from analysis.sud_labels import load_reviews

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
KEYWORDS = {  # patient-voice dictionary from analysis/task1_signal_detection.py (without ICD-10 augmentation)
    'opioid': ['opioid', 'opiate', 'heroin', 'fentanyl', 'morphine', 'hydrocodone', 'oxycodone', 'oxycontin',
               'vicodin', 'suboxone', 'buprenorphine', 'methadone', 'naloxone', 'naltrexone', 'tramadol',
               'codeine', 'dilaudid', 'opioid addiction', 'pain pill', 'pain pills', 'pill mill',
               'medication-assisted', 'mat program', 'on suboxone', 'on methadone', 'need it to feel normal',
               'cant function without', "can't function without"],
    'alcohol': ['alcohol dependence', 'alcoholism', 'alcohol abuse', 'alcohol use disorder', 'alcohol withdrawal',
                'drinking problem', 'alcoholic', 'cant stop drinking', "can't stop drinking", 'sober', 'sobriety',
                'aa meeting', 'alcoholics anonymous', 'fell off the wagon'],
    'withdrawal': ['withdrawal', 'detoxification', 'detox', 'withdrawal hell', 'withdrawal symptoms', 'coming off',
                   'getting off', 'kicking', 'cold turkey', 'withdrawals', 'rebound', 'stopping cold'],
    'stimulant': ['cocaine', 'amphetamine', 'methamphetamine', 'stimulant dependence', 'stimulant use disorder',
                  'meth', 'crystal', 'coke', 'crack', 'speed', 'stimulant addiction'],
    'other_sud': ['substance abuse', 'drug abuse', 'drug dependence', 'drug addiction', 'addiction treatment',
                  'polysubstance', 'benzodiazepine dependence', 'sedative dependence', 'xanax dependence',
                  'valium dependence', 'addict', 'addiction', 'addicted', 'in recovery', 'relapse', 'relapsed',
                  'using again', 'fell off', 'rock bottom', 'rehab', 'treatment center', 'need it to function',
                  'dependent on', 'have to have it', 'cant live without', "can't live without", 'getting clean',
                  'staying clean', 'clean and sober', 'drug seeking', 'tolerance built up', 'need higher dose',
                  'taking more than prescribed'],
}


def build_splits(df: pd.DataFrame, per_category: int = 60, neg_ratio: float = 2.0):
    df = df.drop_duplicates('review_text').reset_index(drop=True)
    sud = df[df.is_sud_relevant & (df.review_text.str.len() > 50)]
    pos_eval = (sud.sort_values(['usefulCount', 'uniqueID'], ascending=[False, True])
                .groupby('signal_category', group_keys=False).head(per_category))
    neg_eval = df[~df.is_sud_relevant & (df.review_text.str.len() > 50)].sample(300, random_state=SEED)
    eval_df = pd.concat([pos_eval, neg_eval]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    rest = df[~df.review_text.isin(set(eval_df.review_text))]
    pos = rest[rest.is_sud_relevant]
    neg = rest[~rest.is_sud_relevant].sample(int(len(pos) * neg_ratio), random_state=SEED)
    train = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val = int(len(train) * 0.1)
    return train.iloc[n_val:], train.iloc[:n_val], eval_df


TEXT_ONLY = False

# Counterfactual augmentation (--augment-identity). Training identities are deliberately disjoint from the
# ones analysis/bert_bias.py tests, so the bias test measures generalization, not memorized phrases.
TRAIN_IDENTITIES = ['Black man', 'white woman', 'Latina woman', 'Asian man', 'Indigenous woman', '19-year-old',
                    '45-year-old', '65-year-old', 'college student', 'retired nurse', 'lesbian', 'bisexual woman',
                    'transgender man', 'Christian man', 'Jewish woman', 'Hindu man', 'wheelchair user',
                    'father of three', 'grandmother', 'immigrant', 'person', 'patient']


def augment_identity(frame: pd.DataFrame, rate: float = 0.5, seed: int = SEED) -> pd.DataFrame:
    """Rewrite a share of reviews in place: an identity statement in front and, half the time, swapped gender words.
    Labels are unchanged, so the model learns that who is writing does not decide relevance."""
    from analysis.bert_bias import swap_gender, with_identity
    rng = np.random.default_rng(seed)
    out = frame.copy()
    texts = []
    for t in out.review_text:
        if rng.random() < rate:
            t = with_identity(swap_gender(t) if rng.random() < 0.5 else t,
                              TRAIN_IDENTITIES[rng.integers(len(TRAIN_IDENTITIES))])
        texts.append(t)
    out['review_text'] = texts
    return out


def text_of(row) -> str:
    return row.review_text if TEXT_ONLY else f"drug: {row.drugName}. review: {row.review_text}"


def rule_baselines(eval_df: pd.DataFrame) -> dict:
    """Keyword rules on the same set, with the drug name (as in the original benchmark) and on review text only."""
    y = eval_df.is_sud_relevant.astype(int).values
    with_drug = [rule_classify(t, d, KEYWORDS)[0] for t, d in zip(eval_df.review_text, eval_df.drugName)]
    text_only = [rule_classify(t, '', KEYWORDS)[0] for t in eval_df.review_text]
    return {'rule_based_static_keywords': metrics(y, with_drug), 'rule_based_text_only': metrics(y, text_only)}


def metrics(y, pred, score=None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {'precision': round(float(p), 3), 'recall': round(float(r), 3), 'f1': round(float(f), 3),
           'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
    if score is not None:
        out['auroc'] = round(float(roc_auc_score(y, score)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', default='distilbert-base-uncased')
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--out', default='outputs/bert_eval.json')
    ap.add_argument('--text-only', action='store_true', help='review text only, without the drug name')
    ap.add_argument('--augment-identity', action='store_true',
                    help='counterfactual augmentation: rewrite half of the training reviews with identity statements')
    ap.add_argument('--rules-only', action='store_true',
                    help='recompute the rule baselines on the same evaluation set and merge them into --out')
    args = ap.parse_args()
    if args.rules_only:
        _, _, eval_df = build_splits(load_reviews(args.csv))
        report = json.loads(Path(args.out).read_text())
        report['results_same_600'].update(rule_baselines(eval_df))
        # reports written before --text-only existed trained on drug name + review text
        report.setdefault('inputs', 'review text only' if report['config'].get('text_only') else 'drug name + review text')
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(json.dumps(report['results_same_600'], indent=1))
        return
    global TEXT_ONLY
    TEXT_ONLY = args.text_only

    import torch  # heavy imports stay inside main so the split logic is testable without them

    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_PROJECT', 'cliniq-sud-detection')
    os.environ.setdefault('WANDB_DIR', str(ROOT / 'outputs'))
    import wandb
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    train, val, eval_df = build_splits(load_reviews(args.csv))
    if args.augment_identity:
        train = augment_identity(train)
    y_eval = eval_df.is_sud_relevant.astype(int).values

    rules = rule_baselines(eval_df)

    tok = AutoTokenizer.from_pretrained(args.model)
    def enc(frame):
        ds = Dataset.from_dict({'text': [text_of(r) for r in frame.itertuples()],
                                'label': frame.is_sud_relevant.astype(int).tolist()})
        return ds.map(lambda b: tok(b['text'], truncation=True, max_length=args.max_len), batched=True,
                      remove_columns=['text'])
    ds_train, ds_val, ds_eval = enc(train), enc(val), enc(eval_df)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def compute(p):
        logits, labels = p
        prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
        m = metrics(labels, (prob >= 0.5).astype(int), prob)
        return {k: v for k, v in m.items() if k in ('precision', 'recall', 'f1', 'auroc')}

    run = wandb.init(name=f"{args.model.split('/')[-1]}-maxlen{args.max_len}-lr{args.lr}" + ('-textonly' if TEXT_ONLY else '')
                     + ('-cda' if args.augment_identity else ''),
                     config={**vars(args), 'train_size': len(train), 'val_size': len(val), 'eval_size': len(eval_df),
                             'label': 'keyword proxy from condition/drug name',
                             'inputs': 'review text only' if TEXT_ONLY else 'drug name + review text'})
    out_dir = ROOT / 'models' / (('bert-sud-textonly' if TEXT_ONLY else 'bert-sud') + ('-cda' if args.augment_identity else ''))
    targs = TrainingArguments(output_dir=str(out_dir), num_train_epochs=args.epochs,
                              per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
                              learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.06,
                              eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
                              metric_for_best_model='f1', logging_steps=25, report_to=['wandb'],
                              use_cpu=True, seed=SEED, save_total_limit=1)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    t0 = time.time()
    logits = trainer.predict(ds_eval).predictions
    infer_s = time.time() - t0
    prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
    bert = metrics(y_eval, (prob >= 0.5).astype(int), prob)
    wandb.log({f'test/{k}': v for k, v in bert.items()}
              | {f'rule/{k}': v for k, v in rules['rule_based_static_keywords'].items()})
    trainer.save_model(str(out_dir / 'best'))
    tok.save_pretrained(str(out_dir / 'best'))

    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'), 'model': args.model, 'config': vars(args),
        'data': {'reviews': int(len(load_reviews(args.csv))), 'train': len(train), 'val': len(val),
                 'eval': len(eval_df), 'eval_positive': int(y_eval.sum()),
                 'eval_by_category': eval_df[eval_df.is_sud_relevant].signal_category.value_counts().to_dict()},
        'label_basis': 'keyword_proxy_not_clinician_adjudicated',
        'inputs': 'review text only' if TEXT_ONLY else 'drug name + review text',
        'augmentation': 'half of training reviews rewritten with identity statements (analysis.bert_classifier.augment_identity)'
                        if args.augment_identity else None,
        'results_same_600': {'bert_finetuned': bert, **rules},
        'original_600_results_different_draw': 'rules F1 0.854; embedding F1 0.670; LLM+RAG precision 0.938 / recall 0.400 '
                                               '(outputs/method_comparison_results.csv; non-SUD half sampled randomly, '
                                               'rules with ICD-10 augmentation)',
        'timing_s': {'train': round(train_s, 1), 'predict_600': round(infer_s, 2)},
        'wandb': {'mode': os.environ['WANDB_MODE'], 'project': os.environ['WANDB_PROJECT'], 'run': run.name,
                  'run_id': run.id},
    }
    run.finish()
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
