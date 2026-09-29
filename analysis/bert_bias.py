"""
Counterfactual bias test for the fine-tuned DistilBERT (text-only) SUD classifier.

    python -m analysis.bert_bias --csv /path/to/drugsComTest_raw.csv

Whether a review is about substance use should not depend on who wrote it. Each of the 600 test reviews
(the same set as analysis/bert_classifier.py) is rewritten with swapped gender terms, and separately with an
identity statement added at the start ("As a Black woman, ..."). The test counts how often the predicted
label flips and how far the probability moves. Pass criterion, fixed before running: at most 2% of
predictions flip for every perturbation.

Adding a prefix also pushes the end of long reviews past the 128-token limit, so neutral control prefixes
("As a person, ...") are run too; the identity effect is the flip rate above the control's.

Needs the local weights from `python -m analysis.bert_classifier --text-only` (models/bert-sud-textonly/best).
The gender swap is word-level and approximate: "her" always becomes "his", even where "him" would be right.
Reviews contain no demographic fields, so this tests sensitivity to identity words, not outcome parity
between real groups.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_FLIP_RATE = 0.02

GENDER_PAIRS = [('he', 'she'), ('him', 'her'), ('his', 'her'), ('himself', 'herself'), ('man', 'woman'),
                ('men', 'women'), ('husband', 'wife'), ('boyfriend', 'girlfriend'), ('son', 'daughter'),
                ('brother', 'sister'), ('father', 'mother'), ('dad', 'mom'), ('male', 'female'), ('guy', 'girl')]
CONTROLS = ['person', 'patient']        # neutral prefixes: same length change, no identity
IDENTITIES = ['Black woman', 'white man', 'Hispanic man', 'Asian woman', 'Native American man', '22-year-old',
              '70-year-old', 'veteran', 'single mother', 'gay man', 'Muslim woman', 'person on disability']


def _gender_map() -> dict[str, str]:
    m = {}
    for a, b in GENDER_PAIRS:
        m.setdefault(a, b)
        m.setdefault(b, 'his' if b == 'her' else a)
    return m


GENDER = _gender_map()
_WORD = re.compile(r"\b(" + '|'.join(sorted(GENDER, key=len, reverse=True)) + r")\b", re.I)


def swap_gender(text: str) -> str:
    def repl(m):
        w = m.group(0)
        out = GENDER[w.lower()]
        return out.capitalize() if w[0].isupper() else out
    return _WORD.sub(repl, text)


def with_identity(text: str, identity: str) -> str:
    article = 'an' if identity[0].lower() in 'aeiou' else 'a'
    return f'As {article} {identity}, {text[0].lower() + text[1:] if text else text}'


def flip_stats(p0: np.ndarray, p1: np.ndarray, changed: np.ndarray | None = None) -> dict:
    y0, y1 = p0 >= 0.5, p1 >= 0.5
    d = np.abs(p1 - p0)
    out = {'flip_rate': round(float((y0 != y1).mean()), 4), 'flips': int((y0 != y1).sum()),
           'to_relevant': int((~y0 & y1).sum()), 'to_not_relevant': int((y0 & ~y1).sum()),
           'mean_abs_delta': round(float(d.mean()), 4), 'max_abs_delta': round(float(d.max()), 4)}
    if changed is not None:
        out['reviews_changed'] = int(changed.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model-dir', default=str(ROOT / 'models' / 'bert-sud-textonly' / 'best'))
    ap.add_argument('--out', default='outputs/bert_bias.json')
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from analysis.bert_classifier import build_splits
    from analysis.sud_labels import load_reviews

    t0 = time.time()
    _, _, test = build_splits(load_reviews(args.csv))
    texts = test.review_text.tolist()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
    torch.set_num_threads(max(1, (__import__('os').cpu_count() or 2) - 2))

    @torch.no_grad()
    def prob(batch_texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(batch_texts), 64):
            enc = tok(batch_texts[i:i + 64], truncation=True, max_length=128, padding=True, return_tensors='pt')
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
        return np.concatenate(out)

    p0 = prob(texts)
    y = test.is_sud_relevant.astype(int).values
    results = {}
    swapped = [swap_gender(t) for t in texts]
    changed = np.array([a != b for a, b in zip(texts, swapped)])
    results['gender_swap'] = flip_stats(p0[changed], prob([s for s, c in zip(swapped, changed) if c]), changed)
    for identity in CONTROLS + IDENTITIES:
        kind = 'control' if identity in CONTROLS else 'identity'
        results[f'{kind}: {identity}'] = flip_stats(p0, prob([with_identity(t, identity) for t in texts]))
    control_rate = max(r['flip_rate'] for k, r in results.items() if k.startswith('control'))
    for k, r in results.items():
        if k.startswith('identity'):
            r['excess_over_control'] = round(r['flip_rate'] - control_rate, 4)
    n_tokens = [len(tok(t, truncation=False)['input_ids']) for t in texts]
    results = dict(results)
    worst = max(((k, r) for k, r in results.items() if not k.startswith('control')), key=lambda kv: kv[1]['flip_rate'])
    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'model': 'DistilBERT fine-tuned on review text only (outputs/bert_eval_textonly.json)',
        'test_set': {'reviews': len(texts), 'positives': int(y.sum()), 'baseline_positive_predictions': int((p0 >= 0.5).sum()),
                     'longer_than_128_tokens': int(sum(n > 128 for n in n_tokens))},
        'criterion': f'flip rate <= {MAX_FLIP_RATE:.0%} for every perturbation',
        'passed': all(r['flip_rate'] <= MAX_FLIP_RATE for k, r in results.items() if not k.startswith('control')),
        'control_flip_rate': control_rate,
        'worst': {'perturbation': worst[0], **worst[1]},
        'perturbations': results,
        'limits': 'Word-level gender swap is approximate; identity statements are synthetic; reviews carry no demographic '
                  'fields, so this measures sensitivity to identity words, not outcome parity between real groups.',
        'runtime_s': round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('test_set', 'passed', 'control_flip_rate', 'worst')}, indent=1))
    for k, r in results.items():
        print(f"{k:32s} flips {r['flips']:3d} ({r['flip_rate']:.1%})  mean|dp| {r['mean_abs_delta']:.3f}  max {r['max_abs_delta']:.3f}")


if __name__ == '__main__':
    main()
