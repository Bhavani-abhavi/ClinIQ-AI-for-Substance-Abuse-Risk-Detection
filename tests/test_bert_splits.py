"""Split and label logic behind the DistilBERT benchmark (no torch or model download needed)."""
import pandas as pd

from analysis import bert_classifier
from analysis.bert_classifier import build_splits, metrics, rule_baselines
from analysis.sud_labels import contains_keyword, signal_category

FILLER = " and the side effects were manageable for most of the first month of treatment overall"


def _frame():
    rows = []
    sud_drugs = [('Opioid Dependence', 'Suboxone'), ('Alcohol Dependence', 'Naltrexone'),
                 ('Opiate Withdrawal', 'Clonidine'), ('Drug Dependence', 'Xanax')]
    uid = 0
    for i in range(40):
        for condition, drug in sud_drugs:
            uid += 1
            rows.append({'uniqueID': uid, 'condition': condition, 'drugName': drug, 'usefulCount': i,
                         'review_text': f"review {uid} about {drug}" + FILLER})
    for i in range(900):
        uid += 1
        rows.append({'uniqueID': uid, 'condition': 'Acne', 'drugName': 'Doxycycline', 'usefulCount': i % 7,
                     'review_text': f"review {uid} about my skin" + FILLER})
    # the dataset repeats reviews under brand and generic names; duplicates must not straddle splits
    rows.append(dict(rows[0], uniqueID=uid + 1, drugName='Buprenorphine / naloxone'))
    df = pd.DataFrame(rows)
    df['is_sud_relevant'] = df.condition.apply(contains_keyword) | df.drugName.apply(contains_keyword)
    df['signal_category'] = [signal_category(c, d, t) if s else 'non_sud' for c, d, t, s in
                             zip(df.condition, df.drugName, df.review_text, df.is_sud_relevant)]
    return df


def test_eval_set_is_stratified_and_disjoint_from_training():
    train, val, eval_df = build_splits(_frame(), per_category=10)
    # opioid (Suboxone, opiate withdrawal), polysubstance (alcohol + naltrexone), other_sud (Xanax): 10 each
    per_cat = eval_df[eval_df.is_sud_relevant].signal_category.value_counts().to_dict()
    assert per_cat == {'opioid': 10, 'polysubstance': 10, 'other_sud': 10}
    assert (~eval_df.is_sud_relevant).sum() == 300
    seen = set(train.review_text) | set(val.review_text)
    assert not seen & set(eval_df.review_text)
    assert not set(train.review_text) & set(val.review_text)
    assert eval_df.review_text.is_unique


def test_training_negatives_are_twice_the_positives():
    train, val, _ = build_splits(_frame(), per_category=10)
    both = pd.concat([train, val])
    pos = int(both.is_sud_relevant.sum())
    assert len(both) - pos == 2 * pos


def test_labels_come_from_condition_or_drug_name_not_review_text():
    assert contains_keyword('Opioid Dependence') and contains_keyword('Suboxone')
    assert not contains_keyword('Acne') and not contains_keyword(None)
    assert signal_category('Opioid Dependence', 'Suboxone', '') == 'opioid'
    assert signal_category('Drug Dependence', 'Xanax', 'also drinking, alcoholism in family') == 'polysubstance'


def test_text_only_flag_drops_the_drug_name():
    row = pd.Series({'drugName': 'Suboxone', 'review_text': 'helped me stay clean'})
    try:
        bert_classifier.TEXT_ONLY = False
        assert bert_classifier.text_of(row).startswith('drug: Suboxone.')
        bert_classifier.TEXT_ONLY = True
        assert bert_classifier.text_of(row) == 'helped me stay clean'
    finally:
        bert_classifier.TEXT_ONLY = False


def test_metrics_reports_confusion_counts():
    m = metrics([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.1, 0.6])
    assert (m['tp'], m['fn'], m['tn'], m['fp']) == (1, 1, 1, 1)
    assert m['precision'] == m['recall'] == 0.5 and m['auroc'] == 0.75


def test_text_only_rules_ignore_the_drug_name():
    frame = pd.DataFrame({'review_text': ['this medicine kept me stable for months', 'I relapsed after rehab'],
                          'drugName': ['Suboxone', 'Doxycycline'], 'is_sud_relevant': [True, False]})
    r = rule_baselines(frame)
    assert (r['rule_based_static_keywords']['tp'], r['rule_based_static_keywords']['fp']) == (1, 1)
    assert (r['rule_based_text_only']['tp'], r['rule_based_text_only']['fp']) == (0, 1)


def test_meth_matches_only_as_a_whole_word():
    from analysis.detection import rule_classify
    for drug in ('Methylphenidate', 'Sulfamethoxazole / trimethoprim', 'Dextromethorphan', 'Indomethacin',
                 'Methylprednisolone', 'Promethazine'):
        assert not contains_keyword(drug), drug
    for text in ('Methadone', 'Methamphetamine', 'relapsed on meth', 'Meth addiction'):
        assert contains_keyword(text), text
    assert signal_category('ADHD', 'Methylphenidate', 'helped my focus') == 'other_sud'   # only called for positives
    kw = {'stimulant': ['meth']}
    assert rule_classify('methylphenidate helped my focus', 'Methylphenidate', kw)[0] == 0
    assert rule_classify('I used meth for years', 'Suboxone', kw)[0] == 1
