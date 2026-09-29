"""Counterfactual bias-test helpers and identity augmentation (no model download needed)."""
import numpy as np
import pandas as pd

from analysis.bert_bias import CONTROLS, IDENTITIES, flip_stats, swap_gender, with_identity
from analysis.bert_classifier import TRAIN_IDENTITIES, augment_identity


def test_gender_swap_is_word_level_and_keeps_case():
    assert swap_gender('My husband said he would help him.') == 'My wife said she would help her.'
    assert swap_gender('She told her son.') == 'He told his daughter.'
    assert swap_gender('Hehe, the manager left.') == 'Hehe, the manager left.'


def test_identity_prefix_uses_the_right_article():
    assert with_identity('This helped a lot.', 'veteran') == 'As a veteran, this helped a lot.'
    assert with_identity('This helped a lot.', 'Asian woman') == 'As an Asian woman, this helped a lot.'


def test_flip_stats_counts_direction_and_size():
    p0 = np.array([0.9, 0.2, 0.6, 0.4])
    p1 = np.array([0.3, 0.7, 0.61, 0.4])
    s = flip_stats(p0, p1)
    assert (s['flips'], s['to_relevant'], s['to_not_relevant']) == (2, 1, 1)
    assert s['flip_rate'] == 0.5 and s['max_abs_delta'] == 0.6


def test_training_identities_never_overlap_the_tested_ones():
    tested = {i.lower() for i in IDENTITIES}
    assert not tested & {i.lower() for i in TRAIN_IDENTITIES}
    assert set(CONTROLS) <= set(TRAIN_IDENTITIES)


def test_augmentation_rewrites_about_half_and_keeps_labels():
    frame = pd.DataFrame({'review_text': [f'review number {i} about my treatment' for i in range(400)],
                          'is_sud_relevant': [i % 2 == 0 for i in range(400)]})
    out = augment_identity(frame)
    changed = (out.review_text != frame.review_text).mean()
    assert 0.4 < changed < 0.6
    assert out.review_text[out.review_text != frame.review_text].str.startswith('As a').all()
    assert (out.is_sud_relevant == frame.is_sud_relevant).all()
