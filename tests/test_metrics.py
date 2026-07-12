# tests/test_metrics.py
import torch

from src.train.metrics import (bleu1, closed_binary_report, masked_closed_argmax,
                               score_answers, token_f1)


def test_token_f1_matches_notebook_protocol():
    assert token_f1(["right", "lung"], ["right", "lung"]) == 1.0
    assert token_f1(["yes"], ["no"]) == 0.0
    assert 0 < token_f1(["left", "upper", "lobe"], ["left", "lobe"]) < 1


def test_bleu1_unigram_with_brevity_penalty():
    assert bleu1(["yes"], ["yes"]) == 1.0
    assert bleu1(["right", "lung"], ["lung"]) < 1.0    # hyp ngan hon -> BP phat
    assert bleu1(["yes"], ["no"]) == 0.0


def test_score_answers_full_report():
    refs = ["yes", "no", "right lung", "axial"]
    hyps = ["yes", "yes", "right lung", "coronal"]
    r = score_answers(refs, hyps)
    assert r["em_overall"] == 0.5                       # 2/4
    assert r["em_closed"] == 0.5 and r["n_closed"] == 2
    assert r["em_open"] == 0.5 and r["n_open"] == 2
    assert 0 <= r["token_f1"] <= 1 and 0 <= r["bleu1"] <= 1


def test_closed_binary_report():
    #                yes=1, no=0
    y_true = [1, 1, 0, 0]
    p_yes = [0.9, 0.4, 0.2, 0.6]
    r = closed_binary_report(y_true, p_yes, threshold=0.5)
    assert r["sensitivity"] == 0.5                      # 1/2 yes bat duoc
    assert r["specificity"] == 0.5                      # 1/2 no dung
    assert 0 <= r["auc"] <= 1 and 0 <= r["precision"] <= 1


def test_masked_closed_argmax():
    logits = torch.tensor([[0.1, 0.2, 5.0, 4.0],
                           [3.0, 0.1, 0.2, 0.3]])
    preds = masked_closed_argmax(logits, yes_id=0, no_id=1)
    assert preds.tolist() == [1, 0]
