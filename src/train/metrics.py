# src/train/metrics.py
"""Evaluator DUNG CHUNG: cham moi model (13 runs + prediction cua Gemma cu)
bang cung mot bo metric tren string da normalize -> so sanh dong bo.

Protocol EM/token-F1/BLEU-1 theo dung Open_Ended_VQA_fixed.ipynb.
"""
import math
from collections import Counter

import torch

from ..data.vqa_rad import is_closed, normalize_answer


def token_f1(ref_toks, hyp_toks):
    """Dung ham compute_f1 cua notebook cu (SQuAD-style token F1)."""
    common = Counter(ref_toks) & Counter(hyp_toks)
    num_same = sum(common.values())
    if len(ref_toks) == 0 or len(hyp_toks) == 0:
        return float(ref_toks == hyp_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(hyp_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def bleu1(ref_toks, hyp_toks):
    """Unigram BLEU: clipped precision x brevity penalty (khong can nltk)."""
    if not hyp_toks:
        return 0.0
    ref_counts = Counter(ref_toks)
    clipped = sum(min(c, ref_counts[t]) for t, c in Counter(hyp_toks).items())
    precision = clipped / len(hyp_toks)
    bp = 1.0 if len(hyp_toks) >= len(ref_toks) else math.exp(1 - len(ref_toks) / len(hyp_toks))
    return bp * precision


def score_answers(refs, hyps):
    """Cham cap string: EM/token-F1/BLEU-1 — moi chi so deu tach overall/closed/open.

    refs/hyps: list cau tra loi (chua/da normalize deu duoc).
    """
    assert len(refs) == len(hyps)
    em_o = em_c = em_open_hits = 0.0
    f1_sum = bleu_sum = 0.0
    f1_c = f1_o = bleu_c = bleu_o = 0.0
    n_closed = n_open = 0
    for ref, hyp in zip(refs, hyps):
        ref_n, hyp_n = normalize_answer(ref), normalize_answer(hyp)
        hit = float(ref_n == hyp_n)
        rt, ht = ref_n.split(), hyp_n.split()
        f1, bl = token_f1(rt, ht), bleu1(rt, ht)
        em_o += hit
        f1_sum += f1
        bleu_sum += bl
        if is_closed(ref_n):
            n_closed += 1
            em_c += hit
            f1_c += f1
            bleu_c += bl
        else:
            n_open += 1
            em_open_hits += hit
            f1_o += f1
            bleu_o += bl
    n = len(refs)
    return {
        "em_overall": em_o / n if n else 0.0,
        "em_closed": em_c / n_closed if n_closed else 0.0,
        "em_open": em_open_hits / n_open if n_open else 0.0,
        "token_f1": f1_sum / n if n else 0.0,
        "token_f1_closed": f1_c / n_closed if n_closed else 0.0,
        "token_f1_open": f1_o / n_open if n_open else 0.0,
        "bleu1": bleu_sum / n if n else 0.0,
        "bleu1_closed": bleu_c / n_closed if n_closed else 0.0,
        "bleu1_open": bleu_o / n_open if n_open else 0.0,
        "n_closed": n_closed, "n_open": n_open,
    }


def pyes_from_text(hyp):
    """Suy xac suat lop 'yes' tu chuoi sinh khi khong co logit (vd Gemma CSV).

    Giu dung bo cot clinical cho moi run: yes->1.0, no->0.0, con lai->0.5.
    """
    toks = normalize_answer(hyp).split()
    if "yes" in toks:
        return 1.0
    if "no" in toks:
        return 0.0
    return 0.5


def score_predictions(refs, hyps, p_yes=None):
    """Bo cham DUY NHAT cho moi run (checkpoint + CSV) -> dam bao trung bo cot.

    Gop string metrics (score_answers) + clinical binary tren subset closed.
    p_yes: xac suat lop 'yes' cung do dai refs/hyps. None -> suy tu chuoi sinh
    (pyes_from_text) de run CSV van co du cot clin_*.
    """
    m = score_answers(refs, hyps)
    refs_n = [normalize_answer(r) for r in refs]
    closed_idx = [i for i, r in enumerate(refs_n) if is_closed(r)]
    if closed_idx:
        y_true_yes = [int(refs_n[i] == "yes") for i in closed_idx]
        if p_yes is not None:
            p_closed = [float(p_yes[i]) for i in closed_idx]
        else:
            p_closed = [pyes_from_text(hyps[i]) for i in closed_idx]
        rep = closed_binary_report(y_true_yes, p_closed)
    else:
        rep = {"sensitivity": 0.0, "specificity": 0.0, "precision": 0.0, "auc": 0.5}
    m.update({f"clin_{k}": v for k, v in rep.items()})
    return m


def closed_binary_report(y_true_yes, p_yes, threshold=0.5):
    """Metric y te tren subset yes/no; yes = positive class.

    y_true_yes: list 0/1; p_yes: xac suat lop yes (softmax han che tren {yes,no}).
    """
    from sklearn.metrics import roc_auc_score

    preds = [int(p >= threshold) for p in p_yes]
    tp = sum(1 for t, p in zip(y_true_yes, preds) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true_yes, preds) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true_yes, preds) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_yes, preds) if t == 0 and p == 1)
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,   # recall lop yes
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "auc": roc_auc_score(y_true_yes, p_yes) if len(set(y_true_yes)) > 1 else 0.5,
    }


def masked_closed_argmax(logits, yes_id, no_id):
    """Constrained decoding cho classification: cau closed chi duoc chon yes/no."""
    sub = logits[:, [yes_id, no_id]]
    ids = torch.tensor([yes_id, no_id], device=logits.device)
    return ids[sub.argmax(dim=1)]


def score_predictions_csv(csv_path):
    """Cham lai file prediction cua model khac (vd: Gemma 2B/4B) cho DONG BO.

    CSV can cot: answer_ref, answer_pred. Tra ve dict cung bo cot voi run checkpoint
    (clin_* suy tu chuoi sinh vi CSV khong co logit).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    return score_predictions(df["answer_ref"].tolist(), df["answer_pred"].tolist())
