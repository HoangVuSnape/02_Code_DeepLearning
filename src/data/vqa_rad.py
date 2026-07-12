# src/data/vqa_rad.py
"""Tien xu ly VQA-RAD cho bai toan classification tren answer vocabulary."""
import hashlib
import re
from collections import Counter

import numpy as np


def normalize_answer(ans):
    ans = str(ans).lower().strip()
    ans = re.sub(r"\s+", " ", ans)
    return ans.rstrip(".").strip()


def is_closed(ans_norm):
    return ans_norm in ("yes", "no")


def build_answer_vocab(train_answers):
    """Map answer -> class id; sap theo tan suat giam dan roi alphabet de deterministic."""
    counts = Counter(normalize_answer(a) for a in train_answers)
    ordered = sorted(counts, key=lambda a: (-counts[a], a))
    return {a: i for i, a in enumerate(ordered)}


def tokenize_question(q):
    q = str(q).lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return q.split()


def build_question_vocab(train_questions, min_freq=1):
    counts = Counter(t for q in train_questions for t in tokenize_question(q))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok in sorted(counts, key=lambda t: (-counts[t], t)):
        if counts[tok] >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_question(q, vocab, max_len=32):
    ids = [vocab.get(t, 1) for t in tokenize_question(q)][:max_len]
    return ids + [0] * (max_len - len(ids))


def image_key(pil_image):
    """Khoa dinh danh anh de split image-disjoint (1 anh co nhieu cau hoi)."""
    return hashlib.md5(pil_image.convert("RGB").tobytes()).hexdigest()


def group_split(keys, val_ratio=0.1, seed=42):
    """Chia index train/val sao cho cung mot anh khong nam o ca hai phia."""
    uniq = sorted(set(keys))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_ratio)))
    val_keys = set(uniq[:n_val])
    train_idx = [i for i, k in enumerate(keys) if k not in val_keys]
    val_idx = [i for i, k in enumerate(keys) if k in val_keys]
    return train_idx, val_idx
