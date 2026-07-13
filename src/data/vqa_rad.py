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
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok in sorted(counts, key=lambda t: (-counts[t], t)):
        if tok not in vocab and counts[tok] >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_question(q, vocab, max_len=32):
    """Encode a question string into token IDs without special BOS/EOS tags."""
    tokens = tokenize_question(str(q))
    ids = [vocab.get(t, 1) for t in tokens][:max_len]
    return ids + [0] * (max_len - len(ids))


def build_vocab(train_questions, train_answers, min_freq=1):
    """Build unified vocabulary containing both questions and answers with special tokens."""
    q_tokens = [t for q in train_questions for t in tokenize_question(q)]
    ans_tokens = [t for a in train_answers for t in tokenize_question(a)]
    counts = Counter(q_tokens + ans_tokens)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok in sorted(counts, key=lambda t: (-counts[t], t)):
        if tok not in vocab and counts[tok] >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_sequence(seq_str, vocab, max_len=16, is_target=False):
    """Encode a sequence string into list of word token IDs with BOS/EOS tags."""
    tokens = tokenize_question(str(seq_str))
    ids = [vocab.get(t, 1) for t in tokens]  # 1 is <unk>
    if is_target:
        # Target sequence: tokens + <eos> (3)
        ids = ids[:max_len-1] + [3]
    else:
        # Input sequence: <bos> (2) + tokens
        ids = [2] + ids[:max_len-1]
    return ids + [0] * (max_len - len(ids))  # 0 is <pad>


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

