# VQA-RAD Attention & Encoder Ablation (v2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v2 (2026-07-12):** sửa theo feedback của user — (1) làm rõ cơ chế có/không attention, (2) tách text encoder thành LSTM **và** Transformer, (3) đủ 8 tổ hợp attention, (4) thêm nhóm frozen-CNN (ResNet-18 pretrained, chỉ train text + fusion), (5) SFT trước → Reinforcement Learning sau, (6) liệt kê rõ model sử dụng, (7) chốt bảng metric rõ ràng — dùng chung một evaluator để chấm lại cả prediction của Gemma cũ cho đồng bộ.

> **v2.1 (2026-07-12):** thêm hạ tầng vận hành (spec nhỏ ở `docs/plan/`, code ở `src/config.py` + `src/integrations/` + `src/utils/`): config động (không fix cứng), smoke test, checkpoint, thông báo Discord, push HuggingFace, log Comet ML, tqdm, nạp secret Kaggle/.env. Hướng dẫn chạy: `HUONG_DAN_CHAY.md`.

**Goal:** Xây dựng pipeline classification nhẹ trên VQA-RAD với thiết kế thực nghiệm 4 nhóm (A: 8 tổ hợp attention; B: LSTM vs Transformer text encoder; C: frozen ResNet-18; D: SFT→RL), đo bộ metric thống nhất (EM / token-F1 / BLEU-1 / metric y tế) để so sánh trực tiếp với Gemma 2B/4B đã train.

**Architecture:** VQA quy về **classification trên answer vocabulary của train set** (chuẩn MedVInT-TE). Image encoder (CustomCNN hoặc ResNet-18 frozen) → `v_img [B,128]`; text encoder (BiLSTM hoặc TransformerEncoder) → `v_txt [B,64]`; concat `[B,192]` → MLP decoder → logits `[B,C]`. Sau SFT, fine-tune bằng REINFORCE self-critical với reward EM/token-F1.

**Tech Stack:** PyTorch + torchvision (ResNet-18 pretrained), HuggingFace `datasets`, scikit-learn (AUC/macro-F1), pytest. **Môi trường:** máy local KHÔNG có Python — mọi verify (pytest) chạy ở cell đầu notebook trên Kaggle/Colab: `!python -m pytest tests/ -q`.

---

## 0. Model sử dụng (chốt rõ — feedback #6)

| Vai trò | Model | Nguồn | Train thế nào | Params |
|---|---|---|---|---|
| Image encoder (chính) | **CustomCNN** 3 block (3→32→64→128) | tự viết | train từ đầu | ~0.1M |
| Image encoder (nhóm C) | **ResNet-18** pretrained ImageNet | `torchvision.models.resnet18` | **đóng băng toàn bộ**, chỉ train projection + text + fusion | 11.2M frozen + 0.07M train |
| Text encoder (nhánh 1) | **BiLSTM** (emb 64, hidden 32×2 → 64-d) | tự viết | train từ đầu | ~0.1M |
| Text encoder (nhánh 2) | **TransformerEncoder** (d=64, 4 head, 2 layer → 64-d) | `nn.TransformerEncoder`, tự viết wrapper | train từ đầu | ~0.15M |
| Fusion + Decoder | Concat + MLP 192→128→C (+ Gated Attention tùy chọn) | tự viết | train từ đầu | ~0.1M |
| RL fine-tune | REINFORCE self-critical (SCST) trên model SFT tốt nhất | tự viết | fine-tune decoder + text encoder, lr nhỏ | — |
| Đối chứng (đã có) | Gemma/PaliGemma 2B & 4B + LoRA | đã train trước (`v3-vqa-*.ipynb`) | chỉ chấm lại prediction bằng evaluator chung | 2B/4B |
| Đối chứng (đã có) | PubMedCLIP-ViT + DistilGPT2 prefix | `Open_Ended_VQA_fixed.ipynb` | chỉ chấm lại nếu có prediction CSV | ~200M |

## 0b. Có attention vs không attention — khác nhau thế nào (feedback #1)

Mỗi block có đúng MỘT điểm khác biệt cơ chế, mọi thứ khác giữ nguyên:

| Block | KHÔNG attention | CÓ attention | Ý nghĩa |
|---|---|---|---|
| Image (CNN/ResNet) | `v = GAP(F)` — average pooling coi mọi kênh như nhau | **SE-Block:** `s = σ(W₂·ReLU(W₁·GAP(F)))`, `v = GAP(s ⊙ F)` — kênh nào quan trọng được khuếch đại | Học "kênh đặc trưng nào đáng tin" (cạnh, đốm mờ, texture xương...) |
| Text (BiLSTM) | `v = [h→_T ; h←_1]` — chỉ lấy hidden state cuối | **Temporal Attention:** `αₜ = softmax(w·hₜ)` (mask PAD), `v = Σ αₜhₜ` | Từ khóa quan trọng trong câu hỏi ("effusion", "left lobe") được trọng số cao thay vì phụ thuộc vị trí cuối câu |
| Text (Transformer) | `v = masked-mean(H)` — trung bình mọi token | **Attention pooling:** cùng công thức Temporal Attention trên output H | Self-attention bên trong đã có sẵn; ablation ở đây là **cách pooling** |
| Decoder | `z' = z` — dùng thẳng fused vector | **Gated Attention:** `z' = z ⊙ σ(Wz)` | Cổng lọc chiều nhiễu sau khi concat 2 modality |

Phân tích trong báo cáo: vẽ trọng số `αₜ` trên từng từ của câu hỏi (model có text-attention) và phân bố trọng số kênh SE — trả lời "attention nhìn vào đâu".

## 0c. Ma trận thực nghiệm — đủ trường hợp (feedback #2, #3, #4, #5)

**Nhóm A — Attention ablation ĐỦ 8 tổ hợp** (image=CustomCNN, text=BiLSTM; cờ `(img, txt, dec)`):

| Run | Cờ | Run | Cờ |
|---|---|---|---|
| A1_000 (baseline) | (0,0,0) | A5_110 | (1,1,0) |
| A2_100 | (1,0,0) | A6_101 | (1,0,1) |
| A3_010 | (0,1,0) | A7_011 | (0,1,1) |
| A4_001 | (0,0,1) | A8_111 (full) | (1,1,1) |

**Nhóm B — Text encoder: LSTM vs Transformer** (image=CustomCNN; lấy cờ attention tốt nhất theo val EM từ nhóm A):

| Run | Text encoder | Pooling |
|---|---|---|
| B1_trans_mean | Transformer | masked mean (không attention pooling) |
| B2_trans_attn | Transformer | attention pooling |

→ So cặp: BiLSTM tốt nhất (từ A) vs B1 vs B2, gần bằng nhau về params.

**Nhóm C — Đóng băng CNN, chỉ train text + fusion** (theo yêu cầu + `code_references.ipynb` §so sánh pretrained):

| Run | Image encoder | Text encoder |
|---|---|---|
| C1_resnet_lstm | ResNet-18 frozen (224×224, ImageNet norm) | BiLSTM (cờ tốt nhất từ A) |
| C2_resnet_best | ResNet-18 frozen | encoder text tốt nhất từ A/B |

**Nhóm D — SFT trước, RL sau:**

| Run | Nội dung |
|---|---|
| D1_scst | Lấy checkpoint SFT tốt nhất toàn cục (theo val EM) → REINFORCE self-critical 3–5 epoch, lr=1e-5, reward = 0.5·EM + 0.5·token-F1 |

**Tổng: 8 + 2 + 2 + 1 = 13 runs.** Model ~0.3M params, ~5–15 phút/run trên T4 → vừa quota Kaggle/Colab Free. Chạy tuần tự A→B→C→D, checkpoint + CSV lưu ngay từng run.

**Biến cố định mọi run (trừ điều đang ablate):** data + split image-disjoint (seed 42), vocab, MAX_LEN=32, batch 32, AdamW lr=1e-3 wd=1e-4, CE loss, 12 epoch, early-stop patience 3 theo val EM, test đánh giá đúng một lần cuối. Nhóm C khác ảnh 224 + ImageNet norm (bắt buộc của pretrained — ghi rõ trong báo cáo là thay đổi đi kèm).

## 0d. Metric đánh giá — chốt rõ (feedback #7)

**Nguyên tắc đồng bộ:** MỌI model (13 runs + Gemma 2B/4B cũ) được chấm bằng **cùng một module** `src/train/metrics.py` trên cùng test set 451 câu. Với Gemma cũ: export prediction ra CSV (`question, answer_ref, answer_pred, question_type`) rồi chấm lại bằng `score_predictions_csv()` — đây chính là bước "đánh giá lại thêm metric cho đồng bộ" user đã nói.

| # | Metric | Định nghĩa | Dùng so sánh với |
|---|---|---|---|
| 1 | **EM overall / closed / open** (chính) | pred string == ref string sau normalize (lowercase, strip, bỏ dấu chấm) | Gemma 2B/4B (Open 23%/20%, Closed 68.53%/70.92%) |
| 2 | **Token-F1** | F1 trên tập token overlap (đúng hàm `compute_f1` của `Open_Ended_VQA_fixed.ipynb`) | baseline PubMedCLIP+GPT-2, Gemma |
| 3 | **BLEU-1** | unigram precision có clip × brevity penalty (protocol cũ, tự implement không cần nltk) | baseline cũ |
| 4 | **Accuracy + Macro-F1** trên answer classes | chất lượng classification, công bằng giữa lớp hiếm | giữa 13 runs |
| 5 | **Closed-subset y tế:** Precision, Recall (Sensitivity), Specificity, ROC-AUC | nhị phân yes/no, yes = positive; AUC từ softmax hạn chế trên {yes,no} | tiêu chí an toàn y tế giữa các run |
| 6 | **Chi phí:** params total / trainable, giây/epoch | trade-off hiệu năng–chi phí | mọi model kể cả Gemma |

Metric chọn checkpoint & chọn "best config": **val EM overall** (metric 1). Metric 2–6 chỉ báo cáo, không dùng để tune.

---

## File Structure

```
02_Code_DeepLearning/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vqa_rad.py        ← normalize, tokenize, vocab, image-disjoint split
│   │   └── dataset.py        ← VQARADClsDataset + transform (0.5-norm | ImageNet-norm)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py      ← ChannelAttention (SE), TemporalAttention, GatedAttention
│   │   ├── encoders.py       ← ImageEncoderCNN, ImageEncoderResNet18Frozen,
│   │   │                        TextEncoderBiLSTM, TextEncoderTransformer
│   │   └── fusion.py         ← FusionModel + GROUP_A registry + build helpers
│   └── train/
│       ├── __init__.py
│       ├── metrics.py        ← EM/token-F1/BLEU-1 string-level + closed binary + CSV re-scorer
│       ├── engine.py         ← SFT loop, early stop, checkpoint, CSV log, trainable params
│       └── rl.py             ← REINFORCE self-critical (SCST)
├── tests/
│   ├── test_vqa_rad.py  test_dataset.py  test_attention.py
│   ├── test_encoders.py test_fusion.py   test_metrics.py
│   ├── test_engine.py   test_rl.py
├── notebook/
│   └── ablation_vqarad.ipynb ← runner: pytest → pilot → A(8) → B(2) → C(2) → D(1) → chấm lại Gemma → tổng hợp
└── docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md
```

**Hyperparameters cố định:**

```python
CFG = dict(
    IMAGE_SIZE=128, IMAGE_SIZE_RESNET=224, MAX_LEN=32, BATCH_SIZE=32,
    EPOCHS=12, LR=1e-3, WEIGHT_DECAY=1e-4, PATIENCE=3, SEED=42,
    EMB_DIM=64, LSTM_HIDDEN=32, TRANS_LAYERS=2, TRANS_HEADS=4,
    IMG_DIM=128, VAL_RATIO=0.1,
    RL_EPOCHS=5, RL_LR=1e-5,
)
```

---

### Task 0: Khởi tạo git repo

**Files:** Create: `.gitignore`, `requirements-dev.txt`

- [ ] **Step 1: git init**

```powershell
git init
git branch -m main
```

- [ ] **Step 2: `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
runs/
*.pt
data/hf_cache/
.env
```

- [ ] **Step 3: `requirements-dev.txt`**

```
torch
torchvision
datasets
pillow
numpy
pandas
scikit-learn
pytest
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: init repo for VQA-RAD ablation v2"
```

---

### Task 1: Data utilities

**Files:** Create: `src/__init__.py`, `src/data/__init__.py`, `src/data/vqa_rad.py` · Test: `tests/test_vqa_rad.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_vqa_rad.py
from src.data.vqa_rad import (
    normalize_answer, is_closed, build_answer_vocab,
    tokenize_question, build_question_vocab, encode_question, group_split,
)

def test_normalize_answer():
    assert normalize_answer("  Yes. ") == "yes"
    assert normalize_answer("Right  Lung") == "right lung"

def test_is_closed():
    assert is_closed("yes") and is_closed("no")
    assert not is_closed("right lung")

def test_answer_vocab_deterministic_and_freq_ordered():
    v = build_answer_vocab(["yes", "no", "yes", "axial"])
    assert v["yes"] == 0
    assert set(v) == {"yes", "no", "axial"}
    assert v == build_answer_vocab(["yes", "no", "yes", "axial"])

def test_tokenize_and_encode_question():
    vocab = build_question_vocab(["Is this an MRI?"])
    ids = encode_question("Is this an MRI?", vocab, max_len=8)
    assert len(ids) == 8 and ids[0] != 0
    assert encode_question("unseen words", vocab, max_len=4)[0] == 1  # <unk>

def test_group_split_image_disjoint():
    keys = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    tr, va = group_split(keys, val_ratio=0.3, seed=42)
    assert {keys[i] for i in tr}.isdisjoint({keys[i] for i in va})
    assert sorted(tr + va) == list(range(len(keys)))
```

- [ ] **Step 2: Run (Kaggle/Colab): `python -m pytest tests/test_vqa_rad.py -v`** — Expected: FAIL ImportError.

- [ ] **Step 3: Implement `src/data/vqa_rad.py`** (kèm `__init__.py` rỗng)

```python
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
    return hashlib.md5(pil_image.convert("RGB").tobytes()).hexdigest()


def group_split(keys, val_ratio=0.1, seed=42):
    """Cung mot anh khong bao gio nam o ca train va val."""
    uniq = sorted(set(keys))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_ratio)))
    val_keys = set(uniq[:n_val])
    train_idx = [i for i, k in enumerate(keys) if k not in val_keys]
    val_idx = [i for i, k in enumerate(keys) if k in val_keys]
    return train_idx, val_idx
```

- [ ] **Step 4: Run test — Expected: 5 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: VQA-RAD preprocessing utils"`

---

### Task 2: PyTorch Dataset (hỗ trợ 2 chế độ normalize)

**Files:** Create: `src/data/dataset.py` · Test: `tests/test_dataset.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_dataset.py
import torch
from PIL import Image

from src.data.dataset import VQARADClsDataset, default_transform
from src.data.vqa_rad import build_answer_vocab, build_question_vocab


def _fake_records(n=4):
    img = Image.new("L", (64, 80), color=128)
    answers = ["yes", "no", "right lung", "yes"]
    return [{"image": img, "question": f"is this q {i}", "answer": answers[i]}
            for i in range(n)]


def test_dataset_shapes_and_types():
    recs = _fake_records()
    a2i = build_answer_vocab([r["answer"] for r in recs])
    qv = build_question_vocab([r["question"] for r in recs])
    ds = VQARADClsDataset(recs, a2i, qv, default_transform(128, train=False), max_len=16)
    img, tokens, label, closed = ds[0]
    assert img.shape == (3, 128, 128)
    assert tokens.shape == (16,) and tokens.dtype == torch.long
    assert label == a2i["yes"] and closed is True


def test_imagenet_transform_size_224():
    recs = _fake_records()
    a2i = build_answer_vocab([r["answer"] for r in recs])
    qv = build_question_vocab([r["question"] for r in recs])
    tf = default_transform(224, train=False, imagenet=True)   # cho ResNet-18 frozen
    ds = VQARADClsDataset(recs, a2i, qv, tf)
    assert ds[0][0].shape == (3, 224, 224)


def test_unseen_answer_gets_minus_one():
    recs = _fake_records()
    a2i = build_answer_vocab(["yes", "no"])
    qv = build_question_vocab([r["question"] for r in recs])
    ds = VQARADClsDataset(recs, a2i, qv, default_transform(128, train=False))
    _, _, label, closed = ds[2]
    assert label == -1 and closed is False
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/data/dataset.py`**

```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .vqa_rad import encode_question, is_closed, normalize_answer

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def default_transform(image_size=128, train=True, imagenet=False):
    """imagenet=True cho backbone pretrained (ResNet-18); nguoc lai norm 0.5."""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet else ([0.5] * 3, [0.5] * 3)
    aug = [transforms.Resize((image_size, image_size)),
           transforms.Grayscale(num_output_channels=3)]
    if train:
        aug.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)))
    aug += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(aug)


class VQARADClsDataset(Dataset):
    """Item: (image[3,S,S], tokens[MAX_LEN], label, closed).

    label = -1 khi answer khong nam trong train vocab (chi o test set):
    khong bao gio match duoc -> tinh sai trong EM, dong protocol voi generation.
    """

    def __init__(self, records, answer2id, q_vocab, transform, max_len=32):
        self.records = records
        self.answer2id = answer2id
        self.q_vocab = q_vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = self.transform(r["image"].convert("RGB"))
        tokens = torch.tensor(
            encode_question(r["question"], self.q_vocab, self.max_len),
            dtype=torch.long)
        ans = normalize_answer(r["answer"])
        label = self.answer2id.get(ans, -1)
        return img, tokens, label, is_closed(ans)
```

- [ ] **Step 4: Run — Expected: 3 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: dataset with 0.5/ImageNet normalize modes"`

---

### Task 3: Attention modules

**Files:** Create: `src/models/__init__.py`, `src/models/attention.py` · Test: `tests/test_attention.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_attention.py
import torch

from src.models.attention import ChannelAttention, GatedAttention, TemporalAttention


def test_channel_attention_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    assert ChannelAttention(32)(x).shape == x.shape


def test_temporal_attention_masks_pad():
    attn = TemporalAttention(hidden_dim=16)
    out = torch.randn(2, 5, 16)
    pad_mask = torch.tensor([[False, False, True, True, True],
                             [False, False, False, False, False]])
    ctx, w = attn(out, pad_mask)
    assert ctx.shape == (2, 16) and w.shape == (2, 5)
    assert torch.allclose(w[0, 2:], torch.zeros(3), atol=1e-6)
    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)


def test_gated_attention_preserves_shape():
    x = torch.randn(4, 48)
    assert GatedAttention(48)(x).shape == x.shape
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/models/attention.py`** (kèm `__init__.py` rỗng)

```python
# src/models/attention.py
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """SE-Block: v = GAP(s * F), s = sigmoid(W2 ReLU(W1 GAP(F)))."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.fc(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        return x * weights


class TemporalAttention(nn.Module):
    """v = sum_t alpha_t h_t, alpha = softmax(w.h_t) voi mask PAD."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score_fn = nn.Linear(hidden_dim, 1)

    def forward(self, seq_outputs, pad_mask=None):
        scores = self.score_fn(seq_outputs).squeeze(-1)            # [B, T]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (seq_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class GatedAttention(nn.Module):
    """z' = z * sigmoid(Wz) — cong loc thong tin sau fusion."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)
```

- [ ] **Step 4: Run — Expected: 3 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: SE, temporal, gated attention modules"`

---

### Task 4: Encoders — 2 image (CNN, ResNet18-frozen) + 2 text (BiLSTM, Transformer)

**Files:** Create: `src/models/encoders.py` · Test: `tests/test_encoders.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_encoders.py
import torch

from src.models.encoders import (
    ImageEncoderCNN, ImageEncoderResNet18Frozen,
    TextEncoderBiLSTM, TextEncoderTransformer,
)


def test_cnn_output_dim_with_and_without_attention():
    x = torch.randn(2, 3, 128, 128)
    assert ImageEncoderCNN(use_attention=False)(x).shape == (2, 128)
    assert ImageEncoderCNN(use_attention=True)(x).shape == (2, 128)


def test_resnet18_frozen_backbone_and_output():
    enc = ImageEncoderResNet18Frozen(pretrained=False)   # test khong tai weights
    assert enc(torch.randn(2, 3, 224, 224)).shape == (2, 128)
    assert all(not p.requires_grad for p in enc.features.parameters())  # dong bang
    assert any(p.requires_grad for p in enc.proj.parameters())          # proj van train


def test_bilstm_output_dim():
    enc = TextEncoderBiLSTM(vocab_size=50, use_attention=False)
    assert enc(torch.randint(1, 50, (2, 32))).shape == (2, 64)


def test_bilstm_attention_handles_padding():
    enc = TextEncoderBiLSTM(vocab_size=50, use_attention=True)
    tokens = torch.zeros(2, 32, dtype=torch.long)
    tokens[:, :3] = torch.randint(1, 50, (2, 3))
    out = enc(tokens)
    assert out.shape == (2, 64) and torch.isfinite(out).all()


def test_transformer_output_dim_both_poolings():
    tokens = torch.zeros(2, 32, dtype=torch.long)
    tokens[:, :5] = torch.randint(1, 50, (2, 5))
    for attn in (False, True):    # mean pooling vs attention pooling
        enc = TextEncoderTransformer(vocab_size=50, use_attention=attn)
        out = enc(tokens)
        assert out.shape == (2, 64) and torch.isfinite(out).all()
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/models/encoders.py`**

```python
# src/models/encoders.py
import torch
import torch.nn as nn

from .attention import ChannelAttention, TemporalAttention


class ImageEncoderCNN(nn.Module):
    """CustomCNN 3 tang: 3->32->64->128, tuy chon SE-block o feature map cuoi."""

    def __init__(self, use_attention=False, out_dim=128):
        super().__init__()
        self.use_attention = use_attention

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, out_dim))
        if use_attention:
            self.attention = ChannelAttention(out_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.features(x)
        if self.use_attention:
            x = self.attention(x)
        return self.norm(self.pool(x).flatten(1))


class ImageEncoderResNet18Frozen(nn.Module):
    """ResNet-18 pretrained DONG BANG toan bo backbone; chi train proj (+SE neu bat).

    Nhom C cua thi nghiem: freeze CNN, train text encoder + fusion/decoder.
    """

    def __init__(self, use_attention=False, out_dim=128, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:                       # torchvision cu
            backbone = resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # [B,512,h,w]
        for p in self.features.parameters():
            p.requires_grad = False
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        self.features.eval()                      # BatchNorm dung running stats
        with torch.no_grad():
            f = self.features(x)
        if self.use_attention:
            f = self.attention(f)
        return self.norm(self.proj(self.pool(f).flatten(1)))


class TextEncoderBiLSTM(nn.Module):
    """Embedding(64) + BiLSTM(32x2) -> v_txt [B,64].

    Khong attention: v = [h_forward_cuoi ; h_backward_dau].
    Co attention: temporal attention tren moi buoc thoi gian (mask PAD).
    """

    def __init__(self, vocab_size, use_attention=False, emb_dim=64, hidden=32):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        out_dim = hidden * 2
        if use_attention:
            self.attention = TemporalAttention(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, tokens):
        pad_mask = tokens == 0
        lstm_out, (h_n, _) = self.lstm(self.embedding(tokens))
        if self.use_attention:
            out, _ = self.attention(lstm_out, pad_mask)
        else:
            out = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.norm(out)


class TextEncoderTransformer(nn.Module):
    """TransformerEncoder (d=64, 4 head, 2 layer) -> v_txt [B,64].

    Khong attention: masked mean pooling. Co attention: attention pooling
    (cung TemporalAttention de doi xung voi nhanh BiLSTM).
    """

    def __init__(self, vocab_size, use_attention=False, emb_dim=64,
                 nhead=4, nlayers=2, max_len=32):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        if use_attention:
            self.attention = TemporalAttention(emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, tokens):
        pad_mask = tokens == 0                                   # [B, T]
        x = self.embedding(tokens) + self.pos[:, :tokens.size(1)]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        if self.use_attention:
            out, _ = self.attention(x, pad_mask)
        else:
            keep = (~pad_mask).unsqueeze(-1).float()
            out = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # masked mean
        return self.norm(out)
```

- [ ] **Step 4: Run — Expected: 5 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: 2 image encoders (CNN, frozen ResNet18) + 2 text encoders (BiLSTM, Transformer)"`

---

### Task 5: FusionModel + registry nhóm thực nghiệm

**Files:** Create: `src/models/fusion.py` · Test: `tests/test_fusion.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_fusion.py
import torch

from src.models.fusion import GROUP_A, FusionModel, build_model


def test_group_a_has_all_8_combos():
    assert len(GROUP_A) == 8
    flags = {(v["image_attention"], v["text_attention"], v["decoder_attention"])
             for v in GROUP_A.values()}
    assert len(flags) == 8                        # du 2^3 to hop, khong trung


def test_forward_shape_all_group_a():
    for name, flags in GROUP_A.items():
        model = build_model(vocab_size=50, num_classes=10, **flags)
        logits = model(torch.randn(2, 3, 128, 128), torch.randint(1, 50, (2, 32)))
        assert logits.shape == (2, 10), name


def test_transformer_text_encoder_variant():
    model = build_model(50, 10, text_encoder="transformer", text_attention=True)
    logits = model(torch.randn(2, 3, 128, 128), torch.randint(1, 50, (2, 32)))
    assert logits.shape == (2, 10)


def test_resnet_frozen_variant_trainable_params_small():
    model = build_model(50, 10, image_encoder="resnet18_frozen", pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < total                      # backbone bi dong bang
    logits = model(torch.randn(2, 3, 224, 224), torch.randint(1, 50, (2, 32)))
    assert logits.shape == (2, 10)


def test_backward_runs():
    model = build_model(50, 10, image_attention=True, text_attention=True,
                        decoder_attention=True)
    logits = model(torch.randn(2, 3, 128, 128), torch.randint(1, 50, (2, 32)))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/models/fusion.py`**

```python
# src/models/fusion.py
import itertools

import torch
import torch.nn as nn

from .attention import GatedAttention
from .encoders import (ImageEncoderCNN, ImageEncoderResNet18Frozen,
                       TextEncoderBiLSTM, TextEncoderTransformer)

# Nhom A: DU 8 to hop attention (image=cnn, text=lstm). Ten ma hoa co (i,t,d).
GROUP_A = {
    f"A{n+1}_{i}{t}{d}": dict(image_attention=bool(i), text_attention=bool(t),
                              decoder_attention=bool(d))
    for n, (i, t, d) in enumerate(itertools.product([0, 1], repeat=3))
}


class FusionModel(nn.Module):
    """image_enc -> v_img[128] ++ text_enc -> v_txt[64] -> [192]
    -> (GatedAttention?) -> MLP 192->128->C."""

    def __init__(self, image_enc, text_enc, num_classes, decoder_attention=False):
        super().__init__()
        self.image = image_enc
        self.text = text_enc
        fused_dim = 128 + 64
        self.decoder_attention = GatedAttention(fused_dim) if decoder_attention else None
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, images, tokens):
        fused = torch.cat([self.image(images), self.text(tokens)], dim=1)
        if self.decoder_attention is not None:
            fused = self.decoder_attention(fused)
        return self.classifier(fused)


def build_model(vocab_size, num_classes, image_encoder="cnn", text_encoder="lstm",
                image_attention=False, text_attention=False, decoder_attention=False,
                pretrained=True, max_len=32):
    """Factory dung chung cho ca 4 nhom thi nghiem A/B/C/D."""
    if image_encoder == "cnn":
        img = ImageEncoderCNN(use_attention=image_attention)
    elif image_encoder == "resnet18_frozen":
        img = ImageEncoderResNet18Frozen(use_attention=image_attention,
                                         pretrained=pretrained)
    else:
        raise ValueError(image_encoder)

    if text_encoder == "lstm":
        txt = TextEncoderBiLSTM(vocab_size, use_attention=text_attention)
    elif text_encoder == "transformer":
        txt = TextEncoderTransformer(vocab_size, use_attention=text_attention,
                                     max_len=max_len)
    else:
        raise ValueError(text_encoder)

    return FusionModel(img, txt, num_classes, decoder_attention=decoder_attention)
```

- [ ] **Step 4: Run — Expected: 5 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: FusionModel factory + GROUP_A full 8-combo registry"`

---

### Task 6: Metrics — evaluator dùng chung cho mọi model (kể cả Gemma cũ)

**Files:** Create: `src/train/__init__.py`, `src/train/metrics.py` · Test: `tests/test_metrics.py`

- [ ] **Step 1: Failing tests**

```python
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
    p_yes  = [0.9, 0.4, 0.2, 0.6]
    r = closed_binary_report(y_true, p_yes, threshold=0.5)
    assert r["sensitivity"] == 0.5                      # 1/2 yes bat duoc
    assert r["specificity"] == 0.5                      # 1/2 no dung
    assert 0 <= r["auc"] <= 1 and 0 <= r["precision"] <= 1


def test_masked_closed_argmax():
    logits = torch.tensor([[0.1, 0.2, 5.0, 4.0],
                           [3.0, 0.1, 0.2, 0.3]])
    preds = masked_closed_argmax(logits, yes_id=0, no_id=1)
    assert preds.tolist() == [1, 0]
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/train/metrics.py`** (kèm `__init__.py` rỗng)

```python
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
    """Cham cap string: EM overall/closed/open + token-F1 + BLEU-1 trung binh.

    refs/hyps: list cau tra loi (chua/da normalize deu duoc).
    """
    assert len(refs) == len(hyps)
    em_o = em_c = f1_sum = bleu_sum = 0.0
    n_closed = n_open = em_open_hits = 0
    for ref, hyp in zip(refs, hyps):
        ref_n, hyp_n = normalize_answer(ref), normalize_answer(hyp)
        hit = float(ref_n == hyp_n)
        em_o += hit
        if is_closed(ref_n):
            n_closed += 1
            em_c += hit
        else:
            n_open += 1
            em_open_hits += hit
        rt, ht = ref_n.split(), hyp_n.split()
        f1_sum += token_f1(rt, ht)
        bleu_sum += bleu1(rt, ht)
    n = len(refs)
    return {
        "em_overall": em_o / n if n else 0.0,
        "em_closed": em_c / n_closed if n_closed else 0.0,
        "em_open": em_open_hits / n_open if n_open else 0.0,
        "token_f1": f1_sum / n if n else 0.0,
        "bleu1": bleu_sum / n if n else 0.0,
        "n_closed": n_closed, "n_open": n_open,
    }


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

    CSV can cot: answer_ref, answer_pred. Tra ve dict nhu score_answers.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    return score_answers(df["answer_ref"].tolist(), df["answer_pred"].tolist())
```

- [ ] **Step 4: Run — Expected: 5 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: unified evaluator (EM/token-F1/BLEU-1/clinical) + CSV re-scorer for Gemma"`

---

### Task 7: SFT training engine

**Files:** Create: `src/train/engine.py` · Test: `tests/test_engine.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_engine.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_model
from src.train.engine import run_experiment


def _loader(n=16):
    imgs = torch.randn(n, 3, 128, 128)
    toks = torch.randint(1, 30, (n, 32))
    labels = torch.randint(0, 4, (n,))
    closed = labels < 2
    return DataLoader(TensorDataset(imgs, toks, labels, closed), batch_size=8)


def test_run_experiment_returns_history_and_saves(tmp_path):
    model = build_model(vocab_size=30, num_classes=4)
    result = run_experiment(
        name="unit", model=model,
        train_loader=_loader(), val_loader=_loader(),
        epochs=2, lr=1e-3, weight_decay=1e-4, patience=3,
        out_dir=str(tmp_path), device="cpu")
    assert len(result["history"]["train_loss"]) == 2
    assert (tmp_path / "unit_best.pt").exists()
    assert (tmp_path / "unit_history.csv").exists()
    assert result["params_total"] >= result["params_trainable"] > 0
    assert result["best_val_em"] >= 0.0
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/train/engine.py`**

```python
# src/train/engine.py
import csv
import os
import time

import torch
import torch.nn as nn

from .metrics import masked_closed_argmax


def _em_from_ids(preds, labels, closed):
    correct = (preds == labels) & (labels >= 0)

    def rate(mask):
        n = int(mask.sum())
        return float(correct[mask].sum()) / n if n else 0.0

    return {"em_overall": rate(torch.ones_like(closed)),
            "em_closed": rate(closed), "em_open": rate(~closed)}


def _epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, labels_all, closed_all = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for images, tokens, labels, closed in loader:
            images, tokens = images.to(device), tokens.to(device)
            labels = labels.to(device)
            logits = model(images, tokens)
            loss = criterion(logits, labels.clamp(min=0))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds.append(logits.argmax(1).cpu())
            labels_all.append(labels.cpu())
            closed_all.append(torch.as_tensor(closed, dtype=torch.bool))
    rep = _em_from_ids(torch.cat(preds), torch.cat(labels_all), torch.cat(closed_all))
    return total_loss / len(loader), rep


def run_experiment(name, model, train_loader, val_loader, epochs, lr,
                   weight_decay, patience, out_dir, device, class_weights=None):
    """SFT mot cau hinh: early stop theo val EM overall, luu best.pt + history CSV."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Chi optimize tham so trainable (backbone frozen bi loai tu dong)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ["train_loss", "train_em", "val_loss", "val_em",
                               "val_em_closed", "val_em_open", "epoch_sec"]}
    best_val_em, bad = -1.0, 0
    ckpt = os.path.join(out_dir, f"{name}_best.pt")

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr = _epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va = _epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_em"].append(tr["em_overall"])
        history["val_loss"].append(va_loss)
        history["val_em"].append(va["em_overall"])
        history["val_em_closed"].append(va["em_closed"])
        history["val_em_open"].append(va["em_open"])
        history["epoch_sec"].append(time.time() - t0)
        print(f"[{name}] {epoch+1}/{epochs} train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} val_em={va['em_overall']:.4f} "
              f"(closed={va['em_closed']:.4f} open={va['em_open']:.4f})")
        if va["em_overall"] > best_val_em:
            best_val_em, bad = va["em_overall"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop @ epoch {epoch+1}")
                break

    with open(os.path.join(out_dir, f"{name}_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(history))
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] for k in history])

    return {"name": name, "history": history, "best_val_em": best_val_em,
            "params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            "checkpoint": ckpt}


@torch.no_grad()
def predict_answers(model, loader, id2answer, device,
                    yes_id=None, no_id=None, constrained_closed=False):
    """Chay inference -> tra ve (hyp_strings, p_yes_closed) de cham bang metrics chung.

    hyp_strings: cau tra loi du doan (map class id -> string).
    p_yes_closed: xac suat lop yes cho cac cau closed (danh cho AUC).
    """
    model.to(device).eval()
    hyps, p_yes_list = [], []
    for images, tokens, labels, closed in loader:
        logits = model(images.to(device), tokens.to(device))
        preds = logits.argmax(1)
        closed_t = torch.as_tensor(closed, dtype=torch.bool)
        if constrained_closed and yes_id is not None:
            masked = masked_closed_argmax(logits, yes_id, no_id)
            preds = torch.where(closed_t.to(device), masked, preds)
        hyps += [id2answer[int(i)] for i in preds.cpu()]
        if yes_id is not None:
            p = torch.softmax(logits[:, [yes_id, no_id]], dim=1)[:, 0]
            p_yes_list += p.cpu().tolist()
    return hyps, p_yes_list
```

- [ ] **Step 4: Run — Expected: 1 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: SFT engine with frozen-aware optimizer and string prediction"`

---

### Task 8: REINFORCE self-critical (SFT → RL, feedback #5)

**Files:** Create: `src/train/rl.py` · Test: `tests/test_rl.py`

Chỉ chạy SAU khi SFT xong (nhóm D). Reward không khả vi (EM, token-F1) được tối ưu bằng policy gradient: sample answer từ softmax, advantage = reward(sample) − reward(greedy) (self-critical baseline, không cần value network).

- [ ] **Step 1: Failing test**

```python
# tests/test_rl.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_model
from src.train.rl import answer_reward, scst_epoch


def test_answer_reward_range():
    id2answer = {0: "yes", 1: "no", 2: "right lung"}
    r = answer_reward(torch.tensor([0, 2]), torch.tensor([0, 1]), id2answer)
    assert r[0] == 1.0                       # exact match
    assert 0.0 <= r[1] < 1.0                 # sai -> reward thap
    assert r.shape == (2,)


def test_scst_epoch_updates_and_returns_reward(tmp_path):
    imgs = torch.randn(16, 3, 128, 128)
    toks = torch.randint(1, 30, (16, 32))
    labels = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(imgs, toks, labels, labels < 2), batch_size=8)
    model = build_model(vocab_size=30, num_classes=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    id2answer = {0: "yes", 1: "no", 2: "right lung", 3: "axial"}
    stats = scst_epoch(model, loader, opt, id2answer, device="cpu")
    assert "mean_reward" in stats and "loss" in stats
    assert 0.0 <= stats["mean_reward"] <= 1.0
```

- [ ] **Step 2: Run — Expected: FAIL ImportError.**
- [ ] **Step 3: Implement `src/train/rl.py`**

```python
# src/train/rl.py
"""REINFORCE self-critical (SCST) fine-tune sau SFT.

reward = 0.5*EM + 0.5*token-F1 giua answer sample va answer dung.
advantage = reward(sample) - reward(greedy)  -> giam variance, khong can critic.
"""
import torch
from torch.distributions import Categorical

from ..data.vqa_rad import normalize_answer
from .metrics import token_f1


def answer_reward(pred_ids, label_ids, id2answer):
    """Reward tren string; label -1 (unseen) cho reward 0."""
    rewards = []
    for p, l in zip(pred_ids.tolist(), label_ids.tolist()):
        if l < 0:
            rewards.append(0.0)
            continue
        hyp = normalize_answer(id2answer[p])
        ref = normalize_answer(id2answer[l])
        em = float(hyp == ref)
        f1 = token_f1(ref.split(), hyp.split())
        rewards.append(0.5 * em + 0.5 * f1)
    return torch.tensor(rewards)


def scst_epoch(model, loader, optimizer, id2answer, device):
    model.to(device).train()
    total_loss = total_reward = n_batches = 0.0
    for images, tokens, labels, _ in loader:
        images, tokens = images.to(device), tokens.to(device)
        logits = model(images, tokens)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        greedy = logits.argmax(1)
        r_sample = answer_reward(sample.cpu(), labels, id2answer).to(device)
        r_greedy = answer_reward(greedy.cpu(), labels, id2answer).to(device)
        advantage = r_sample - r_greedy
        loss = -(dist.log_prob(sample) * advantage).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_reward += r_sample.mean().item()
        n_batches += 1
    return {"loss": total_loss / n_batches, "mean_reward": total_reward / n_batches}


def scst_finetune(model, train_loader, val_eval_fn, epochs, lr, id2answer,
                  device, out_dir, name="D1_scst"):
    """Chay SCST nhieu epoch; giu checkpoint co val EM cao nhat (val_eval_fn tra EM)."""
    import os
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_em = val_eval_fn(model)
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[{name}] val EM truoc RL: {best_em:.4f}")
    for ep in range(epochs):
        stats = scst_epoch(model, train_loader, optimizer, id2answer, device)
        em = val_eval_fn(model)
        print(f"[{name}] RL epoch {ep+1}/{epochs} loss={stats['loss']:.4f} "
              f"reward={stats['mean_reward']:.4f} val_em={em:.4f}")
        if em > best_em:
            best_em = em
            torch.save(model.state_dict(), ckpt)
    return {"best_val_em": best_em, "checkpoint": ckpt}
```

- [ ] **Step 4: Run — Expected: 2 passed.**
- [ ] **Step 5: Commit** `git commit -m "feat: SCST/REINFORCE fine-tune after SFT"`

---

### Task 9: Notebook runner — pytest → pilot → A → B → C → D → chấm lại Gemma → tổng hợp

**Files:** Create: `notebook/ablation_vqarad.ipynb`

Cell theo thứ tự (mỗi cell là một bước có thể chạy lại độc lập):

- [ ] **Step 1: Cell 1 — setup + CHẠY PYTEST (verify toàn bộ src/ trước khi train)**

```python
# !pip install -q datasets
import subprocess, sys
sys.path.insert(0, "/kaggle/input/medvqa-src")   # hoac repo da clone
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                   capture_output=True, text=True)
print(r.stdout[-3000:]); assert r.returncode == 0, "TESTS FAIL - dung lai, khong train"

import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset
ds = load_dataset("flaviagiammarino/vqa-rad")
print(ds)   # train 1793 / test 451
```

- [ ] **Step 2: Cell 2 — vocab + split image-disjoint (chỉ từ train)**

```python
from src.data.vqa_rad import (build_answer_vocab, build_question_vocab,
                              image_key, group_split, normalize_answer)

train_recs = [dict(r) for r in ds["train"]]
test_recs  = [dict(r) for r in ds["test"]]
keys = [image_key(r["image"]) for r in train_recs]
tr_idx, va_idx = group_split(keys, val_ratio=0.1, seed=SEED)
tr = [train_recs[i] for i in tr_idx]
va = [train_recs[i] for i in va_idx]

answer2id = build_answer_vocab([r["answer"] for r in tr])
id2answer = {i: a for a, i in answer2id.items()}
q_vocab   = build_question_vocab([r["question"] for r in tr])
NUM_CLASSES, VOCAB_SIZE = len(answer2id), len(q_vocab)
yes_id, no_id = answer2id["yes"], answer2id["no"]

unseen = sum(normalize_answer(r["answer"]) not in answer2id for r in test_recs)
print(f"train={len(tr)} val={len(va)} test={len(test_recs)} "
      f"C={NUM_CLASSES} V={VOCAB_SIZE} | unseen test answers: "
      f"{unseen}/{len(test_recs)} = {unseen/len(test_recs):.1%}  <- tran EM cua phuong phap")
```

- [ ] **Step 3: Cell 3 — dataloaders (2 bộ: 128-norm0.5 cho A/B/D, 224-ImageNet cho C)**

```python
from torch.utils.data import DataLoader
from src.data.dataset import VQARADClsDataset, default_transform

MAX_LEN, BATCH = 32, 32
def loaders(size, imagenet):
    mk = lambda recs, train: VQARADClsDataset(
        recs, answer2id, q_vocab, default_transform(size, train, imagenet), MAX_LEN)
    return (DataLoader(mk(tr, True),  BATCH, shuffle=True,  num_workers=2),
            DataLoader(mk(va, False), BATCH, shuffle=False, num_workers=2),
            DataLoader(mk(test_recs, False), BATCH, shuffle=False))

train_128, val_128, test_128 = loaders(128, imagenet=False)   # nhom A, B, D
train_224, val_224, test_224 = loaders(224, imagenet=True)    # nhom C (ResNet frozen)
```

- [ ] **Step 4: Cell 4 — pilot + overfit sanity (bắt buộc pass trước khi chạy 13 run)**

```python
from src.models.fusion import build_model, GROUP_A
from src.train.engine import run_experiment
import torch.utils.data as tud

torch.manual_seed(SEED)
pilot = run_experiment("pilot", build_model(VOCAB_SIZE, NUM_CLASSES),
                       train_128, val_128, epochs=2, lr=1e-3, weight_decay=1e-4,
                       patience=3, out_dir="runs", device=device)
assert pilot["history"]["train_loss"][-1] < pilot["history"]["train_loss"][0]

tiny_ds = tud.Subset(val_128.dataset, range(64))
tiny = DataLoader(tiny_ds, batch_size=16, shuffle=True)
torch.manual_seed(SEED)
ov = run_experiment("overfit_check", build_model(VOCAB_SIZE, NUM_CLASSES),
                    tiny, tiny, epochs=60, lr=1e-3, weight_decay=0.0,
                    patience=60, out_dir="runs", device=device)
assert ov["history"]["train_em"][-1] > 0.9, "khong overfit duoc 64 mau -> bug pipeline"
```

- [ ] **Step 5: Cell 5 — NHÓM A: đủ 8 tổ hợp attention**

```python
results = {}
for name, flags in GROUP_A.items():
    torch.manual_seed(SEED)
    m = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **flags)
    results[name] = run_experiment(name, m, train_128, val_128, epochs=12,
                                   lr=1e-3, weight_decay=1e-4, patience=3,
                                   out_dir="runs", device=device)

best_A = max(GROUP_A, key=lambda k: results[k]["best_val_em"])
best_flags = GROUP_A[best_A]
print("Best nhom A theo val EM:", best_A, best_flags)
```

- [ ] **Step 6: Cell 6 — NHÓM B: Transformer text encoder (2 pooling)**

```python
GROUP_B = {
    "B1_trans_mean": dict(best_flags, text_encoder="transformer", text_attention=False),
    "B2_trans_attn": dict(best_flags, text_encoder="transformer", text_attention=True),
}
for name, kw in GROUP_B.items():
    torch.manual_seed(SEED)
    m = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **kw)
    results[name] = run_experiment(name, m, train_128, val_128, epochs=12,
                                   lr=1e-3, weight_decay=1e-4, patience=3,
                                   out_dir="runs", device=device)
```

- [ ] **Step 7: Cell 7 — NHÓM C: ResNet-18 frozen (train text + fusion)**

```python
best_B = max(GROUP_B, key=lambda k: results[k]["best_val_em"])
use_transformer = results[best_B]["best_val_em"] > results[best_A]["best_val_em"]
best_text = "transformer" if use_transformer else "lstm"
best_text_attn = (GROUP_B[best_B]["text_attention"] if use_transformer
                  else best_flags["text_attention"])
GROUP_C = {
    "C1_resnet_lstm": dict(best_flags, image_encoder="resnet18_frozen",
                           text_encoder="lstm"),
    "C2_resnet_best": dict(best_flags, image_encoder="resnet18_frozen",
                           text_encoder=best_text, text_attention=best_text_attn),
}
for name, kw in GROUP_C.items():
    torch.manual_seed(SEED)
    m = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **kw)
    results[name] = run_experiment(name, m, train_224, val_224, epochs=12,
                                   lr=1e-3, weight_decay=1e-4, patience=3,
                                   out_dir="runs", device=device)
```

- [ ] **Step 8: Cell 8 — NHÓM D: SFT → RL (SCST) trên checkpoint tốt nhất toàn cục**

```python
from src.train.rl import scst_finetune
from src.train.engine import predict_answers
from src.train.metrics import score_answers

best_global = max(results, key=lambda k: results[k]["best_val_em"])
cfg = (GROUP_A | GROUP_B | GROUP_C)[best_global] if best_global in {**GROUP_B, **GROUP_C} \
      else GROUP_A[best_global]
is_resnet = cfg.get("image_encoder") == "resnet18_frozen"
tr_ld, va_ld = (train_224, val_224) if is_resnet else (train_128, val_128)

model = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **cfg)
model.load_state_dict(torch.load(results[best_global]["checkpoint"]))

va_refs = [normalize_answer(r["answer"]) for r in va]
def val_em(m):
    hyps, _ = predict_answers(m, va_ld, id2answer, device)
    return score_answers(va_refs, hyps)["em_overall"]

rl_result = scst_finetune(model, tr_ld, val_em, epochs=5, lr=1e-5,
                          id2answer=id2answer, device=device, out_dir="runs")
print("SFT best:", results[best_global]["best_val_em"], "-> sau RL:", rl_result["best_val_em"])
```

- [ ] **Step 9: Cell 9 — Test MỘT LẦN + chấm lại Gemma bằng evaluator chung**

```python
import pandas as pd
from src.train.metrics import closed_binary_report, score_predictions_csv

test_refs = [normalize_answer(r["answer"]) for r in test_recs]
rows = []
def eval_run(name, cfg, ckpt, test_ld):
    m = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **cfg)
    m.load_state_dict(torch.load(ckpt))
    hyps, p_yes = predict_answers(m, test_ld, id2answer, device, yes_id, no_id)
    hyps_c, _ = predict_answers(m, test_ld, id2answer, device, yes_id, no_id,
                                constrained_closed=True)
    s = score_answers(test_refs, hyps)
    sc = score_answers(test_refs, hyps_c)
    closed_idx = [i for i, r in enumerate(test_refs) if r in ("yes", "no")]
    bin_rep = closed_binary_report(
        [int(test_refs[i] == "yes") for i in closed_idx],
        [p_yes[i] for i in closed_idx])
    rows.append(dict(model=name, **s,
                     em_closed_constrained=sc["em_closed"],
                     **{f"clin_{k}": v for k, v in bin_rep.items()},
                     params_total=results.get(name, {}).get("params_total"),
                     params_trainable=results.get(name, {}).get("params_trainable")))

all_cfgs = {**GROUP_A, **GROUP_B, **GROUP_C}
for name in results:
    if name in ("pilot", "overfit_check"):
        continue
    cfg = all_cfgs[name]
    test_ld = test_224 if cfg.get("image_encoder") == "resnet18_frozen" else test_128
    eval_run(name, cfg, results[name]["checkpoint"], test_ld)
eval_run("D1_scst", cfg, rl_result["checkpoint"],
         test_224 if is_resnet else test_128)

# Cham lai prediction Gemma cu (export tu v3-vqa-*.ipynb ra CSV: answer_ref, answer_pred)
for gname, path in [("gemma-2b-lora", "gemma_preds/gemma2b_test.csv"),
                    ("gemma-4b-lora", "gemma_preds/gemma4b_test.csv")]:
    try:
        rows.append(dict(model=gname, **score_predictions_csv(path)))
    except FileNotFoundError:
        print(f"(chua co {path} - xuat prediction tu notebook Gemma roi cham lai sau)")

df = pd.DataFrame(rows)
df.to_csv("runs/metrics_summary.csv", index=False)
df.sort_values("em_overall", ascending=False)
```

- [ ] **Step 10: Cell 10 — phân tích: curves, bảng delta vs baseline, visualize attention**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for name in GROUP_A:
    axes[0].plot(results[name]["history"]["val_loss"], label=name)
    axes[1].plot(results[name]["history"]["val_em"], label=name)
axes[0].set_title("Val Loss - nhom A (8 to hop attention)")
axes[1].set_title("Val EM - nhom A")
for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
plt.savefig("runs/groupA_curves.png", dpi=150, bbox_inches="tight"); plt.show()

base_em = df.loc[df.model == "A1_000", "em_overall"].iloc[0]
df["delta_vs_baseline"] = df["em_overall"] - base_em

# Visualize temporal attention: cau hoi + trong so tung tu (model co text_attention)
from src.data.vqa_rad import encode_question, tokenize_question
best_m = build_model(VOCAB_SIZE, NUM_CLASSES, max_len=MAX_LEN, **all_cfgs[best_global])
best_m.load_state_dict(torch.load(results[best_global]["checkpoint"]))
best_m.to(device).eval()
if getattr(best_m.text, "use_attention", False):
    sample = test_recs[0]
    toks = torch.tensor([encode_question(sample["question"], q_vocab, MAX_LEN)]).to(device)
    emb_out, _ = best_m.text.lstm(best_m.text.embedding(toks)) \
        if hasattr(best_m.text, "lstm") else (None, None)
    if emb_out is not None:
        _, w = best_m.text.attention(emb_out, toks == 0)
        words = tokenize_question(sample["question"])
        for word, weight in zip(words, w[0][:len(words)].tolist()):
            print(f"{word:15s} {'#' * int(weight * 60)} {weight:.3f}")
```

- [ ] **Step 11: Commit notebook** `git commit -m "feat: runner notebook groups A/B/C/D + unified Gemma re-scoring"`

---

### Task 10: Xuất prediction Gemma cũ để chấm đồng bộ

**Files:** Modify: cuối `notebook/v3-vqa-2b-it.ipynb` và `v3-vqa-4b-it.ipynb` (thêm 1 cell)

- [ ] **Step 1: Thêm cell export vào cuối mỗi notebook Gemma** (user chạy khi có GPU):

```python
# Xuat prediction test-set ra CSV de cham dong bo bang src/train/metrics.py
import pandas as pd
rows = []
for sample in test_dataset:           # loop eval co san trong notebook
    pred = generate_answer(sample)    # ham inference co san cua notebook
    rows.append(dict(question=sample["question"], answer_ref=sample["answer"],
                     answer_pred=pred,
                     question_type="closed" if str(sample["answer"]).lower().strip(". ")
                                   in ("yes", "no") else "open"))
pd.DataFrame(rows).to_csv("gemma2b_test.csv", index=False)   # 4b: gemma4b_test.csv
```

(Điều chỉnh tên biến theo đúng notebook thật khi mở ra — cell eval đã có sẵn loop prediction; chỉ cần gom kết quả vào list thay vì chỉ print.)

- [ ] **Step 2: Đặt 2 file CSV vào `gemma_preds/` cạnh notebook runner** → Cell 9 tự chấm lại.

---

### Task 11: Cập nhật ai-memory sau khi có kết quả

- [ ] Điền bảng metric 13 runs + Gemma (đồng bộ) vào `ai-memory/active-context.md`, tick `tasks.md`, ghi phân tích vào `handoff.md`: (1) attention đặt đâu lợi nhất & 8 tổ hợp nói gì, (2) LSTM vs Transformer, (3) frozen ResNet-18 có đáng không, (4) RL cải thiện bao nhiêu so với SFT.
- [ ] Cập nhật bảng kết quả trong `html_report/index.html`.
- [ ] Commit: `git commit -m "docs: record ablation v2 results"`

---

## Rủi ro & tiêu chí dừng

- **13 runs > quota?** Ưu tiên cắt theo thứ tự: giữ nguyên A (8 run, cốt lõi); B/C mỗi nhóm tối thiểu 1 run; D có thể hoãn. Không cắt bớt trong nhóm A.
- **RL không cải thiện / bất ổn:** chấp nhận báo cáo kết quả âm tính — vẫn là finding hợp lệ (REINFORCE variance cao trên reward thưa); giữ checkpoint SFT làm final.
- **EM Open thấp:** trần phương pháp = % answer test ngoài train vocab (Cell 2 in ra) — ghi vào limitations.
- **Không tune theo test; không đổi kiến trúc giữa chừng; best-config chọn theo val EM.**
- **Session đứt:** mỗi run tự lưu checkpoint + CSV ngay; các cell 5–8 có thể chạy lại từng nhóm độc lập nếu giữ `runs/`.
