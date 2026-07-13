# tests/test_dataset.py
import torch
from PIL import Image

from src.data.dataset import VQARADClsDataset, default_transform
from src.data.vqa_rad import build_answer_vocab, build_question_vocab


def _fake_records(n=4):
    img = Image.new("L", (64, 80), color=128)   # grayscale, size lech chuan
    answers = ["yes", "no", "right lung", "yes"]
    return [{"image": img, "question": f"is this q {i}", "answer": answers[i]}
            for i in range(n)]


def test_dataset_shapes_and_types():
    recs = _fake_records()
    a2i = build_answer_vocab([r["answer"] for r in recs])
    qv = build_question_vocab([r["question"] for r in recs])
    ds = VQARADClsDataset(recs, a2i, qv, default_transform(128, train=False), max_len=16)
    img, tokens, label, closed, dec_in, dec_tgt = ds[0]
    assert img.shape == (3, 128, 128)           # grayscale -> 3 kenh, resize 128
    assert tokens.shape == (16,) and tokens.dtype == torch.long
    assert label == a2i["yes"] and closed is True
    assert dec_in.shape == (16,) and dec_tgt.shape == (16,)


def test_imagenet_transform_size_224():
    recs = _fake_records()
    a2i = build_answer_vocab([r["answer"] for r in recs])
    qv = build_question_vocab([r["question"] for r in recs])
    tf = default_transform(224, train=False, imagenet=True)   # cho ResNet-18 frozen
    ds = VQARADClsDataset(recs, a2i, qv, tf)
    assert ds[0][0].shape == (3, 224, 224)


def test_unseen_answer_gets_minus_one():
    recs = _fake_records()
    a2i = build_answer_vocab(["yes", "no"])      # vocab KHONG chua "right lung"
    qv = build_question_vocab([r["question"] for r in recs])
    ds = VQARADClsDataset(recs, a2i, qv, default_transform(128, train=False))
    _, _, label, closed, _, _ = ds[2]
    assert label == -1 and closed is False       # unseen answer -> khong bao gio dung
