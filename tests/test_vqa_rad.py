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
    assert v["yes"] == 0            # tan suat cao nhat dung dau
    assert set(v) == {"yes", "no", "axial"}
    assert v == build_answer_vocab(["yes", "no", "yes", "axial"])


def test_tokenize_and_encode_question():
    vocab = build_question_vocab(["Is this an MRI?"])
    ids = encode_question("Is this an MRI?", vocab, max_len=8)
    assert len(ids) == 8 and ids[0] != 0        # co noi dung + pad ve sau
    assert encode_question("unseen words", vocab, max_len=4)[0] == 1  # <unk>


def test_group_split_image_disjoint():
    keys = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    tr, va = group_split(keys, val_ratio=0.3, seed=42)
    assert {keys[i] for i in tr}.isdisjoint({keys[i] for i in va})
    assert sorted(tr + va) == list(range(len(keys)))
