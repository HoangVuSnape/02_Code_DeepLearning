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
