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
    assert torch.allclose(w[0, 2:], torch.zeros(3), atol=1e-6)  # PAD nhan trong so 0
    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)


def test_gated_attention_preserves_shape():
    x = torch.randn(4, 48)
    assert GatedAttention(48)(x).shape == x.shape
