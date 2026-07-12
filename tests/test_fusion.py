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
