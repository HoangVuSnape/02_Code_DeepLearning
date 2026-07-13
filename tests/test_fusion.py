# tests/test_fusion.py
import torch
from src.models.fusion import build_model


def test_mlp_classification_shapes():
    model = build_model(vocab_size=50, num_classes=10, decoder="mlp")
    images = torch.randn(2, 3, 128, 128)
    tokens = torch.randint(1, 50, (2, 32))
    
    logits = model(images, tokens)
    assert logits.shape == (2, 10)
    
    preds = model.generate(images, tokens, max_len=16)
    assert preds.shape == (2, 1)


def test_generative_gru_decoder_shapes():
    model = build_model(vocab_size=50, num_classes=10, decoder="gru")
    images = torch.randn(2, 3, 128, 128)
    tokens = torch.randint(1, 50, (2, 32))
    dec_in = torch.randint(1, 50, (2, 16))
    
    logits = model(images, tokens, dec_in)
    assert logits.shape == (2, 16, 50)
    
    preds = model.generate(images, tokens, max_len=16)
    assert preds.shape == (2, 16)


def test_generative_lstm_decoder_shapes():
    model = build_model(vocab_size=50, num_classes=10, decoder="lstm")
    images = torch.randn(2, 3, 128, 128)
    tokens = torch.randint(1, 50, (2, 32))
    dec_in = torch.randint(1, 50, (2, 16))
    
    logits = model(images, tokens, dec_in)
    assert logits.shape == (2, 16, 50)
    
    preds = model.generate(images, tokens, max_len=16)
    assert preds.shape == (2, 16)


def test_generative_transformer_decoder_shapes():
    model = build_model(vocab_size=50, num_classes=10, decoder="transformer")
    images = torch.randn(2, 3, 128, 128)
    tokens = torch.randint(1, 50, (2, 32))
    dec_in = torch.randint(1, 50, (2, 16))
    
    logits = model(images, tokens, dec_in)
    assert logits.shape == (2, 16, 50)
    
    preds = model.generate(images, tokens, max_len=16)
    assert preds.shape == (2, 16)


def test_resnet_frozen_variant_trainable_params_small():
    model = build_model(50, 10, image_encoder="resnet18_frozen", pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < total                      # backbone bi dong bang
    
    logits = model(torch.randn(2, 3, 224, 224), torch.randint(1, 50, (2, 32)))
    assert logits.shape == (2, 10)
