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
                decoder="mlp", image_attention=False, text_attention=False,
                decoder_attention=False, pretrained=True, max_len=32):
    """Factory dung chung cho ca 4 nhom thi nghiem A/B/C/D/E."""
    if decoder != "mlp":
        raise ValueError(f"Unsupported decoder: {decoder}")
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
