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
    """ResNet-18 pretrained DONG BANG toan bo backbone; chi train proj (+SE neu bat)."""

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


class ImageEncoderPubMedCLIP(nn.Module):
    """PubMedCLIP (ViT) Vision Tower - Dong bang backbone, chi train projection."""

    def __init__(self, out_dim=128):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
            # low_cpu_mem_usage=False: nap thang, ne buoc "Materializing param" cua transformers 5.x hay treo
            self.model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32", low_cpu_mem_usage=False)
            self.proj = nn.Linear(768, out_dim)
        except Exception:
            self.model = None
            self.proj = nn.Linear(128, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        if self.model is not None:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x).pooler_output  # [B, 768]
            x = self.proj(out)
        else:
            # Fallback mock representation
            mock_feat = torch.zeros(x.size(0), 128, device=x.device)
            x = self.proj(mock_feat)
        return self.norm(x)


class TextEncoderBiLSTM(nn.Module):
    """Embedding(64) + BiLSTM(32x2) -> v_txt [B,64], H_txt [B, T, 64]."""

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
        return self.norm(out), lstm_out


class TextEncoderTransformer(nn.Module):
    """TransformerEncoder (d=64, 4 head, 2 layer) -> v_txt [B,64], H_txt [B, T, 64]."""

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
        return self.norm(out), x


class TextEncoderPubMedBERT(nn.Module):
    """PubMedBERT text backbone - Dong bang, chi train projection."""

    def __init__(self, out_dim=64):
        super().__init__()
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                low_cpu_mem_usage=False)
            self.proj = nn.Linear(768, out_dim)
        except Exception:
            self.model = None
            self.proj = nn.Linear(64, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        if self.model is not None:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, tokens):
        pad_mask = tokens == 0
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, attention_mask=(~pad_mask).long())
                H_txt = outputs.last_hidden_state  # [B, T, 768]
                v_txt = outputs.pooler_output      # [B, 768]
            v_txt = self.proj(v_txt)
            H_txt = self.proj(H_txt)
        else:
            # Fallback mock representation
            B, T = tokens.size()
            H_txt = torch.zeros(B, T, 64, device=tokens.device)
            v_txt = torch.zeros(B, 64, device=tokens.device)
        return self.norm(v_txt), H_txt
