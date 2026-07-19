# src/models/qformer.py
"""Q-Former nhe (theo tinh than BLIP-2) dung lam projector anh->LLM.

Thay cho `Linear(192->768)` nen toan bo anh+chu thanh 1 token, module nay giu
lai CHUOI patch token cua CLIP-ViT va dung K learnable query + cross-attention
de rut thanh K visual token (giu duoc chi tiet khong gian anh).

Input : patch tokens [B, N, in_dim]  (CLIP last_hidden_state, in_dim=768)
Output: visual tokens [B, K, out_dim] (out_dim=768 = chieu cua GPT-2)
"""
import torch
import torch.nn as nn


class QFormerProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, n_query=16, n_layer=2,
                 n_head=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.n_query = n_query
        # K learnable query token
        self.query = nn.Parameter(torch.randn(1, n_query, out_dim) * 0.02)
        # chieu patch -> chieu LLM (neu khac)
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict({
                # query attend len patch token (cross-attention)
                "cross_attn": nn.MultiheadAttention(out_dim, n_head, dropout=dropout,
                                                    batch_first=True),
                "ln1": nn.LayerNorm(out_dim),
                "ffn": nn.Sequential(
                    nn.Linear(out_dim, ffn_dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(ffn_dim, out_dim)),
                "ln2": nn.LayerNorm(out_dim),
            }))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, patches):
        # patches: [B, N, in_dim]
        B = patches.size(0)
        kv = self.in_proj(patches)                    # [B, N, out_dim]
        q = self.query.expand(B, -1, -1)              # [B, K, out_dim]
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](q, kv, kv)   # query <- patch
            q = layer["ln1"](q + attn_out)                 # residual + LN
            ffn_out = layer["ffn"](q)
            q = layer["ln2"](q + ffn_out)                  # residual + LN
        return self.norm(q)                           # [B, K, out_dim]
