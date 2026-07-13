# src/models/decoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim=128):
        super().__init__()
        self.W_q = nn.Linear(query_dim, attn_dim)
        self.W_k = nn.Linear(key_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)

    def forward(self, query, keys, key_padding_mask=None):
        # query: [B, query_dim]
        # keys: [B, T, key_dim]
        q_proj = self.W_q(query).unsqueeze(1)  # [B, 1, attn_dim]
        k_proj = self.W_k(keys)                # [B, T, attn_dim]
        scores = self.V(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [B, T]

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # [B, T]
        context = (keys * weights.unsqueeze(-1)).sum(dim=1)  # [B, key_dim]
        return context, weights


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, encoder_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.cell = nn.GRUCell(emb_dim + encoder_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, dec_input, initial_state, encoder_states, pad_mask=None):
        # dec_input: [B, S_len] (token IDs starting with <bos>)
        # initial_state: [B, hidden_dim]
        # encoder_states: [B, T_len, encoder_dim]
        batch_size, seq_len = dec_input.size()
        h = initial_state
        logits_seq = []

        for t in range(seq_len):
            x = self.embedding(dec_input[:, t])  # [B, emb_dim]
            context, _ = self.attention(h, encoder_states, pad_mask)  # [B, encoder_dim]
            rnn_in = torch.cat([x, context], dim=-1)  # [B, emb_dim + encoder_dim]
            h = self.cell(rnn_in, h)  # [B, hidden_dim]
            logits = self.fc(h)  # [B, vocab_size]
            logits_seq.append(logits.unsqueeze(1))

        return torch.cat(logits_seq, dim=1)  # [B, S_len, vocab_size]

    def generate(self, initial_state, encoder_states, max_len=16, pad_mask=None, bos_idx=2, eos_idx=3):
        batch_size = initial_state.size(0)
        h = initial_state
        device = initial_state.device
        
        # Start with <bos> token
        curr_token = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)
        generated = []
        
        for _ in range(max_len):
            x = self.embedding(curr_token)  # [B, emb_dim]
            context, _ = self.attention(h, encoder_states, pad_mask)  # [B, encoder_dim]
            rnn_in = torch.cat([x, context], dim=-1)
            h = self.cell(rnn_in, h)
            logits = self.fc(h)  # [B, vocab_size]
            curr_token = logits.argmax(dim=-1)  # [B]
            generated.append(curr_token.unsqueeze(1))
            
        return torch.cat(generated, dim=1)  # [B, max_len]


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, encoder_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.cell = nn.LSTMCell(emb_dim + encoder_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, dec_input, initial_state, encoder_states, pad_mask=None):
        batch_size, seq_len = dec_input.size()
        h = initial_state
        c = torch.zeros_like(h)  # LSTM cell state
        logits_seq = []

        for t in range(seq_len):
            x = self.embedding(dec_input[:, t])
            context, _ = self.attention(h, encoder_states, pad_mask)
            rnn_in = torch.cat([x, context], dim=-1)
            h, c = self.cell(rnn_in, (h, c))
            logits = self.fc(h)
            logits_seq.append(logits.unsqueeze(1))

        return torch.cat(logits_seq, dim=1)

    def generate(self, initial_state, encoder_states, max_len=16, pad_mask=None, bos_idx=2, eos_idx=3):
        batch_size = initial_state.size(0)
        h = initial_state
        c = torch.zeros_like(h)
        device = initial_state.device
        
        curr_token = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)
        generated = []
        
        for _ in range(max_len):
            x = self.embedding(curr_token)
            context, _ = self.attention(h, encoder_states, pad_mask)
            rnn_in = torch.cat([x, context], dim=-1)
            h, c = self.cell(rnn_in, (h, c))
            logits = self.fc(h)
            curr_token = logits.argmax(dim=-1)
            generated.append(curr_token.unsqueeze(1))
            
        return torch.cat(generated, dim=1)


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, encoder_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.proj_enc = nn.Linear(encoder_dim, emb_dim) if encoder_dim != emb_dim else nn.Identity()
        self.proj_hidden = nn.Linear(hidden_dim, emb_dim) if hidden_dim != emb_dim else nn.Identity()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, dec_input, initial_state, encoder_states, pad_mask=None):
        # dec_input: [B, S_len]
        # initial_state: [B, hidden_dim]
        # encoder_states: [B, T_len, encoder_dim]
        
        # Prepare targets and memory
        tgt = self.embedding(dec_input)  # [B, S_len, emb_dim]
        memory = self.proj_enc(encoder_states)  # [B, T_len, emb_dim]
        
        # Generate causal mask for decoder
        seq_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=dec_input.device)
        
        # Run transformer decoder
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=pad_mask)  # [B, S_len, emb_dim]
        return self.fc(out)  # [B, S_len, vocab_size]

    def generate(self, initial_state, encoder_states, max_len=16, pad_mask=None, bos_idx=2, eos_idx=3):
        batch_size = initial_state.size(0)
        device = initial_state.device
        
        # Autoregressive generation
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        memory = self.proj_enc(encoder_states)
        
        for _ in range(max_len):
            tgt = self.embedding(generated)  # [B, curr_len, emb_dim]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=pad_mask)
            logits = self.fc(out[:, -1, :])  # [B, vocab_size]
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            
        # Return generated sequence excluding the initial <bos>
        return generated[:, 1:]
