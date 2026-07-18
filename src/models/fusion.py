# src/models/fusion.py
import torch
import torch.nn as nn

from .attention import GatedAttention
from .encoders import (ImageEncoderCNN, ImageEncoderResNet18Frozen, ImageEncoderPubMedCLIP,
                       TextEncoderBiLSTM, TextEncoderTransformer, TextEncoderPubMedBERT)
from .decoders import GRUDecoder, LSTMDecoder, TransformerDecoderWrapper


class FusionModel(nn.Module):
    """Unified VQA Model: supports classification (MLP) and generation (GRU, LSTM, Transformer, GPT2)."""

    def __init__(self, image_enc, text_enc, vocab_size, num_classes, decoder_type="mlp",
                 decoder_attention=False):
        super().__init__()
        self.image = image_enc
        self.text = text_enc
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        
        fused_dim = 128 + 64
        
        if decoder_type == "mlp":
            self.decoder_attention = GatedAttention(fused_dim) if decoder_attention else None
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, num_classes))
        elif decoder_type == "gpt2":
            self.gpt2_proj = nn.Linear(fused_dim, 768)
            try:
                from transformers import GPT2LMHeadModel
                # low_cpu_mem_usage=False -> tranh vong "Materializing param..." treo tren Kaggle
                self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", low_cpu_mem_usage=False)
                # Resize token embeddings to vocab_size
                self.gpt2.resize_token_embeddings(vocab_size)
            except Exception:
                self.gpt2 = None
        else:
            self.init_decoder = nn.Linear(fused_dim, 128)
            if decoder_type == "gru":
                self.decoder = GRUDecoder(vocab_size=vocab_size)
            elif decoder_type == "lstm":
                self.decoder = LSTMDecoder(vocab_size=vocab_size)
            elif decoder_type == "transformer":
                self.decoder = TransformerDecoderWrapper(vocab_size=vocab_size)
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(self, images, question_tokens, dec_input=None):
        v_img = self.image(images)  # [B, 128]
        v_txt, H_txt = self.text(question_tokens)  # v_txt: [B, 64], H_txt: [B, T, 64]
        
        if self.decoder_type == "mlp":
            fused = torch.cat([v_img, v_txt], dim=1)
            if self.decoder_attention is not None:
                fused = self.decoder_attention(fused)
            return self.classifier(fused)
            
        elif self.decoder_type == "gpt2":
            fused = torch.cat([v_img, v_txt], dim=1)  # [B, 192]
            prefix = self.gpt2_proj(fused).unsqueeze(1)  # [B, 1, 768]
            
            if self.gpt2 is not None:
                tgt_emb = self.gpt2.transformer.wte(dec_input)  # [B, S_len, 768]
                inputs_embeds = torch.cat([prefix, tgt_emb], dim=1)  # [B, 1 + S_len, 768]
                outputs = self.gpt2(inputs_embeds=inputs_embeds)
                return outputs.logits[:, 1:, :]  # [B, S_len, vocab_size]
            else:
                B, S_len = dec_input.size()
                return torch.zeros(B, S_len, self.vocab_size, device=dec_input.device)
                
        else:
            fused = torch.cat([v_img, v_txt], dim=1)
            h0 = self.init_decoder(fused)
            pad_mask = question_tokens == 0
            return self.decoder(dec_input, h0, H_txt, pad_mask=pad_mask)

    def generate(self, images, question_tokens, max_len=16, bos_idx=2, eos_idx=3):
        self.eval()
        with torch.no_grad():
            v_img = self.image(images)
            v_txt, H_txt = self.text(question_tokens)
            
            if self.decoder_type == "mlp":
                fused = torch.cat([v_img, v_txt], dim=1)
                logits = self.classifier(fused)
                return logits.argmax(dim=-1, keepdim=True)
                
            elif self.decoder_type == "gpt2":
                fused = torch.cat([v_img, v_txt], dim=1)
                prefix = self.gpt2_proj(fused).unsqueeze(1)
                device = images.device
                
                if self.gpt2 is not None:
                    generated = torch.full((images.size(0), 1), bos_idx, dtype=torch.long, device=device)
                    for _ in range(max_len):
                        tgt_emb = self.gpt2.transformer.wte(generated)
                        inputs_embeds = torch.cat([prefix, tgt_emb], dim=1)
                        outputs = self.gpt2(inputs_embeds=inputs_embeds)
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                    return generated[:, 1:]
                else:
                    return torch.zeros(images.size(0), max_len, dtype=torch.long, device=device)
                    
            else:
                fused = torch.cat([v_img, v_txt], dim=1)
                h0 = self.init_decoder(fused)
                pad_mask = question_tokens == 0
                return self.decoder.generate(h0, H_txt, max_len=max_len, pad_mask=pad_mask,
                                             bos_idx=bos_idx, eos_idx=eos_idx)

    def sample(self, images, question_tokens, max_len=16, bos_idx=2, eos_idx=3):
        """Autoregressive policy sampling for RL fine-tuning. Returns (tokens, log_probs)."""
        v_img = self.image(images)
        v_txt, H_txt = self.text(question_tokens)
        device = images.device
        
        if self.decoder_type == "mlp":
            logits = self.classifier(torch.cat([v_img, v_txt], dim=1))
            dist = torch.distributions.Categorical(logits=logits)
            sample_tokens = dist.sample()  # [B]
            log_probs = dist.log_prob(sample_tokens)  # [B]
            return sample_tokens.unsqueeze(1), log_probs
            
        elif self.decoder_type == "gpt2":
            fused = torch.cat([v_img, v_txt], dim=1)
            prefix = self.gpt2_proj(fused).unsqueeze(1)
            
            if self.gpt2 is not None:
                generated = torch.full((images.size(0), 1), bos_idx, dtype=torch.long, device=device)
                log_probs = torch.zeros(images.size(0), device=device)
                
                for _ in range(max_len):
                    tgt_emb = self.gpt2.transformer.wte(generated)
                    inputs_embeds = torch.cat([prefix, tgt_emb], dim=1)
                    outputs = self.gpt2(inputs_embeds=inputs_embeds)
                    next_token_logits = outputs.logits[:, -1, :]  # [B, V]
                    dist = torch.distributions.Categorical(logits=next_token_logits)
                    next_token = dist.sample()  # [B]
                    log_probs += dist.log_prob(next_token)
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    
                return generated[:, 1:], log_probs
            else:
                return torch.zeros(images.size(0), max_len, dtype=torch.long, device=device), torch.zeros(images.size(0), device=device)
                
        else:
            # Custom Decoders (GRU/LSTM/Transformer)
            fused = torch.cat([v_img, v_txt], dim=1)
            h = self.init_decoder(fused)
            pad_mask = question_tokens == 0
            
            curr_token = torch.full((images.size(0),), bos_idx, dtype=torch.long, device=device)
            generated = []
            log_probs = torch.zeros(images.size(0), device=device)
            c = torch.zeros_like(h) if self.decoder_type == "lstm" else None
            
            for _ in range(max_len):
                x = self.decoder.embedding(curr_token)  # [B, emb_dim]
                context, _ = self.decoder.attention(h, H_txt, pad_mask)
                rnn_in = torch.cat([x, context], dim=-1)
                
                if self.decoder_type == "gru":
                    h = self.decoder.cell(rnn_in, h)
                elif self.decoder_type == "lstm":
                    h, c = self.decoder.cell(rnn_in, (h, c))
                elif self.decoder_type == "transformer":
                    tgt = self.decoder.embedding(curr_token.unsqueeze(1))
                    memory = self.decoder.proj_enc(H_txt)
                    out = self.decoder.decoder(tgt, memory)
                    h = out[:, -1, :]
                    
                logits = self.decoder.fc(h)  # [B, vocab_size]
                dist = torch.distributions.Categorical(logits=logits)
                curr_token = dist.sample()  # [B]
                log_probs += dist.log_prob(curr_token)
                generated.append(curr_token.unsqueeze(1))
                
            return torch.cat(generated, dim=1), log_probs


def build_model(vocab_size, num_classes, image_encoder="cnn", text_encoder="lstm",
                decoder="mlp", image_attention=False, text_attention=False,
                decoder_attention=False, pretrained=True, max_len=32):
    """Model factory for SFT and RL models."""
    
    # 1. Image Encoder
    if image_encoder == "cnn":
        img = ImageEncoderCNN(use_attention=image_attention)
    elif image_encoder == "resnet18_frozen":
        img = ImageEncoderResNet18Frozen(use_attention=image_attention, pretrained=pretrained)
    elif image_encoder == "pubmedclip":
        img = ImageEncoderPubMedCLIP(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown image encoder: {image_encoder}")

    # 2. Text Encoder
    if text_encoder == "lstm":
        txt = TextEncoderBiLSTM(vocab_size, use_attention=text_attention)
    elif text_encoder == "transformer":
        txt = TextEncoderTransformer(vocab_size, use_attention=text_attention, max_len=max_len)
    elif text_encoder == "pubmedbert":
        txt = TextEncoderPubMedBERT(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown text encoder: {text_encoder}")

    # 3. Complete Fusion model
    return FusionModel(img, txt, vocab_size, num_classes, decoder_type=decoder,
                       decoder_attention=decoder_attention)
