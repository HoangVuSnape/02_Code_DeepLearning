# src/train/engine.py
import csv
import os
import time

import torch
import torch.nn as nn

from .metrics import masked_closed_argmax, score_answers


def decode_sequence(token_ids, id2word):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == 3:  # <eos>
            break
        if tid > 3 and tid in id2word:
            words.append(id2word[tid])
    return " ".join(words)


def _epoch(model, loader, criterion, seq_criterion, device, optimizer=None,
           desc="", disable_pbar=True, id2answer=None, id2word=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    ref_strings, hyp_strings = [], []
    
    with torch.set_grad_enabled(training):
        for images, tokens, labels, closed, dec_in, dec_tgt in loader:
            images, tokens = images.to(device), tokens.to(device)
            labels = labels.to(device)
            
            if model.decoder_type == "mlp":
                logits = model(images, tokens)
                loss = criterion(logits, labels.clamp(min=0))
                
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                preds = logits.argmax(dim=-1).cpu().tolist()
                batch_hyps = [id2answer.get(p, "") for p in preds]
            else:
                dec_in = dec_in.to(device)
                dec_tgt = dec_tgt.to(device)
                
                logits = model(images, tokens, dec_in)
                loss = seq_criterion(logits.view(-1, logits.size(-1)), dec_tgt.view(-1))
                
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # Autoregressive generation for calculating val EM
                with torch.no_grad():
                    pred_tokens = model.generate(images, tokens, max_len=dec_tgt.size(1))
                pred_tokens = pred_tokens.cpu().tolist()
                batch_hyps = [decode_sequence(pt, id2word) for pt in pred_tokens]
                
            total_loss += loss.item()
            batch_refs = [id2answer.get(int(lbl), "") for lbl in labels]
            
            ref_strings.extend(batch_refs)
            hyp_strings.extend(batch_hyps)
            
    rep = score_answers(ref_strings, hyp_strings)
    return total_loss / len(loader), rep


def run_experiment(name, model, train_loader, val_loader, epochs, lr,
                   weight_decay, patience, out_dir, device, id2answer, q_vocab,
                   class_weights=None, callbacks=None, show_progress=True):
    """SFT one configuration: early stops based on val EM overall, saves best.pt + history CSV."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    seq_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    id2word = {i: w for w, i in q_vocab.items()}
    
    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ["train_loss", "train_em", "val_loss", "val_em",
                               "val_em_closed", "val_em_open", "epoch_sec"]}
    best_val_em, bad = -1.0, 0
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if callbacks is not None:
        callbacks.on_run_start(name, {"params_trainable": params_trainable})

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr = _epoch(model, train_loader, criterion, seq_criterion, device, optimizer,
                             desc=f"{name} e{epoch+1} train",
                             disable_pbar=not show_progress,
                             id2answer=id2answer, id2word=id2word)
        va_loss, va = _epoch(model, val_loader, criterion, seq_criterion, device,
                             desc=f"{name} e{epoch+1} val",
                             disable_pbar=not show_progress,
                             id2answer=id2answer, id2word=id2word)
        
        history["train_loss"].append(tr_loss)
        history["train_em"].append(tr["em_overall"])
        history["val_loss"].append(va_loss)
        history["val_em"].append(va["em_overall"])
        history["val_em_closed"].append(va["em_closed"])
        history["val_em_open"].append(va["em_open"])
        history["epoch_sec"].append(time.time() - t0)
        
        print(f"[{name}] {epoch+1}/{epochs} train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} val_em={va['em_overall']:.4f} "
              f"(closed={va['em_closed']:.4f} open={va['em_open']:.4f})")
              
        if callbacks is not None:
            callbacks.on_epoch_end(name, epoch + 1, epochs, {
                "epoch": epoch + 1, "train_loss": tr_loss,
                "train_em": tr["em_overall"], "val_loss": va_loss,
                "val_em": va["em_overall"], "val_em_closed": va["em_closed"],
                "val_em_open": va["em_open"], "epoch_sec": history["epoch_sec"][-1]})
                
        # Save regular checkpoint at the end of every epoch
        epoch_ckpt = os.path.join(out_dir, f"{name}_checkpoint.pt")
        torch.save(model.state_dict(), epoch_ckpt)

        if va["em_overall"] > best_val_em:
            best_val_em, bad = va["em_overall"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop @ epoch {epoch+1}")
                break

    with open(os.path.join(out_dir, f"{name}_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(history))
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] for k in history])

    result = {"name": name, "history": history, "best_val_em": best_val_em,
              "params_total": sum(p.numel() for p in model.parameters()),
              "params_trainable": params_trainable,
              "checkpoint": ckpt,
              "latest_checkpoint": epoch_ckpt}
              
    if callbacks is not None:
        callbacks.on_run_end(name, result)
    return result


@torch.no_grad()
def predict_answers(model, loader, id2answer, q_vocab, device,
                    yes_id=None, no_id=None, constrained_closed=False):
    """Run inference to generate text answers and return lists of strings for comparison."""
    model.to(device).eval()
    id2word = {i: w for w, i in q_vocab.items()}
    hyps = []
    
    for images, tokens, labels, closed, dec_in, dec_tgt in loader:
        images, tokens = images.to(device), tokens.to(device)
        
        if model.decoder_type == "mlp":
            logits = model(images, tokens)
            preds = logits.argmax(dim=-1)
            closed_t = torch.as_tensor(closed, dtype=torch.bool).to(device)
            if constrained_closed and yes_id is not None:
                masked = masked_closed_argmax(logits, yes_id, no_id)
                preds = torch.where(closed_t, masked, preds)
            batch_hyps = [id2answer.get(int(i), "") for i in preds.cpu()]
        else:
            # Generative mode
            if constrained_closed:
                pred_tokens = model.generate(images, tokens, max_len=dec_tgt.size(1))
                pred_tokens = pred_tokens.cpu().tolist()
                batch_hyps = []
                for i, pt in enumerate(pred_tokens):
                    pred_str = decode_sequence(pt, id2word)
                    if closed[i]:
                        pred_clean = pred_str.lower().strip()
                        if "yes" in pred_clean or "no" in pred_clean:
                            pass
                        else:
                            # Fallback: check logits of yes vs no for the first token
                            dec_in_single = torch.full((1, 1), 2, dtype=torch.long, device=device) # <bos>
                            logits_single = model(images[i:i+1], tokens[i:i+1], dec_in_single)
                            yes_tok_id = q_vocab.get("yes", 1)
                            no_tok_id = q_vocab.get("no", 1)
                            yes_score = logits_single[0, 0, yes_tok_id].item()
                            no_score = logits_single[0, 0, no_tok_id].item()
                            pred_str = "yes" if yes_score > no_score else "no"
                    batch_hyps.append(pred_str)
            else:
                pred_tokens = model.generate(images, tokens, max_len=dec_tgt.size(1))
                pred_tokens = pred_tokens.cpu().tolist()
                batch_hyps = [decode_sequence(pt, id2word) for pt in pred_tokens]
                
        hyps.extend(batch_hyps)
    return hyps, [0.5] * len(hyps)
