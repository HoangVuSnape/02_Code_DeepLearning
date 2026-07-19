# src/train/rl.py
import os
import csv
import time
import torch

from ..data.vqa_rad import normalize_answer
from .metrics import token_f1, semantic_sim_each
from .engine import decode_sequence


def answer_reward(pred_tokens, label_ids, id2answer, id2word, is_mlp=False, reward_mode="lexical"):
    """Reward tren string; label -1 (unseen) nhan reward 0.

    reward_mode:
      'lexical'  = 0.5*EM + 0.5*token_f1 (mac dinh, tu vung - thua & nhi phan)
      'semantic' = cosine sentence-embedding (min, lien tuc - hop cau MO)
      'mixed'    = 0.5*lexical + 0.5*semantic
    """
    hyps, refs, valid = [], [], []
    for p_tokens, l_id in zip(pred_tokens, label_ids.tolist()):
        l_id = int(l_id)
        if l_id < 0:
            hyps.append(""); refs.append(""); valid.append(False); continue
        if is_mlp:
            hyp = normalize_answer(id2answer.get(int(p_tokens[0]), ""))
        else:
            hyp = normalize_answer(decode_sequence(p_tokens, id2word))
        hyps.append(hyp)
        refs.append(normalize_answer(id2answer.get(l_id, "")))
        valid.append(True)

    # thanh phan tu vung (lexical)
    lex = [0.5 * float(h == r) + 0.5 * token_f1(r.split(), h.split())
           for h, r in zip(hyps, refs)]

    if reward_mode == "lexical":
        rewards = lex
    else:
        sem = semantic_sim_each(refs, hyps)          # list cosine, hoac None
        if sem is None:
            rewards = lex                            # fallback neu thieu sentence-model
        elif reward_mode == "semantic":
            rewards = list(sem)
        else:                                        # mixed
            rewards = [0.5 * l + 0.5 * s for l, s in zip(lex, sem)]

    rewards = [rw if v else 0.0 for rw, v in zip(rewards, valid)]
    return torch.tensor(rewards, dtype=torch.float32)


def scst_epoch(model, loader, optimizer, id2answer, q_vocab, device, reward_mode="lexical"):
    model.to(device).train()
    id2word = {i: w for w, i in q_vocab.items()}
    is_mlp = (model.decoder_type == "mlp")
    total_loss = total_reward = n_batches = 0.0

    for images, tokens, labels, _, dec_in, dec_tgt in loader:
        images, tokens = images.to(device), tokens.to(device)

        # Policy sampling (returns sampled tokens and log probabilities)
        sample, log_probs = model.sample(images, tokens, max_len=dec_tgt.size(1))

        # Greedy decoding baseline
        with torch.no_grad():
            greedy = model.generate(images, tokens, max_len=dec_tgt.size(1))

        r_sample = answer_reward(sample.cpu(), labels, id2answer, id2word, is_mlp, reward_mode).to(device)
        r_greedy = answer_reward(greedy.cpu(), labels, id2answer, id2word, is_mlp, reward_mode).to(device)
        
        advantage = r_sample - r_greedy
        loss = -(log_probs * advantage).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_reward += r_sample.mean().item()
        n_batches += 1
        
    return {"loss": total_loss / n_batches, "mean_reward": total_reward / n_batches}


def scst_finetune(model, train_loader, val_eval_fn, epochs, lr, id2answer, q_vocab,
                  device, out_dir, name="D1_scst", callbacks=None, reward_mode="lexical"):
    """Run SCST reinforcement learning fine-tuning and keep checkpoint with best val EM.
    reward_mode: 'lexical' | 'semantic' | 'mixed' (xem answer_reward)."""
    print(f"[{name}] SCST reward_mode = {reward_mode}", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
        
    # val_eval_fn co the tra ve float (chi em_overall) HOAC dict day du (em_overall/
    # em_closed/em_open) -> chuan hoa ve dict de log breakdown vao history.csv.
    def _scores(m):
        v = val_eval_fn(m)
        if isinstance(v, dict):
            return v
        return {"em_overall": float(v), "em_closed": "", "em_open": ""}

    base = _scores(model)
    best_em = base["em_overall"]
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[{name}] val EM before RL: {best_em:.4f}")

    # Lich su theo epoch -> xuat history.csv (dong bo voi cac run SFT, on_run_end tu push).
    # RL khong co train_em/val_loss theo nghia SFT -> thay bang rl_loss + mean_reward.
    # epoch 0 = baseline truoc RL (chua co loss/reward).
    HIST_COLS = ["epoch", "rl_loss", "mean_reward", "val_em",
                 "val_em_closed", "val_em_open", "epoch_sec"]
    hist_path = os.path.join(out_dir, f"{name}_history.csv")
    history = [{"epoch": 0, "rl_loss": "", "mean_reward": "", "val_em": best_em,
                "val_em_closed": base["em_closed"], "val_em_open": base["em_open"],
                "epoch_sec": ""}]

    def _write_history():
        with open(hist_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HIST_COLS)
            for row in history:
                w.writerow([row[c] for c in HIST_COLS])

    _write_history()   # ghi baseline ngay (phong crash truoc khi xong epoch 1)

    for ep in range(epochs):
        t0 = time.time()
        stats = scst_epoch(model, train_loader, optimizer, id2answer, q_vocab, device, reward_mode)
        sc = _scores(model)
        em = sc["em_overall"]
        epoch_sec = round(time.time() - t0, 2)
        print(f"[{name}] RL epoch {ep+1}/{epochs} loss={stats['loss']:.4f} "
              f"reward={stats['mean_reward']:.4f} val_em={em:.4f}")

        history.append({"epoch": ep + 1, "rl_loss": stats["loss"],
                        "mean_reward": stats["mean_reward"], "val_em": em,
                        "val_em_closed": sc["em_closed"], "val_em_open": sc["em_open"],
                        "epoch_sec": epoch_sec})
        _write_history()   # ghi lai sau moi epoch (phong Kaggle crash van con du lieu)

        # Save regular checkpoint at the end of every epoch
        epoch_ckpt = os.path.join(out_dir, f"{name}_checkpoint.pt")
        torch.save(model.state_dict(), epoch_ckpt)

        if callbacks is not None:
            callbacks.on_epoch_end(name, ep + 1, epochs, {
                "epoch": ep + 1, "loss": stats["loss"],
                "mean_reward": stats["mean_reward"], "val_em": em
            })

        if em > best_em:
            best_em = em
            torch.save(model.state_dict(), ckpt)

    return {"best_val_em": best_em, "checkpoint": ckpt, "latest_checkpoint": epoch_ckpt}
