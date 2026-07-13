# src/train/rl.py
import os
import torch

from ..data.vqa_rad import normalize_answer
from .metrics import token_f1
from .engine import decode_sequence


def answer_reward(pred_tokens, label_ids, id2answer, id2word, is_mlp=False):
    """Reward on string; label -1 (unseen) receives reward 0."""
    rewards = []
    for p_tokens, l_id in zip(pred_tokens, label_ids.tolist()):
        l_id = int(l_id)
        if l_id < 0:
            rewards.append(0.0)
            continue
        
        if is_mlp:
            p_id = int(p_tokens[0])
            hyp = normalize_answer(id2answer.get(p_id, ""))
        else:
            hyp = normalize_answer(decode_sequence(p_tokens, id2word))
            
        ref = normalize_answer(id2answer.get(l_id, ""))
        em = float(hyp == ref)
        f1 = token_f1(ref.split(), hyp.split())
        rewards.append(0.5 * em + 0.5 * f1)
    return torch.tensor(rewards)


def scst_epoch(model, loader, optimizer, id2answer, q_vocab, device):
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
            
        r_sample = answer_reward(sample.cpu(), labels, id2answer, id2word, is_mlp).to(device)
        r_greedy = answer_reward(greedy.cpu(), labels, id2answer, id2word, is_mlp).to(device)
        
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
                  device, out_dir, name="D1_scst", callbacks=None):
    """Run SCST reinforcement learning fine-tuning and keep checkpoint with best val EM."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
        
    best_em = val_eval_fn(model)
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[{name}] val EM before RL: {best_em:.4f}")
    
    for ep in range(epochs):
        stats = scst_epoch(model, train_loader, optimizer, id2answer, q_vocab, device)
        em = val_eval_fn(model)
        print(f"[{name}] RL epoch {ep+1}/{epochs} loss={stats['loss']:.4f} "
              f"reward={stats['mean_reward']:.4f} val_em={em:.4f}")
              
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
