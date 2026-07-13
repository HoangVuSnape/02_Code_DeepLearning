# src/train/rl.py
"""REINFORCE self-critical (SCST) fine-tune sau SFT — nhom D cua thi nghiem.

reward = 0.5*EM + 0.5*token-F1 giua answer sample va answer dung.
advantage = reward(sample) - reward(greedy)  -> giam variance, khong can critic.
"""
import torch
from torch.distributions import Categorical

from ..data.vqa_rad import normalize_answer
from .metrics import token_f1


def answer_reward(pred_ids, label_ids, id2answer):
    """Reward tren string; label -1 (unseen) cho reward 0."""
    rewards = []
    for p, l in zip(pred_ids.tolist(), label_ids.tolist()):
        if l < 0:
            rewards.append(0.0)
            continue
        hyp = normalize_answer(id2answer[p])
        ref = normalize_answer(id2answer[l])
        em = float(hyp == ref)
        f1 = token_f1(ref.split(), hyp.split())
        rewards.append(0.5 * em + 0.5 * f1)
    return torch.tensor(rewards)


def scst_epoch(model, loader, optimizer, id2answer, device):
    model.to(device).train()
    total_loss = total_reward = n_batches = 0.0
    for images, tokens, labels, _ in loader:
        images, tokens = images.to(device), tokens.to(device)
        logits = model(images, tokens)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        greedy = logits.argmax(1)
        r_sample = answer_reward(sample.cpu(), labels, id2answer).to(device)
        r_greedy = answer_reward(greedy.cpu(), labels, id2answer).to(device)
        advantage = r_sample - r_greedy
        loss = -(dist.log_prob(sample) * advantage).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_reward += r_sample.mean().item()
        n_batches += 1
    return {"loss": total_loss / n_batches, "mean_reward": total_reward / n_batches}


def scst_finetune(model, train_loader, val_eval_fn, epochs, lr, id2answer,
                  device, out_dir, name="D1_scst", callbacks=None):
    """Chay SCST nhieu epoch; giu checkpoint co val EM cao nhat (val_eval_fn tra EM)."""
    import os
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_em = val_eval_fn(model)
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[{name}] val EM truoc RL: {best_em:.4f}")
    for ep in range(epochs):
        stats = scst_epoch(model, train_loader, optimizer, id2answer, device)
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
