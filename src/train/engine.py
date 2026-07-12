# src/train/engine.py
import csv
import os
import time

import torch
import torch.nn as nn

from .metrics import masked_closed_argmax
from ..utils.progress import get_pbar


def _em_from_ids(preds, labels, closed):
    correct = (preds == labels) & (labels >= 0)

    def rate(mask):
        n = int(mask.sum())
        return float(correct[mask].sum()) / n if n else 0.0

    return {"em_overall": rate(torch.ones_like(closed)),
            "em_closed": rate(closed), "em_open": rate(~closed)}


def _epoch(model, loader, criterion, device, optimizer=None,
           desc="", disable_pbar=True):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, preds, labels_all, closed_all = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for images, tokens, labels, closed in get_pbar(loader, desc, disable_pbar):
            images, tokens = images.to(device), tokens.to(device)
            labels = labels.to(device)
            logits = model(images, tokens)
            loss = criterion(logits, labels.clamp(min=0))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds.append(logits.argmax(1).cpu())
            labels_all.append(labels.cpu())
            closed_all.append(torch.as_tensor(closed, dtype=torch.bool))
    rep = _em_from_ids(torch.cat(preds), torch.cat(labels_all), torch.cat(closed_all))
    return total_loss / len(loader), rep


def run_experiment(name, model, train_loader, val_loader, epochs, lr,
                   weight_decay, patience, out_dir, device, class_weights=None,
                   callbacks=None, show_progress=True):
    """SFT mot cau hinh: early stop theo val EM overall, luu best.pt + history CSV.

    callbacks: object co on_run_start/on_epoch_end/on_run_end (optional, None -> bo qua).
    show_progress: bat tqdm cho tung epoch.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Chi optimize tham so trainable (backbone frozen bi loai tu dong)
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
        tr_loss, tr = _epoch(model, train_loader, criterion, device, optimizer,
                             desc=f"{name} e{epoch+1} train",
                             disable_pbar=not show_progress)
        va_loss, va = _epoch(model, val_loader, criterion, device,
                             desc=f"{name} e{epoch+1} val",
                             disable_pbar=not show_progress)
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
              "checkpoint": ckpt}
    if callbacks is not None:
        callbacks.on_run_end(name, result)
    return result


@torch.no_grad()
def predict_answers(model, loader, id2answer, device,
                    yes_id=None, no_id=None, constrained_closed=False):
    """Chay inference -> tra ve (hyp_strings, p_yes) de cham bang metrics chung.

    hyp_strings: cau tra loi du doan (map class id -> string).
    p_yes: xac suat lop yes cho MOI sample (danh cho AUC tren subset closed).
    """
    model.to(device).eval()
    hyps, p_yes_list = [], []
    for images, tokens, labels, closed in loader:
        logits = model(images.to(device), tokens.to(device))
        preds = logits.argmax(1)
        closed_t = torch.as_tensor(closed, dtype=torch.bool)
        if constrained_closed and yes_id is not None:
            masked = masked_closed_argmax(logits, yes_id, no_id)
            preds = torch.where(closed_t.to(device), masked, preds)
        hyps += [id2answer[int(i)] for i in preds.cpu()]
        if yes_id is not None:
            p = torch.softmax(logits[:, [yes_id, no_id]], dim=1)[:, 0]
            p_yes_list += p.cpu().tolist()
    return hyps, p_yes_list
