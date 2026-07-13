# tests/test_engine.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_model
from src.train.engine import run_experiment


def _loader(n=16):
    imgs = torch.randn(n, 3, 128, 128)
    toks = torch.randint(1, 30, (n, 32))
    labels = torch.randint(0, 4, (n,))
    closed = labels < 2
    dec_in = torch.randint(1, 30, (n, 16))
    dec_tgt = torch.randint(1, 30, (n, 16))
    return DataLoader(TensorDataset(imgs, toks, labels, closed, dec_in, dec_tgt), batch_size=8)


def test_run_experiment_returns_history_and_saves(tmp_path):
    model = build_model(vocab_size=30, num_classes=4)
    id2answer = {0: "yes", 1: "no", 2: "left", 3: "right"}
    q_vocab = {f"word_{i}": i for i in range(30)}
    q_vocab["yes"] = 4
    q_vocab["no"] = 5
    
    result = run_experiment(
        name="unit", model=model,
        train_loader=_loader(), val_loader=_loader(),
        epochs=2, lr=1e-3, weight_decay=1e-4, patience=3,
        out_dir=str(tmp_path), device="cpu",
        id2answer=id2answer, q_vocab=q_vocab
    )
    assert len(result["history"]["train_loss"]) == 2
    assert (tmp_path / "unit_best.pt").exists()
    assert (tmp_path / "unit_history.csv").exists()
    assert result["params_total"] >= result["params_trainable"] > 0
    assert result["best_val_em"] >= 0.0
