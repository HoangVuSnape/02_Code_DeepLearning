# tests/test_rl.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_model
from src.train.rl import answer_reward, scst_epoch


def test_answer_reward_range():
    id2answer = {0: "yes", 1: "no", 2: "right lung"}
    r = answer_reward(torch.tensor([0, 2]), torch.tensor([0, 1]), id2answer)
    assert r[0] == 1.0                       # exact match
    assert 0.0 <= r[1] < 1.0                 # sai -> reward thap
    assert r.shape == (2,)


def test_scst_epoch_updates_and_returns_reward():
    imgs = torch.randn(16, 3, 128, 128)
    toks = torch.randint(1, 30, (16, 32))
    labels = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(imgs, toks, labels, labels < 2), batch_size=8)
    model = build_model(vocab_size=30, num_classes=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    id2answer = {0: "yes", 1: "no", 2: "right lung", 3: "axial"}
    stats = scst_epoch(model, loader, opt, id2answer, device="cpu")
    assert "mean_reward" in stats and "loss" in stats
    assert 0.0 <= stats["mean_reward"] <= 1.0
