# tests/test_rl.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_model
from src.train.rl import answer_reward, scst_epoch


def test_answer_reward_range():
    id2answer = {0: "yes", 1: "no", 2: "right lung"}
    id2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>", 4: "yes", 5: "no", 6: "right", 7: "lung"}
    
    # MLP mode: pred_tokens are class IDs wrapped in a list
    r_mlp = answer_reward(torch.tensor([[0], [2]]), torch.tensor([0, 1]), id2answer, id2word, is_mlp=True)
    assert r_mlp[0] == 1.0                       # exact match
    assert 0.0 <= r_mlp[1] < 1.0                 # wrong
    assert r_mlp.shape == (2,)

    # Generative mode: pred_tokens are word token IDs
    # e.g., token 4 is "yes", tokens [6, 7] is "right lung"
    pred_tokens = torch.tensor([[4, 3, 0], [6, 7, 3]])
    r_gen = answer_reward(pred_tokens, torch.tensor([0, 2]), id2answer, id2word, is_mlp=False)
    assert r_gen[0] == 1.0                       # exact match for "yes"
    assert r_gen[1] == 1.0                       # exact match for "right lung"
    assert r_gen.shape == (2,)


def test_scst_epoch_updates_and_returns_reward():
    imgs = torch.randn(16, 3, 128, 128)
    toks = torch.randint(1, 30, (16, 32))
    labels = torch.randint(0, 4, (16,))
    closed = labels < 2
    dec_in = torch.randint(1, 30, (16, 16))
    dec_tgt = torch.randint(1, 30, (16, 16))
    loader = DataLoader(TensorDataset(imgs, toks, labels, closed, dec_in, dec_tgt), batch_size=8)
    
    model = build_model(vocab_size=30, num_classes=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    id2answer = {0: "yes", 1: "no", 2: "right lung", 3: "axial"}
    q_vocab = {f"word_{i}": i for i in range(30)}
    
    stats = scst_epoch(model, loader, opt, id2answer, q_vocab, device="cpu")
    assert "mean_reward" in stats and "loss" in stats
    assert 0.0 <= stats["mean_reward"] <= 1.0
