# tests/test_integrations.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.integrations import discord, hf_push
from src.integrations.callbacks import BaseCallbacks, ExperimentCallbacks
from src.models.fusion import build_model
from src.train.engine import run_experiment
from src.utils.progress import get_pbar


def test_config_smoke_resolved_reduces_epochs():
    cfg = Config(epochs=12, batch_size=32, smoke=True)
    r = cfg.resolved()
    assert r.epochs == cfg.smoke_epochs and r.batch_size <= 8
    assert cfg.epochs == 12                       # ban goc khong bi doi


def test_config_to_dict_has_dynamic_fields():
    d = Config(project_name="abc", hf_repo_id="u/r").to_dict()
    assert d["project_name"] == "abc" and d["hf_repo_id"] == "u/r"


def test_discord_notify_noop_on_empty_webhook():
    assert discord.notify("", "hello") is False   # webhook rong -> no-op


def test_hf_push_noop_without_repo_or_token():
    assert hf_push.push_file("", "x.pt", "tok") is False
    assert hf_push.push_file("u/r", "nonexistent.pt", "") is False


def test_get_pbar_disabled_returns_iterable():
    data = [1, 2, 3]
    assert list(get_pbar(data, "x", disable=True)) == data


def test_callbacks_receive_hooks(tmp_path):
    events = []

    class Spy(BaseCallbacks):
        def on_run_start(self, run_name, info=None):
            events.append(("start", run_name))

        def on_epoch_end(self, run_name, epoch, total, logs):
            events.append(("epoch", epoch, logs["val_em"]))

        def on_run_end(self, run_name, result):
            events.append(("end", result["best_val_em"]))

    imgs = torch.randn(16, 3, 128, 128)
    toks = torch.randint(1, 30, (16, 32))
    labels = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(imgs, toks, labels, labels < 2), batch_size=8)
    run_experiment("cb", build_model(30, 4), loader, loader, epochs=2, lr=1e-3,
                   weight_decay=1e-4, patience=3, out_dir=str(tmp_path),
                   device="cpu", callbacks=Spy(), show_progress=False)
    assert events[0][0] == "start"
    assert sum(1 for e in events if e[0] == "epoch") == 2
    assert events[-1][0] == "end"


def test_experiment_callbacks_all_optional_noop(tmp_path):
    # Tat ca None -> khong crash
    cb = ExperimentCallbacks()
    imgs = torch.randn(8, 3, 128, 128)
    toks = torch.randint(1, 30, (8, 32))
    labels = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(imgs, toks, labels, labels < 2), batch_size=8)
    res = run_experiment("cb2", build_model(30, 4), loader, loader, epochs=1,
                         lr=1e-3, weight_decay=1e-4, patience=3,
                         out_dir=str(tmp_path), device="cpu", callbacks=cb,
                         show_progress=False)
    assert res["best_val_em"] >= 0.0
