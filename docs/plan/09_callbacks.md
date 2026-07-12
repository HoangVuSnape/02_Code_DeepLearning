# 09 — Callbacks (gom Discord + Comet + HF)

**Code:** `src/integrations/callbacks.py`.

`run_experiment(..., callbacks=CB)` gọi 3 hook. `callbacks=None` → không làm gì (test vẫn chạy).

```python
class BaseCallbacks:
    def on_run_start(self, run_name, info): ...
    def on_epoch_end(self, run_name, epoch, total, logs): ...
    def on_run_end(self, run_name, result): ...
```

## `ExperimentCallbacks` làm gì

| Hook | Hành động |
|---|---|
| `on_run_start` | Discord: 🚀 bắt đầu run |
| `on_epoch_end` | Comet: log_metrics (prefix = run_name, step = epoch) |
| `on_run_end` | Discord: ✅ tóm tắt (best val_em, closed/open, params); HF: push `{run}_best.pt` |

## Tạo trong notebook

```python
from src.integrations.callbacks import ExperimentCallbacks
CB = ExperimentCallbacks(
    discord_webhook=WEBHOOK_URL,
    comet_experiment=experiment,
    hf_repo_id=CFG.hf_repo_id if CFG.use_hf_push else None,
    hf_token=os.environ.get("HF_TOKEN"),
    project_name=CFG.project_name)
```

Mọi thành phần đều optional → thiếu cái nào tắt cái đó, không crash train.
