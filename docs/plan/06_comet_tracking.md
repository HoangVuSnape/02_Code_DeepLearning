# 06 — Comet ML tracking

**Khởi tạo (trong notebook, sau khi có secret):**

```python
import comet_ml
os.environ["COMET_PROJECT_NAME"] = CFG.project_name
experiment = comet_ml.Experiment(project_name=CFG.project_name,
                                 auto_metric_logging=True,
                                 auto_param_logging=True) if CFG.use_comet else None
```

- Truyền `experiment` vào `ExperimentCallbacks` ([09_callbacks.md](09_callbacks.md)).
- `on_epoch_end` → `experiment.log_metrics({...}, prefix=run_name, step=epoch)`.
- Đầu phiên: `experiment.log_parameters(asdict(CFG))`.
- `use_comet=False` hoặc thiếu key → bỏ qua, không crash.

## Metric log lên Comet

`train_loss, train_em, val_loss, val_em, val_em_closed, val_em_open` mỗi epoch, prefix theo tên run để phân biệt 13 runs trong 1 experiment.
