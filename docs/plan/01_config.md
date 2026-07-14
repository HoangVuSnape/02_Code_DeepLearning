# 01 — Config động

**Vấn đề:** đừng fix cứng tên project / repo / hyperparameter trong code.

**Giải pháp:** 1 dataclass `Config` trong `src/config.py`, truyền qua kwargs.

```python
from src.config import Config
CFG = Config(project_name="gemma4-medical-vqa", use_hf_push=True,
             hf_repo_id="username/medvqa-ablation", smoke=False)
```

## Nhóm field

| Nhóm | Field | Mặc định |
|---|---|---|
| Data | `hf_dataset`, `val_ratio`, `seed`, `image_size`, `image_size_resnet`, `max_len` | vqa-rad, 0.1, 42, 128, 224, 32 |
| Train | `batch_size`, `epochs`, `lr`, `weight_decay`, `patience` | 32, 12, 1e-3, 1e-4, 3 |
| RL | `rl_epochs`, `rl_lr` | 5, 1e-5 |
| Run | `run_groups` (A/B/C/D) | ("A","B","C","D") |
| Smoke | `smoke`, `smoke_subset`, `smoke_epochs`, `smoke_runs_per_group` | False, 64, 1, 2 |
| IO | `out_dir`, `project_name` | "runs", "medvqa-generative-ablation" |
| Tích hợp | `use_comet`, `use_discord`, `use_hf_push`, `hf_repo_id`, `num_workers`, `show_progress` | True, True, False, "", 2, True |

## Quy tắc

- Mọi tên/đường dẫn/số → đọc từ `CFG.*`, không viết literal trong vòng train.
- `CFG.resolved()` trả về bản đã áp smoke (giảm epoch/subset/số run) nếu `smoke=True`.
- Log `asdict(CFG)` ra `runs/config.json` + Comet params đầu mỗi phiên.
