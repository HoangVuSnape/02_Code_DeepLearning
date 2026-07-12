# 05 — Push HuggingFace Hub

**Code:** `src/integrations/hf_push.py`.

```python
from src.integrations import hf_push
hf_push.push_file(repo_id="username/medvqa-ablation",
                  local_path="runs/A8_111_best.pt",
                  token=os.environ["HF_TOKEN"],
                  path_in_repo="A8_111_best.pt")
```

## Quy tắc

- `repo_id` / bật-tắt lấy từ `CFG.hf_repo_id`, `CFG.use_hf_push` — **không fix cứng**.
- `token` lấy từ secret `HF_TOKEN` (login qua [08_secrets.md](08_secrets.md)).
- `create_repo(exist_ok=True, private=True)` — an toàn gọi lại nhiều lần.
- `repo_id` rỗng hoặc thiếu token → **no-op**, in cảnh báo, không crash.
- Đẩy tự động ở `on_run_end` (mỗi run xong → 1 file `.pt`) + đẩy `metrics_summary.csv` cuối cùng.

## Cái gì nên đẩy

- Checkpoint best mỗi run (`{run}_best.pt`).
- `runs/metrics_summary.csv`, `runs/config.json`, hình `runs/*.png`.
