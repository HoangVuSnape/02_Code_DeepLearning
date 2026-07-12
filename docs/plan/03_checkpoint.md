# 03 — Checkpoint

**Ai lưu:** `run_experiment()` trong `src/train/engine.py`.

## Quy tắc

- Lưu `{out_dir}/{run_name}_best.pt` **chỉ khi** `val_em` cải thiện (early-stop patience).
- Lưu `{out_dir}/{run_name}_history.csv` mỗi epoch (loss/em/thời gian) — phòng session Kaggle/Colab đứt.
- `torch.save(model.state_dict())` (nhẹ, không lưu optimizer để tiết kiệm dung lượng).

## Resume / eval lại

```python
from src.models.fusion import build_model
m = build_model(VOCAB_SIZE, NUM_CLASSES, **cfg_kwargs)
m.load_state_dict(torch.load("runs/A8_111_best.pt"))
```

## Lưu ý Kaggle/Colab

- Kaggle: `runs/` nằm trong `/kaggle/working` → tải về ở tab Output.
- Colab: mount Drive rồi đặt `out_dir="/content/drive/MyDrive/medvqa_runs"` để không mất khi ngắt.
- Bật `use_hf_push=True` để đẩy checkpoint tốt nhất lên HF Hub ngay khi mỗi run xong (xem [05_huggingface.md](05_huggingface.md)).
