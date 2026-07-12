# Index — Plan nhỏ (mỗi file 1 việc)

> Nguyên tắc: mỗi file ngắn, 1 chủ đề, dễ đọc lại. Không nhồi ngữ cảnh dài.

| File | Chủ đề | Code liên quan |
|---|---|---|
| [01_config.md](01_config.md) | Config động (dataclass, không fix cứng) | `src/config.py` |
| [02_smoke_test.md](02_smoke_test.md) | Chạy nhanh để test pipeline | `src/config.py` (`smoke=True`) |
| [03_checkpoint.md](03_checkpoint.md) | Lưu / resume checkpoint | `src/train/engine.py` |
| [04_discord.md](04_discord.md) | Thông báo Discord webhook | `src/integrations/discord.py` |
| [05_huggingface.md](05_huggingface.md) | Push checkpoint lên HF Hub | `src/integrations/hf_push.py` |
| [06_comet_tracking.md](06_comet_tracking.md) | Log metric lên Comet ML | `src/integrations/callbacks.py` |
| [07_progress_tqdm.md](07_progress_tqdm.md) | Thanh tiến trình tqdm | `src/utils/progress.py` |
| [08_secrets.md](08_secrets.md) | Nạp secret (Kaggle/.env), login HF | `src/integrations/secrets.py` |
| [09_callbacks.md](09_callbacks.md) | Gom Discord+Comet+HF thành callbacks | `src/integrations/callbacks.py` |
| [10_train_eval_cli.md](10_train_eval_cli.md) | **CLI tách train / eval / report** | `train.py`, `eval.py`, `report.py` |

**Cách chạy chính:** CLI `!python train.py --args` / `!python eval.py --args` / `!python report.py`.
Notebook lệnh sẵn: [../../notebook/kaggle_cli.ipynb](../../notebook/kaggle_cli.ipynb).

**Hướng dẫn chạy:** [../../HUONG_DAN_CHAY.md](../../HUONG_DAN_CHAY.md)
**Plan gốc (thực nghiệm A/B/C/D):** [../superpowers/plans/2026-07-12-vqarad-attention-ablation.md](../superpowers/plans/2026-07-12-vqarad-attention-ablation.md)
