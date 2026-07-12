# 🚀 Hướng dẫn chạy — VQA-RAD Attention Ablation (CLI)

> Cách chạy: mỗi việc 1 lệnh `!python <script>.py --args` trong cell Kaggle.
> **train / eval / report tách riêng.** Chi tiết từng phần: `docs/plan/`.

## 0. Ba script

| Script | Việc |
|---|---|
| `train.py` | huấn luyện **1 run** (SFT, hoặc RL với `--rl`) → `runs/{name}_best.pt` |
| `eval.py` | đánh giá **1 checkpoint** trên test → `runs/{name}_metrics.json` |
| `report.py` | gom mọi `runs/*_metrics.json` → `runs/metrics_summary.csv` |

⚠️ **Quy tắc vàng:** cờ kiến trúc (`--image_encoder --text_encoder --*_attention`) khi `eval.py` phải **trùng y hệt** khi `train.py`, nếu không load checkpoint sẽ lỗi.

## 1. Chuẩn bị secret (1 lần)

| Tên | Lấy ở đâu |
|---|---|
| `COMET_API_KEY` | comet.com → API Keys |
| `DISCORD_WEBHOOK_URL` | Discord channel → Integrations → Webhooks |
| `HF_TOKEN` | huggingface.co → Access Tokens (quyền **write** nếu push) |

**Kaggle:** Add-ons → Secrets. **Colab/local:** file `.env` cạnh code (đã trong `.gitignore`):

```
COMET_API_KEY=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
HF_TOKEN=hf_...
```

Thiếu secret nào → phần đó tự tắt, train vẫn chạy.

## 2. Upload code

- **Kaggle:** nén cả repo (chứa `train.py`, `src/`, `tests/`) → Dataset `medvqa-src` → attach. Giữ `REPO="/kaggle/input/medvqa-src"` (Cell 1 của notebook).
- **Colab:** `!git clone` hoặc mount Drive → `os.chdir` tới thư mục chứa `train.py`.

## 3. Chạy — dùng notebook lệnh sẵn

Mở [notebook/kaggle_cli.ipynb](notebook/kaggle_cli.ipynb) — đã có sẵn toàn bộ lệnh. Thứ tự:

1. **Cell 1:** setup + pytest (đỏ thì dừng).
2. **Smoke:** `!python train.py --run_name smoke_A1 --smoke` + eval + report (~1–2 phút).
3. **Nhóm A (8 run):** train 8 tổ hợp → eval 8 → `report.py` xem best A.
4. **Nhóm B/C:** điền cờ của best A vào template → train + eval.
5. **Nhóm D (RL):** `--rl --load_checkpoint runs/<best>_best.pt` + lặp cờ của best.
6. **Gemma:** `!python eval.py --predictions_csv gemma_preds/gemma2b_test.csv ...` → `report.py`.

## 4. Ví dụ lệnh

```bash
# Train (SFT) 1 run — smoke rồi bỏ --smoke để chạy thật
python train.py --run_name A8_111 --image_attention --text_attention --decoder_attention --use_discord --use_comet

# Train RL sau khi có SFT tốt nhất
python train.py --run_name D1_scst --rl --load_checkpoint runs/A8_111_best.pt --image_attention --text_attention --decoder_attention

# Eval (cờ phải trùng train). Thêm --constrained_closed cho bản ràng buộc yes/no
python eval.py --checkpoint runs/A8_111_best.pt --image_attention --text_attention --decoder_attention --save_metrics runs/A8_111_metrics.json --save_predictions runs/A8_111_preds.csv

# Chấm lại Gemma cũ cho đồng bộ (không cần checkpoint)
python eval.py --predictions_csv gemma_preds/gemma2b_test.csv --save_metrics runs/gemma-2b-lora_metrics.json

# Gom bảng so sánh
python report.py
```

## 5. Đẩy checkpoint lên HuggingFace

Thêm vào lệnh train: `--use_hf_push --hf_repo_id username/medvqa-ablation`.
`train.py` tự push `{name}_best.pt` khi run xong (qua callbacks).

## 6. Bản đồ tên run ↔ cờ (nhóm A)

| run_name | --image_attention | --text_attention | --decoder_attention |
|---|:--:|:--:|:--:|
| A1_000 | | | |
| A2_001 | | | ✓ |
| A3_010 | | ✓ | |
| A4_011 | | ✓ | ✓ |
| A5_100 | ✓ | | |
| A6_101 | ✓ | | ✓ |
| A7_110 | ✓ | ✓ | |
| A8_111 | ✓ | ✓ | ✓ |

## 7. So sánh Gemma cũ

Mở `v3-vqa-2b-it.ipynb`/`4b`, thêm cell cuối export prediction test-set ra CSV 2 cột `answer_ref, answer_pred`, đặt vào `gemma_preds/`. Rồi `eval.py --predictions_csv ...` chấm cùng thước đo.

## 8. Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| `TESTS FAIL` (Cell 1) | Sửa `src/` theo log pytest, KHÔNG train khi test đỏ |
| `Error(s) in loading state_dict` | Cờ eval KHÔNG trùng cờ train — sửa cho khớp |
| `ModuleNotFoundError: src` | Sai đường dẫn — `os.chdir(REPO)` tới thư mục chứa `train.py` |
| OOM CUDA | `--batch_size 16` |
| Discord/HF/Comet im lặng | Thiếu secret tương ứng (train vẫn chạy) |
