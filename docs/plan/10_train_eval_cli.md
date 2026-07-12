# 10 — CLI: train / eval / report (tách riêng)

3 script độc lập, gọi kiểu `!python <file>.py --args` trong cell Kaggle:

| Script | Việc | KHÔNG làm |
|---|---|---|
| `train.py` | huấn luyện **1 run** (SFT hoặc RL) → lưu `runs/{name}_best.pt` | không eval test |
| `eval.py` | đánh giá **1 checkpoint** trên test → in/lưu metric | không train |
| `report.py` | gom `runs/*_metrics.json` → 1 bảng CSV | không train/eval |

⚠️ **Quy tắc vàng:** cờ kiến trúc khi `eval.py` PHẢI trùng khi `train.py` (nếu không `load_state_dict` sẽ lỗi). Đặt tên run theo cờ để khỏi nhầm.

## train.py — flags chính

```
--run_name X                 # bắt buộc, tên run (= tên checkpoint)
--image_encoder cnn|resnet18_frozen
--text_encoder  lstm|transformer
--image_attention            # bật SE-block
--text_attention             # bật temporal/attention pooling
--decoder_attention          # bật gated attention
--rl --load_checkpoint P     # RL (SCST) từ checkpoint SFT P
--smoke                      # chạy nhanh (64 mẫu, 1 epoch)
--epochs --lr --batch_size --patience --seed
--use_comet --use_discord --use_hf_push --hf_repo_id
```

## eval.py — flags chính

```
--checkpoint runs/X_best.pt  # + LẶP LẠI đúng cờ encoder/attention của X
--constrained_closed         # câu closed chỉ chọn yes/no
--save_metrics runs/X_metrics.json     # để report.py gom
--save_predictions runs/X_preds.csv
--predictions_csv gemma.csv  # chế độ 2: chấm lại CSV Gemma (không cần checkpoint)
```

## Bản đồ tên run ↔ cờ (nhóm A, 8 tổ hợp)

| run_name | image_attention | text_attention | decoder_attention |
|---|:--:|:--:|:--:|
| A1_000 | | | |
| A2_001 | | | ✓ |
| A3_010 | | ✓ | |
| A4_011 | | ✓ | ✓ |
| A5_100 | ✓ | | |
| A6_101 | ✓ | | ✓ |
| A7_110 | ✓ | ✓ | |
| A8_111 | ✓ | ✓ | ✓ |

## Runbook đầy đủ 13 thí nghiệm

Xem lệnh copy-paste trong notebook: [../../notebook/kaggle_cli.ipynb](../../notebook/kaggle_cli.ipynb).
Thứ tự: smoke → A(8) train+eval → chọn best A → B(2) → C(2) → chọn best toàn cục → D(1 RL) → chấm Gemma → `report.py`.
