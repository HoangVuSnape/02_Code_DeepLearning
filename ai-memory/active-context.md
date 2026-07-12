# Active Context

> Cập nhật: 2026-07-12  
> Agent cuối cùng chỉnh sửa: Claude

## Đang làm gì

- **🔴 QUYẾT ĐỊNH LỚN (2026-07-12): chuyển hẳn sang GENERATIVE VQA** (sinh answer token-by-token bằng GRU decoder + cross-attention), bỏ hướng classification. Xem [decisions/2026-07-12_switch-to-generative.md](decisions/2026-07-12_switch-to-generative.md) + đề cương `html_report/bo_sung_de_cuong_generative_medvqa.html`.
- **Đã sửa HTML sang generative** (theo quy tắc "HTML trước, code sau"): index.html, technical_deep_dive.html, architecture.html (banner + decoder + workflow + matrix G0–G3/E1–E4 + metric bỏ AUC).
- **Chưa sửa code** — code hiện tại vẫn là classification. Bước tiếp: rework src/ theo ADR.

## Vừa hoàn thành

- Tách biệt hoàn toàn logic Huấn luyện (`train.py`) và logic Đánh giá (`eval.py`):
  - `train.py` nhận các đối số CLI như `--run_name`, `--image_encoder`, `--text_encoder`, các cờ attention (`--image_attention`, `--text_attention`, `--decoder_attention`), `--rl` (huấn luyện REINFORCE self-critical), `--smoke` và các siêu tham số khác.
  - `eval.py` hỗ trợ hai chế độ đánh giá: (1) Nạp mô hình checkpoint PyTorch (`.pt`) chạy suy luận trên tập Test và (2) Nạp trực tiếp file predictions CSV (ví dụ từ Gemma) để đánh giá đồng bộ. Hỗ trợ xuất kết quả dự đoán ra CSV (`--save_predictions`) và lưu chỉ số metrics ra JSON (`--save_metrics`).
- Runner notebook chỉ chứa các cell `!python`: **`notebook/kaggle_cli.ipynb`** (train.py/eval.py/report.py theo nhóm A/B/C/D + chấm Gemma). Thêm `report.py` gom `runs/*_metrics.json` → bảng.
- ⚠️ ĐÃ GỠ `notebook/ablation_vqarad.ipynb` (library-style, trùng) + `src/cli.py`, `src/data/prepare.py`, `src/train/runner.py`. Entry chính thức: 3 script CLI ở root.

## Tiếp theo

- **Việc kế tiếp của user:** Tải dự án lên Kaggle/Colab, chạy `notebook/kaggle_cli.ipynb` cell-by-cell (13 runs: A8+B2+C2+D1). ⚠️ cờ eval phải trùng cờ train.
- Thêm cell xuất prediction từ `v3-vqa-2b/4b-it.ipynb` sang CSV (`answer_ref, answer_pred`), đặt vào `gemma_preds/` → `eval.py --predictions_csv` chấm đồng bộ.
- Thu thập và cập nhật bảng so sánh cuối (`report.py` → `runs/metrics_summary.csv`).

## Blocker

- Chưa có (cập nhật khi gặp)

## Context quan trọng cho agent tiếp theo

- Tập dữ liệu VQA-RAD: 1793 Train / 451 Test.
- Trình đánh giá thống nhất (`src/train/metrics.py`) sẽ chấm điểm tất cả các run và cả baseline Gemma cũ trên cùng một tập Test để so sánh chính xác nhất.
