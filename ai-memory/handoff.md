# Handoff — Agent A → Agent B

> Cập nhật: 2026-07-12

## Ai đã làm gì

| Thời gian | Agent | Việc đã làm |
|---|---|---|
| 2026-07-12 | Antigravity | Dọn dẹp memory cũ, tái khởi tạo cấu trúc `ai-memory` cho dự án Medical-VQA |
| 2026-07-12 | Claude | Đọc hiểu toàn bộ context (planningcode 01–03, code_references, Open_Ended_VQA_fixed, kết quả Gemma); viết ADR "VQA → classification"; viết implementation plan 10 task cho ablation M0–M4 trên VQA-RAD; cập nhật tasks.md + active-context.md |

## Việc đang dở

- **Plan đã viết, chưa thực thi.** Bắt đầu từ Task 0 trong [docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md](../docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md).
- Task 1–7 chạy được local (pytest, CPU, synthetic data — KHÔNG cần tải VQA-RAD). Task 8 cần GPU Kaggle/Colab.
- Lưu ý cho agent tiếp theo: giữ nguyên hyperparameters cố định trong plan (seed 42, batch 32, lr 1e-3...); không đổi kiến trúc giữa 5 run chính; test set chỉ đánh giá một lần cuối.
