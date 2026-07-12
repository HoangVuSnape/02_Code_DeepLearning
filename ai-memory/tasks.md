# Tasks — Medical-VQA

> Cập nhật: 2026-07-12

## ⚡ Track ưu tiên: Attention-Placement Ablation (CNN–BiLSTM–Decoder trên VQA-RAD)

> Plan chi tiết: [docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md](../docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md)

- [x] Đọc hiểu context + viết implementation plan
- [x] Quyết định kiến trúc: VQA → classification trên answer vocab (ADR 2026-07-12)
- [x] **Plan v2** theo feedback: đủ 8 tổ hợp attention, LSTM vs Transformer, frozen ResNet-18, SFT→RL, model list, metric đồng bộ (13 runs: A8+B2+C2+D1)
- [x] Task 0: .gitignore + requirements-dev.txt (git init: chưa, chờ user)
- [x] Task 1: src/data/vqa_rad.py + tests/test_vqa_rad.py
- [x] Task 2: src/data/dataset.py (2 chế độ normalize) + tests/test_dataset.py
- [x] Task 3: src/models/attention.py + tests/test_attention.py
- [x] Task 4: src/models/encoders.py (4 encoders) + tests/test_encoders.py
- [x] Task 5: src/models/fusion.py (build_model + GROUP_A 8 tổ hợp) + tests/test_fusion.py
- [x] Task 6: src/train/metrics.py (evaluator chung) + tests/test_metrics.py
- [x] Task 7: src/train/engine.py (SFT loop) + tests/test_engine.py
- [x] Task 8: src/train/rl.py (SCST) + tests/test_rl.py
- [x] Tạo kịch bản CLI độc lập `train.py` và `eval.py` để chạy huấn luyện và đánh giá riêng biệt
- [x] Task 9: `report.py` (gom runs/*_metrics.json) + `notebook/kaggle_cli.ipynb` (chỉ các lệnh !python train/eval/report)
- [x] **Hạ tầng v2.1:** config động (src/config.py), smoke test, checkpoint, Discord notify, HF push, Comet, tqdm, secrets Kaggle/.env (src/integrations/, src/utils/) + tests/test_integrations.py
- [x] Spec nhỏ docs/plan/00-10 + HUONG_DAN_CHAY.md
- [x] Gỡ file trùng: src/cli.py, src/data/prepare.py, src/train/runner.py, notebook/ablation_vqarad.ipynb
- [ ] ⚠️ CHẠY pytest trên Kaggle/Colab (Cell 1) — chưa verify vì máy local không có Python
- [ ] Task 9-run: chạy notebook trên GPU → runs/metrics_summary.csv
- [ ] Task 10: export prediction Gemma 2B/4B ra CSV → chấm lại bằng evaluator chung
- [ ] Task 11: ghi kết quả vào ai-memory + html_report
- [ ] (Tùy chọn, nếu còn quota) Rerun top-2 với seed 123, 2026 → mean ± std

## 🟢 Baseline & Setup (track VLM cũ)
- [x] Thiết lập `ai-memory/` dùng chung
- [x] Ghi nhận metric baseline Gemma (EM): 2B Open 23.00% / Closed 68.53%; 4B Open 20.00% / Closed 70.92%

## 🔵 Kiến trúc & LoRA Optimization (track VLM — tạm hoãn)
- [ ] Cấu hình LoRA target modules cho cả Vision Encoder (ViT/SigLIP)
- [ ] Huấn luyện thử nghiệm LoRA kết hợp (Vision + Text) và so sánh hiệu quả

## 🟡 Decoding & Routing (track VLM — tạm hoãn)
- [ ] Cài đặt Logits Processor cho Constrained Decoding (chỉ cho phép "yes"/"no" đối với câu Closed)
- [ ] Viết bộ định tuyến (Router) phân tách câu hỏi: Open -> VLM 2B, Closed -> VLM 4B
- [ ] Đánh giá hiệu suất tổng hợp sau khi áp dụng Ensemble Routing & Constrained Decoding

## 🔴 Image Enhancement & RAG (track VLM — tạm hoãn)
- [ ] Áp dụng CLAHE tăng tương phản ảnh X-quang/MRI trước khi feed vào Vision Tower
- [ ] Thiết kế kiến trúc Multimodal RAG (Vector DB ca lâm sàng tương tự)
- [ ] Triển khai prompt template hỗ trợ RAG cho Paligemma

## 🟣 Reinforcement Learning (track VLM — tạm hoãn)
- [ ] Xây dựng khung huấn luyện RLHF/DPO cải thiện chất lượng câu trả lời của VLM
