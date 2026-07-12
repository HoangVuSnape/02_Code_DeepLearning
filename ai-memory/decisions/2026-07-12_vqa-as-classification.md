# ADR — Quy VQA-RAD về classification để so sánh với Gemma baseline

> Ngày: 2026-07-12 · Agent: Claude · Status: Accepted

## Bối cảnh

- Kết quả trước đó (v3-vqa-2b/4b-it.ipynb, Gemma LoRA, EM): 2B Open 23.00% / Closed 68.53%; 4B Open 20.00% / Closed 70.92%.
- Hướng mới (docs/planningcode/01–03.md): ablation vị trí attention trong CNN–BiLSTM–Decoder nhẹ, ràng buộc Kaggle/Colab Free.
- Câu hỏi (docs/overview/02.md): dùng lại VQA-RAD (1793 train / 451 test) để so sánh với kết quả cũ được không?

## Quyết định

**Được.** Quy VQA về **classification trên answer vocabulary của train set** (chuẩn MedVInT-TE):
- Input: ảnh 128×128 (grayscale→3 kênh) + câu hỏi tokenize (MAX_LEN=32).
- Output: logits trên C answer duy nhất (chuẩn hóa lowercase/strip) trong train split.
- Metric: **EM overall / EM closed (yes-no) / EM open** — cùng protocol exact-match với benchmark Gemma nên so sánh trực tiếp được.
- Answer của test nằm ngoài answer space → prediction tự động sai (đúng protocol, ghi vào limitations kèm % unseen).

## Ràng buộc thực nghiệm

- 5 model M0–M4, chỉ thay đổi 3 cờ attention (SE-block CNN / temporal LSTM / gated decoder); mọi thứ khác cố định (seed 42, split image-disjoint 90/10 từ train, AdamW lr=1e-3, batch 32, 12 epoch, patience 3).
- Test set đánh giá đúng một lần cuối.
- Constrained closed argmax (chỉ chọn yes/no cho câu closed) = phiên bản classification của Hướng 2 (Constrained Decoding) trong docs/overview/01_before.md — báo cáo như một biến thể inference.

## Bổ sung v2 (cùng ngày, theo feedback user)

- Thiết kế mở rộng thành 4 nhóm, 13 runs: **A** = đủ 2³ = 8 tổ hợp attention (CustomCNN + BiLSTM); **B** = Transformer text encoder (mean vs attention pooling) so với BiLSTM; **C** = ResNet-18 pretrained **đóng băng**, chỉ train text + fusion (ảnh 224, ImageNet norm); **D** = SFT trước → REINFORCE self-critical (SCST) sau, reward = 0.5·EM + 0.5·token-F1.
- Metric chốt 6 nhóm (EM, token-F1, BLEU-1, Acc/macro-F1, y tế closed-subset, chi phí) — evaluator dùng chung `src/train/metrics.py`, chấm lại cả prediction Gemma cũ qua `score_predictions_csv()`.

## Hệ quả

- Plan chi tiết: `docs/superpowers/plans/2026-07-12-vqarad-attention-ablation.md` (src/ modules + tests + notebook/ablation_vqarad.ipynb).
- Track VLM (LoRA Vision, RAG, Ensemble Routing) vẫn giữ trong tasks.md nhưng không phải ưu tiên hiện tại.
