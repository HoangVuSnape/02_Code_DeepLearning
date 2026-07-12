# ADR — Chuyển từ Classification sang Generative VQA

> Ngày: 2026-07-12 · Agent: Claude · Status: Accepted (thay thế [2026-07-12_vqa-as-classification.md])

## Bối cảnh

- Bản classification (VQA → phân loại trên answer vocab, `Linear(fused→C)`) đã code xong + test.
- User đưa đề cương mới `html_report/bo_sung_de_cuong_generative_medvqa.html` đề xuất chuyển hẳn sang **generative**.
- User đã chốt: **Chuyển hẳn sang Generative**.

## Quyết định

Bài toán = **image + question → sinh answer token-by-token** `p(y₁:T|I,q)=Π p(yₜ|y<t,I,q)`. Bỏ classification decoder.

- **Decoder:** GRU 1 layer, hidden 128, sinh tới `<eos>`/`MAX_ANSWER_LEN`; hidden khởi tạo từ `Linear(fused→128)`; **cross-attention** lên image spatial tokens + question states mỗi bước.
- **Data:** 2 vocab train-only (question + answer), token đặc biệt `<pad>=0,<unk>=1,<bos>=2,<eos>=3`; dataset trả `decoder_input`/`decoder_target`; bỏ `answer→class_id` và `label=-1`. MAX_QUESTION_LEN=32, MAX_ANSWER_LEN=12.
- **Image encoder:** xuất **spatial tokens** `[B,128,16,16]→[B,256,128]` (không GAP sớm) để cross-attention.
- **Train:** teacher forcing + `CrossEntropyLoss(ignore_index=PAD)`, AdamW lr 1e-3, grad clip 1.0, AMP, batch 16, ≤15 epoch, patience 3.
- **Inference:** greedy (mặc định), beam=3 chỉ ở best model.
- **Thực nghiệm Core:** G0 question-only · G1 image-only · G2 multimodal baseline · G3 full-attention (SE+temporal+cross-attn). G0/G1 bắt buộc để chống question-bias (RQ1).
- **Extension:** E1 BiLSTM vs Transformer · E2 CNN vs ResNet-18 frozen · E3 concat vs projector · E4 SFT vs SFT+SCST.
- **Metric:** EM overall/closed/open, token-F1, BLEU-1, sens/spec closed. **Bỏ ROC-AUC** (không hợp sequence).

## Lý do

Trung thành với Gemma (generative) + baseline PubMedCLIP+GPT-2 → so sánh **cùng loại task**. G0/G1 là thiết kế khoa học mạnh.

## Đánh đổi (đã chấp nhận)

EM tuyệt đối có thể **thấp hơn** classification & Gemma (data nhỏ 1793, decoder train từ đầu). Giá trị = ablation có kiểm soát (RQ1–RQ5), không phải số cao nhất — phải ghi rõ trong báo cáo.

## Hệ quả code

- Tái dùng được: `src/train/metrics.py` (score_answers chấm trên string), `src/train/rl.py` (reward), phần secrets/integrations/config/CLI.
- Sửa lại: `src/data/dataset.py` (decoder_input/target), `src/data/vqa_rad.py` (answer vocab token-level + special tokens), `src/models/encoders.py` (spatial tokens), `src/models/fusion.py` → GRU decoder + cross-attention, `src/train/engine.py` (teacher forcing, token CE, greedy eval), train.py/eval.py.
- Giữ code classification cũ làm baseline phụ (git history / nhánh riêng).
