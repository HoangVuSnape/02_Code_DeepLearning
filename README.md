# 🩺 Medical Visual Question Answering (MedVQA): Attention Ablation & Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-VQA--RAD-yellow.svg)](https://huggingface.co/datasets/VQA-DeepLearning/vqa-rad)
[![Comet.ml](https://img.shields.io/badge/Comet.ml-Tracking-brightgreen.svg)](https://www.comet.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Mã nguồn nghiên cứu và thực nghiệm bài toán **Hỏi - Đáp trên Ảnh Y tế (Medical Visual Question Answering - MedVQA)** trên bộ dữ liệu [VQA-RAD](https://huggingface.co/datasets/VQA-DeepLearning/vqa-rad). Dự án tập trung phân tích định lượng đóng góp của các cơ chế **Attention** (Hình ảnh, Văn bản, Fusion Decoder), mô hình chiếu thị giác **Q-Former**, bộ giải mã sinh câu trả lời (**GPT-2, GRU, LSTM, Transformer**), và tinh chỉnh bằng **Học tăng cường (Self-Critical Sequence Training - SCST/REINFORCE)**.

---

## 📋 Mục lục

1. [Tính năng Nổi bật](#-tính-năng-nổi-bật)
2. [Cấu trúc Thư mục](#-cấu-trúc-thư-mục)
3. [Mô hình & Ma trận Thực nghiệm (Ablation Study)](#-mô-hình--ma-trận-thực-nghiệm-ablation-study)
4. [Cài đặt & Chuẩn bị Môi trường](#-cài-đặt--chuẩn-bị-môi-trường)
5. [Hướng dẫn Chạy Thực nghiệm](#-hướng-dẫn-chạy-thực-nghiệm)
   - [Chạy Thử Nhanh (Smoke Test)](#1-chạy-thử-nhanh-smoke-test)
   - [Huấn luyện Mô hình (Supervised Fine-Tuning - SFT)](#2-huấn-luyện-mô-hình-supervised-fine-tuning---sft)
   - [Huấn luyện Học tăng cường (RL / SCST)](#3-huấn-luyện-học-tăng-cường-rl--scst)
   - [Đánh giá Checkpoint & Dự đoán Gemma](#4-đánh-giá-checkpoint--dự-đoán-gemma)
   - [Gom Báo cáo So sánh (Report)](#5-gom-báo-cáo-so-sánh-report)
6. [Thước đo Đánh giá (Metrics)](#-thước-đo-đánh-giá-metrics)
7. [Kiểm thử Đơn vị (Unit Testing)](#-kiểm-thử-đơn-vị-unit-testing)
8. [Tích hợp Hệ thống (Integrations)](#-tích-hợp-hệ-thống-integrations)

---

## ✨ Tính năng Nổi bật

- **Kiến trúc Fusion Đa phương thức (Multimodal Fusion)**:
  - **Image Encoders**: CNN chuẩn, ResNet18 (Frozen), PubMedCLIP.
  - **Text Encoders**: BiLSTM, Transformer Encoder, PubMedBERT.
  - **Decoders**: Classification MLP, GRU, LSTM, Transformer Decoder, GPT-2 LM Head.
  - **Q-Former Visual Projector**: Nối kết feature map của ảnh thành $K$ visual tokens cho GPT-2.
- **Thực nghiệm Ablation Study Hệ thống (Matrix A1 - A8)**:
  - Tách bạch và khảo sát hiệu quả của 3 tầng Attention: **Image Channel/SE Attention**, **Text Temporal Attention**, và **Decoder Gated Fusion Attention**.
- **Học tăng cường SCST (Self-Critical Sequence Training)**:
  - Tối ưu hóa trực tiếp các chỉ số đánh giá không thể tính đạo hàm như Exact Match (EM) và Token F1 bằng giải thuật REINFORCE.
- **Hệ thống Đánh giá Toàn diện**:
  - Hỗ trợ câu hỏi mở (Open-ended) và câu hỏi đóng (Closed/Yes-No).
  - Tích hợp chuẩn hóa đáp án tự động (Normalization) và chuẩn hóa ràng buộc Yes/No (Constrained Closed parsing).
  - Đánh giá đồng bộ kết quả dự đoán từ các mô hình ngôn ngữ thị giác lớn (LLaVA / Gemma-2B/4B LoRA).
- **Tự động hóa & Giám sát**:
  - Theo dõi thực nghiệm trực tuyến qua **Comet.ml**.
  - Gửi thông báo kết quả & tiến độ real-time qua **Discord Webhook**.
  - Đẩy checkpoint và file metrics trực tiếp lên **Hugging Face Model Hub**.

---

## 📁 Cấu trúc Thư mục

```text
.
├── config.py                 # File cấu hình trung tâm (Hyperparameters, Paths, Defaults)
├── train.py                  # Script CLI chính huấn luyện 1 run (SFT hoặc RL)
├── eval.py                   # Script CLI đánh giá 1 checkpoint hoặc CSV predictions
├── report.py                 # Script gom kết quả từ runs/*_metrics.json ra CSV/HTML
├── HUONG_DAN_CHAY.md         # Hướng dẫn chi tiết các bước chạy lệnh CLI
├── requirements-dev.txt      # Thư viện phụ thuộc phục vụ phát triển & kiểm thử
│
├── src/                      # Mã nguồn cốt lõi
│   ├── data/                 # Xử lý dữ liệu VQA-RAD (Dataset, Tokenizer, Splits)
│   ├── models/               # Định nghĩa Encoders, Decoders, Attention, Q-Former, Fusion
│   ├── train/                # Engine huấn luyện (SFT Engine, RL Engine, Metrics)
│   ├── integrations/         # Callback Comet.ml, Discord notify, HuggingFace Hub push
│   └── utils/                # Utility helpers (Logging, I/O)
│
├── docs/                     # Tài liệu thiết kế chi tiết (Plan, Secret, Callbacks, etc.)
├── notebook/                 # Jupyter Notebooks (kaggle_cli.ipynb cho chạy máy ảo)
├── tests/                    # Bộ kiểm thử tự động với Pytest
└── runs/                     # Thư mục chứa Checkpoints (.pt), Metrics (.json), Predictions (.csv)
```

---

## 🧬 Mô hình & Ma trận Thực nghiệm (Ablation Study)

Ma trận thí nghiệm nhóm A khảo sát sự đóng góp của từng thành phần Attention trong kiến trúc Fusion:

| Run Name | `--image_attention` | `--text_attention` | `--decoder_attention` | Mô tả |
| :---: | :---: | :---: | :---: | :--- |
| **A1_000** | ❌ | ❌ | ❌ | Baseline Fusion (Không dùng Attention) |
| **A2_001** | ❌ | ❌ | ✅ | Chỉ bật Gated Fusion Attention ở Decoder |
| **A3_010** | ❌ | ✅ | ❌ | Chỉ bật Temporal Attention ở Text Encoder |
| **A4_011** | ❌ | ✅ | ✅ | Kết hợp Text Attention + Decoder Attention |
| **A5_100** | ✅ | ❌ | ❌ | Chỉ bật SE Channel Attention ở Image Encoder |
| **A6_101** | ✅ | ❌ | ✅ | Kết hợp Image Attention + Decoder Attention |
| **A7_110** | ✅ | ✅ | ❌ | Kết hợp Image Attention + Text Attention |
| **A8_111** | ✅ | ✅ | ✅ | Bật toàn bộ 3 tầng Attention |

---

## ⚙️ Cài đặt & Chuẩn bị Môi trường

### 1. Cài đặt Thư viện Phụ thuộc

Yêu cầu môi trường Python $\ge 3.10$:

```bash
git clone https://github.com/VQA-DeepLearning/medvqa-ablation.git
cd medvqa-ablation

# Cài đặt gói phụ thuộc
pip install -r requirements-dev.txt
```

### 2. Thiết lập Khóa bí mật (Secrets / API Keys)

Tạo file `.env` tại thư mục gốc của dự án (được bỏ qua bởi `.gitignore`):

```env
COMET_API_KEY=your_comet_api_key_here
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
HF_TOKEN=hf_your_huggingface_write_token
```

> **Lưu ý:** Nếu thiếu secret nào, tính năng tương ứng (Comet tracking, Discord alert, HF Push) sẽ tự động tắt mà không làm ngắt quãng quá trình huấn luyện.

---

## 🚀 Hướng dẫn Chạy Thực nghiệm

### 1. Chạy Thử Nhanh (Smoke Test)

Đảm bảo pipeline không gặp lỗi runtime chỉ trong 1-2 phút:

```bash
python train.py --run_name smoke_test --smoke --use_discord
```

### 2. Huấn luyện Mô hình (Supervised Fine-Tuning - SFT)

Huấn luyện 1 cấu hình cụ thể (Ví dụ: `A8_111` bật full Attention):

```bash
python train.py \
  --run_name A8_111 \
  --image_encoder cnn \
  --text_encoder lstm \
  --decoder mlp \
  --image_attention \
  --text_attention \
  --decoder_attention \
  --epochs 20 \
  --batch_size 32 \
  --use_comet \
  --use_discord
```

*Huấn luyện mô hình GPT-2 cùng visual projector Q-Former:*

```bash
python train.py \
  --run_name QFormer_GPT2 \
  --image_encoder pubmedclip \
  --decoder gpt2 \
  --image_proj qformer \
  --epochs 15
```

### 3. Huấn luyện Học tăng cường (RL / SCST)

Tinh chỉnh checkpoint tốt nhất thu được từ pha SFT bằng REINFORCE algorithm:

```bash
python train.py \
  --run_name D1_scst_A8 \
  --rl \
  --load_checkpoint runs/A8_111_best.pt \
  --image_attention \
  --text_attention \
  --decoder_attention \
  --rl_epochs 5 \
  --rl_lr 1e-5 \
  --rl_reward lexical
```

### 4. Đánh giá Checkpoint & Dự đoán Gemma

> ⚠️ **Quy tắc quan trọng:** Các cờ kiến trúc (`--image_attention`, `--text_attention`, `--decoder_attention`, v.v.) khi gọi `eval.py` phải **trùng khớp 100%** với cờ đã dùng khi `train.py`.

```bash
# Đánh giá checkpoint
python eval.py \
  --checkpoint runs/A8_111_best.pt \
  --image_attention \
  --text_attention \
  --decoder_attention \
  --constrained_closed \
  --save_metrics runs/A8_111_metrics.json \
  --save_predictions runs/A8_111_preds.csv

# Đánh giá file kết quả dự đoán CSV từ mô hình ngoài (ví dụ: Gemma-2B LoRA)
python eval.py \
  --predictions_csv gemma_preds/gemma2b_test.csv \
  --save_metrics runs/gemma-2b-lora_metrics.json
```

### 5. Gom Báo cáo So sánh (Report)

Tự động tổng hợp tất cả các file `runs/*_metrics.json` thành bảng so sánh thống nhất:

```bash
python report.py
```

Kết quả xuất ra tại `runs/metrics_summary.csv` và hiển thị trực quan dạng Markdown table.

---

## 📊 Thước đo Đánh giá (Metrics)

Dự án đánh giá chất lượng câu trả lời MedVQA dựa trên các thước đo chuẩn:

- **Exact Match (EM)**: Tỷ lệ đáp án dự đoán khớp hoàn toàn với đáp án chuẩn (sau khi qua hàm `normalize_answer`).
  - **EM All**: Tính trên toàn bộ tập test.
  - **EM Open**: Tính riêng cho các câu hỏi mở (Open-ended questions).
  - **EM Closed**: Tính riêng cho các câu hỏi đóng (Closed / Yes-No questions).
  - **Constrained Closed EM**: Tính sau khi chuẩn hóa cứng câu trả lời về dạng Yes/No.
- **Token F1**: Chỉ số F1 ở mức độ token giữa chuỗi dự đoán và chuỗi gốc.
- **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4 đo đạc độ tương đồng n-gram.

---

## 🧪 Kiểm thử Đơn vị (Unit Testing)

Hệ thống được đảm bảo chất lượng với bộ kiểm thử Pytest. Trước khi tiến hành train thật trên GPU, luôn chạy bộ test:

```bash
pytest tests/
```

Bộ test kiểm tra:
- `test_attention.py`: Hoạt động của Gated Attention & SE Attention.
- `test_dataset.py` & `test_vqa_rad.py`: Đọc và tiền xử lý dữ liệu VQA-RAD.
- `test_encoders.py` & `test_fusion.py`: Shape của tensor qua Encoders & Decoders.
- `test_metrics.py`: Độ chính xác của các thuật toán tính EM, BLEU, F1.
- `test_rl.py`: Thuật toán tính Reward và Loss trong SCST.
- `test_integrations.py`: Tích hợp Comet, Discord, HuggingFace.

---

## 🤝 Tích hợp Hệ thống (Integrations)

- **[Comet.ml](https://www.comet.com/)**: Ghi nhận loss, learning rate, validation accuracy, BLEU score theo từng epoch dưới dạng biểu đồ tương tác.
- **[Discord Webhooks](https://discord.com/)**: Tự động thông báo trạng thái khi bắt đầu run, báo lỗi khẩn cấp, hoặc báo cáo chỉ số cuối cùng khi hoàn thành run.
- **[Hugging Face Hub](https://huggingface.co/)**: Đẩy tự động checkpoint tốt nhất (`{run_name}_best.pt`) và file báo cáo `_metrics.json` lên repository cá nhân/tổ chức.

---

## 📜 Giấy phép & Liên hệ

Dự án phục vụ mục đích nghiên cứu học thuật tại **Trường Đại học Tôn Đức Thắng (TDTU)**.

- **Tác giả**: HoangVuSnape & MedVQA Research Team
- **Liên hệ**: [GitHub Issues](https://github.com/VQA-DeepLearning/medvqa-ablation/issues)
