# 🩺 Medical Visual Question Answering (MedVQA): Ablation Study, Pretrained Multimodal & Reinforcement Learning (SCST)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Models](https://img.shields.io/badge/HuggingFace-VQA--DeepLearning-yellow.svg)](https://huggingface.co/VQA-DeepLearning/vqa-rad-generative-ep100)
[![Dataset](https://img.shields.io/badge/Dataset-RadImageNet--500--Test-blue.svg)](https://huggingface.co/datasets/VQA-DeepLearning/radimagenet-vqa-500-test)
[![Comet.ml](https://img.shields.io/badge/Comet.ml-Tracking-brightgreen.svg)](https://www.comet.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Mã nguồn nghiên cứu và thực nghiệm bài toán **Hỏi - Đáp trên Ảnh Y tế (Medical Visual Question Answering - MedVQA)** trên bộ dữ liệu [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) và tập kiểm thử audit độc lập **RadImageNet-VQA 500 Test**.

Dự án tập trung phân tích định lượng:
1. Hiệu quả của cơ chế **Attention** (Hình ảnh & Văn bản) qua so sánh **Nhóm A1** (Không Attention) và **Nhóm A2** (Có Attention).
2. Ảnh hưởng của chiến lược đóng băng trọng số (**Freeze All / Image / Text / Decoder**).
3. Đóng góp của các backbone y tế tiền huấn luyện (**PubMedCLIP**, **PubMedBERT**, **ResNet18**) kết hợp bộ giải mã sinh câu trả lời (**GPT-2**, **MLP**, **BiLSTM**, **Transformer Encoder**).
4. Tinh chỉnh bằng **Học tăng cường (Self-Critical Sequence Training - SCST / REINFORCE)**.
5. Đánh giá đối sánh với các mô hình ngôn ngữ thị giác lớn (**Gemma-2B-IT**, **Gemma-4B-IT**).

---

## 📋 Mục lục

1. [Tính năng Nổi bật](#-tính-năng-nổi-bật)
2. [Báo cáo HTML Trực quan & Kiến trúc Mô hình](#-báo-cáo-html-trực-quan--kiến-trúc-mô-hình)
3. [Notebooks Huấn luyện, Đánh giá & Suy luận](#-notebooks-huấn-luyện-đánh-giá--suy-luận)
4. [Cấu trúc Thư mục Dự án](#-cấu-trúc-thư-mục-dự-án)
5. [Ma trận Thực nghiệm (Experiment Run Matrix)](#-ma-trận-thực-nghiệm-experiment-run-matrix)
   - [Nhóm A1: Baseline (Không Attention)](#1-nhóm-a1-baseline-không-attention)
   - [Nhóm A2: Attention Matrix](#2-nhóm-a2-attention-matrix)
   - [Nhóm P: Pretrained Backbones & Generative GPT-2](#3-nhóm-p-pretrained-backbones--generative-gpt-2)
   - [Nhóm R & D: Constrained Closed & RL SCST Fine-Tuning](#4-nhóm-r--d-constrained-closed--rl-scst-fine-tuning)
6. [Kịch bản Đánh giá Kép (Dual-Dataset Evaluation Benchmark)](#-kịch-bản-đánh-giá-kép-dual-dataset-evaluation-benchmark)
7. [Cài đặt & Chuẩn bị Môi trường](#-cài-đặt--chuẩn-bị-môi-trường)
8. [Hướng dẫn Chạy Thực nghiệm CLI](#-hướng-dẫn-chạy-thực-nghiệm-cli)
9. [Kiểm thử Đơn vị (Unit Testing)](#-kiểm-thử-đơn-vị-unit-testing)
10. [Tích hợp Hệ thống & Hugging Face Repositories](#-tích-hợp-hệ-thống--hugging-face-repositories)

---

## ✨ Tính năng Nổi bật

- **Kiến trúc Fusion Đa phương thức Linh hoạt**:
  - **Visual Encoders**: CNN chuẩn, ResNet18 (Frozen ImageNet), PubMedCLIP ViT-B/32.
  - **Text Encoders**: BiLSTM, Transformer Encoder, PubMedBERT.
  - **Decoders**: Classification MLP Head, GPT-2 LM Head (Prefix Tuning / Direct Generation).
- **Thực nghiệm Tách biệt Đóng băng Trọng số (Freezing Ablation)**:
  - Tách bạch tác động của việc freeze toàn bộ (`all`), đóng băng visual encoder (`img`), text encoder (`txt`), hoặc decoder (`dec`).
- **Học tăng cường SCST (Self-Critical Sequence Training)**:
  - Tinh chỉnh mô hình sinh câu trả lời bằng giải thuật REINFORCE, trực tiếp tối ưu chỉ số Exact Match và Token F1 trên tập dữ liệu VQA.
- **Đánh giá Kép trên 2 Tập Test Độc lập (Dual Evaluation Benchmark)**:
  - **VQA-RAD Test Set**: Tập test chuẩn gốc.
  - **RadImageNet-VQA 500 Test Set**: Tập 500 mẫu audit độc lập kiểm tra khả năng suy luận thực tế.
- **Theo dõi & Tự động hóa**:
  - Quản lý thực nghiệm trực tuyến với **Comet.ml** (`medvqa-generative-ablation-ep100_pat15`).
  - Gửi thông báo kết quả & tiến độ real-time qua **Discord Webhook**.
  - Đẩy checkpoints và file metrics tự động lên **Hugging Face Hub** (`VQA-DeepLearning/vqa-rad-generative-ep100`).

---

## 🌐 Báo cáo HTML Trực quan & Kiến trúc Mô hình

Dự án cung cấp bộ **Báo cáo HTML Trực quan (Interactive HTML Report)** tại thư mục [`html_report/`](html_report/). Bạn có thể tải về hoặc mở trực tiếp các file `.html` trong trình duyệt để xem sơ đồ kiến trúc chi tiết, phân tích kỹ thuật và bảng so sánh trực quan:

| Trang Báo cáo HTML | Nội dung Chính |
| :--- | :--- |
| 📑 **[`00_muc_luc.html`](html_report/00_muc_luc.html)** | Trang tổng hợp mục lục và liên kết toàn bộ báo cáo |
| 📋 **[`01_ke_hoach_tong_quan.html`](html_report/01_ke_hoach_tong_quan.html)** | Kế hoạch tổng quan và các pha thực nghiệm |
| 🎯 **[`02_de_cuong_generative_medvqa.html`](html_report/02_de_cuong_generative_medvqa.html)** | Đề cương nghiên cứu Generative MedVQA |
| 🧱 **[`03_kien_truc_model.html`](html_report/03_kien_truc_model.html)** | **Sơ đồ & Trực quan hóa Kiến trúc Mô hình** (Encoders, Fusion, GPT-2, Attention) |
| 🔬 **[`04_technical_deep_dive.html`](html_report/04_technical_deep_dive.html)** | Phân tích chuyên sâu về mã nguồn, luồng dữ liệu & Loss functions |
| 🤖 **[`05_gemma4_architecture.html`](html_report/05_gemma4_architecture.html)** | Kiến trúc & Fine-tuning mô hình Gemma |
| 📊 **[`06_so_sanh_va_data_audit.html`](html_report/06_so_sanh_va_data_audit.html)** | Báo cáo so sánh tổng hợp các run thực nghiệm |
| 🩺 **[`07_vqa_rad_data_audit.html`](html_report/07_vqa_rad_data_audit.html)** | Kiểm toán & Phân tích đặc trưng dữ liệu VQA-RAD |
| 🏥 **[`08_radimagenet_vqa_audit.html`](html_report/08_radimagenet_vqa_audit.html)** | Kiểm toán & Phân tích tập test audit RadImageNet-VQA 500 |
| 📈 **[`09_evaluation_metrics.html`](html_report/09_evaluation_metrics.html)** | Chi tiết các thước đo (Exact Match, Token F1, BLEU, Constrained Closed) |

> 💡 **Cách xem**: Chỉ cần click mở file `.html` bất kỳ trong thư mục [`html_report/`](html_report/) bằng trình duyệt web (Chrome, Edge, Firefox) để trải nghiệm giao diện tương tác động.

---

## 📓 Notebooks Huấn luyện, Đánh giá & Suy luận

Nếu bạn muốn đọc mã nguồn thực thi đầy đủ, kiểm tra quá trình **Huấn luyện (Train)**, **Đánh giá (Eval)** và **Suy luận (Inference)**, hãy truy cập thư mục:

👉 **[`notebook/final_deep_result/`](notebook/final_deep_result/)**

### Danh sách Notebooks chính:

1. 🚀 **[`deeplearning-ep100.ipynb-reinforce.ipynb`](notebook/final_deep_result/deeplearning-ep100.ipynb-reinforce.ipynb)**:
   - Notebook trung tâm chứa toàn bộ quy trình train & eval cho ma trận **A1**, **A2**, **P** (100 epochs) và tinh chỉnh RL **SCST** (`D1_scst`).
   - Tự động đánh giá 2-in-1 trên cả **VQA-RAD Test** và **RadImageNet-VQA 500 Test**, tự động đẩy checkpoints & metrics lên Hugging Face.
2. 📊 **[`deeplearning-ep100.ipynb-eval.ipynb`](notebook/final_deep_result/deeplearning-ep100.ipynb-eval.ipynb)** & **[`deeplearning-ep100.ipynb-eval_2_test.ipynb`](notebook/final_deep_result/deeplearning-ep100.ipynb-eval_2_test.ipynb)**:
   - Đánh giá độc lập và chấm điểm các checkpoint đã huấn luyện.
3. ⚡ **Thư mục Suy luận [`notebook/final_deep_result/inference/`](notebook/final_deep_result/inference/)**:
   - Chứa mã nguồn load checkpoint và chạy suy luận (Inference) trên ảnh y tế & câu hỏi mới.
4. 🤖 **[`v3-vqa-2b-it.ipynb`](notebook/final_deep_result/v3-vqa-2b-it.ipynb)** & **[`v3-vqa-4b-it.ipynb`](notebook/final_deep_result/v3-vqa-4b-it.ipynb)**:
   - Thực nghiệm Fine-tuning & Đánh giá mô hình **Gemma-2B-IT** và **Gemma-4B-IT** làm baseline so sánh.
5. 📈 **[`benchmark_4.ipynb`](notebook/final_deep_result/benchmark_4.ipynb)**:
   - Tổng hợp & vẽ đồ thị so sánh hiệu năng benchmark giữa các mô hình.

---

## 📁 Cấu trúc Thư mục Dự án

```text
.
├── config.py                 # Cấu hình hằng số & hyperparameters chung
├── train.py                  # CLI Script chính huấn luyện (SFT hoặc RL)
├── eval.py                   # CLI Script đánh giá checkpoint hoặc file prediction CSV
├── report.py                 # Script gom kết quả từ runs/*_metrics.json ra CSV/HTML
├── push_dataset_to_hf.py     # Script đẩy dữ liệu audit RadImageNet-500 lên HF Dataset Org
├── HUONG_DAN_CHAY.md         # Hướng dẫn chi tiết thứ tự chạy lệnh CLI
├── requirements-dev.txt      # Gói phụ thuộc kiểm thử & phát triển
│
├── html_report/              # Báo cáo HTML trực quan & Trực quan hóa Kiến trúc Mô hình
│   ├── 00_muc_luc.html       # Mục lục tổng quan báo cáo HTML
│   ├── 03_kien_truc_model.html# Sơ đồ & Trực quan hóa Kiến trúc Mô hình
│   └── ...                   # Các trang báo cáo chuyên sâu (Data Audit, Technical, Metrics)
│
├── notebook/                 # Notebooks thực nghiệm
│   ├── kaggle_cli.ipynb      # Notebook điều khiển chạy CLI trên Kaggle/Colab
│   └── final_deep_result/    # Thư mục chính chứa Notebooks Train, Eval & Inference
│
├── src/                      # Mã nguồn cốt lõi
│   ├── data/                 # Data loaders, tokenizers, VQA-RAD preprocessors
│   ├── models/               # Encoders (CNN, ResNet, PubMedCLIP, PubMedBERT), Fusion & GPT2
│   ├── train/                # Engine SFT, Engine RL SCST, Metrics (EM, BLEU, F1)
│   ├── integrations/         # Callbacks Comet.ml, Discord Webhooks, HF Hub push
│   └── utils/                # Helpers & I/O
│
├── radimagenet_vqa_audit/    # File dữ liệu audit radimagenet_vqa_500_test.parquet & metadata
├── docs/                     # Tài liệu thiết kế & hướng dẫn push dataset
├── tests/                    # Bộ unit test Pytest tự động
└── runs/                     # Thư mục chứa Checkpoints (.pt), Metrics (.json), Summary CSVs
```

---

## 🧬 Ma trận Thực nghiệm (Experiment Run Matrix)

Hệ thống được thiết kế theo 4 nhóm thực nghiệm chính (theo cấu hình từ notebook `deeplearning-ep100.ipynb-reinforce.ipynb`):

### 1. Nhóm A1: Baseline (Không Attention - `attn: False`)

| Run Name | Visual Encoder | Text Encoder | Decoder | Freeze Strategy |
| :--- | :---: | :---: | :---: | :---: |
| `A1_lstm_all` | CNN | BiLSTM | MLP | None (Un-freezed) |
| `A1_lstm_img` | CNN | BiLSTM | MLP | Freeze Image Encoder |
| `A1_lstm_txt` | CNN | BiLSTM | MLP | Freeze Text Encoder |
| `A1_lstm_dec` | CNN | BiLSTM | MLP | Freeze Decoder |
| `A1_trans_all` | CNN | Transformer | MLP | None (Un-freezed) |
| `A1_trans_img` | CNN | Transformer | MLP | Freeze Image Encoder |
| `A1_trans_txt` | CNN | Transformer | MLP | Freeze Text Encoder |
| `A1_trans_dec` | CNN | Transformer | MLP | Freeze Decoder |

### 2. Nhóm A2: Attention Matrix (`attn: True`)

| Run Name | Visual Encoder | Text Encoder | Decoder | Freeze Strategy |
| :--- | :---: | :---: | :---: | :---: |
| `A2_lstm_all` | CNN + Attention | BiLSTM + Attention | MLP | None (Un-freezed) |
| `A2_lstm_img` | CNN + Attention | BiLSTM + Attention | MLP | Freeze Image Encoder |
| `A2_lstm_txt` | CNN + Attention | BiLSTM + Attention | MLP | Freeze Text Encoder |
| `A2_lstm_dec` | CNN + Attention | BiLSTM + Attention | MLP | Freeze Decoder |
| `A2_trans_all` | CNN + Attention | Transformer + Attention | MLP | None (Un-freezed) |
| `A2_trans_img` | CNN + Attention | Transformer + Attention | MLP | Freeze Image Encoder |
| `A2_trans_txt` | CNN + Attention | Transformer + Attention | MLP | Freeze Text Encoder |
| `A2_trans_dec` | CNN + Attention | Transformer + Attention | MLP | Freeze Decoder |

### 3. Nhóm P: Pretrained Backbones & Generative GPT-2

| Run Name | Visual Encoder | Text Encoder | Decoder | Mô tả |
| :--- | :---: | :---: | :---: | :--- |
| `P_resnet_lstm` | ResNet18 (Frozen) | BiLSTM | MLP | Baseline ResNet18 + BiLSTM |
| `P_resnet_trans` | ResNet18 (Frozen) | Transformer | MLP | Baseline ResNet18 + Transformer |
| `P_pubmed_mlp` | PubMedCLIP | PubMedBERT | MLP | Domain-adapted Classification |
| `P_pubmed_gpt2` | PubMedCLIP | PubMedBERT | GPT-2 | Domain-adapted Generative LM |

### 4. Nhóm R & D: Constrained Closed & RL SCST Fine-Tuning

| Run Name | Base Model | Thuật toán / Chế độ | Mục tiêu |
| :--- | :---: | :---: | :--- |
| `R_constrained` | `P_pubmed_gpt2` | Constrained Closed Parsing | Chuẩn hóa cứng câu trả lời closed về Yes/No |
| `D1_scst` | `P_pubmed_gpt2` | RL (SCST / REINFORCE) | Tinh chỉnh trực tiếp metric Exact Match & Token F1 |

---

## 🔬 Kịch bản Đánh giá Kép (Dual-Dataset Evaluation Benchmark)

Mỗi mô hình sau khi huấn luyện xong trên **VQA-RAD** sẽ được đánh giá tự động trên **2 tập test độc lập**:

1. **VQA-RAD Test Set** (`flaviagiammarino/vqa-rad`): Lưu kết quả tại `runs/{name}_metrics.json` $\rightarrow$ Tổng hợp tại `runs/summary_vqarad.csv`.
2. **RadImageNet-VQA 500 Test Set** (`radimagenet_vqa_audit/radimagenet_vqa_500_test.parquet`): Lưu kết quả tại `runs_radimagenet/{name}_metrics.json` $\rightarrow$ Tổng hợp tại `runs_radimagenet/summary_radimagenet.csv`.

---

## ⚙️ Cài đặt & Chuẩn bị Môi trường

### 1. Cài đặt Phụ thuộc

Sử dụng `uv` (khuyên dùng) hoặc `pip`:

```bash
git clone https://github.com/VQA-DeepLearning/medvqa-ablation.git
cd medvqa-ablation

# Cài đặt môi trường với uv
uv sync
```

### 2. Cấu hình Khóa bí mật (`.env`)

Tạo file `.env` tại thư mục gốc dự án:

```env
COMET_API_KEY=your_comet_api_key
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook
HF_TOKEN=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
```

---

## 🚀 Hướng dẫn Chạy Thực nghiệm CLI

### 1. Chạy Thử Nhanh (Smoke Test)

```bash
python train.py --run_name smoke_pubmed --image_encoder pubmedclip --text_encoder pubmedbert --decoder mlp --smoke
```

### 2. Huấn luyện Cấu hình (SFT)

Huấn luyện 1 run cụ thể thuộc nhóm **A1**, **A2**, hoặc **P**:

```bash
# Train A1 (Baseline không attention)
python train.py --run_name A1_lstm_all --image_encoder cnn --text_encoder lstm --decoder mlp --epochs 100 --batch_size 128

# Train A2 (Có attention)
python train.py --run_name A2_lstm_all --image_encoder cnn --text_encoder lstm --decoder mlp --image_attention --text_attention --epochs 100 --batch_size 128

# Train P_pubmed_gpt2 (Generative Model)
python train.py --run_name P_pubmed_gpt2 --image_encoder pubmedclip --text_encoder pubmedbert --decoder gpt2 --epochs 100 --batch_size 32
```

### 3. Huấn luyện Học tăng cường (RL / SCST)

Fine-tune bằng SCST trên checkpoint generative tốt nhất:

```bash
python train.py \
  --run_name D1_scst \
  --rl \
  --load_checkpoint runs/P_pubmed_gpt2_best.pt \
  --image_encoder pubmedclip \
  --text_encoder pubmedbert \
  --decoder gpt2 \
  --rl_epochs 20 \
  --rl_lr 1e-5
```

### 4. Đánh giá Kép (Dual Evaluation)

```bash
# Đánh giá trên VQA-RAD Test Set
python eval.py --checkpoint runs/P_pubmed_gpt2_best.pt --image_encoder pubmedclip --text_encoder pubmedbert --decoder gpt2 --save_metrics runs/P_pubmed_gpt2_metrics.json

# Đánh giá trên RadImageNet-VQA 500 Audit Set
python eval.py --checkpoint runs/P_pubmed_gpt2_best.pt --image_encoder pubmedclip --text_encoder pubmedbert --decoder gpt2 --test_parquet radimagenet_vqa_audit/radimagenet_vqa_500_test.parquet --save_metrics runs_radimagenet/P_pubmed_gpt2_metrics.json
```

### 5. Tổng hợp Báo cáo (Report Generator)

```bash
# Báo cáo VQA-RAD Test
python report.py --out-dir runs --out-csv runs/summary_vqarad.csv

# Báo cáo RadImageNet Audit Test
python report.py --out-dir runs_radimagenet --out-csv runs_radimagenet/summary_radimagenet.csv
```

---

## 🧪 Kiểm thử Đơn vị (Unit Testing)

Chạy bộ kiểm thử tự động Pytest:

```bash
uv run python -m pytest tests/
```

Các module được kiểm thử:
- `test_attention.py`: Kiểm tra Channel & Temporal Attention mechanisms.
- `test_dataset.py` & `test_vqa_rad.py`: Đọc dữ liệu, tokenizer & data splits.
- `test_encoders.py` & `test_fusion.py`: Shape tensor đầu ra của Encoders & GPT-2 Fusion.
- `test_metrics.py`: Kiểm tra thuật toán Exact Match, Token F1, BLEU.
- `test_rl.py`: Thuật toán Reward & Policy Loss trong SCST.

---

## 🤝 Tích hợp Hệ thống & Hugging Face Repositories

- **Model Checkpoints & Metrics Hub**:
  🔗 [https://huggingface.co/VQA-DeepLearning/vqa-rad-generative-ep100](https://huggingface.co/VQA-DeepLearning/vqa-rad-generative-ep100)
- **RadImageNet Audit Dataset Hub**:
  🔗 [https://huggingface.co/datasets/VQA-DeepLearning/radimagenet-vqa-500-test](https://huggingface.co/datasets/VQA-DeepLearning/radimagenet-vqa-500-test)
- **Comet.ml Tracking**:
  Project `medvqa-generative-ablation-ep100_pat15`
- **Discord Webhook**:
  Tự động báo tiến độ & hoàn thành từng run.

---

## 📜 Giấy phép & Liên hệ

Dự án phục vụ mục đích nghiên cứu tại **Trường Đại học Tôn Đức Thắng (TDTU)**.

- **Tác giả**: HoangVuSnape & MedVQA Research Team
- **Organization**: [VQA-DeepLearning](https://huggingface.co/VQA-DeepLearning)
