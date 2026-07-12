# Project Overview — Medical-VQA

> Cập nhật lần cuối: 2026-07-12

## Mục tiêu

Tập trung vào việc **tăng độ chính xác (Accuracy - Exact Match & F1)** cho 2 dạng câu hỏi cốt lõi trên bộ dữ liệu [**VQA-RAD**](https://huggingface.co/datasets/flaviagiammarino/vqa-rad):
- **Open-ended**: Câu hỏi mở về y khoa (abnormality, size, position, plane...).
- **Closed-ended**: Câu hỏi đóng dạng Yes/No.

## Scope dự án

| Thuộc dự án này | KHÔNG thuộc dự án này |
|---|---|
| Cân chỉnh (Fine-tune) mô hình VLM (Gemma / Paligemma 2B & 4B) | Thu thập ảnh y khoa mới từ đầu |
| Áp dụng LoRA vào cả Vision Encoder và Text Decoder | Huấn luyện mô hình từ đầu (Pretraining) |
| Ràng buộc từ vựng đầu ra (Constrained Decoding) cho Yes/No | Phát triển ứng dụng Web thương mại |
| Ensemble Routing giữa mô hình 2B (cho Open) & 4B (cho Closed) | |
| Tiền xử lý ảnh (CLAHE, Histogram Equalization) | |
| Xây dựng Multimodal RAG dựa trên ca lâm sàng lịch sử | |
| Huấn luyện Reinforcement Learning (RLHF) cải thiện VLM | |

## Cấu trúc dự án

```
02_Code_DeepLearning/
├── ai-memory/      ← 🧠 Bộ nhớ dùng chung giữa các Agent và User
├── data/           ← Dữ liệu dataset (VQA-RAD)
├── docs/           ← Tài liệu nghiên cứu, ghi chép kỹ thuật
│   └── overview/   ← Ý tưởng cải tiến (LoRA Vision, RAG, Ensemble...)
├── notebook/       ← Jupyter Notebooks chạy huấn luyện và đánh giá
│   ├── v3-vqa-2b-it.ipynb
│   ├── v3-vqa-4b-it.ipynb
│   ├── Open_Ended_VQA_fixed.ipynb
│   └── code_references.ipynb
└── src/            ← Source code mô-đun hóa (nếu có)
```

## Các hướng cải tiến kỹ thuật chính

1. **LoRA Vision Encoder**: Áp dụng LoRA cho các lớp của Vision Tower (SigLIP/ViT) thay vì chỉ Text Decoder để cải thiện khả năng đọc ảnh y khoa.
2. **Constrained Decoding**: Sử dụng Logits Processor để ép mô hình chỉ chọn giữa các token "yes" / "no" cho câu hỏi Closed-ended, cải thiện điểm Exact Match.
3. **Ensemble Routing**: Định tuyến câu hỏi Open sang mô hình 2B (hiệu quả hơn trên Open-ended) và câu Closed sang mô hình 4B (hiệu quả hơn trên Closed-ended).
4. **Medical Image Enhancement**: Tăng cường tương phản bằng CLAHE để làm rõ tổn thương trước khi đưa vào Vision Encoder.
5. **Multimodal RAG**: Truy vấn các ca lâm sàng tương tự từ Vector DB để cung cấp context chẩn đoán phụ trợ.
6. **RLHF**: Áp dụng reinforcement learning để căn chỉnh hành vi tạo câu trả lời của VLM tốt hơn.
