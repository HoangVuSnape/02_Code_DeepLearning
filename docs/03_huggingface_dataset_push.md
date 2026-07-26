# 📦 Hướng dẫn Đẩy Dataset lên Hugging Face Organization (`VQA-DeepLearning`)

Tài liệu này hướng dẫn cách đưa tập dữ liệu kiểm thử audit **RadImageNet MedVQA 500 Test** (`radimagenet_vqa_500_test.parquet`) lên Hugging Face Hub thuộc tổ chức **VQA-DeepLearning**.

---

## 📌 1. Thông tin Dataset

- **Tên Parquet gốc**: `radimagenet_vqa_audit/radimagenet_vqa_500_test.parquet` (Dung lượng: ~23.3 MB)
- **Tên Metadata CSV**: `radimagenet_vqa_audit/radimagenet_vqa_500_test_metadata.csv`
- **Số lượng mẫu**: 500 cặp VQA (Visual Question Answering)
- **Cấu trúc trường dữ liệu (Features)**:
  - `image`: Dữ liệu ảnh y tế (PIL / Binary Image stream)
  - `question`: Câu hỏi bằng tiếng Anh
  - `choices`: Các lựa chọn đáp án (nếu là câu hỏi trắc nghiệm/closed)
  - `answer`: Đáp án chuẩn (Ground truth answer)
  - `question_type`: Phân loại câu hỏi (`open` / `closed`)
  - `metadata`: Thông tin bổ sung (modalities: CT/MRI/X-ray, location, pathology, abnormal status)

---

## 🔑 2. Cấu hình Khóa Bí Mật (HF_TOKEN)

Script tự động đọc khóa truy cập Hugging Face từ file `.env` ở thư mục gốc:

```env
HF_TOKEN=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
```

> **Quyền yêu cầu**: Khóa API phải có quyền **Write** (hoặc thuộc thành viên của Organization `VQA-DeepLearning`) để tạo và upload dataset repository.

---

## 🚀 3. Thực thi Script Đẩy Dataset (`push_dataset_to_hf.py`)

Chạy script hỗ trợ tự động bằng `uv`:

```bash
uv run --with huggingface_hub --with pandas --with pyarrow python push_dataset_to_hf.py
```

### Các bước script thực hiện:
1. Load token `HF_TOKEN` bắt đầu bằng `hf_ay...` từ file `.env`.
2. Tạo/Kiểm tra dataset repository `VQA-DeepLearning/radimagenet-vqa-500-test` trên Hugging Face Hub (nếu chưa có).
3. Đẩy file dữ liệu chính `radimagenet_vqa_500_test.parquet` vào đường dẫn tiêu chuẩn `data/test-00000-of-00001.parquet`.
4. Đẩy file `radimagenet_vqa_500_test_metadata.csv` và tạo file `README.md` mô tả thẻ dataset card chuẩn Hugging Face.

---

## 🔗 4. Đường dẫn & Cách tải Dataset sau khi tải lên

- **URL Dataset trên Hugging Face Hub**:
  👉 [https://huggingface.co/datasets/VQA-DeepLearning/radimagenet-vqa-500-test](https://huggingface.co/datasets/VQA-DeepLearning/radimagenet-vqa-500-test)

### Cách 1: Sử dụng thư viện `datasets` của Hugging Face (Khuyên dùng)

```python
from datasets import load_dataset

# Tải tự động dataset từ Organization VQA-DeepLearning
dataset = load_dataset("VQA-DeepLearning/radimagenet-vqa-500-test")

print(dataset)
# Output: DatasetDict({ test: Dataset({ features: [...], num_rows: 500 }) })

# Xem sample đầu tiên
sample = dataset["test"][0]
print("Question:", sample["question"])
print("Answer:", sample["answer"])
```

### Cách 2: Đọc trực tiếp file Parquet bằng Pandas

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/VQA-DeepLearning/radimagenet-vqa-500-test/data/test-00000-of-00001.parquet")
print(df.head())
```
