---
license: cc-by-4.0
task_categories:
- visual-question-answering
language:
- en
tags:
- medical
- vqa
- radimagenet
- radiology
size_categories:
- n<1K
dataset_info:
  features:
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: question_type
    dtype: string
  - name: content_type
    dtype: string
  - name: is_abnormal
    dtype: bool
  - name: location
    dtype: string
  - name: modality
    dtype: string
  - name: pathology
    dtype: string
  - name: question_id
    dtype: string
  splits:
  - name: test
    num_bytes: 23360557
    num_examples: 500
---

# 🩺 RadImageNet VQA 500 Test Subset

This dataset contains a curated 500-example test audit subset for Medical Visual Question Answering based on **RadImageNet**.

## 📊 Dataset Summary

- **Total Samples:** 500 test VQA pairs
- **Modalities:** CT, MRI, X-ray (Abdomen, Brain, Chest/Lung, Ankle/Foot, Hip, Knee)
- **Question Types:** Open-ended & Closed (Yes/No)
- **Organization:** [VQA-DeepLearning](https://huggingface.co/VQA-DeepLearning)

## 💻 Usage

```python
from datasets import load_dataset

dataset = load_dataset("VQA-DeepLearning/radimagenet-vqa-500-test")
print(dataset)
```

Or load directly with Pandas / PyArrow:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/VQA-DeepLearning/radimagenet-vqa-500-test/data/test-00000-of-00001.parquet")
print(df.head())
```

## 📁 File Structure

- `data/test-00000-of-00001.parquet`: Parquet file containing 500 test records with embedded images.
- `radimagenet_vqa_500_test_metadata.csv`: Metadata CSV reference.
