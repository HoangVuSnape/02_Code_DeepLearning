import os
import sys
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def load_env_token():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token.startswith("hf_ay"):
                        return token
    # Fallback to os.environ
    token = os.environ.get("HF_TOKEN", "")
    return token

def main():
    token = load_env_token()
    if not token:
        print("[ERROR] HF_TOKEN starting with 'hf_ay' not found in .env!")
        sys.exit(1)
    
    print(f"[INFO] Using HF_TOKEN: {token[:7]}...{token[-4:]}")

    repo_id = "VQA-DeepLearning/radimagenet-vqa-500-test"
    parquet_path = "radimagenet_vqa_audit/radimagenet_vqa_500_test.parquet"
    metadata_path = "radimagenet_vqa_audit/radimagenet_vqa_500_test_metadata.csv"

    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet file not found at: {parquet_path}")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[ERROR] huggingface_hub module missing. Please install it.")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"[INFO] Creating / Verifying Hugging Face Dataset Repository: {repo_id}...")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            token=token,
            exist_ok=True
        )
        print(f"[OK] Repository ready: {repo_url}")
    except Exception as e:
        print(f"[WARNING] Warning during repo creation: {e}")

    # Generate dataset README card
    readme_content = f"""---
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

dataset = load_dataset("{repo_id}")
print(dataset)
```

Or load directly with Pandas / PyArrow:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/{repo_id}/data/test-00000-of-00001.parquet")
print(df.head())
```

## 📁 File Structure

- `data/test-00000-of-00001.parquet`: Parquet file containing 500 test records with embedded images.
- `radimagenet_vqa_500_test_metadata.csv`: Metadata CSV reference.
"""

    readme_path = "radimagenet_vqa_audit/DATASET_README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("[INFO] Uploading files to Hugging Face Hub...")
    
    # 1. Upload main data parquet file
    print(f"[INFO] Uploading {parquet_path} -> data/test-00000-of-00001.parquet ...")
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo="data/test-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )

    # 2. Upload metadata CSV file
    if os.path.exists(metadata_path):
        print(f"[INFO] Uploading {metadata_path} -> radimagenet_vqa_500_test_metadata.csv ...")
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="radimagenet_vqa_500_test_metadata.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )

    # 3. Upload dataset README
    print("[INFO] Uploading README.md ...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )

    print("\n[SUCCESS] Dataset successfully uploaded to Hugging Face Organization VQA-DeepLearning!")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
