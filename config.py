# config.py
"""
Cấu hình hằng số và siêu tham số (hyperparameters) cho dự án Medical-VQA.
Đây là file duy nhất chứa toàn bộ cấu hình dự án.
"""
import copy
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- Data ---
    hf_dataset: str = "VQA-DeepLearning/vqa-rad"
    val_ratio: float = 0.1
    seed: int = 42
    image_size: int = 224          # Kích thước ảnh chuẩn hóa (224 cho tất cả mô hình)
    max_len: int = 32              # Độ dài tối đa của câu hỏi (tokens)
    max_ans_len: int = 16          # Độ dài tối đa của câu trả lời sinh ra (tokens)

    # ==================== TRAINING CONFIG ====================
    batch_size: int = 32
    epochs: int = 20               # Số epoch tối đa khi train SFT
    lr: float = 1e-3               # Learning rate cho SFT
    weight_decay: float = 1e-4     # AdamW weight decay
    patience: int = 5              # Early stopping patience (theo val EM)

    # --- RL (SCST) ---
    rl_epochs: int = 5             # Số epoch tối đa khi train RL
    rl_lr: float = 1e-5            # Learning rate cho RL (SCST)

    # --- Smoke test ---
    smoke: bool = False
    smoke_subset: int = 64
    smoke_epochs: int = 1
    smoke_runs_per_group: int = 2

    # --- IO & Logging ---
    out_dir: str = "runs"          # Thư mục lưu kết quả và checkpoints
    project_name: str = "medvqa-attention-ablation"

    # --- Tích hợp hệ thống ---
    use_comet: bool = True
    use_discord: bool = True
    use_hf_push: bool = False
    hf_repo_id: str = ""
    num_workers: int = 2
    show_progress: bool = True

    def resolved(self):
        """Trả về bản sao cấu hình đã áp dụng tham số chạy thử nhanh (smoke test) nếu bật."""
        cfg = copy.deepcopy(self)
        if cfg.smoke:
            cfg.epochs = cfg.smoke_epochs
            cfg.rl_epochs = 1
            cfg.batch_size = min(cfg.batch_size, 8)
        return cfg

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
