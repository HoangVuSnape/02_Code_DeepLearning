# src/config.py
"""Config dong cho toan bo thi nghiem — KHONG fix cung ten/duong dan/hyperparameter.

Dung: CFG = Config(project_name=..., use_hf_push=True, hf_repo_id=...).resolved()
"""
import copy
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- Data ---
    hf_dataset: str = "flaviagiammarino/vqa-rad"
    val_ratio: float = 0.1
    seed: int = 42
    image_size: int = 128
    image_size_resnet: int = 224
    max_len: int = 32

    # --- Train ---
    batch_size: int = 32
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3

    # --- RL (nhom D) ---
    rl_epochs: int = 5
    rl_lr: float = 1e-5

    # --- Run ---
    run_groups: tuple = ("A", "B", "C", "D")

    # --- Smoke test ---
    smoke: bool = False
    smoke_subset: int = 64
    smoke_epochs: int = 1
    smoke_runs_per_group: int = 2

    # --- IO ---
    out_dir: str = "runs"
    project_name: str = "medvqa-attention-ablation"

    # --- Tich hop (deu optional) ---
    use_comet: bool = True
    use_discord: bool = True
    use_hf_push: bool = False
    hf_repo_id: str = ""
    num_workers: int = 2
    show_progress: bool = True

    def resolved(self):
        """Tra ve ban da ap smoke (khong sua ban goc)."""
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
