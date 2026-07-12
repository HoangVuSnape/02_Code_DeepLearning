# train.py
import argparse
import json
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import Config
from src.data.vqa_rad import build_answer_vocab, build_question_vocab, image_key, group_split, normalize_answer
from src.data.dataset import VQARADClsDataset, default_transform
from src.models.fusion import build_model
from src.integrations.secrets import load_secrets, hf_login_if_possible
from src.integrations.callbacks import ExperimentCallbacks
from src.train.engine import run_experiment, predict_answers
from src.train.metrics import score_answers


def main():
    parser = argparse.ArgumentParser(description="Train VQA-RAD Ablation Model")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for the run")
    parser.add_argument("--image_encoder", type=str, default="cnn", choices=["cnn", "resnet18_frozen"])
    parser.add_argument("--text_encoder", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--image_attention", action="store_true", help="Use channel/SE attention in image encoder")
    parser.add_argument("--text_attention", action="store_true", help="Use temporal attention in text encoder")
    parser.add_argument("--decoder_attention", action="store_true", help="Use gated attention in decoder fusion")
    parser.add_argument("--rl", action="store_true", help="Run REINFORCE self-critical fine-tuning after/instead SFT")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load model state dict checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode (small data/epochs)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of SFT epochs")
    parser.add_argument("--rl_epochs", type=int, default=5, help="Number of RL epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for SFT")
    parser.add_argument("--rl_lr", type=float, default=1e-5, help="Learning rate for RL")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--project_name", type=str, default="medvqa-attention-ablation", help="Comet project name")
    parser.add_argument("--use_comet", action="store_true", help="Log metrics to Comet ML")
    parser.add_argument("--use_discord", action="store_true", help="Send progress notifications to Discord webhook")
    parser.add_argument("--use_hf_push", action="store_true", help="Push best checkpoint to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default="", help="Hugging Face repo ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda/cpu)")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"🚀 Initializing run: {args.run_name}")

    # Load secrets
    secrets = load_secrets(verbose=True)

    # Hugging Face Hub login if requested
    if args.use_hf_push and args.hf_repo_id:
        hf_login_if_possible(secrets.get("HF_TOKEN"), verbose=True)

    # Comet ML initialization
    comet_exp = None
    if args.use_comet and secrets.get("COMET_API_KEY"):
        try:
            import comet_ml
            comet_exp = comet_ml.Experiment(
                api_key=secrets["COMET_API_KEY"],
                project_name=args.project_name,
                auto_metric_logging=False
            )
            comet_exp.log_parameters(vars(args))
            print("🚀 Comet ML experiment initialized.")
        except Exception as e:
            print(f"⚠️ Cannot initialize Comet: {e}")

    # Load dataset
    print("📦 Loading VQA-RAD dataset from Hugging Face...")
    ds = load_dataset("flaviagiammarino/vqa-rad")
    train_recs = [dict(r) for r in ds["train"]]
    keys = [image_key(r["image"]) for r in train_recs]

    # Split train/val
    tr_idx, va_idx = group_split(keys, val_ratio=0.1, seed=args.seed)
    tr = [train_recs[i] for i in tr_idx]
    va = [train_recs[i] for i in va_idx]

    # Apply smoke test limits
    if args.smoke:
        print("🧪 Smoke test mode enabled! Subsetting data and reducing epochs.")
        tr = tr[:64]
        va = va[:64]
        args.epochs = 1
        args.rl_epochs = 1
        args.batch_size = min(args.batch_size, 8)

    # Build vocabularies
    answer2id = build_answer_vocab([r["answer"] for r in tr])
    q_vocab = build_question_vocab([r["question"] for r in tr])
    id2answer = {i: a for a, i in answer2id.items()}

    # Save vocab files for eval.py consistency
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab_answer2id.json"), "w") as f:
        json.dump(answer2id, f, indent=2)
    with open(os.path.join(args.out_dir, "vocab_q_vocab.json"), "w") as f:
        json.dump(q_vocab, f, indent=2)
    print(f"📂 Saved vocabularies: C={len(answer2id)} classes, V={len(q_vocab)} words.")

    # Data loaders
    imagenet_norm = (args.image_encoder == "resnet18_frozen")
    image_size = 224 if imagenet_norm else 128

    train_tf = default_transform(image_size, train=True, imagenet=imagenet_norm)
    val_tf = default_transform(image_size, train=False, imagenet=imagenet_norm)

    train_ds = VQARADClsDataset(tr, answer2id, q_vocab, train_tf, max_len=32)
    val_ds = VQARADClsDataset(va, answer2id, q_vocab, val_tf, max_len=32)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model
    model = build_model(
        vocab_size=len(q_vocab),
        num_classes=len(answer2id),
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        image_attention=args.image_attention,
        text_attention=args.text_attention,
        decoder_attention=args.decoder_attention,
        pretrained=True,
        max_len=32
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"💻 Running on device: {device}")

    # Set up callbacks
    cb = ExperimentCallbacks(
        discord_webhook=secrets.get("DISCORD_WEBHOOK_URL") if args.use_discord else None,
        comet_experiment=comet_exp,
        hf_repo_id=args.hf_repo_id if args.use_hf_push else None,
        hf_token=secrets.get("HF_TOKEN") if args.use_hf_push else None,
        project_name=args.project_name
    )

    if args.rl:
        # Load SFT checkpoint for RL
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
            print(f"✅ Loaded SFT checkpoint for RL: {args.load_checkpoint}")
        else:
            print("⚠️ WARNING: Running RL from scratch (no load_checkpoint provided)!")

        # Define eval function for RL validation
        va_refs = [normalize_answer(r["answer"]) for r in va]
        def val_eval_fn(m):
            hyps, _ = predict_answers(m, val_loader, id2answer, device)
            return score_answers(va_refs, hyps)["em_overall"]

        # Run REINFORCE self-critical fine-tuning
        from src.train.rl import scst_finetune
        print("🏋️ Starting REINFORCE Self-Critical training loop...")
        rl_result = scst_finetune(
            model=model,
            train_loader=train_loader,
            val_eval_fn=val_eval_fn,
            epochs=args.rl_epochs,
            lr=args.rl_lr,
            id2answer=id2answer,
            device=device,
            out_dir=args.out_dir,
            name=args.run_name
        )
        print(f"🎉 RL Training Completed. Best Val EM: {rl_result['best_val_em']:.4f}")
    else:
        # Load checkpoint if provided (resume/finetune)
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
            print(f"✅ Loaded checkpoint: {args.load_checkpoint}")

        # Run supervised fine-tuning (SFT)
        print("🏋️ Starting Supervised Fine-Tuning (SFT) training loop...")
        sft_result = run_experiment(
            name=args.run_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            out_dir=args.out_dir,
            device=device,
            callbacks=cb,
            show_progress=True
        )
        print(f"🎉 SFT Training Completed. Best Val EM: {sft_result['best_val_em']:.4f}")

    if comet_exp is not None:
        comet_exp.end()


if __name__ == "__main__":
    main()
