# eval.py
import argparse
import json
import os
import time
import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.dataset import VQARADClsDataset, default_transform
from src.data.vqa_rad import normalize_answer
from src.models.fusion import build_model
from src.train.engine import predict_answers
from src.train.metrics import score_answers, closed_binary_report, score_predictions_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA-RAD Ablation Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (.pt file)")
    parser.add_argument("--predictions_csv", type=str, default=None, help="Path to raw predictions CSV to evaluate (e.g. Gemma)")
    parser.add_argument("--image_encoder", type=str, default="cnn", choices=["cnn", "resnet18_frozen", "pubmedclip"])
    parser.add_argument("--text_encoder", type=str, default="lstm", choices=["lstm", "transformer", "pubmedbert"])
    parser.add_argument("--decoder", type=str, default="mlp", choices=["mlp", "gru", "lstm", "transformer", "gpt2"], help="Type of decoder")
    parser.add_argument("--image_attention", action="store_true", help="Use channel/SE attention in image encoder")
    parser.add_argument("--text_attention", action="store_true", help="Use temporal attention in text encoder")
    parser.add_argument("--decoder_attention", action="store_true", help="Use gated attention in decoder fusion")
    parser.add_argument("--constrained_closed", action="store_true", help="Constrain Closed-ended answers to yes/no only")
    parser.add_argument("--out_dir", type=str, default="runs", help="Output/Vocab directory")
    parser.add_argument("--save_predictions", type=str, default=None, help="Path to save prediction CSV (e.g. runs/preds_A1.csv)")
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save evaluation metrics as JSON (e.g. runs/metrics_A1.json)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--freeze_image", action="store_true", help="Freeze entire image encoder parameters")
    parser.add_argument("--freeze_text", action="store_true", help="Freeze entire text encoder parameters")
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze entire decoder/classifier parameters")
    args = parser.parse_args()

    # Mode 1: Evaluate raw prediction CSV (e.g., from Gemma)
    if args.predictions_csv:
        print(f"📊 Evaluating raw prediction CSV: {args.predictions_csv}")
        if not os.path.exists(args.predictions_csv):
            print(f"❌ File not found: {args.predictions_csv}")
            return
        
        metrics = score_predictions_csv(args.predictions_csv)
        print("\n✨ Unified Evaluation Metrics for CSV:")
        print(json.dumps(metrics, indent=2))
        
        if args.save_metrics:
            os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
            with open(args.save_metrics, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"💾 Saved CSV metrics to: {args.save_metrics}")
        return

    # Mode 2: Evaluate PyTorch checkpoint
    if not args.checkpoint:
        print("❌ Error: Must provide either --checkpoint or --predictions_csv")
        return

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return

    # Load vocabularies
    vocab_answer2id_path = os.path.join(args.out_dir, "vocab_answer2id.json")
    vocab_q_vocab_path = os.path.join(args.out_dir, "vocab_q_vocab.json")
    
    if not os.path.exists(vocab_answer2id_path) or not os.path.exists(vocab_q_vocab_path):
        print("❌ Error: Vocabulary JSON files not found in output directory! Run train.py first to generate vocabs.")
        return

    with open(vocab_answer2id_path) as f:
        answer2id = json.load(f)
    with open(vocab_q_vocab_path) as f:
        q_vocab = json.load(f)
    
    id2answer = {int(i): a for a, i in answer2id.items()}
    yes_id = answer2id.get("yes")
    no_id = answer2id.get("no")

    print(f"🔍 Loading checkpoint: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(
        vocab_size=len(q_vocab),
        num_classes=len(answer2id),
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        decoder=args.decoder,
        image_attention=args.image_attention,
        text_attention=args.text_attention,
        decoder_attention=args.decoder_attention,
        pretrained=False,
        max_len=32
    )
    
    # Load state dict
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Apply manual freezing based on arguments (for correct parameter counts)
    if args.freeze_image:
        for p in model.image.parameters():
            p.requires_grad = False
    if args.freeze_text:
        for p in model.text.parameters():
            p.requires_grad = False
    if args.freeze_decoder:
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = False
        if hasattr(model, "decoder"):
            for p in model.decoder.parameters():
                p.requires_grad = False
        if hasattr(model, "gpt2"):
            for p in model.gpt2.parameters():
                p.requires_grad = False
        if hasattr(model, "gpt2_proj"):
            for p in model.gpt2_proj.parameters():
                p.requires_grad = False
        if hasattr(model, "init_decoder"):
            for p in model.init_decoder.parameters():
                p.requires_grad = False

    model.to(device).eval()

    # Load dataset
    print("📦 Loading VQA-RAD test dataset from Hugging Face...")
    ds = load_dataset("flaviagiammarino/vqa-rad")
    test_recs = [dict(r) for r in ds["test"]]

    # Setup transform — CHUAN HOA 224x224 cho moi model (fair comparison)
    imagenet_norm = (args.image_encoder == "resnet18_frozen")
    image_size = 224                # cung resolution cho CNN va ResNet
    test_tf = default_transform(image_size, train=False, imagenet=imagenet_norm)
    
    test_ds = VQARADClsDataset(test_recs, answer2id, q_vocab, test_tf, max_len=32)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("🏃 Running model inference on test set...")
    t0 = time.time()
    
    # Predict answers
    hyps, p_yes = predict_answers(
        model=model,
        loader=test_loader,
        id2answer=id2answer,
        q_vocab=q_vocab,
        device=device,
        yes_id=yes_id,
        no_id=no_id,
        constrained_closed=args.constrained_closed
    )
    
    inference_time = time.time() - t0
    print(f"⚡ Inference finished in {inference_time:.2f} seconds.")

    # Calculate metrics
    test_refs = [normalize_answer(r["answer"]) for r in test_recs]
    metrics = score_answers(test_refs, hyps)
    
    # Add clinical binary metrics for closed-ended questions
    closed_idx = [i for i, r in enumerate(test_refs) if r in ("yes", "no")]
    if closed_idx and yes_id is not None:
        y_true_yes = [int(test_refs[i] == "yes") for i in closed_idx]
        p_yes_closed = [p_yes[i] for i in closed_idx]
        bin_rep = closed_binary_report(y_true_yes, p_yes_closed)
        metrics.update({f"clin_{k}": v for k, v in bin_rep.items()})

    # Count parameters
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["params_total"] = params_total
    metrics["params_trainable"] = params_trainable
    metrics["inference_time_sec"] = inference_time

    print("\n✨ Unified Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    # Save predictions if path provided
    if args.save_predictions:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        df_out = pd.DataFrame({
            "question": [r["question"] for r in test_recs],
            "answer_ref": [r["answer"] for r in test_recs],
            "answer_pred": hyps,
            "question_type": ["closed" if test_refs[i] in ("yes", "no") else "open" for i in range(len(test_refs))]
        })
        df_out.to_csv(args.save_predictions, index=False)
        print(f"💾 Saved predictions to: {args.save_predictions}")

    # Save metrics if path provided
    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        with open(args.save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved metrics to: {args.save_metrics}")


if __name__ == "__main__":
    main()
