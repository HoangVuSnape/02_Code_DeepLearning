# src/integrations/callbacks.py
"""Callbacks gom Discord + Comet + HF. Truyen vao run_experiment(..., callbacks=CB).

callbacks=None -> khong lam gi (unit test van chay). Moi thanh phan optional.
"""
from . import discord, hf_push


class BaseCallbacks:
    def on_run_start(self, run_name, info=None):
        pass

    def on_epoch_end(self, run_name, epoch, total, logs):
        pass

    def on_run_end(self, run_name, result):
        pass


class ExperimentCallbacks(BaseCallbacks):
    def __init__(self, discord_webhook=None, comet_experiment=None,
                 hf_repo_id=None, hf_token=None, project_name="medvqa", out_dir="runs"):
        self.webhook = discord_webhook
        self.comet = comet_experiment
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.project_name = project_name
        self.out_dir = out_dir

    def on_run_start(self, run_name, info=None):
        n = (info or {}).get("params_trainable")
        extra = f" | params train={n:,}" if n else ""
        discord.notify(self.webhook,
                       f"🚀 [{self.project_name}] Bat dau **{run_name}**{extra}")

    def on_epoch_end(self, run_name, epoch, total, logs):
        # 1. Log to Comet
        if self.comet is not None:
            try:
                self.comet.log_metrics(
                    {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    prefix=run_name, step=epoch)
            except Exception as e:
                print(f"(comet log bo qua: {e})")

        # 2. Push checkpoints to Hugging Face Hub at the end of EVERY epoch
        if self.hf_repo_id and self.hf_token:
            import os
            best_ckpt = os.path.join(self.out_dir, f"{run_name}_best.pt")
            latest_ckpt = os.path.join(self.out_dir, f"{run_name}_checkpoint.pt")
            
            if os.path.exists(best_ckpt):
                print(f"🤗 [Epoch {epoch}] Pushing best checkpoint to Hugging Face Hub...")
                hf_push.push_file(self.hf_repo_id, best_ckpt, self.hf_token, verbose=False)
            if os.path.exists(latest_ckpt):
                print(f"🤗 [Epoch {epoch}] Pushing latest checkpoint to Hugging Face Hub...")
                hf_push.push_file(self.hf_repo_id, latest_ckpt, self.hf_token, verbose=False)

        # 3. Auto-sync to Google Drive if mounted (Colab)
        import os
        import shutil
        drive_dir = "/content/drive/MyDrive/VQA-DeepLearning/runs"
        if os.path.exists("/content/drive/MyDrive"):
            try:
                os.makedirs(drive_dir, exist_ok=True)
                for fname in [f"{run_name}_best.pt", f"{run_name}_checkpoint.pt", f"{run_name}_history.csv"]:
                    local_p = os.path.join(self.out_dir, fname)
                    if os.path.exists(local_p):
                        shutil.copy(local_p, os.path.join(drive_dir, fname))
                print(f"💾 [Epoch {epoch}] Auto-synced checkpoints and logs to Google Drive!")
            except Exception as e:
                print(f"⚠️ Google Drive sync failed: {e}")

    def on_run_end(self, run_name, result):
        h = result.get("history", {})
        msg = (f"✅ [{self.project_name}] **{run_name}** xong\n"
               f"best val_em: {result.get('best_val_em'):.4f}\n")
        if h.get("val_em_closed"):
            msg += (f"closed: {h['val_em_closed'][-1]:.4f} | "
                    f"open: {h['val_em_open'][-1]:.4f}\n")
        
        train_params = result.get('params_trainable')
        params_str = f"{train_params:,}" if train_params is not None else "N/A"
        epochs_run = len(h.get('train_loss', [])) if h else "N/A"
        msg += f"params train: {params_str} | epochs chay: {epochs_run}"
        discord.notify(self.webhook, msg)

        # Push checkpoints to Hugging Face Hub if configured
        if self.hf_repo_id and self.hf_token:
            if result.get("checkpoint"):
                print(f"🤗 Pushing best checkpoint to Hugging Face Hub: {result['checkpoint']}")
                hf_push.push_file(self.hf_repo_id, result["checkpoint"], self.hf_token)
            if result.get("latest_checkpoint"):
                print(f"🤗 Pushing latest epoch checkpoint to Hugging Face Hub: {result['latest_checkpoint']}")
                hf_push.push_file(self.hf_repo_id, result["latest_checkpoint"], self.hf_token)

        # Auto-sync to Google Drive if mounted (Colab)
        import os
        import shutil
        drive_dir = "/content/drive/MyDrive/VQA-DeepLearning/runs"
        if os.path.exists("/content/drive/MyDrive"):
            try:
                os.makedirs(drive_dir, exist_ok=True)
                for fname in [f"{run_name}_best.pt", f"{run_name}_checkpoint.pt", f"{run_name}_history.csv"]:
                    local_p = os.path.join(self.out_dir, fname)
                    if os.path.exists(local_p):
                        shutil.copy(local_p, os.path.join(drive_dir, fname))
                print("💾 Final auto-sync to Google Drive completed!")
            except Exception as e:
                print(f"⚠️ Google Drive final sync failed: {e}")
