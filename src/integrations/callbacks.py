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
                 hf_repo_id=None, hf_token=None, project_name="medvqa"):
        self.webhook = discord_webhook
        self.comet = comet_experiment
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.project_name = project_name

    def on_run_start(self, run_name, info=None):
        n = (info or {}).get("params_trainable")
        extra = f" | params train={n:,}" if n else ""
        discord.notify(self.webhook,
                       f"🚀 [{self.project_name}] Bat dau **{run_name}**{extra}")

    def on_epoch_end(self, run_name, epoch, total, logs):
        if self.comet is not None:
            try:
                self.comet.log_metrics(
                    {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    prefix=run_name, step=epoch)
            except Exception as e:
                print(f"(comet log bo qua: {e})")

    def on_run_end(self, run_name, result):
        h = result.get("history", {})
        msg = (f"✅ [{self.project_name}] **{run_name}** xong\n"
               f"best val_em: {result.get('best_val_em'):.4f}\n")
        if h.get("val_em_closed"):
            msg += (f"closed: {h['val_em_closed'][-1]:.4f} | "
                    f"open: {h['val_em_open'][-1]:.4f}\n")
        msg += (f"params train: {result.get('params_trainable'):,} | "
                f"epochs chay: {len(h.get('train_loss', []))}")
        discord.notify(self.webhook, msg)

        if self.hf_repo_id and self.hf_token and result.get("checkpoint"):
            hf_push.push_file(self.hf_repo_id, result["checkpoint"], self.hf_token)
