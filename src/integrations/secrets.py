# src/integrations/secrets.py
"""Nap secret theo thu tu: Kaggle Secrets -> .env -> os.environ san co.

Tim duoc thi set vao os.environ de comet_ml / huggingface_hub tu doc.
Thieu cai nao thi phan tich hop tuong ung tu tat (no-op).
"""
import os

DEFAULT_NAMES = ("COMET_API_KEY", "DISCORD_WEBHOOK_URL", "HF_TOKEN")


def load_secrets(names=DEFAULT_NAMES, verbose=True):
    out = {}

    # 1) Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for n in names:
            try:
                out[n] = client.get_secret(n)
            except Exception:
                pass
        if out and verbose:
            print(f"🔑 Nap tu Kaggle Secrets: {sorted(out)}")
    except Exception:
        pass

    # 2) file .env
    if len(out) < len(names):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        for n in names:
            if n not in out and os.environ.get(n):
                out[n] = os.environ[n]

    # 3) export sang os.environ
    for n, v in out.items():
        if v:
            os.environ[n] = v

    missing = [n for n in names if not out.get(n)]
    if missing and verbose:
        print(f"⚠️ Thieu secret (phan lien quan se bi tat): {missing}")
    return out


def hf_login_if_possible(token, verbose=True):
    if not token or token.strip() in ("", "YOUR_HF_TOKEN"):
        if verbose:
            print("⚠️ Thieu HF_TOKEN — khong push len HuggingFace Hub.")
        return False
    try:
        from huggingface_hub import login
        login(token=token)
        if verbose:
            print("🤗 Dang nhap HuggingFace Hub thanh cong!")
        return True
    except Exception as e:
        if verbose:
            print(f"⚠️ Login HF that bai: {e}")
        return False
