# 08 — Secrets (Kaggle / .env)

**Code:** `src/integrations/secrets.py`.

```python
from src.integrations.secrets import load_secrets, hf_login_if_possible
sec = load_secrets(["COMET_API_KEY", "DISCORD_WEBHOOK_URL", "HF_TOKEN"])
WEBHOOK_URL = sec.get("DISCORD_WEBHOOK_URL")
hf_login_if_possible(sec.get("HF_TOKEN"))
```

## Thứ tự tìm (fallback dần)

1. **Kaggle Secrets** — `UserSecretsClient().get_secret(name)`.
2. **file `.env`** — `python-dotenv` `load_dotenv()` (chạy local/Colab).
3. **os.environ** sẵn có.

Secret tìm được → set vào `os.environ[name]` để thư viện khác (comet_ml, huggingface_hub) tự đọc.

## Tên secret cần khai báo

| Tên | Dùng cho |
|---|---|
| `COMET_API_KEY` | Comet ML |
| `DISCORD_WEBHOOK_URL` | Discord notify |
| `HF_TOKEN` | login + push HuggingFace |

Thiếu cái nào → phần tích hợp tương ứng tự tắt (no-op), phần train vẫn chạy.

⚠️ File `.env` đã nằm trong `.gitignore` — không commit token.
