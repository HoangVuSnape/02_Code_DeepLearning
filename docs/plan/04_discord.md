# 04 — Thông báo Discord

**Code:** `src/integrations/discord.py` — chỉ dùng `urllib` (không cần cài `requests`).

```python
from src.integrations import discord
discord.notify(WEBHOOK_URL, "🚀 Bắt đầu train A1_000")
```

- `WEBHOOK_URL` lấy từ secret `DISCORD_WEBHOOK_URL` (xem [08_secrets.md](08_secrets.md)).
- Webhook rỗng / lỗi mạng → **no-op**, không làm crash train (bọc try/except).

## Khi nào bắn tin (qua callbacks — [09_callbacks.md](09_callbacks.md))

| Sự kiện | Nội dung |
|---|---|
| `on_run_start` | 🚀 bắt đầu `{run}` — project, số epoch |
| `on_run_end` | ✅ `{run}` xong — best val_em, closed/open, params, thời gian |
| lỗi (except trong notebook) | ❌ báo lỗi + tên run |

Giữ nội dung ngắn (<1900 ký tự — giới hạn Discord).
