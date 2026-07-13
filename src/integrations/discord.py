# src/integrations/discord.py
"""Thong bao Discord qua webhook — chi dung urllib (khong can cai requests).

Webhook rong / loi mang -> no-op, khong lam crash train.
"""
import json
import urllib.request

MAX_LEN = 1900   # gioi han Discord ~2000 ky tu


def notify(webhook_url, content, username="MedVQA-Bot"):
    if not webhook_url:
        return False
    webhook_url = str(webhook_url).strip()
    try:
        payload = json.dumps({"content": str(content)[:MAX_LEN],
                              "username": username}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            })
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"(discord notify bo qua: {e})")
        return False
