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
    try:
        payload = json.dumps({"content": str(content)[:MAX_LEN],
                              "username": username}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"(discord notify bo qua: {e})")
        return False
