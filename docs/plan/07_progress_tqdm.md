# 07 — Progress bar (tqdm)

**Vấn đề:** code cũ CHƯA có tqdm → không thấy tiến độ mỗi epoch.

**Code:** `src/utils/progress.py` — `get_pbar(iterable, desc, disable)`.

- Import `tqdm.auto` (đẹp trên cả notebook lẫn terminal).
- Nếu môi trường KHÔNG có tqdm → fallback trả về iterable gốc (không crash).

## Dùng trong engine

```python
for batch in get_pbar(loader, desc=f"{name} e{epoch+1} train",
                      disable=not CFG.show_progress):
    ...
```

- Bật/tắt qua `CFG.show_progress`.
- `leave=False` để không rác log khi chạy 13 run liên tiếp.
- Trong smoke/CI có thể `show_progress=False` cho gọn.
