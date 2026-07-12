# 02 — Smoke test (chạy nhanh)

**Mục đích:** kiểm tra cả pipeline chạy trơn (data→train→eval→notify→push) trong ~1–2 phút, TRƯỚC khi chạy thật 13 runs.

## Bật

```python
CFG = Config(smoke=True)      # 1 dòng
```

`CFG.resolved()` khi `smoke=True` sẽ override:

| Field | Thật | Smoke |
|---|---|---|
| dữ liệu | full 1793/451 | cắt còn `smoke_subset` (64) mỗi tập |
| `epochs` | 12 | `smoke_epochs` (1) |
| `rl_epochs` | 5 | 1 |
| số run mỗi nhóm | đủ (A=8...) | `smoke_runs_per_group` (2) đầu |
| `batch_size` | 32 | min(32, 8) |

## Checklist smoke pass

- [ ] pytest Cell 1 pass
- [ ] loss giảm sau 1 epoch, không NaN
- [ ] checkpoint `*_best.pt` + `*_history.csv` tạo ra trong `runs/`
- [ ] Discord nhận được 1 tin nhắn test
- [ ] (nếu bật) HF Hub có file đẩy lên
- [ ] Comet có experiment mới

→ Pass hết thì đặt `smoke=False` chạy thật.
