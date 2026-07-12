# 🧠 AI Shared Memory — Medical-VQA

> **Mục đích**: Lớp bộ nhớ dùng chung giữa Antigravity và Claude cho dự án này.
> Thay vì nhét full chat history vào prompt, cả hai agent đọc repo này để khôi phục context.
> Mỗi file là **bản nén tri thức** — ngắn, có cấu trúc, dễ diff.

---

## 📁 Cấu trúc

```
ai-memory/
├── README.md                 ← Bạn đang đọc file này
├── 00_project_overview.md    ← Mục tiêu, stack, metric, constraint
├── active-context.md         ← Trạng thái hiện tại (sprint/phiên gần nhất)
├── handoff.md                ← Agent A làm dở gì → Agent B tiếp
├── tasks.md                  ← TODO master của dự án
├── decisions/                ← Architecture Decision Records (ADR)
│   └── TEMPLATE.md
├── failures/                 ← Lỗi đã gặp, nguyên nhân, cách tránh
│   └── TEMPLATE.md
└── prompts/                  ← Prompt template chuẩn hóa
    └── TEMPLATE.md
```

---

## 📖 Quy ước sử dụng

### Cho Agent (Antigravity / Claude)

1. **Bắt đầu phiên mới** → Đọc `active-context.md` + `handoff.md` trước.
2. **Khi đưa quyết định kiến trúc** → Tạo file mới trong `decisions/`.
3. **Khi gặp lỗi khó debug** → Tạo file mới trong `failures/`.
4. **Khi kết thúc phiên** → Cập nhật `active-context.md` và `handoff.md`.
5. **Khi thay đổi TODO** → Cập nhật `tasks.md`.

### Cho người dùng

- Review diff qua Git để biết agent đã thay đổi memory gì.
- Có thể tự edit bất kỳ file nào — agent sẽ đọc bản mới nhất.
- Giữ mỗi file **dưới 200 dòng** để tiết kiệm token tối đa.

---

## 🏷️ Naming convention

| Thư mục | Format tên file | Ví dụ |
|---|---|---|
| `decisions/` | `YYYY-MM-DD_slug.md` | `2026-07-12_constrained-decoding.md` |
| `failures/` | `YYYY-MM-DD_slug.md` | `2026-07-12_lora-divergence.md` |
| `prompts/` | `purpose_slug.md` | `vqa_inference_prompt.md` |

---

## 💡 Nguyên tắc tiết kiệm token

1. **Không lưu raw conversation** — chỉ lưu kết luận.
2. **Mỗi file ≤ 200 dòng** — nếu dài hơn, tách hoặc nén.
3. **Dùng bullet point** — không viết văn xuôi dài.
4. **Link tới source** thay vì copy nội dung.
5. **Chỉ đọc file cần thiết** — không đọc cả thư mục nếu không cần.
