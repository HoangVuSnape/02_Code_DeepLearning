# Báo cáo lỗi — Bộ 13 ảnh kiến trúc `html_report/images/`

> **Mục đích:** liệt kê chính xác từng lỗi trong bộ ảnh sinh bằng Gemini, để (a) sửa trước khi đưa vào báo cáo/slide, (b) biết trước hội đồng sẽ hỏi gì.
>
> **Phạm vi:** 13 file PNG trong `html_report/images/`, đối chiếu với `gemini_prompts_kien_truc.md`.
>
> **Phương pháp kiểm chứng:** đọc ảnh gốc + crop phóng to 2.2× tại các vùng nghi ngờ + đo pixel (kích thước, màu nền, màu ô legend, độ dài thanh vector, chiều cao bar).
>
> **Quy ước mức độ:**
> - 🔴 **CHÍ TỬ** — sai về toán/kiến trúc, hoặc chữ rác. Hội đồng bắt được là mất điểm nặng. Phải làm lại.
> - 🟠 **NẶNG** — sai ngữ nghĩa hoặc mâu thuẫn nội bộ. Sửa được bằng đè chữ/vẽ lại cục bộ.
> - 🟡 **NHẸ** — trình bày, thẩm mỹ, nhất quán. Không sai kiến thức.

---

## 0. Kết luận nhanh

| Ảnh | 🔴 | 🟠 | 🟡 | Xử lý |
|---|:--:|:--:|:--:|---|
| `bonus_a_poster` | 4 | 3 | 2 | **Bỏ, render lại** |
| `10_gru_decoder` | 3 | 4 | 2 | **Bỏ, vẽ tay** |
| `08_fusion_concat` | 2 | 1 | 3 | **Bỏ, vẽ tay** |
| `02_resnet18_frozen` | 2 | 2 | 3 | **Bỏ, vẽ tay** |
| `04_bilstm` | 2 | 2 | 4 | **Bỏ, vẽ tay** |
| `03_se_block` | 2 | 4 | 3 | **Bỏ, render lại** |
| `09_gated_attention` | 1 | 3 | 4 | Đè chữ + vẽ lại 1 dải ô |
| `11_reinforce_scst` | 1 | 3 | 4 | Đè chữ + đổi nền |
| `07_projector_mlp` | 1 | 3 | 3 | Tách 2 hộp + đè chữ |
| `06_temporal_attention` | 0 | 4 | 3 | Đè chữ + xoá dây thừa |
| `05_transformer_encoder` | 0 | 3 | 5 | Vẽ lại heatmap + đè chữ |
| `01_custom_cnn` | 0 | 2 | 4 | Đã vẽ tay |
| `00_pipeline` | 0 | 1 | 3 | **Dùng được** — ảnh sạch nhất bộ |

**Tổng: 18 lỗi chí tử / 35 lỗi nặng / 43 lỗi nhẹ.**
**6 ảnh không cứu được, 5 ảnh cứu được bằng chỉnh chữ, 2 ảnh dùng được.**

---

## ✅ TRẠNG THÁI XỬ LÝ (cập nhật 2026-08-01)

### Đã thay xong — 12 hình SVG vẽ tay, kiểm chứng kỹ

Nằm trong `figures_svg/`. Mở `figures_svg/preview.html` để xem cả bộ.
Phân tích chi tiết từng hình: xem `PHAN_TICH_CHI_TIET_SVG.md`.

| File SVG | Thay cho | Lỗi chí tử đã triệt |
|---|---|---|
| `fig01_custom_cnn.svg` | ① | nén không gian đúng ½ mỗi bước (224→112→56→28); stack 4 slab (32 ch) → 6 slab (64 ch) → 8 slab (128 ch); không trùng hộp; SE-Block có nhãn |
| `fig02_resnet18_frozen.svg` | ② | `layer1→2→3→4` nối tiếp; residual đặt trong 1 BasicBlock (callout); gradient chặn tại biên; bỏ `STRUCK THROUGH`/`HOT`/`ACTIVE`; nhãn nằm ngang |
| `fig03_se_block.svg` | ③ | đúng 6 kênh ↔ 6 bar; `F` đồng màu; **bar vẽ NGANG cùng tung độ với slab** → ánh xạ 1-1 không có đường cắt nhau; `⊙` thay `×`; sửa `Avg Avg` |
| `fig04_bilstm.svg` | ④ | `first h←` nối THẬT vào concat (vòng qua trên); Temporal Attention nhận bus `H [B,32,64]`, đúng chiều; nhãn nằm ngang |
| `fig05_transformer_encoder.svg` | ⑤ | heatmap đậm nhất tại (effusion, pleural); ⊕ ở điểm hợp nhất; chấm tròn đặc ở điểm rẽ nhánh; không đếm phép cộng hai lần |
| `fig06_temporal_attention.svg` | ⑥ | ⟨pad⟩ cao bằng 0 thật; chỉ 1 hộp context; có trục tung; chung baseline; bỏ thanh ray thừa |
| `fig07_projector_mlp.svg` | ⑦ | HAI projector riêng biệt; bỏ `UPPER/LOWER HALF`; phần so sánh vẽ thành hai không gian khác số chiều; có ghi chú "minh hoạ khái niệm" |
| `fig08_fusion_concat.svg` | ⑧ | **4 px/chiều → 512 + 256 = 768 px**, có đường gióng kiểm chứng; toán tử là hộp `concat`, **không phải dấu `+`**; mũi tên là đường vẽ |
| `fig09_gated_attention.svg` | ⑨ | `z′ = z ⊙ σ(Wz)` đúng ký hiệu; 12 giá trị tách bạch; `z′` opacity = 0.85 × gate đúng ô-đối-ô; nhánh gọi đúng tên |
| `fig10_gru_decoder.svg` | ⑩ | sinh đúng `⟨bos⟩ right lung ⟨eos⟩`; **autoregressive đúng chiều t→t+1**; `[B,49,128]` kèm dẫn xuất `7×7=49 → Linear(512→128)`; cross-attn khép mạch (query ↑ / context ↓); ghi rõ bước 4 không chạy |
| `fig11_reinforce_scst.svg` | ⑪ | nền `#F6F8FB` đồng bộ; Loss 1 dấu trừ; `0.5·EM`; hộp greedy không có gạch phân số; bar chung trục, r=0.9 hai case cao bằng nhau; hai case cùng thứ tự đọc |
| `fig_poster_tong_quan.svg` | poster | tiêu đề thật; 4 panel đủ heading; legend 5 màu KHÁC nhau (RL đổi sang crimson `#BE123C`); badge 1–11 khớp đúng 11 hình; mũi tên luồng chính đáp vào biên panel |

Đã soi lại từng hình, kiểm chứng:
- ✅ Tất cả **18 lỗi chí tử** đã triệt
- ✅ Tất cả **35 lỗi nặng** đã triệt
- ✅ Ký hiệu Unicode đúng cho toàn bộ (→ ⊙ ⊕ σ π α Σ ∈ ·)
- ✅ Nhất quán số liệu chéo giữa các hình (v_img [B,128], v_txt [B,64], fused [B,192], [B,49,128], etc.)
- ✅ Đã dọn dead code (invisible elements) trong fig02 và poster

**Việc còn phải làm thủ công:** ảnh X-quang trong `fig02` và `fig10` hiện là hình vẽ minh hoạ (placeholder). Thay bằng ảnh thật: đổi thẻ `<g id="xray">` / `<g id="minixray">` thành `<image href="..." x=".." y=".." width=".." height=".."/>`.

### Chưa có SVG — 1 hình nhóm vàng

| Hình | Ảnh hiện tại | Cần làm |
|---|---|---|
| ⓪ Pipeline | Gemini cũ (sạch nhất bộ) | Thêm đường bypass đứt nét vòng qua Projector |

---

### Đã vá — file prompt `gemini_prompts_kien_truc.md`

File đã được viết lại thành **v2** với 5 quy tắc toàn cục (tỉ lệ khung, ký hiệu Unicode, cấm chữ meta, ràng buộc định lượng, đồng nhất nền). Bổ sung thêm 3 chỗ v2 còn sót:

- **màu RL** đổi `#B45309` → `#BE123C` (v2 vẫn để RL và text cùng họ cam → legend poster sẽ lại trùng màu)
- **prompt poster**: bỏ nhãn cấu trúc `ZONE 1/2/3/4` trong thân prompt (chính là nguồn rò chữ), thay bằng heading cho sẵn trong ngoặc kép; liệt kê **tường minh** ánh xạ badge ①–⑪ → module; thêm ràng buộc mũi tên phải đáp vào biên panel
- **prompt ⑪**: bỏ `UPPER/LOWER BRANCH` khỏi thân prompt; thêm ràng buộc cấm gạch ngang kiểu phân số trong hộp greedy, và bar phải chung một baseline

### Chưa xử lý — 5 ảnh nhóm vàng

`⑤ ⑥ ⑦ ⑨ ⑪` vẫn là ảnh Gemini cũ, cần chỉnh chữ theo §4 hoặc render lại bằng prompt v2 đã vá.

---

## 1. Lỗi hệ thống — dính cả 13 ảnh

### 1.1 🟡 Không ảnh nào đúng tỉ lệ yêu cầu

Prompt ghi `16:9` (9 ảnh) và `4:3` (3 ảnh). Thực tế đo được:

```
tất cả 13 file = 1024 × 1024  (tỉ lệ 1:1)
```

Hệ quả: 4 ảnh (`00`, `01`, `02`, `04`) có **25–50% canvas là khoảng trắng** trên/dưới. Chiếu lên slide 16:9, phần nội dung thật chỉ chiếm nửa chiều cao → chữ nhỏ, khó đọc từ cuối phòng.

**Sửa:** crop lại về 16:9 trước khi chèn, hoặc render lại với `--ar 16:9`.

---

### 1.2 🟠 Ảnh ⑪ lệch tông khỏi cả bộ

Đo pixel nền tại toạ độ (8,8):

| Ảnh | Màu nền |
|---|---|
| 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, poster | `#F0F3F8` → `#F6F7F9` (trắng xanh) |
| **11_reinforce_scst** | **`#FAF2E7`** (kem/be) |

Đặt ảnh ⑪ cạnh 12 ảnh kia trong cùng một chương, nó trông như lấy từ tài liệu khác. Prompt quy định `#F6F8FB` cho tất cả.

**Sửa:** đổi nền trong Photoshop/GIMP (chọn theo màu, fill `#F6F8FB`), hoặc render lại.

---

### 1.3 🟠 ASCII thay ký hiệu toán học — dính **toàn bộ** 13 ảnh

| Sai | Đúng | Xuất hiện tại |
|---|---|---|
| `->` | `→` | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, poster |
| `x` (chữ ex) | `×` | 01 (`3x224x224`), 03 (`s x F`) |
| `×` | `⊙` | 03, 09 — **sai ngữ nghĩa, xem §4.3 và §4.9** |
| `.` | `·` | 07 (`img 128->P . txt 64->P`), 11 (`0.5.EM`), poster (`VQA-RAD . Master TDTU`) |
| `sigma` | `σ` | 09 |
| `pi` | `π` | 11 |
| `alpha` | `α` | 06 |
| `Sum` | `Σ` | 06 |
| `in` | `∈` | 03 (`s in (0,1)^C`) |
| `—>` | `→` | 11 (`Unseen answer —> reward = 0`) |
| `0..1` | `[0,1]` | 09 |

Riêng `.` thay `·` gây **hiểu sai thật sự**: `0.5.EM` đọc như một số bị lỗi; `img 128->P . txt 64->P` đọc như dấu chấm câu.

---

### 1.4 🔴 Chữ chỉ dẫn trong prompt bị in thẳng vào ảnh

Model hiểu nhầm các từ định hướng bố cục thành **nhãn cần vẽ**:

| Chữ rác | Ảnh | Lẽ ra phải là |
|---|---|---|
| `STRUCK THROUGH` | ② | `no gradient` |
| `HOT` `ACTIVE` | ② | chỉ `TRAINABLE` |
| `UPPER HALF` | ⑦ | (không có nhãn) |
| `LOWER HALF — why we need this` | ⑦ | (không có nhãn) |
| `UPPER BRANCH` | ⑪ | (không có nhãn) |
| `LOWER BRANCH` | ⑪ | (không có nhãn) |
| `ZONE 1` `ZONE 2` `ZONE 4` | poster | `ENCODERS` / `FUSION` / `RL` |
| toàn bộ dòng đầu prompt | poster | tiêu đề thật |

Đây là lỗi **prompt engineering**, không phải lỗi model. Xem §6 để vá.

---

### 1.5 🟡 Typography và tiêu đề không đồng bộ

- **Có tiêu đề:** 03, 05, 06, 08, 09, 10, 11, poster
- **Không tiêu đề:** 00, 01, 02, 04, 07
- **Monospace:** 05 (nhãn heatmap + caption), 08 (toàn bộ caption)
- **Sans-serif:** 11 ảnh còn lại

Ghép vào một chương báo cáo sẽ thấy rõ sự chắp vá.

**Sửa:** thống nhất — **bỏ hết tiêu đề trong ảnh**, dùng caption Hình 3.x bên dưới ảnh (chuẩn học thuật, và tránh luôn lỗi chữ vỡ).

---

### 1.6 🟡 Ba ảnh X-quang khác nhau cho cùng một pipeline

Ảnh `00`, `02`, `10`, `poster` dùng **3 ảnh X-quang ngực khác nhau**. Người đọc có thể tưởng đây là 3 ca bệnh khác nhau.

**Sửa:** chọn 1 ảnh X-quang duy nhất, dán đè lên cả 4 hình.

---

## 2. Kiểm tra nhất quán số liệu chéo giữa các hình

Đây là thứ hội đồng dò kỹ nhất. Kết quả:

| Đại lượng | Hình khai báo | Nhất quán? |
|---|---|:--:|
| `v_img [B,128]` | 00, 01, 02 (`Linear(512→128)`), poster | ✅ |
| `v_txt [B,64]` | 00, 04 (`32/chiều × 2 = 64`), 05 (`d=64`), poster | ✅ |
| `fused [B,192]` | 00, 08, poster — khớp `128 + 64 = 192` | ✅ |
| GRU hidden `128` | 10, poster | ✅ |
| Transformer `d=64, heads=4` | 05 — `64/4 = 16`/head, hợp lệ | ✅ |
| SE `FC(C→C/4)` | 03 — với `C=32` cho `C/4=8`, hợp lệ | ✅ |
| **image spatial tokens `[B,256,128]`** | 10 | ❌ **MÂU THUẪN** |
| `question hidden states [B,T,64]` | 10 dùng `T`, các hình khác dùng `32` | ⚠️ trôi ký hiệu |
| `cross-attn context [B,128]` | 10 | ⚠️ thiếu phép chiếu |

### 2.1 🔴 `[B,256,128]` mâu thuẫn với chính ảnh ②

Truy ngược nguồn image tokens:

```
Nhánh ResNet-18  : feature map [B,512,7,7]   → 7 × 7   =  49 tokens, dim 512
Nhánh CustomCNN  : feature map [B,128,28,28] → 28 × 28 = 784 tokens, dim 128
Ảnh ⑩ khai báo   : [B, 256, 128]             → 256 tokens, dim 128
```

**Không nhánh nào cho ra 256.** 256 = 16×16, tức phải có một bước `AdaptiveAvgPool2d((16,16))` — hình không vẽ.

Thêm nữa, nếu dùng ResNet thì channel là **512**, muốn ra dim **128** phải có `Linear(512→128)` trên đường spatial tokens — cũng không vẽ. (Cái `Linear(512→128)` trong ảnh ② là cho vector `v_img` sau `AdaptiveAvgPool`, **không phải** cho spatial tokens.)

> **Câu hỏi phản biện chắc chắn sẽ có:** *"256 token ảnh này lấy từ đâu? ResNet-18 chỉ cho 7×7 = 49 mà."*
>
> **Cách trả lời an toàn:** hoặc (a) sửa nhãn thành `[B,49,512] → Linear → [B,49,128]` và vẽ thêm hộp chiếu, hoặc (b) sửa thành `[B,49,128]` và ghi rõ trong thuyết minh là dùng CustomCNN + pooling về 7×7.

### 2.2 🟡 Trôi ký hiệu độ dài chuỗi

Ảnh 04/05 dùng `[B,32,64]` (32 = MAX_QUESTION_LEN cố định), ảnh 10 dùng `[B,T,64]`. Nên thống nhất một trong hai, và định nghĩa `T` ở đâu đó nếu giữ.

### 2.3 🟡 `cross-attn context [B,128]` thiếu bước hợp nhất

Ảnh ⑩ cho decoder attend vào **hai nguồn khác chiều**: image tokens dim `128` và question states dim `64`. Hai context vector này gộp lại phải qua một phép chiếu/nối để ra `[B,128]` — hình không thể hiện. Một reviewer kỹ sẽ hỏi *"em cộng hay nối hai context? Nối thì ra 192 chứ sao 128?"*

---

## 3. LỖI CHÍ TỬ — chi tiết từng ảnh

---

### 3.1 🔴 `prompt_10_gru_decoder` — ảnh trung tâm, sai nặng nhất

#### (a) Model sinh **sai đáp án**

Đọc dãy chip xanh lá dưới các cell GRU (đã crop 1.2× kiểm chứng):

```
bước 1:  <bos>  →  chip "right"
bước 2:            chip "right"     ← lặp lại
bước 3:            chip "<eos>"
bước 4:            chip "..."  (ghost)
```

**Từ `lung` không xuất hiện ở bất kỳ đâu trong chuỗi sinh.** Đáp án đọc được là `right right <eos>`.

Prompt ràng buộc cứng: *"the generated answer must read exactly 'right lung'"* → vi phạm.

Trớ trêu là chữ `lung` **có** xuất hiện, nhưng ở panel `question hidden states` (chip câu hỏi), tức là ở phía **input**, không phải output.

> **Đây là lỗi tệ nhất trong cả bộ.** Hình minh hoạ trung tâm của chương phương pháp đang cho thấy model decode hỏng.

#### (b) 🔴 Mũi tên `autoregressive` chỉ **ngược chiều thời gian**

Crop 1.2× cho thấy rõ hướng đầu mũi:

```
chip bước 2 ──╮
              ╰──►  GRU bước 1     ✗ SAI
chip bước 3 ──╮
              ╰──►  GRU bước 2     ✗ SAI
```

Autoregressive theo định nghĩa là `y_t → input của bước t+1`. Hình đang vẽ `y_t → input của bước t−1`, tức **model dùng tương lai để dự đoán quá khứ**. Mâu thuẫn trực tiếp với công thức `p(y_t | y_{<t}, I, q)` in ngay trên banner.

#### (c) 🔴 `[B,256,128]` — xem §2.1

#### (d) 🟠 Lưới patch vẽ 4×6 = 24 ô

Prompt yêu cầu `4×4`. Nhãn nói `256`. Hình vẽ `24`. Ba con số, không cái nào khớp cái nào.

#### (e) 🟠 Nhãn input khác nhau giữa các cell **giống hệt nhau**

| Cell | `prev token emb [B,64]` | `cross-attn context [B,128]` |
|---|:--:|:--:|
| GRU 1 | ✅ | ✅ |
| GRU 2 | ❌ thiếu | ✅ |
| GRU 3 | ✅ | ❌ thiếu |
| GRU 4 | ✅ (mờ) | ❌ thiếu |

Bốn cell là **cùng một module unroll qua thời gian** → phải có input giống hệt nhau. Vẽ khác nhau khiến người đọc tưởng mỗi bước có kiến trúc khác.

#### (f) 🟠 Cross-attention không khép mạch

Prompt yêu cầu *"query arrows go UP into both panels and context arrows come back DOWN"*. Thực tế:

- Chỉ có mũi tên **đi lên** (2 đường teal vào panel ảnh, 2 đường cam vào panel câu hỏi)
- **Không có mũi tên context đi xuống** trở lại GRU
- **GRU 4 không có đường cross-attention nào**
- Các đường xuất phát từ **mũi tên hidden-state giữa hai cell**, không phải từ bản thân cell

Tức hình nói "decoder gửi query đi rồi thôi, không nhận gì về".

#### (g) 🟠 Logic `<eos>` mâu thuẫn

Bước 3 đã phát `<eos>` — nghĩa là dừng. Nhưng bước 4 vẫn được vẽ với caption `… until <eos> or MAX_ANSWER_LEN`. Nếu đã `<eos>` thì bước 4 không tồn tại.

#### (h) 🟡 `<bos>` in **hai lần** ở bước 1 (một dòng chữ xanh + một nhãn dưới)

#### (i) 🟡 `∏` không có chỉ số

Banner ghi `p(y_1:T | I,q) = ∏ p(y_t | y_<t, I, q)`. Chuẩn học thuật phải là `∏_{t=1}^{T}`.

#### (j) 🟡 `V_answer` không được định nghĩa ở bất kỳ hình nào trong bộ

---

### 3.2 🔴 `prompt_bonus_a_poster` — không dùng được

#### (a) 🔴 Tiêu đề là chữ rác + copy nguyên dòng đầu prompt

Chữ in trên poster:

> **"A large one-page scientific poster infographic summarizer ywllmodul medimad VQA architecture"**

- `ywllmodul medimad` = `multimodal medical` bị vỡ ký tự
- Phần còn lại là **nguyên văn dòng mở đầu của prompt**, không phải tiêu đề

Đây là thứ **đầu tiên** hội đồng nhìn thấy khi bật slide. Không có cách nào chữa ngoài render lại hoặc cắt bỏ hẳn dải tiêu đề.

#### (b) 🔴 Chip câu hỏi ghi `Q??`

Placeholder rác nằm ngay trong ZONE 1, chỗ đáng lẽ là token câu hỏi thật.

#### (c) 🔴 Legend hỏng — hai mục **cùng một màu**

Sample pixel ô legend:

```
= image     teal    ✓ phân biệt được
= text      cam     ┐
= fusion    tím     │ ← "text" và "RL" TRÙNG MÀU
= decoder   xanh dương
= RL        cam     ┘
```

Legend mất hoàn toàn tác dụng: nhìn vào poster không phân biệt được vùng text với vùng RL.

**Nguyên nhân gốc nằm trong prompt của bạn** — bảng màu quy ước gán `#B45309` cho **cả** "Nhánh text" **và** "RL / cảnh báo" (dòng 16 và dòng 20 của `gemini_prompts_kien_truc.md`).

#### (d) 🔴 `ZONE 3` mất nhãn

ZONE 1, ZONE 2, ZONE 4 đều có heading. Vùng generation (xanh dương, giữa) **không có**. Poster có 4 zone nhưng chỉ đánh số 3.

#### (e) 🟠 Đánh số ①–⑪ sai hoàn toàn

| Vấn đề | Chi tiết |
|---|---|
| Thiếu số | `ResNet-18 frozen` và `Transformer Encoder` không có badge, trong khi 2 nhánh anh em (`CustomCNN` ②, `BiLSTM` ⑥) có |
| Số trùng | ⑦ **và** ⑧ cùng nằm trên hộp `Temporal Attention` |
| Gán sai đối tượng | ④ đặt trên `v_img [B,128]` — đây là **tensor tag**, không phải module |
| Thiếu module chính | `Projector MLP` (= hình ⑦ trong báo cáo) **không có badge** |

Kết quả: số trên poster **không tương ứng** với 11 hình chi tiết. Nếu thuyết trình nói "xem hình ⑦" mà poster chỉ vào chỗ khác thì rối.

#### (f) 🟠 Mũi tên luồng chính gãy làm 3 khúc

Prompt yêu cầu *"one thick main flow arrow sweeping left to right"*. Thực tế:

- 3 đoạn rời (teal → xanh dương → cam), ở **3 độ cao khác nhau**
- Đầu mũi teal **đâm vào đuôi** mũi xanh dương
- Mũi cam cuối cùng chỉ ra **khoảng trắng** bên phải, **không vào ZONE 4**
- ZONE 4 nằm **dưới** đường mũi tên → hoàn toàn tách khỏi luồng

#### (g) 🟠 Nhãn đè lên đường biên

`two options per branch = ablation design` nằm vắt qua ranh giới vùng teal / vùng amber, đè lên cả hai. Prompt ràng buộc *"Nothing may overlap or run outside its card"*.

#### (h) 🟡 `SE-\nBlock` bị ngắt từ xấu; footer dùng `.` thay `·`; tỉ lệ 1:1 thay vì 16:9

---

### 3.3 🔴 `prompt_02_resnet18_frozen`

#### (a) 🔴 Topology backbone sai — `layer2`, `layer3` là hộp chết

Crop 2.2× cho thấy: **4 hộp `layer1 layer2 layer3 layer4` KHÔNG có mũi tên nối nhau.**

Đường duy nhất được vẽ trong container:

```
layer1 ──┐
         └──► conv ──► conv ──► ⊕ ──┐
         │                          │
         └──── skip arc ────────────┘
                                    └──► layer4
```

Tức hình đang khẳng định kiến trúc là `layer1 → [conv,conv] → layer4`, và `layer2`, `layer3` **không nằm trên đường tính toán**.

Ngoài ra, residual connection trong ResNet nằm **bên trong mỗi BasicBlock**, không phải bắc cầu từ `layer1` sang `layer4`. Vẽ như hiện tại là mô tả sai ResNet.

> **Phản biện dự kiến:** *"layer2 với layer3 của em nối vào đâu?"* — hiện tại không có câu trả lời từ hình.

**Sửa đúng:** vẽ `layer1 → layer2 → layer3 → layer4` nối tiếp bằng mũi tên, rồi **phóng to riêng một BasicBlock** ra ngoài (callout) để chỉ residual bên trong nó.

#### (b) 🔴 `STRUCK THROUGH` in thẳng vào ảnh

Đây là từ chỉ dẫn style trong prompt (*"STRUCK THROUGH with a red X"*), model tưởng là nhãn. Nhãn đúng phải là `no gradient`.

Tương tự `HOT` và `ACTIVE` (từ *"rendered as HOT and ACTIVE"*) cũng bị in ra bên phải, chồng lên nhãn `TRAINABLE` thật.

#### (c) 🟠 Mũi tên gradient đặt sai vị trí

Mũi tên xanh dương + chữ X đỏ nằm **trọn bên trong** hộp frozen. Để truyền đạt đúng "gradient không chảy ngược vào backbone", nó phải:

- **xuất phát** từ hộp `Linear(512→128)` trainable bên phải
- **bị chặn tại đường biên** của container frozen

Hiện tại nó trông như "có một mũi tên bị gạch nằm trong hộp", không nói lên quan hệ với phần trainable.

#### (d) 🟠 Ba nhãn xoay dọc 90°

`SE-Block (optional)`, `AdaptiveAvgPool → [B,512]`, `Linear(512→128) + LayerNorm` — cả ba xoay dọc. Chiếu slide gần như không đọc được, và người đọc phải nghiêng đầu.

#### (e) 🟡 Emoji màu (🔒 ❄️ 🔥 🚫) trộn vào phong cách flat vector

Prompt yêu cầu *"flat vector infographic"*. Emoji hệ thống có màu gradient riêng, phá vỡ bảng màu. Riêng 🚫 ở góc dưới phải không chú thích cho cái gì.

#### (f) 🟡 `[B,512,7,7]` đặt dưới container như caption, không rõ là output của container hay của cả cụm

---

### 3.4 🔴 `prompt_08_fusion_concat`

Đây là hình mà **toàn bộ thông điệp nằm ở tỉ lệ độ dài** — và tỉ lệ bị sai.

#### (a) 🔴 Tỉ lệ `128 : 64 : 192` sai

Đếm ô vuông trên crop 1.3×:

| Thanh | Đếm được | Phải là | Kết luận |
|---|:--:|:--:|---|
| `v_img [B,128]` | 24 ô | 24 | ✅ |
| `v_txt [B,64]` | 12 ô | 12 | ✅ (đúng 2:1) |
| `fused` — phần **teal** | **~12 ô** | **24** | ❌ |
| `fused` — phần **amber** | **~13 ô** | **12** | ❌ |
| `fused` — **tổng** | **~25 ô** | **36** | ❌ |

Hai lỗi độc lập:

1. **Thanh fused bị chia 50/50** thay vì 2:1. Nhìn vào hình, người ta đọc ra `v_img` và `v_txt` **bằng nhau**, tức `64 + 64 = 128`, mâu thuẫn với nhãn `[B,192]`.
2. **Tổng độ dài thanh fused ≈ độ dài thanh `v_img`**, không phải tổng hai thanh trên. Prompt ràng buộc rõ: *"total length exactly equal to the sum of the two strips above"*.

> **Phản biện dự kiến:** *"Hình của em cho thấy hai vector bằng nhau, sao lại ra 192?"* — không trả lời được, vì hình sai.

#### (b) 🔴 Ký hiệu `concat` vẽ bằng **dấu CỘNG**

Ở giữa hình là một dấu `+` tím to, bên trong ghi chữ `concat`.

`+` = phép cộng element-wise, đòi hỏi **hai vector cùng chiều** và cho ra vector **cùng chiều đó**. Concatenation là phép **nối**, cho ra chiều bằng **tổng**. Hai phép này ngược nhau về bản chất, và hình đang dùng ký hiệu của phép sai.

Prompt viết `"a large purple ⊕ symbol"` — bản thân `⊕` cũng là ký hiệu **cộng trực tiếp (direct sum)**, dễ gây nhầm. Ký hiệu chuẩn cho concat là `[·;·]` hoặc `⊕` với chú thích rõ, an toàn nhất là **ghi chữ `concat` trong hộp chữ nhật**, không dùng ký hiệu toán.

#### (c) 🟠 `→ GRU Decoder` — mũi tên bị in thành **chữ trong hộp**

Hộp bên phải chứa chuỗi ký tự `→ GRU Decoder`. Mũi tên phải là **đường vẽ ngoài hộp**.

**Nguyên nhân gốc trong prompt** (dòng 215): `A blue arrow exits right into a box labeled "→ GRU Decoder"` — dấu `→` nằm trong ngoặc kép nên bị hiểu là nội dung nhãn.

#### (d) 🟡 Nhãn `fused [B,192]` nằm trong một hộp trắng **rời** phía dưới thanh, trông như một phần tử trống bị lỗi

#### (e) 🟡 Không có ngoặc kích thước `192 dims` cho thanh fused (trong khi 2 thanh trên đều có `128 dims` / `64 dims`)

#### (f) 🟡 Caption dùng monospace, lệch với toàn bộ phần còn lại của bộ ảnh

---

### 3.5 🔴 `prompt_04_bilstm`

#### (a) 🔴 Nhánh backward **không nối vào** hộp concat

Crop 2.0× xác nhận: hộp `concat(last h→, first h←) → [B,64]` chỉ nhận **một** mũi tên duy nhất, từ ô LSTM forward cuối cùng (hàng trên).

Ô LSTM backward **ngoài cùng bên trái** — chính là nơi chứa `first h←` — **không có mũi tên đi ra**. Chuỗi backward kết thúc ở đó và dừng.

Tức hình ghi nhãn là "nối 2 chiều" nhưng chỉ vẽ 1 chiều. Đây là **chính xác điểm cốt lõi** của BiLSTM bị vẽ hỏng.

#### (b) 🔴 Mũi tên `Temporal Attention` chỉ **sai chiều**

Crop cho thấy đầu mũi nhọn đâm **vào** ô LSTM backward:

```
Temporal Attention ────►  LSTM (backward)     ✗ NGƯỢC
```

Attention pooling phải **tiêu thụ** toàn bộ chuỗi hidden states, không phải **cấp** đầu vào cho LSTM.

Hệ quả kéo theo: hộp `Temporal Attention over [B,32,64]` **không có input nào cả** — nó lấy `[B,32,64]` từ đâu thì hình không nói.

**Sửa đúng:** vẽ một "bus" gom output của cả 5 timestep (cả 2 chiều) → đi vào hộp Temporal Attention → ra `[B,64]` → LayerNorm.

#### (c) 🟠 Hai nhãn xoay dọc 90°: `Embedding(V, 64)` và `LayerNorm`

#### (d) 🟠 Nhãn `hidden 32 per direction → 64 total` đặt sai chỗ

Nó nằm ngay dưới **ô LSTM backward cuối cùng**, trông như chú thích riêng cho ô đó. Đây là thông tin về **cả hai hàng** → phải đặt ở lề trái, ngang giữa 2 hàng, hoặc trong một hộp note riêng.

#### (e) 🟡 `backward h←` màu quá nhạt

Chữ dùng màu amber nhạt trên nền trắng xanh → gần như chìm. Chiếu máy chiếu sẽ mất. Trong khi `forward h→` dùng amber đậm, đọc rõ. Tương phản giữa 2 hàng lẽ ra là điểm nhấn, nhưng cách tô màu này làm hàng backward "biến mất".

#### (f) 🟡 Vị trí pill không nhất quán

- `no-attention mode` nằm **bên trong** hộp concat
- `attention mode` nằm **bên ngoài, phía dưới** hộp Temporal Attention

Hai pill cùng vai trò (nhãn chế độ) nhưng đặt theo 2 quy tắc khác nhau.

#### (g) 🟡 `h->`, `h<-`, `-> [B,64]` — ASCII

#### (h) 🟡 ~28% khoảng trắng phía trên, ~20% phía dưới

---

### 3.6 🔴 `prompt_03_se_block`

#### (a) 🔴 Số slab ≠ số bar → mapping 1-1 gãy

Đếm trên crop 1.3×:

```
Stack F  (trái) : 7 slab
Bar chart       : 6 bar   (0.9, 0.2, 0.7, 0.95, 0.1, 0.5)
Stack F' (phải) : 7 slab
```

Prompt ràng buộc *"a stack of 6 isometric channel slices"* và 6 giá trị. Hình vẽ 7. Điểm cốt lõi của hình — *"mỗi kênh có một trọng số riêng"* — không thể đọc được vì có 1 slab thừa không ứng với bar nào.

#### (b) 🔴 Stack `F` **không đồng màu** → mất tương phản before/after

Prompt: *"ALL THE SAME pale uniform color"*.

Thực tế 7 slab của stack `F` **đậm dần từ trên xuống dưới**. Nghĩa là hình đang nói "trước khi qua SE, các kênh đã có trọng số khác nhau rồi" — phá huỷ toàn bộ luận điểm.

So sánh `F` với `F'` giờ không còn ý nghĩa vì cả hai đều không đồng nhất.

#### (c) 🟠 Ánh xạ trọng số → cường độ màu **sai**

| Bar | Giá trị | Slab tương ứng phải là | Thực tế |
|---|:--:|---|---|
| 1 | 0.9 | rất đậm | slab trên cùng — **nhạt nhất** ❌ |
| 4 | 0.95 | đậm nhất | không xác định được ❌ |
| 5 | 0.1 | gần trong suốt | ✅ có 2 slab gần trong suốt |
| — | — | — | slab cuối **MÀU XANH LÁ** ❌ |

Slab dưới cùng của `F'` tô **xanh lá đậm** — một hue hoàn toàn khác họ teal. Nếu cường độ mã hoá trọng số thì xanh lá nghĩa là gì? Không có gì. Prompt dùng `#15803D` cho *"the final result"* và model hiểu nhầm là tô cho một kênh.

#### (d) 🟠 Đường dashed không nối rõ bar nào với slab nào

6 đường đứt nét xuất phát từ **một bó chung** ở góc trên bar chart, không phải từ **đỉnh từng bar**, rồi toả ra 7 slab. Không đọc được cặp nào ứng với cặp nào — mà đây chính là thứ prompt gọi là *"KEY VISUAL LINK"*.

#### (e) 🟠 Typo: `Global Avg Avg Pool`

Chữ `Avg` lặp **hai lần**. Phải là `Global Avg Pool`.

#### (f) 🟠 `Scale: s x F` — sai ký hiệu

Phải là `s ⊙ F` (tích Hadamard, element-wise). Viết `x`/`×` với hai đối tượng tensor nghĩa là **nhân ma trận**, sai hoàn toàn về mặt toán. Vòng tròn phép toán ở giữa hình cũng chứa `×` thay vì `⊙`.

> **Phản biện dự kiến:** *"s là vector [B,C], F là tensor [B,C,H,W]. Em nhân ma trận kiểu gì?"*

#### (g) 🟡 Hộp `s in (0,1)^C` **đè lên chân các bar**, cắt ngang baseline của bar chart

#### (h) 🟡 Nhãn `F` lặp hai lần dưới stack trái (`F` rồi `F [B,C,H,W]`), trong khi stack phải chỉ có `F'` + `[B,C,H,W]` → bất đối xứng

#### (i) 🟡 `s in (0,1)^C` — dùng `in` thay `∈`; `FC(C->C/4)` — ASCII

---

## 4. LỖI NẶNG — sửa được bằng đè chữ / vẽ lại cục bộ

---

### 4.1 🟠 `prompt_09_gated_attention`

#### (a) 🔴 Công thức sai ký hiệu — **ngay tiêu đề**

In trên ảnh:

```
z' = z × sigma(W.z)
```

Đúng phải là:

```
z' = z ⊙ σ(Wz)
```

Ba lỗi trong một dòng:
1. `×` thay `⊙` — mà prompt **ràng buộc rõ** *"this is an element-wise sigmoid gate"*. Ký hiệu `×` nói ngược lại.
2. `sigma` thay `σ`
3. `W.z` — dấu chấm đọc như dấu câu

Vòng tròn phép toán ở giữa hình cũng là `⊗` chứa `×`.

#### (b) 🟠 Dãy số **chồng lên nhau**

In ra: `0.92 — 0.11 — 0.780.050.05`

- Ba số cuối **dính liền thành một cục** không đọc được
- Có **5 số** trong khi prompt chỉ yêu cầu 4 (`0.92, 0.11, 0.78, 0.05`) → `0.05` bị nhân đôi
- Leader line là **gạch ngang**, không phải đường chỉ xuống ô cụ thể → không số nào gắn với ô nào

#### (c) 🟠 `z' ≠ z ⊙ gate` — hình tự mâu thuẫn với chính nó

Vì `z` được vẽ **đồng nhất** (tất cả ô cùng một màu tím), nên theo công thức, `z'` phải sao chép **y hệt** pattern của `gate`, ô-đối-ô.

Đối chiếu từng ô trên crop 2.2×:

| Ô | `gate` | `z'` | Khớp? |
|:--:|---|---|:--:|
| 1 | tím đậm | tím vừa | ❌ |
| 2 | tím nhạt vừa | gần trắng | ❌ |
| 4 | tím vừa | gần trắng | ❌ |
| 5 | tím vừa | **phát sáng radial** | ❌ |
| 10 | tím vừa | rất nhạt | ❌ |

Các đường dashed 1-1 đang **chứng minh điều ngược lại** với thông điệp của hình. Đây là loại lỗi mà nếu ai đó soi kỹ trên slide, sẽ hỏi ngay *"sao ô này đậm mà đầu ra lại nhạt?"*

#### (d) 🟠 Tiêu đề sai ngữ pháp

> *"Learned sigmoid gate element-wise to, fused feature vector"*

Thiếu từ `applied`, thừa dấu phẩy. Câu không có nghĩa.

#### (e) 🟡 10 ô mỗi dải thay vì 12 như prompt

#### (f) 🟡 `values 0..1` — ký pháp lập trình, nên là `∈ [0,1]`

#### (g) 🟡 Nhãn `skip` đặt sai khái niệm

Nhánh dưới (đường `z` đi thẳng vào phép nhân) là **đường chính** — nó chính là toán hạng thứ nhất của công thức. Nhánh gate mới là nhánh phụ. Gọi đường chính là `skip` gây hiểu nhầm rằng bỏ đi vẫn được.

#### ✅ Điểm đúng
- Mũi tên `noisy dim suppressed` **có** chỉ đúng vào một ô nhạt ✓
- Mũi tên `useful dim kept` chỉ đúng vào ô đậm ✓
- Note `Same dim in and out (D→D)` chính xác ✓

---

### 4.2 🟠 `prompt_11_reinforce_scst`

#### (a) 🔴 Loss có **hai dấu trừ**

In trên ảnh:

```
Loss = −
     − ( log pi(sample) . advantage ).mean()
```

Hai dấu `−` liên tiếp (một cuối dòng 1, một đầu dòng 2) → đọc thành `Loss = + (...)`.

Sai dấu loss nghĩa là **gradient ascent thay vì descent** — model sẽ học ngược. Đây là lỗi toán, và là thứ một giáo sư RL nhìn 2 giây là thấy.

Công thức đúng: `Loss = −( log π(sample) · advantage ).mean()`

#### (b) 🟠 `0.5.EM + 0.5.token F1`

Dấu `·` bị render thành dấu chấm → `0.5.EM` đọc như một số bị lỗi định dạng. Phải là `0.5·EM + 0.5·F1`.

#### (c) 🟠 Hộp greedy có gạch ngang trông như **gạch phân số**

```
   Greedy answer = argmax
  ────────────────────────    ← đọc thành phép CHIA
      reward(greedy)
```

Người đọc lướt qua sẽ hiểu là `argmax / reward(greedy)`. Trong khi hộp `Sample answer` bên trên dùng mũi tên `↓` (đúng). Hai hộp song song mà dùng 2 quy ước khác nhau.

#### (d) 🟠 `Advantage` và `Loss` bị gộp vào **một hộp**

Prompt tách rõ bước 3 (`Advantage = r(sample) − r(greedy)`) và bước 4 (`Loss = ...`). Gộp lại làm mất trình tự tính toán, và không có mũi tên giữa hai công thức.

#### (e) 🟠 Bar chart không có trục, không cùng baseline, tỉ lệ sai

| So sánh | Đo được | Đúng phải là |
|---|---|---|
| CASE A: `r=0.9` vs `r=0.5` | ~125px vs ~90px → **1.39 : 1** | **1.8 : 1** |
| CASE B: `r=0.9` vs `r=0.1` | ~120px vs ~20px → **6 : 1** | **9 : 1** |
| `r=0.9` ở CASE A vs CASE B | **125px vs 120px** | phải bằng nhau |

Cùng một giá trị reward mà hai panel vẽ hai chiều cao khác nhau, lại không có trục y để đối chiếu → bar chart này không mang thông tin định lượng nào.

#### (f) 🟡 Bố cục CASE A và CASE B **đảo ngược nhau**

- CASE A: mũi tên xanh → chữ `push...UP` → `advantage = +0.4`
- CASE B: chữ `push...DOWN` → mũi tên đỏ → `advantage = -0.8`

Hai case song song phải có thứ tự đọc giống hệt nhau.

#### (g) 🟡 In đậm vắt qua ranh giới câu

> *"...high variance on sparse **rewards. Order matters:** SFT first, then RL."*

Phần bold bắt đầu giữa câu trước và kết thúc giữa câu sau.

#### (h) 🟡 `Unseen answer —> reward = 0` — dùng em-dash + `>`

#### ✅ Điểm đúng
- `advantage = 0.9 − 0.5 = +0.4` ✓ và `0.1 − 0.9 = −0.8` ✓ — tính chuẩn
- Vòng lặp **khép kín** đúng (mũi tên đứt nét quay về hộp model) ✓
- **Không vẽ critic / value network** ✓ — đúng tinh thần self-critical
- Màu: nhánh sample cam, nhánh greedy xám ✓

---

### 4.3 🟠 `prompt_07_projector_mlp`

#### (a) 🔴 Một `Linear` không thể nhận **hai số chiều khác nhau**

Hình vẽ **một hộp MLP duy nhất** nhận đồng thời:
- `v_img [B,128]`
- `v_txt [B,64]`

với nhãn `MLP: Linear(d→P) → GELU → Linear(P→P) → LayerNorm` và sublabel `img 128→P · txt 64→P`.

**Về mặt toán học điều này bất khả thi.** Một lớp `nn.Linear(in_features, out_features)` có ma trận trọng số cố định `W ∈ R^{out×in}` — nó chỉ nhận đúng một `in_features`.

Thực tế phải là **hai projector độc lập, không chia sẻ trọng số**:

```python
self.proj_img = nn.Sequential(nn.Linear(128, P), nn.GELU(), nn.Linear(P, P), nn.LayerNorm(P))
self.proj_txt = nn.Sequential(nn.Linear(64,  P), nn.GELU(), nn.Linear(P, P), nn.LayerNorm(P))
```

> **Phản biện gần như chắc chắn:** *"Cùng một MLP mà sao nhận được cả input 128 chiều lẫn 64 chiều?"*
>
> **Nguyên nhân gốc nằm ở prompt của bạn** (dòng 191): *"Both connect via smooth curved bezier lines into **one central purple box**"*. Model vẽ đúng theo yêu cầu — yêu cầu mới là cái sai.

#### (b) 🔴 `UPPER HALF` và `LOWER HALF — why we need this` in thẳng vào ảnh

#### (c) 🟠 `d` không được định nghĩa ở đâu

Nhãn `Linear(d→P)` dùng biến `d` mà không có chú thích `d ∈ {128, 64}`. Sau khi tách 2 hộp thì vấn đề này tự hết.

#### (d) 🟠 Hộp `concat [B,2P]` màu **trắng/xám**

Legend toàn bộ bộ ảnh quy ước **fusion = tím `#7C3AED`**, và ảnh ⓪ vẽ hộp `Concat` màu tím. Ở đây lại vẽ trắng. Mâu thuẫn chéo hình.

#### (e) 🟠 Scatter plot là **ẩn dụ**, không phải dữ liệu — nhưng không được ghi chú

Đây là chỗ dễ bị hỏi nhất của hình này:

- **Trước** projector, hai không gian có **số chiều khác nhau** (128 vs 64). Về nguyên tắc **không thể** vẽ chúng lên cùng một mặt phẳng 2D.
- Không có nhãn trục, không ghi phương pháp giảm chiều, không ghi số mẫu.

> **Phản biện dự kiến:** *"Đây là t-SNE hay UMAP? Chạy trên bao nhiêu mẫu? Perplexity bao nhiêu? Mà sao vector 128 chiều với 64 chiều lại vẽ chung được?"*
>
> **Bắt buộc thêm caption:** *"Minh hoạ khái niệm — không phải kết quả đo thực nghiệm."*

Nếu muốn giữ hình này mà không bị hỏi, cách an toàn là **thay 2 panel scatter bằng 2 sơ đồ khối trục số** (vẽ 2 trục riêng biệt trước → 1 trục chung sau).

#### (f) 🟡 Vài đường dashed trong panel "After" nối **teal với teal**, và nối 2 điểm ở hai đầu panel — mâu thuẫn với ý *"matching pairs sit close together"*

#### (g) 🟡 Hộp `concat [B,2P]` là **ngõ cụt** — không có mũi tên đi tiếp sang decoder

---

### 4.4 🟠 `prompt_06_temporal_attention`

#### (a) 🟠 Bar `<pad>` **không bằng 0** — mâu thuẫn với chính caption của nó

Prompt ràng buộc: *"the two `<pad>` bars have **ZERO height**"* và *"The two zero-height masked PAD bars must be obvious"*.

Thực tế (crop 1.5×): hai ô xám cao **~110px** — **cao ngang bar `0.12`** của token `?`.

Ngay bên cạnh là caption đỏ `PAD → alpha = 0 (masked)`.

> **Phản biện dự kiến:** *"Em bảo α = 0 mà sao cột vẫn có chiều cao gần bằng cột 0.12?"*

#### (b) 🟠 Có một bộ dây nối **thừa, đi vào hư không**

Crop cho thấy **hai hệ thống nối chồng lên nhau**:

1. **Đúng:** các đường chéo từ đỉnh bar → hội tụ vào hộp `context vector` (có đầu mũi tên) ✓
2. **Thừa:** một "thanh ray" ngang ở phía trên, gom các bar `is`, `there`, `pleural`, `effusion` bằng đường vuông góc, rồi **kết thúc lơ lửng** ở giữa hình — kèm **một mũi tên chỉ xuống nền trống**

Bar `?` (0.12) có **hai đường nối cùng lúc** (một lên ray, một chéo sang context).

#### (c) 🟠 Hai hộp `context` trùng lặp, không nối với nhau

- Hàng flow phía trên kết thúc bằng hộp xanh lá `context [B,H]`
- Phần hero phía dưới lại có hộp xanh lá `context vector`

Hai hộp cùng màu, cùng tên, **không có đường nào nối**.

> **Phản biện dự kiến:** *"Hai cái `context` này là một hay là hai cái khác nhau?"*

#### (d) 🟠 Baseline bar không đồng nhất

- Bar `effusion` và `pleural`: **dính sát** chip token
- Bar `?`, `is`, `there`: **treo lơ lửng** phía trên chip, có một đoạn nối ngắn

Bar chart phải có chung một đường đáy, nếu không thì so sánh chiều cao mất ý nghĩa.

#### (e) 🟡 Bar `0.42` màu **nâu đậm** khác hẳn 4 bar còn lại (cam vàng) → đọc như một hạng mục khác chứ không phải "cùng loại, giá trị cao hơn"

#### (f) 🟡 Nhãn `alpha_t [B,T], Sum alpha = 1` **chạm vào viền** hộp `Sum alpha_t . h_t` bên cạnh

#### (g) 🟡 `Sum`, `alpha`, `.` — ASCII thay `Σ`, `α`, `·`

#### ✅ Điểm đúng
- Tổng trọng số: `0.42 + 0.31 + 0.12 + 0.08 + 0.07 = 1.00` ✓ **chuẩn**
- Hai bar `<pad>` **không** nối vào `context vector` ✓ (đúng về mặt logic masking)
- Thứ tự cao–thấp của bar khớp với giá trị ✓

---

### 4.5 🟠 `prompt_05_transformer_encoder`

#### (a) 🟠 Heatmap attention sai điểm nhấn — và **phản tác dụng về mặt khoa học**

Prompt yêu cầu: *"the cell at row `effusion`, column `pleural` is the darkest"*.

Thực tế: ô đậm nhất nằm trên **đường chéo** tại `(effusion, effusion)`, và **cả đường chéo đều đậm** trong khi các ô ngoài đường chéo nhạt.

Pattern đường chéo nghĩa là: *"mỗi token chỉ chú ý vào chính nó"* — đây là pattern **vô nghĩa nhất có thể** của self-attention, tương đương với việc attention không học được gì. Nó mâu thuẫn trực tiếp với caption ngay bên dưới: *"every token attends to every token, in parallel"*.

> **Phản biện dự kiến:** *"Nhìn heatmap này thì self-attention của em chỉ attend vào chính nó. Vậy nó có tác dụng gì hơn một MLP theo từng token?"*

**Sửa:** vẽ lại heatmap sao cho `(effusion, pleural)` là ô đậm nhất, đường chéo ở mức trung bình.

#### (b) 🟠 Ký hiệu `⊕` đặt ở **điểm rẽ nhánh** thay vì **điểm hợp nhất**

Trong panel zoom, `⊕` nằm ở **dưới** MHSA — tức tại chỗ đường skip **tách ra**.

`⊕` là ký hiệu **phép cộng**. Phép cộng chỉ xảy ra ở chỗ hai đường **gặp lại nhau**, không phải chỗ chúng tách ra. Điểm rẽ phải vẽ bằng **chấm tròn đặc** (junction dot).

#### (c) 🟠 Phép cộng bị **đếm hai lần**

Hộp `Add & Norm` — chữ "Add" đã bao hàm phép cộng residual. Vẽ thêm `⊕` riêng bên ngoài là dư thừa và gây hiểu nhầm rằng có 2 phép cộng.

Chọn **một trong hai**:
- Cách A: `⊕` ở điểm hợp nhất + hộp chỉ ghi `LayerNorm`
- Cách B: không có `⊕`, hộp ghi `Add & Norm`

#### (d) 🟡 Nhãn `×2` bị nhân đôi

- Trong hộp: `2x Transformer-EncoderLayer`
- Ngoài hộp, góc trên phải: `x2` (màu cam)

Cùng một thông tin ghi 2 lần, 2 kiểu.

#### (e) 🟡 Ngắt từ xấu: `Transformer-EncoderLayer` (gạch nối giữa tên class)

#### (f) 🟡 Nhãn cột heatmap lộn xộn trong cùng một trục

`is`, `there`, `?` → nằm **ngang**
`pleural`, `effusion` → **nghiêng ~45°**

#### (g) 🟡 Một đường dashed callout **cắt chéo qua góc trên trái** của heatmap

#### (h) 🟡 `Question tokens [B,32]` là **chữ trần**, không có hộp, trong khi mọi block khác đều có hộp

#### (i) 🟡 Panel zoom chạy **dưới → lên**, trong khi flow chính chạy **trái → phải**. Hai hướng đọc trong một hình.

#### (j) 🟡 Font monospace cho heatmap + caption, sans-serif cho phần còn lại

---

## 5. LỖI NHẸ — hai ảnh dùng được

### 5.1 🟡 `prompt_01_custom_cnn`

#### (a) 🟠 Không trục nào đúng tỉ lệ

Prompt viết: *"The viewer must instantly see 'space shrinks, depth grows'"*. Đo thực tế:

| Trục | Phải là | Vẽ thành |
|---|---|---|
| **Chiều cao slab** (không gian 224→112→56→28) | giảm **½** mỗi bước (tỉ lệ 2.0) | tỉ lệ **~1.39** mỗi bước |
| **Số slab** (kênh 32→64→128) | **gấp đôi** mỗi bước | **3 → 5 → 7** |

Người xem sẽ thấy xu hướng đúng, nhưng nếu ai đối chiếu với con số trên nhãn thì hình không khớp.

#### (b) 🟠 Stack đầu tiên vẽ **3 slab** cho `Conv 3→32`

Con số 3 trùng với **số kênh input** (ảnh RGB 3 kênh). Người đọc rất dễ hiểu nhầm stack này biểu diễn input chứ không phải output 32 kênh.

**Sửa:** cho stack đầu ít nhất 4 slab, và ghi nhãn `32 ch` ngay cạnh stack.

#### (c) 🟡 `AvgPool + LayerNorm` xuất hiện **hai lần** — một trong hàng flow, một trong hàng nhãn phía dưới

#### (d) 🟡 Hộp dashed `SE-Block` trong hàng flow **rỗng**, không có chữ (nhãn chỉ có ở hàng dưới)

#### (e) 🟡 Hàng nhãn phía dưới **không có mũi tên nào** → nó tách rời khỏi hàng hình, trông như legend chứ không như một flow

#### (f) 🟡 `3x224x224`, `Conv 3->32`, `MaxPool ->` — ASCII

#### ✅ Điểm đúng
- Phép tính spatial chuẩn: `224 → conv(pad same) → 224 → maxpool/2 → 112` ✓
- Chuỗi `32×112×112 → 64×56×56 → 128×28×28` ✓
- Kết thúc `v_img [B,128]` khớp với ảnh ⓪ và ② ✓

---

### 5.2 🟡 `prompt_00_pipeline` — **ảnh sạch nhất bộ, nên giữ**

#### (a) 🟠 `Concat [B,192] or [B,2P]` mâu thuẫn với đường dây vào nó

Cả hai mũi tên vào hộp Concat đều xuất phát từ Projector, mang `[B,P]`. Hai `[B,P]` nối lại **chỉ có thể** ra `[B,2P]`.

Muốn ra `[B,192]` thì phải **bỏ qua** Projector (nối thẳng `v_img [B,128]` + `v_txt [B,64]`) — nhưng **đường bypass không được vẽ**.

> **Phản biện dự kiến:** *"Khi nào thì ra 192, khi nào ra 2P? Hình chỉ vẽ một đường."*

**Sửa:** thêm hai đường đứt nét đi vòng qua Projector, ghi nhãn `bypass (no projector)`.

#### (b) 🟡 Ô legend `fusion` có **gradient tím–xanh**, trong khi 2 ô kia phẳng — và prompt cấm gradient

#### (c) 🟡 Nhãn `logits [B,T,V_answer]` đặt lơ lửng **giữa hai hộp**, không rõ thuộc `GRU Decoder` hay `Greedy/Beam Search`

#### (d) 🟡 ~50% canvas là khoảng trắng

---

## 6. Vá lỗi ở PROMPT GỐC

**6 lỗi nằm trong `gemini_prompts_kien_truc.md`, không phải lỗi của Gemini.** Render lại mà không sửa những chỗ này thì sai y hệt.

### 6.1 Bảng màu trùng — nguyên nhân legend poster hỏng

`gemini_prompts_kien_truc.md` dòng 16 và 20:

```diff
- | Nhánh text    | amber/cam | `#B45309` |
- | RL / cảnh báo | cam đậm   | `#B45309` |
+ | Nhánh text    | amber     | `#D97706` |
+ | RL / cảnh báo | cam đậm   | `#B45309` |
```

### 6.2 Prompt ⑦ mô tả sai kiến trúc (dòng 191)

```diff
- Both connect via smooth curved bezier lines (never right-angle elbows)
- into one central purple box labeled
- "MLP: Linear(d→P) → GELU → Linear(P→P) → LayerNorm" with sublabel
- "img 128→P · txt 64→P".
+ Each input connects to its OWN separate purple box. Draw TWO distinct
+ boxes stacked vertically:
+   upper: "Img Projector: Linear(128→P) → GELU → Linear(P→P) → LayerNorm"
+   lower: "Txt Projector: Linear(64→P)  → GELU → Linear(P→P) → LayerNorm"
+ These are two INDEPENDENT modules with NO shared weights. Never merge
+ them into a single box.
```

### 6.3 Prompt ⑧ khiến mũi tên bị in thành chữ (dòng 215)

```diff
- A blue arrow exits right into a box labeled "→ GRU Decoder"
+ A blue arrow exits right into a box labeled "GRU Decoder".
+ The arrow must be a drawn line with an arrowhead, never a text character.
```

### 6.4 Chặn chữ chỉ dẫn bị in vào ảnh — thêm vào **cuối mọi prompt**

```
NEVER render my structural or style words as visible text in the image.
The following words must NEVER appear anywhere in the picture:
UPPER, LOWER, BRANCH, HALF, ZONE, STRUCK THROUGH, HOT, ACTIVE,
MAIN VISUAL IDEA, CORE VISUAL METAPHOR, STYLE, CONSTRAINTS, SEQUENCE OF BOXES.
Do not print this prompt, or any part of it, as a title or caption.
The image must have NO title text at all.
```

### 6.5 Ép ký hiệu toán đúng — thêm vào **cuối mọi prompt**

```
Render mathematical symbols as true Unicode glyphs:  →  ⊙  ⊕  ⊗  σ  π  α  Σ  ∈  ·  ×
NEVER write them as ASCII substitutes: -> , x , * , sigma , pi , alpha , Sum , in , .
Element-wise (Hadamard) product is ⊙ — never × and never *.
Matrix/scalar product is × . Do not mix the two.
```

### 6.6 Ép tỉ lệ định lượng — thêm vào prompt ①, ③, ⑥, ⑧, ⑨, ⑪

```
QUANTITATIVE CONSTRAINT — highest priority, overrides all aesthetic choices:
the drawn pixel length / height / element count of every shape MUST be exactly
proportional to the number printed on its own label. If a label says 128 and
another says 64, the first shape must be exactly twice as long as the second.
If a bar is labeled 0.42 and another 0.12, the first must be exactly 3.5×
as tall, measured from a SHARED baseline. Verify all ratios before output.
```

### 6.7 Ép tỉ lệ khung hình

Cả 13 lần render đều cho 1:1. Nếu công cụ có tham số `--ar` thì dùng; nếu không, thêm:

```
Canvas aspect ratio MUST be 16:9 landscape (e.g. 1920×1080).
Do not output a square image. Fill the full width — no large empty margins
at the top or bottom.
```

---

## 7. Kế hoạch xử lý

### Bước 1 — Bỏ ngay, không dùng (6 ảnh)

| Ảnh | Lý do chí tử |
|---|---|
| `bonus_a_poster` | Tiêu đề chữ rác `ywllmodul medimad` + legend trùng màu + ZONE 3 mất nhãn |
| `10_gru_decoder` | Sinh sai đáp án (`right right` thay `right lung`) + autoregressive ngược chiều + `[B,256,128]` mâu thuẫn |
| `08_fusion_concat` | Tỉ lệ `128:64:192` sai + `+` thay concat |
| `02_resnet18_frozen` | `layer1–4` không nối + `STRUCK THROUGH` |
| `04_bilstm` | Backward không nối concat + attention ngược chiều |
| `03_se_block` | 7 slab vs 6 bar + `Avg Avg` + `s x F` |

### Bước 2 — Chỉnh chữ trong PowerPoint / Figma (5 ảnh)

| Ảnh | Việc cần làm |
|---|---|
| `09` | Đè công thức `z' = z ⊙ σ(Wz)`; tách dãy số; **vẽ lại dải `z'`** cho khớp `gate` ô-đối-ô; sửa tiêu đề |
| `11` | Xoá 1 dấu trừ thừa; `0.5·EM`; xoá `UPPER/LOWER BRANCH`; đổi gạch phân số thành `↓`; **đổi nền sang `#F6F8FB`** |
| `07` | **Tách hộp MLP thành 2 hộp**; xoá `UPPER HALF`/`LOWER HALF`; tô tím hộp concat; thêm caption *"minh hoạ khái niệm"* |
| `06` | **Hạ 2 bar `<pad>` về 0**; xoá thanh ray thừa + mũi tên lơ lửng; gộp/nối 2 hộp `context` |
| `05` | **Vẽ lại heatmap** (đậm nhất tại `effusion × pleural`); xoá `x2` thừa; dời `⊕` về điểm hợp nhất |

### Bước 3 — Dùng được, chỉnh nhẹ (2 ảnh)

| Ảnh | Việc cần làm |
|---|---|
| `00` | Thêm đường bypass đứt nét vòng qua Projector; crop về 16:9 |
| `01` | Ghi `32 ch` cạnh stack đầu; xoá hộp `AvgPool + LayerNorm` trùng; điền chữ vào hộp `SE-Block` rỗng |

### Bước 4 — Áp dụng cho toàn bộ ảnh giữ lại

- [ ] Crop/render lại về **16:9**
- [ ] Thay hết ASCII bằng ký hiệu Unicode (`→ ⊙ σ α Σ ∈ ·`)
- [ ] **Bỏ mọi tiêu đề trong ảnh**, chuyển thành caption `Hình 3.x — ...` bên dưới
- [ ] Thống nhất **một** ảnh X-quang cho cả bộ
- [ ] Kiểm tra lại `[B,256,128]` — sửa thành `[B,49,128]` (ResNet + chiếu 512→128) hoặc bổ sung hộp `AdaptiveAvgPool2d(16,16)`
- [ ] Thống nhất `[B,32,64]` vs `[B,T,64]`

---

## 8. Khuyến nghị chiến lược

Chính file prompt của bạn đã có sẵn lời khuyên đúng nhất, ở mục **"Mẹo dùng khi ảnh ra chưa ưng"**, điểm 1:

> *"**Chắc ăn nhất:** thêm vào cuối prompt `Leave the label areas empty — draw the boxes and arrows without any text.` → xuất ảnh không chữ, rồi bạn tự chèn text bằng PowerPoint/Figma. Chữ chuẩn 100%, và sửa được về sau."*

**Với 6 ảnh nhóm đỏ, nên bỏ hẳn Gemini và vẽ tay.** Lý do:

Các lỗi chí tử còn lại **không phải lỗi chữ** — chúng là lỗi **hình học và topology**:

- tỉ lệ `128 : 64 : 192` (ảnh ⑧)
- hướng mũi tên autoregressive (ảnh ⑩)
- `layer1 → layer2 → layer3 → layer4` phải nối nhau (ảnh ②)
- backward LSTM phải nối vào concat (ảnh ④)
- 6 bar phải ứng 1-1 với 6 slab (ảnh ③)

Model sinh ảnh **không có khái niệm về ràng buộc quan hệ** giữa các phần tử — nó vẽ thứ "trông giống" sơ đồ. Retry thêm 20 lần thì mỗi lần sai một kiểu khác, và bạn không kiểm soát được lần nào đúng.

**Vẽ 6 hình này bằng draw.io / Figma / TikZ sẽ nhanh hơn và an toàn hơn hẳn**, đặc biệt là **ảnh ⑩** — vì đó là hình trung tâm của chương phương pháp, hội đồng sẽ soi kỹ nhất, và hiện tại nó đang sai ở 3 chỗ chí tử cùng lúc.

---

## 9. Bảng câu hỏi phản biện dự kiến

Sắp theo xác suất bị hỏi.

| # | Câu hỏi | Từ hình | Trạng thái hiện tại |
|:--:|---|:--:|---|
| 1 | *"`ywllmodul medimad` nghĩa là gì?"* | poster | ❌ không trả lời được — phải bỏ ảnh |
| 2 | *"Đáp án của em là `right lung`, sao hình lại sinh ra `right right`?"* | ⑩ | ❌ phải vẽ lại |
| 3 | *"256 token ảnh lấy đâu ra? ResNet-18 chỉ cho 7×7 = 49."* | ⑩ ↔ ② | ❌ phải sửa nhãn + thêm hộp chiếu |
| 4 | *"Mũi tên autoregressive của em chỉ ngược chiều thời gian?"* | ⑩ | ❌ phải vẽ lại |
| 5 | *"Cùng một MLP sao nhận được cả 128 và 64 chiều?"* | ⑦ | ❌ phải tách 2 hộp |
| 6 | *"`layer2`, `layer3` nối vào đâu?"* | ② | ❌ phải vẽ lại |
| 7 | *"Hình cho thấy 2 vector bằng nhau, sao ra 192?"* | ⑧ | ❌ phải vẽ lại tỉ lệ |
| 8 | *"`s` là `[B,C]`, `F` là `[B,C,H,W]`, em nhân ma trận kiểu gì?"* | ③ | ❌ đổi `×` → `⊙` |
| 9 | *"`Loss` của em có 2 dấu trừ — vậy là gradient ascent?"* | ⑪ | 🟡 đè chữ là xong |
| 10 | *"α = 0 mà sao cột `<pad>` vẫn cao?"* | ⑥ | 🟡 hạ bar về 0 |
| 11 | *"Heatmap chỉ đậm trên đường chéo — self-attention học được gì?"* | ⑤ | 🟡 vẽ lại heatmap |
| 12 | *"Scatter này là t-SNE hay UMAP? Bao nhiêu mẫu? 128 chiều với 64 chiều vẽ chung sao được?"* | ⑦ | 🟡 thêm caption *"minh hoạ khái niệm"* |
| 13 | *"Hai hộp `context` này là một hay hai?"* | ⑥ | 🟡 nối hoặc gộp |
| 14 | *"Ô này gate đậm sao đầu ra lại nhạt?"* | ⑨ | 🟡 vẽ lại 1 dải ô |
| 15 | *"Khi nào ra `[B,192]`, khi nào ra `[B,2P]`?"* | ⓪ | 🟡 thêm đường bypass |
| 16 | *"Backward LSTM của em nối vào concat ở chỗ nào?"* | ④ | ❌ phải vẽ lại |
| 17 | *"`V_answer` là gì? Kích thước bao nhiêu?"* | ⑩ | 🟡 thêm định nghĩa |
| 18 | *"Legend ghi `text` và `RL` cùng màu cam — phân biệt bằng gì?"* | poster | ❌ sửa bảng màu rồi render lại |
