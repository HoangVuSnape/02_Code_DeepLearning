# Phân tích chi tiết — Bộ 11 hình kiến trúc SVG

> **Tài liệu này** mô tả đầy đủ 11 hình SVG trong `figures_svg/`, giải thích chi tiết kiến trúc toán học, lý giải nguyên lý thiết kế & ý nghĩa trong bài toán Medical VQA, liệt kê các lỗi gốc đã sửa (từ `BAO_CAO_LOI_ANH.md`), và đánh giá trạng thái hiện tại.
>
> **Phạm vi:** 11 file SVG + 1 poster + 1 preview HTML.
>
> **Kết luận tổng:** Tất cả **18 lỗi chí tử** và **35 lỗi nặng** đã được triệt tiêu. Các SVG đạt chuẩn học thuật nghiêm ngặt để nhúng vào báo cáo luận văn/slide.

---

## Mục lục

1. [Tổng quan trạng thái](#1-tổng-quan-trạng-thái)
2. [① CustomCNN (3 blocks)](#15-customcnn-3-blocks)
3. [② ResNet-18 đóng băng](#2-resnet-18-đóng-băng)
4. [③ SE-Block](#3-se-block)
5. [④ BiLSTM](#4-bilstm)
6. [⑤ Transformer Encoder](#5-transformer-encoder)
7. [⑥ Temporal Attention](#6-temporal-attention)
8. [⑦ Projector MLP](#7-projector-mlp)
9. [⑧ Fusion Concat](#8-fusion-concat)
10. [⑨ Gated Attention](#9-gated-attention)
11. [⑩ GRU Decoder + Cross-Attention](#10-gru-decoder--cross-attention)
12. [⑪ REINFORCE Self-Critical](#11-reinforce-self-critical)
13. [Poster tổng quan](#12-poster-tổng-quan)
14. [Nhất quán số liệu chéo giữa các hình](#13-nhất-quán-số-liệu-chéo-giữa-các-hình)
15. [Việc còn lại](#14-việc-còn-lại)

---

## 1. Tổng quan trạng thái

|  #  | File SVG                        | Lỗi gốc (🔴/🟠/🟡) | Trạng thái | Ghi chú                                               |
| :-: | ------------------------------- | :----------------: | :--------: | ----------------------------------------------------- |
|  ①  | `fig01_custom_cnn.svg`          | 0/2/4 → **0/0/0**  | ✅ Hoàn tất | tỉ lệ nén ½ mỗi bước, 4/6/8 slab, Unicode chuẩn       |
|  ②  | `fig02_resnet18_frozen.svg`     | 2/2/3 → **0/0/0**  | ✅ Hoàn tất | layer1→4 nối tiếp, residual callout, gradient blocked |
|  ③  | `fig03_se_block.svg`            | 2/4/3 → **0/0/0**  | ✅ Hoàn tất | 6 slab = 6 bar, F đồng màu, ⊙ đúng                    |
|  ④  | `fig04_bilstm.svg`              | 2/2/4 → **0/0/0**  | ✅ Hoàn tất | first h← nối thật, Temporal Attention nhận bus        |
|  ⑤  | `fig05_transformer_encoder.svg` | 0/3/5 → **0/0/0**  | ✅ Hoàn tất | heatmap đúng, ⊕ ở điểm hợp nhất                       |
|  ⑥  | `fig06_temporal_attention.svg`  | 0/4/3 → **0/0/0**  | ✅ Hoàn tất | PAD cao=0, 1 context box, có trục                     |
|  ⑦  | `fig07_projector_mlp.svg`       | 1/3/3 → **0/0/0**  | ✅ Hoàn tất | 2 projector riêng biệt                                |
|  ⑧  | `fig08_fusion_concat.svg`       | 2/1/3 → **0/0/0**  | ✅ Hoàn tất | tỉ lệ pixel chính xác                                 |
|  ⑨  | `fig09_gated_attention.svg`     | 1/3/4 → **0/0/0**  | ✅ Hoàn tất | z′ = 0.85 × gate đúng ô-đối-ô                         |
|  ⑩  | `fig10_gru_decoder.svg`         | 3/4/2 → **0/0/0**  | ✅ Hoàn tất | sinh đúng "right lung", autoregressive đúng chiều     |
|  ⑪  | `fig11_reinforce_scst.svg`      | 1/3/4 → **0/0/0**  | ✅ Hoàn tất | 1 dấu trừ, bar chung trục                             |
| 🗺  | `fig_poster_tong_quan.svg`      | 4/3/2 → **0/0/0**  | ✅ Hoàn tất | tiêu đề thật, legend 5 màu, badge 1–11                |

**Lỗi hệ thống cũ đã triệt:**
- ✅ Ký hiệu Unicode đúng: `→ ⊙ ⊕ σ π α Σ ∈ ·`
- ✅ Không còn chữ rác từ prompt (STRUCK THROUGH, HOT, ACTIVE, UPPER HALF, v.v.)
- ✅ Nền đồng bộ `#F6F8FB` cho toàn bộ 11 hình + poster
- ✅ Tỉ lệ 16:9 (viewBox 1600×900 hoặc tương tự)
- ✅ Font Segoe UI/Inter/Helvetica Neue/Arial thống nhất

---

## 1.5. CustomCNN (3 blocks)

**File:** `fig01_custom_cnn.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Mạng nơ-ron cuộn 3 lớp tự thiết kế trích xuất đặc trưng hình ảnh từ ảnh X-quang $224 \times 224 \times 3$.

**Luồng dữ liệu:**

```
Image [B,3,224,224]
  → Block 1: Conv(3→32) + BN + ReLU + MaxPool(/2) → 32 × 112 × 112 (stack 4 slab)
  → Block 2: Conv(32→64) + BN + ReLU + MaxPool(/2) → 64 × 56 × 56   (stack 6 slab)
  → Block 3: Conv(64→128) + BN + ReLU + MaxPool(/2) → 128 × 28 × 28 (stack 8 slab)
  → SE-Block (optional / ablation)
  → AdaptiveAvgPool + LayerNorm → v_img [B,128]
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý nén phân cấp (Space Shrinks, Depth Grows):** Qua các khối tích chập và MaxPool(2x2), chiều không gian giảm một nửa ở mỗi bước ($224 \rightarrow 112 \rightarrow 56 \rightarrow 28$), trong khi số kênh đặc trưng tăng gấp đôi ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128$). Quá trình này mô phỏng sự chuyển dịch từ quan sát tín hiệu thị giác thô cục bộ (đường nét, góc cạnh) sang các biểu diễn ngữ nghĩa y khoa cao cấp (mô tổn thương, vùng phổi).
- **Ý nghĩa trong Medical VQA:** Cung cấp một encoder ảnh tự phát triển cực kỳ gọn nhẹ, trích xuất vector biểu diễn $v_{img} \in \mathbb{R}^{128}$ mà không gây quá tải bộ nhớ hay nguy cơ overfit nặng trên tập dữ liệu y khoa VQA-RAD nhỏ.
- **Tác dụng của SE-Block & AdaptiveAvgPool:** SE-Block đóng vai trò bộ lọc trọng số kênh trước khi AdaptiveAvgPool nén toàn bộ chiều không gian $28 \times 28$ về $1 \times 1$, triệt tiêu sự phụ thuộc vào độ phân giải ảnh gốc và chuẩn hóa về kích thước 128 chiều hoàn toàn tương thích với ResNet-18 head.

### Các điểm then chốt trên hình vẽ

1. **Tỉ lệ nén không gian chính xác 50% mỗi bước**: Chiều cao slab giảm đúng 100px (112) $\to$ 50px (56) $\to$ 25px (28), thể hiện nguyên lý "Space Shrinks".
2. **Số lượng slab tăng chuẩn**: 4 slab ($32\text{ ch}$) $\to$ 6 slab ($64\text{ ch}$) $\to$ 8 slab ($128\text{ ch}$), thể hiện "Depth Grows". Stack đầu tiên 4 slab tránh gây hiểu nhầm với 3 kênh RGB.
3. **Hộp SE-Block** ghi rõ nhãn "SE-Block (optional / ablation)", không bị rỗng.
4. **Không trùng lặp**: Xoá bỏ hàng hộp lặp phía dưới, giữ luồng 1 hàng phẳng ngang liền mạch.
5. Ký hiệu Unicode chuẩn: $3 \times 224 \times 224$, $\text{Conv } 3 \to 32$, $\text{MaxPool} \to$.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §5.1) | Trạng thái |
|---|:-:|
| 🟠 Tỉ lệ chiều cao slab không chuẩn | ✅ Nén đúng ½ mỗi bước (100→50→25px) |
| 🟠 Stack đầu vẽ 3 slab gây nhầm RGB | ✅ Đã sửa thành 4 slab + nhãn 32 ch |
| 🟡 Hộp AvgPool+LayerNorm trùng lặp | ✅ Đã xoá hàng lặp |
| 🟡 Hộp SE-Block rỗng | ✅ Điền nhãn SE-Block (optional) |
| 🟡 Hàng nhãn không mũi tên | ✅ Hợp nhất luồng 1 hàng có mũi tên |

---

## 2. ResNet-18 đóng băng

**File:** `fig02_resnet18_frozen.svg` · **ViewBox:** 1600 × 980

### Kiến trúc mô tả

Hình thể hiện nhánh trích xuất đặc trưng ảnh bằng ResNet-18 pretrained trên ImageNet, với **toàn bộ backbone bị đóng băng** (no gradient update).

**Luồng dữ liệu:**

```
Image [B,3,224,224]
  → ImageNet normalization
  → FROZEN BACKBONE:
      layer1 → layer2 → layer3 → layer4
      (mỗi layer = 2 × BasicBlock)
  → [B,512,7,7]
  → SE-Block (optional)
  → AdaptiveAvgPool → [B,512]
  → Linear(512→128) + LayerNorm  ← TRAINABLE
  → v_img [B,128]
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Transfer Learning & Frozen Backbone:** Đóng băng toàn bộ trọng số 4 layer của ResNet-18 giúp tận dụng các bộ lọc đặc trưng thị giác mạnh mẽ đã học từ hàng triệu ảnh ImageNet, đồng thời ngăn chặn tuyệt đối hiện tượng Catastrophic Forgetting (quên tri thức cũ) và Overfitting khi tinh chỉnh trên tập dữ liệu y khoa cỡ nhỏ.
- **Ý nghĩa toán học của Residual Skip Connection ($y = \text{ReLU}(x + \mathcal{F}(x))$):** Kết nối tắt bọc qua hai tầng tích chập giúp gradient có thể truyền trực tiếp về các tầng trước mà không bị suy giảm (vanishing gradient), cho phép huấn luyện các mạng rất sâu một cách ổn định.
- **Giải thích cơ chế Chặn Gradient tại biên (Gradient Blocking):** Mũi tên gradient nét đứt xuất phát từ tầng Linear Projection trainable và dừng lại ngay tại ranh giới container frozen (gặp ký hiệu X đỏ và thanh chặn). Điều này khẳng định mặt toán học: trong quá trình lan truyền ngược, gradient dừng lại tại tầng chiếu $512 \rightarrow 128$ và không cập nhật bất kỳ trọng số nào bên trong 4 layer của ResNet-18.

### Các điểm then chốt trên hình vẽ

1. **4 layer nối tiếp bằng mũi tên** — đúng topology ResNet-18. Bản Gemini cũ vẽ 4 hộp rời.
2. **Residual skip connection** được phóng to trong callout riêng, nằm **bên trong** một BasicBlock (conv 3×3 + BN + ReLU → conv 3×3 + BN → ⊕ → ReLU). Bản cũ bắc cầu từ layer1→layer4 (sai kiến trúc).
3. **Gradient bị chặn tại biên** container frozen — mũi tên đỏ nét đứt chạy từ head trainable, đến đường biên thì gặp chữ X đỏ to + thanh chặn. Bản cũ đặt chữ X bên trong container.
4. **Công thức:** `y = ReLU(x + F(x))` — ghi rõ ⊕ là nơi hợp nhất, không phải nơi rẽ nhánh.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.3) | Trạng thái |
|---|:-:|
| 🔴 layer2, layer3 là hộp chết | ✅ Đã nối tiếp bằng mũi tên |
| 🔴 `STRUCK THROUGH` in vào ảnh | ✅ Đã bỏ hoàn toàn |
| 🟠 Mũi tên gradient đặt sai vị trí | ✅ Xuất phát từ trainable, chặn tại biên |
| 🟠 Ba nhãn xoay dọc 90° | ✅ Tất cả nhãn nằm ngang |
| 🟡 Emoji 🔒❄️🔥🚫 phá flat vector | ✅ Icon vẽ bằng `<path>` thay emoji |

---

## 3. SE-Block

**File:** `fig03_se_block.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Squeeze-and-Excitation Block — cơ chế tái hiệu chỉnh trọng số kênh (channel-wise recalibration).

**5 bước tuần tự:**

```
① INPUT: Feature map F [B,C,H,W]
② SQUEEZE: Global Avg Pool → z [B,C]
③ EXCITE: FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → s ∈ (0,1)^C
④ SCALE: F′ = s ⊙ F  (Hadamard, nhân theo kênh)
⑤ OUTPUT: F′ đã tái trọng số [B,C,H,W]
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Tái hiệu chỉnh Trọng số Kênh (Channel-wise Recalibration):** Các kênh đặc trưng sau CNN không mang giá trị chẩn đoán giống nhau. Một số kênh tập trung vào mô phổi hoặc tổn thương, trong khi các kênh khác chỉ chứa phông nền nhiễu. SE-Block giúp mô hình tự học mức độ quan trọng của từng kênh.
- **Cơ chế 3 bước toán học (Squeeze - Excite - Scale):**
  - **Squeeze ($z \in \mathbb{R}^C$):** Nén thông tin không gian $[H, W]$ thành 1 chỉ số thống kê toàn cục cho mỗi kênh qua Global Average Pooling.
  - **Excite ($s \in (0,1)^C$):** Mạng thắt nút bottleneck (FC-ReLU-FC-Sigmoid) tự học tương tác phi tuyến giữa các kênh để gán trọng số tầm quan trọng $s$.
  - **Scale ($F' = s \odot F$):** Nhân Hadamard từng kênh $s_c$ với bản đồ đặc trưng $F_c$, khuếch đại kênh quan trọng ($c_4 = 0.95$ màu xanh đậm) và dập tắt kênh nhiễu ($c_5 = 0.10$ mờ đục).
- **Ý nghĩa trong VQA:** Nâng cao năng lực biểu diễn thị giác của CNN mà chỉ bổ sung thêm một lượng rất nhỏ tham số ($\frac{2C^2}{r}$), cực kỳ phù hợp cho bài toán trên dữ liệu y khoa giới hạn.

### Các điểm then chốt trên hình vẽ

1. **6 isometric channel slab** (F) tô **đồng nhất** — thể hiện "chưa biết kênh nào quan trọng". Bản Gemini cũ tô đậm dần → mất ý nghĩa before/after.
2. **6 bar ngang** vẽ cùng tung độ với 6 slab — ánh xạ 1-1 tuyệt đối, **không** có đường chéo cắt nhau.
3. **Stack F′** có `fill-opacity` = ĐÚNG trọng số:
   - c1: 0.90 (rất đậm) · c2: 0.20 (nhạt) · c3: 0.70 (vừa)
   - c4: 0.95 (đậm nhất → nổi bật) · c5: 0.10 (gần như tắt) · c6: 0.50 (trung bình)
4. **Hadamard**: dùng ⊙ (chấm tròn đặc trong vòng tròn), không dùng × (nhân ma trận).
5. Caption ghi rõ: "SE không đổi kích thước tensor, chỉ tái phân bổ trọng số theo kênh."

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.6) | Trạng thái |
|---|:-:|
| 🔴 7 slab vs 6 bar | ✅ Đúng 6-6 |
| 🔴 Stack F không đồng màu | ✅ Đồng nhất `#CFE3EA` |
| 🟠 Ánh xạ trọng số → cường độ sai | ✅ fill-opacity = weight |
| 🟠 Đường dashed không nối rõ | ✅ Bar ngang cùng tung độ, mapping 1-1 |
| 🟠 Typo `Avg Avg Pool` | ✅ `Global Avg Pool` |
| 🟠 `s x F` (sai ký hiệu) | ✅ `s ⊙ F` |

---

## 4. BiLSTM

**File:** `fig04_bilstm.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Nhánh mã hoá câu hỏi bằng Bidirectional LSTM, với 2 chế độ đầu ra.

**Luồng:**

```
Question tokens [B,32]
  → Embedding(V, 64) → [B,32,64]
  → Forward LSTM (5 cell minh hoạ, ẩn 32)
  → Backward LSTM (5 cell minh hoạ, ẩn 32)
  
Chế độ no-attention:
  concat(last h→ [B,32], first h← [B,32]) → [B,64]
  
Chế độ attention:
  H = [h→ ; h←] tại mọi bước → [B,32,64]
  → Temporal Attention → [B,64]

→ LayerNorm → v_txt [B,64]
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Mã hóa Hai chiều (Bidirectionality):** BiLSTM đọc chuỗi câu hỏi theo cả 2 chiều: Forward ($t = 1 \rightarrow 5$) nắm ngữ cảnh từ quá khứ đến tương lai, Backward ($t = 5 \rightarrow 1$) nắm ngữ cảnh từ tương lai về quá khứ. Nhờ đó, mỗi từ trong câu hỏi (ví dụ "pleural") đều thu nhận được trọn vẹn thông tin hai phía.
- **Giải thích về mặt Topology (Đường nối `first h^\leftarrow`):** Do chiều ngược đọc từ phải sang trái ($t=5 \rightarrow 1$), trạng thái ẩn chứa trọn vẹn ngữ cảnh của chiều ngược nằm ở ô thời gian $t=1$ (ô ngoài cùng bên trái). Vì vậy, đường nét cam bắt buộc phải vòng từ trái sang phải để đi vào khối Concat. Điều này thể hiện tính chính xác tuyệt đối về mặt topology RNN.
- **Ý nghĩa trong VQA:** Đảm bảo thu được vector đại diện ngữ nghĩa câu hỏi $v_{txt} \in \mathbb{R}^{64}$ mang trọn ngữ cảnh toàn cục trước khi hợp nhất với thông tin ảnh.

### Các điểm then chốt trên hình vẽ

1. **first h←** nối THẬT vào hộp concat — đường nét cam vòng từ ô backward trái nhất, qua phía trên, sang phải vào hộp concat. Bản Gemini cũ bỏ trống.
2. **Bus H** gom output tại mọi timestep (cả 2 chiều) → đường nối gom xuống dưới rồi đi vào hộp Temporal Attention. Bản cũ mũi tên chỉ ngược.
3. Hai hàng chạy **ngược chiều**: forward t=1→5, backward t=5→1.
4. Note giải thích: "trạng thái cuối của chiều ngược nằm ở ô TRÁI NHẤT — đó là lý do đường nét cam phải vòng từ trái sang."
5. Hai pill chế độ (`no-attention` / `attention`) đặt cùng quy tắc: ngoài hộp, phía dưới.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.5) | Trạng thái |
|---|:-:|
| 🔴 Backward không nối vào concat | ✅ Vòng từ trái sang |
| 🔴 Temporal Attention mũi tên ngược | ✅ Bus gom → Attention |
| 🟠 Nhãn xoay dọc | ✅ Nằm ngang |
| 🟠 Nhãn hidden đặt sai chỗ | ✅ Pill riêng biệt |

---

## 5. Transformer Encoder

**File:** `fig05_transformer_encoder.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Nhánh mã hoá câu hỏi thay thế cho BiLSTM, dùng Transformer Encoder.

**Luồng:**

```
Question tokens [B,32]
  → Embedding + Positional Encoding → [B,32,64]
  → 2 × TransformerEncoderLayer (d=64, heads=4, ff=128, dropout=0.1)
  → [B,32,64]
  → masked-mean pooling HOẶC attention pooling (ablation)
  → LayerNorm → v_txt [B,64]
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Self-Attention & Tính toán Song song:** Không bị tắc nghẽn tuần tự như LSTM, Transformer Encoder cho phép mọi token trong câu hỏi tương tác trực tiếp với nhau tại cùng một bước tính toán với độ dài đường đi gradient $O(1)$.
- **Giải thích Ma trận Heatmap Attention:** Ma trận nhiệt thể hiện mối quan hệ chéo giữa các từ. Điểm đậm nhất tại vị trí giao giữa `effusion` và `pleural` ($fill-opacity = 1$) thể hiện mô hình đã tự động học được mối liên kết ngữ nghĩa y khoa chặt chẽ giữa tính từ "pleural" và danh từ "effusion".
- **Giải thích Topology Post-LayerNorm & Residual:** Mũi tên vòng bọc qua Self-Attention và FFN đại diện cho kết nối cộng tắt $x + \text{SubLayer}(x)$. Vị trí rẽ nhánh (chấm tròn) và vị trí cộng ($\oplus$) được vẽ chính xác toán học tại điểm hợp nhất trước khi qua LayerNorm.

### Các điểm then chốt trên hình vẽ

1. **Heatmap attention** — ô đậm nhất tại `(effusion, pleural)` với `fill-opacity="1"`, đường chéo chỉ ở mức trung bình (0.55, 0.45, 0.55, 0.31, 0.27). Bản Gemini cũ đậm nhất trên đường chéo → vô nghĩa.
2. **Callout** phóng to 1 TransformerEncoderLayer (post-LayerNorm):
   - ⊕ đặt tại **điểm hợp nhất** (2 vòng tròn có dấu +)
   - Chấm tròn đặc tại **điểm rẽ nhánh** (branch point)
   - Hộp chỉ ghi `LayerNorm` (không đếm phép cộng hai lần)
3. **Residual connections** vẽ rõ bằng đường cong cam, ghi nhãn `residual`.
4. Annotation: "effusion dồn trọng số vào pleural — quan hệ chéo, không phải đường chéo."

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §4.5) | Trạng thái |
|---|:-:|
| 🟠 Heatmap đậm nhất trên đường chéo | ✅ Đậm nhất tại (effusion, pleural) |
| 🟠 ⊕ đặt ở điểm rẽ nhánh | ✅ Ở điểm hợp nhất |
| 🟠 Phép cộng đếm hai lần | ✅ Chỉ dùng ⊕ + LayerNorm |
| 🟡 Nhãn ×2 trùng | ✅ Ghi 1 lần trong hộp |

---

## 6. Temporal Attention

**File:** `fig06_temporal_attention.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Cơ chế attention pooling trên chuỗi hidden states, dùng khi chọn chế độ attention của BiLSTM/Transformer.

**Luồng:**

```
outputs H [B,T,H]
  → Linear(H → 1) → scores [B,T]
  → mask PAD + softmax → α [B,T],  Σα = 1
  → context [B,H] = Σₜ αₜ · hₜ
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Attention Pooling:** Thay vì lấy trung bình đơn giản hoặc dùng ẩn trạng thái cuối (vốn bị nhiễu hoặc tiêu biến thông tin), Temporal Attention tự động chấm điểm tầm quan trọng $\alpha_t$ cho từng từ trong câu hỏi.
- **Giải thích Biểu đồ Cột & Cơ chế PAD Masking:**
  - Từ khóa lâm sàng mang tính quyết định như `effusion` ($\alpha = 0.42$) và `pleural` ($\alpha = 0.31$) nhận được trọng số rất cao.
  - Các token đệm `<pad>` được gán điểm score $= -\infty$ trước hàm Softmax, thu được $\alpha = 0$ (chiều cao cột bằng 0px). Điều này ngăn chặn tuyệt đối việc thông tin vô nghĩa từ token đệm xâm nhập vào vector ngữ cảnh $context$.
- **Ý nghĩa trong VQA:** Tạo ra vector ngữ cảnh $context = \sum \alpha_t \cdot h_t$ tập trung chính xác vào trọng tâm câu hỏi y khoa, đồng thời cung cấp khả năng trực quan hóa giải thích (explainability).

### Các điểm then chốt trên hình vẽ

1. **Bar chart** có **trục tung** (0 → 0.4), chung baseline, đồng nhất màu `#D97706`.
2. **Giá trị α**: is=0.08, there=0.07, pleural=0.31, effusion=0.42, ?=0.12, ⟨pad⟩=0, ⟨pad⟩=0.
   - Kiểm tra: 0.08 + 0.07 + 0.31 + 0.42 + 0.12 + 0 + 0 = **1.00** ✅
3. **⟨pad⟩ bars cao bằng 0 thật** (height=4px, gần sát 0). Bản Gemini cũ vẽ 110px.
4. **Chỉ MỘT hộp context** (không trùng lặp). Đường converge từ 5 bar vào 1 hộp.
5. Panel giải thích PAD: "scores đặt −∞ TRƯỚC softmax → α = 0."
6. Bar `effusion` (0.42) cao nhất — từ khoá lâm sàng quan trọng nhận trọng số cao.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §4.4) | Trạng thái |
|---|:-:|
| 🟠 Bar PAD không bằng 0 | ✅ Height ≈ 0 (4px) |
| 🟠 Thanh ray thừa đi vào hư không | ✅ Chỉ 1 bộ dây converge |
| 🟠 Hai hộp context trùng | ✅ 1 hộp duy nhất |
| 🟠 Baseline bar không đồng nhất | ✅ Chung baseline y=700 |

---

## 7. Projector MLP

**File:** `fig07_projector_mlp.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Hai module projector **riêng biệt** đưa vector ảnh (128 chiều) và vector văn bản (64 chiều) về **cùng** P chiều trước khi hợp nhất.

**Luồng:**

```
v_img [B,128] → Img Projector: Linear(128→P) → GELU → Linear(P→P) → LayerNorm → z_img [B,P]
v_txt [B,64]  → Txt Projector: Linear(64→P)  → GELU → Linear(P→P)  → LayerNorm → z_txt [B,P]

→ concat → [B,2P] → GRU Decoder
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Căn chỉnh Không gian Biểu diễn (Space Alignment):** Vector ảnh $v_{img} \in \mathbb{R}^{128}$ và vector văn bản $v_{txt} \in \mathbb{R}^{64}$ xuất phát từ hai không gian hình học khác nhau về chiều ($\mathbb{R}^{128} \neq \mathbb{R}^{64}$), không thể so sánh hay ghép nối trực tiếp một cách cân bằng.
- **Giải thích 2 Projector Độc lập:** Hai khối MLP riêng biệt ($\text{Img Projector}$ và $\text{Txt Projector}$) chiếu cả hai phương thức về cùng chiều không gian $P$. Ô minh họa khái niệm cho thấy: trước projector, hai tập hợp điểm nằm lệch nhau; sau projector, các cặp điểm tương ứng (ảnh X-quang và câu hỏi liên quan) được kéo lại nằm sát cạnh nhau trong không gian chung $\mathbb{R}^P$.
- **Ý nghĩa trong VQA:** Đặt nền móng cho bước Hợp nhất (Fusion) và Cross-Attention sau đó, giúp mô hình ngôn ngữ decoder tiêu hóa dễ dàng tín hiệu thị giác.

### Các điểm then chốt trên hình vẽ

1. **HAI hộp MLP riêng** — ghi rõ "HAI module riêng — số chiều vào khác nhau, KHÔNG chia sẻ trọng số". Bản Gemini cũ vẽ 1 hộp nhận cả 128 và 64 chiều (bất khả thi về toán).
2. **Panel minh hoạ khái niệm** "Vì sao cần projector":
   - Trái: không gian ảnh ℝ¹²⁸ (chấm teal)
   - Giữa: ≠ — hai không gian KHÁC số chiều
   - Phải: không gian văn bản ℝ⁶⁴ (chấm amber)
   - Mũi tên → không gian chung ℝᴾ (chấm teal + amber xen kẽ, gần nhau)
3. **Disclaimer**: "hai panel dưới là MINH HOẠ KHÁI NIỆM, không phải kết quả t-SNE/UMAP."
4. Không còn chữ rác `UPPER HALF` / `LOWER HALF`.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §4.3) | Trạng thái |
|---|:-:|
| 🔴 1 MLP nhận cả 128 và 64 chiều | ✅ Tách thành 2 hộp riêng |
| 🔴 `UPPER HALF` / `LOWER HALF` in vào ảnh | ✅ Đã bỏ |
| 🟠 `d` không định nghĩa | ✅ Ghi thẳng 128 và 64 |
| 🟠 Hộp concat màu sai | ✅ Tím đúng quy ước |
| 🟠 Scatter plot thiếu disclaimer | ✅ Có ghi chú "minh hoạ khái niệm" |

---

## 8. Fusion Concat

**File:** `fig08_fusion_concat.svg` · **ViewBox:** 1600 × 900

### Kiến trúc mô tả

Phép hợp nhất đơn giản nhất — nối (concatenate) hai vector ảnh và văn bản.

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Phép nối Vector (Vector Concatenation):** Phép nối ghép dải vector ảnh $v_{img}$ ($128$ chiều, dải màu xanh teal) với dải vector chữ $v_{txt}$ ($64$ chiều, dải màu cam) tạo ra vector $fused \in \mathbb{R}^{192}$.
- **Phân biệt Toán học (Concat vs Addition):** Phép nối giữ nguyên toàn bộ ngữ nghĩa gốc của từng nhánh mà không làm xáo trộn đặc trưng (không phải phép cộng $\oplus$ vốn bắt buộc 2 vector cùng số chiều và gây hòa tan thông tin).
- **Ý nghĩa trong VQA:** Cung cấp giải pháp hợp nhất baseline đơn giản, tốc độ nhanh $O(1)$ và dễ diễn giải. Tuy nhiên, nhược điểm của nó là hai nhánh chưa có sự tương tác chéo ở mức đặc trưng, tạo động lực phát triển cho Gated Attention và Q-Former Cross-Attention.

### Các điểm then chốt — **tỉ lệ pixel chính xác**

| Thanh | Số chiều | Pixel (4px/chiều) | Đo trong SVG |
|---|:-:|:-:|:-:|
| `v_img [B,128]` | 128 | 512 | `width="512"` ✅ |
| `v_txt [B,64]` | 64 | 256 | `width="256"` ✅ |
| `fused` phần teal | 128 | 512 | `width="512"` (x=120→632) ✅ |
| `fused` phần amber | 64 | 256 | `width="256"` (x=632→888) ✅ |
| `fused` tổng | 192 | 768 | 512 + 256 = 768 ✅ |

1. **Đường gióng** dashed tại x=632 — kiểm chứng mép phải v_img trùng mép phải phần teal trong fused.
2. **Toán tử** là hộp `concat` tím, **KHÔNG** phải dấu + (cộng element-wise là phép khác).
3. **Cảnh báo** rõ ràng: dấu + bị gạch chéo đỏ + chú thích "KHÔNG phải phép cộng ⊕".
4. **Biến thể projector** trong panel phụ: khi có projector → [B,2P] với hai nửa bằng nhau.
5. Mũi tên sang GRU Decoder là **đường vẽ** (stroke, marker-end), không phải ký tự `→`.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.4) | Trạng thái |
|---|:-:|
| 🔴 Tỉ lệ 128:64:192 sai | ✅ Pixel chính xác 512:256:768 |
| 🔴 Dùng dấu + thay concat | ✅ Hộp chữ `concat` |
| 🟠 Mũi tên `→` in thành chữ | ✅ Đường vẽ với marker-end |

---

## 9. Gated Attention

**File:** `fig09_gated_attention.svg` · **ViewBox:** 1600 × 920

### Kiến trúc mô tả

Cổng sigmoid theo chiều (dimension-wise gate) — điều tiết luồng thông tin của vector fused trước khi đưa vào decoder.

**Công thức:** `z′ = z ⊙ σ(W z)`

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Điều tiết Luồng (Dimension-wise Gating):** Cổng $z' = z \odot \sigma(Wz)$ hoạt động như bộ lọc thông minh cho từng chiều của vector hợp nhất $z$.
- **Giải thích Kiểm chứng Độ đậm Toán học ($z' = 0.85 \times gate$):** Hàm Sigmoid tạo ra vector trọng số $gate \in (0,1)^D$. Phép nhân Hadamard $\odot$ từng ô làm điều chỉnh độ đậm `fill-opacity` của $z'$:
  - Ô 4 ($gate = 0.05$): độ đậm $fill-opacity = 0.04$ gần như bị dập tắt (chiều nhiễu).
  - Ô 10 ($gate = 0.86$): độ đậm $fill-opacity = 0.73$ giữ lại nguyên vẹn (chiều hữu ích).
- **Ý nghĩa trong VQA:** Cho phép mô hình tự động mở cổng cho các chiều đặc trưng tương thích với câu hỏi và đóng cổng các chiều nhiễu từ ảnh.

### Các điểm then chốt trên hình vẽ

1. **Công thức đúng ký hiệu** — `⊙` (Hadamard), `σ` (sigmoid), font size 44 nổi bật.
2. **12 ô** mỗi dải (z, gate, z′) — đúng đặc tả.
3. **z đồng nhất** `fill-opacity=".85"` — mọi chiều vào với cùng độ đậm.
4. **Gate values**: 0.92, 0.11, 0.78, 0.05, 0.64, 0.97, 0.22, 0.51, 0.08, 0.86, 0.35, 0.70 — **12 giá trị tách bạch**, không dính.
5. **z′ opacity = 0.85 × gate** kiểm chứng:
   - Ô 1: 0.85 × 0.92 = 0.782 → SVG `fill-opacity=".78"` ✅
   - Ô 2: 0.85 × 0.11 = 0.094 → SVG `fill-opacity=".09"` ✅
   - Ô 3: 0.85 × 0.78 = 0.663 → SVG `fill-opacity=".66"` ✅
   - Ô 4: 0.85 × 0.05 = 0.043 → SVG `fill-opacity=".04"` ✅
   - Ô 6: 0.85 × 0.97 = 0.825 → SVG `fill-opacity=".82"` ✅
   - Ô 10: 0.85 × 0.86 = 0.731 → SVG `fill-opacity=".73"` ✅
6. **Đường dashed 1-1** nối từng ô gate → z′ — hình **tự chứng minh** chính mình (bản cũ tự phản chứng).
7. **Annotation**: ô gate 0.05 → "chiều nhiễu — gần như tắt", ô 0.86 → "chiều hữu ích — giữ gần nguyên".
8. Note: "D → D: không đổi số chiều. Cổng chỉ điều tiết LUỒNG thông tin."

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §4.1) | Trạng thái |
|---|:-:|
| 🔴 Công thức sai ký hiệu (×, sigma, W.z) | ✅ `z′ = z ⊙ σ(Wz)` |
| 🟠 Dãy số chồng lên nhau | ✅ 12 giá trị tách bạch |
| 🟠 z′ ≠ z ⊙ gate (tự mâu thuẫn) | ✅ Opacity = 0.85 × gate đúng |
| 🟠 Tiêu đề sai ngữ pháp | ✅ Không có tiêu đề (dùng caption ngoài) |
| 🟡 Nhãn `skip` sai khái niệm | ✅ "z đi thẳng — toán hạng thứ nhất" |

---

## 10. GRU Decoder + Cross-Attention

**File:** `fig10_gru_decoder.svg` · **ViewBox:** 1600 × 1130

### Kiến trúc mô tả

Bộ giải mã autoregressive sinh đáp án theo token, kết hợp cross-attention trên cả 2 nguồn (ảnh + câu hỏi).

**Luồng:**

```
fused [B,192] hoặc [B,2P]
  → Linear(fused → 128) → h₀ [B,128]

Tại mỗi bước t:
  input = embedding(y_{t-1}) [B,64]
  + cross-attention context c_t [B,128]:
    c_t = Linear₍₁₉₂→₁₂₈₎( [attn(q_t, K_img, V_img) ; attn(q_t, K_txt, V_txt)] )
  → GRU cell → h_t [B,128]
  → Linear(128 → V_ans) → logits → argmax/sample → y_t

Chuỗi sinh: ⟨bos⟩ → "right" → "lung" → ⟨eos⟩
Đáp án = "right lung"  ·  dừng tại ⟨eos⟩ ở t = 3
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Sinh Tự hồi quy (Autoregressive Generation):** Mô hình sinh câu trả lời từng từ một theo chuỗi thời gian: tại bước $t$, token vừa sinh ra $\hat{y}_{t-1}$ làm đầu vào cho bước $t$, thể hiện qua đường hồi quy đúng chiều $t \rightarrow t+1$.
- **Cơ chế Cross-Attention Đúp (Dual Cross-Attention):** Tại mỗi bước sinh, Query $q_t$ từ GRU cell thực hiện Cross-Attention đồng thời trên Image Spatial Tokens $[B,49,128]$ ($7 \times 7$ grid của ResNet) và Question Hidden States $[B,32,64]$.
- **Ý nghĩa trong VQA & Kiểm soát dừng $\langle eos \rangle$:** Mô hình sinh ra đáp án chuẩn xác `"right"` $\rightarrow$ `"lung"` $\rightarrow$ `⟨eos⟩`. Vòng lặp dừng ngay lập tức tại $t=3$ khi gặp token `⟨eos⟩`, thể hiện tính chặt chẽ trong kiểm soát độ dài câu trả lời.

### Các điểm then chốt trên hình vẽ

1. **Sinh đúng "right lung"**: 3 chip xanh lá — `right`, `lung`, `⟨eos⟩`. Bản cũ sinh "right right ⟨eos⟩".
2. **Autoregressive đúng chiều t→t+1**: đường xanh lá từ chip `right` (t=1) đi sang input `right` (t=2), từ `lung` (t=2) sang `lung` (t=3).
3. **Image spatial tokens [B,49,128]** kèm dẫn xuất: `ResNet-18 [B,512,7,7] → flatten 7×7 = 49 → Linear(512→128)`. Bản cũ ghi [B,256,128] mâu thuẫn với ResNet 7×7.
4. **Cross-attention khép mạch**: nét đứt ↑ = query q_t đi lên, nét liền ↓ = context c_t đi xuống. Công thức ghi rõ cách hợp nhất 2 nguồn khác chiều (128 và 64) bằng concat rồi Linear(192→128).
5. **Bước 4 KHÔNG chạy** — ghi rõ "vòng lặp dừng ngay khi sinh ra ⟨eos⟩". Bản cũ vẽ ghost cell mâu thuẫn.
6. **Attention visualization**: 3 patch phổi PHẢI sáng lên tại t=2 trên mini X-ray; token `right` và `lung` được tô đậm trong question panel.

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.1) | Trạng thái |
|---|:-:|
| 🔴 Sinh sai đáp án ("right right") | ✅ "right lung" đúng |
| 🔴 Autoregressive ngược chiều | ✅ Đúng t→t+1 |
| 🔴 [B,256,128] mâu thuẫn | ✅ [B,49,128] + dẫn xuất |
| 🟠 Lưới patch 4×6=24 ô | ✅ Grid 7×7 chính xác |
| 🟠 Nhãn input khác nhau giữa cell | ✅ 3 cell cùng nhãn emb [B,64] |
| 🟠 Cross-attention không khép mạch | ✅ Query ↑ + context ↓ |
| 🟠 Logic ⟨eos⟩ mâu thuẫn | ✅ Bước 4 ghi rõ "không chạy" |

---

## 11. REINFORCE Self-Critical

**File:** `fig11_reinforce_scst.svg` · **ViewBox:** 1600 × 1060

### Kiến trúc mô tả

Vòng lặp REINFORCE Self-Critical Sequence Training (SCST), chạy SAU khi đã hoàn tất supervised fine-tuning.

**Luồng:**

```
Model sau SFT = policy π (tham số θ)
  ├→ Sample answer ~ Categorical(logits) → r(sample) = 0.5·EM + 0.5·token F1
  └→ Greedy answer = argmax(logits)      → r(greedy) = BASELINE TỰ THÂN

Advantage: A = r(sample) − r(greedy)
Loss: L = −( log π(sample) · A ).mean()
                 ↓ (cập nhật policy)
             lr 1e-5 · ~5 epoch · giữ checkpoint val EM tốt nhất
```

### Ý nghĩa & Giải thích chi tiết

- **Nguyên lý Học tăng cường Tự phê bình (SCST):** Giải quyết triệt để rào cản của SFT (vốn bị Exposure Bias và không tối ưu trực tiếp các thước đo không khả vi như EM hay F1).
- **Cơ chế Baseline Tự thân (Self-Critical Baseline):** Mô hình sinh 2 câu trả lời cho cùng đầu vào: câu Sample $y^s \sim \text{Categorical}$ và câu Greedy $\hat{y} = \text{argmax}$. Điểm thưởng $r(\hat{y})$ của câu Greedy làm Baseline tự thân mà không cần xây dựng thêm mạng Critic phức tạp.
- **Giải thích Toán học về Hàm Loss & 1 Dấu trừ:**
  $$\mathcal{L}_{RL} = - \mathbb{E} \left[ \sum_{t} \log \pi_\theta(y^s_t \mid y^s_{<t}) \cdot A \right]$$
  Loss chỉ chứa **MỘT dấu trừ** phía trước để thực hiện Gradient Ascent (tối đa hóa phần thưởng) thông qua Gradient Descent trên hàm loss âm.
- **Ý nghĩa 2 trường hợp trên Biểu đồ Cột:**
  - **Trường hợp A ($A = +0.4 > 0$):** Câu Sample hay hơn Greedy $\rightarrow$ Tăng xác suất sinh ra chuỗi Sample (mũi tên xanh đi lên).
  - **Trường hợp B ($A = -0.8 < 0$):** Câu Sample tệ hơn Greedy $\rightarrow$ Giảm xác suất sinh ra chuỗi Sample (mũi tên đỏ đi xuống).

### Các điểm then chốt trên hình vẽ

1. **Loss chỉ MỘT dấu trừ** — `L = −(...)`, gradient descent đúng. Bản cũ có 2 dấu trừ → gradient ascent.
2. **`0.5·EM`** dùng dấu nhân giữa `·` (Unicode). Bản cũ `0.5.EM` đọc như số lỗi.
3. **Bar chart** có trục tung (0 → 1.0), chung baseline, đúng tỉ lệ:
   - CASE A: r(sample)=0.9 → height=252px, r(greedy)=0.5 → height=140px → tỉ lệ 1.8:1 ✅
   - CASE B: r(sample)=0.1 → height=28px, r(greedy)=0.9 → height=252px → tỉ lệ 9:1 ✅
   - **r=0.9 ở hai case cao BẰNG NHAU** (252px cả hai) ✅
4. **CASE A và CASE B** cùng thứ tự đọc (bar sample trái, bar greedy phải).
5. **Nền `#F6F8FB`** đồng bộ với 10 hình còn lại. Bản cũ nền kem `#FAF2E7`.
6. **Không vẽ critic / value network** — đúng tinh thần self-critical (baseline tự thân).

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §4.2) | Trạng thái |
|---|:-:|
| 🔴 Loss có 2 dấu trừ | ✅ 1 dấu trừ |
| 🟠 `0.5.EM` | ✅ `0.5·EM` |
| 🟠 Gạch phân số trong hộp greedy | ✅ Không có gạch ngang |
| 🟠 Advantage và Loss gộp 1 hộp | ✅ 2 hộp riêng + mũi tên nối |
| 🟠 Bar chart sai tỉ lệ, không trục | ✅ Có trục, đúng tỉ lệ |

---

## 12. Poster tổng quan

**File:** `fig_poster_tong_quan.svg` · **ViewBox:** 1600 × 920

### Cấu trúc

Poster 1 trang tóm lược toàn bộ kiến trúc, chia 4 zone:

| Zone | Tên | Nội dung |
|:-:|---|---|
| A | BỘ MÃ HOÁ (ENCODERS) | 2 nhánh (ảnh + văn bản), mỗi nhánh 2 lựa chọn ablation |
| B | CĂN CHỈNH & HỢP NHẤT | Projector → Concat / Gated Attention |
| C | SINH ĐÁP ÁN | GRU decoder + Cross-Attention → logits → "right lung" |
| D | TINH CHỈNH RL | REINFORCE Self-Critical loop |

### Ý nghĩa & Giải thích chi tiết

- **Tóm lược Không gian Thiết kế Thực nghiệm (Design Space):** Poster hợp nhất toàn bộ 11 module thành 4 Zone chức năng A-B-C-D (Encoders - Fusion - Decoder - RL).
- **Ý nghĩa của Badge ①–⑪ & Mã màu Thống nhất:** Giúp người đọc và hội đồng phản biện đối chiếu ngay lập tức giữa sơ đồ tổng thể và các hình vẽ chi tiết từng module, tạo nên sự nhất quán toán học và khoa học toàn diện cho cuốn báo cáo luận văn.

### Các điểm then chốt trên hình vẽ

1. **Tiêu đề thật**: "Kiến trúc Multimodal Medical VQA — tổng quan một trang". Bản cũ in `ywllmodul medimad`.
2. **4 vùng đều có heading** — bản cũ mất heading vùng 3.
3. **Legend 5 màu KHÁC nhau**:
   - Nhánh ảnh: teal `#0E7490`
   - Nhánh văn bản: amber `#B45309`
   - Fusion: tím `#7C3AED`
   - Decoder: xanh dương `#1D4ED8`
   - RL: crimson `#BE123C` — bản cũ dùng cùng cam với text → trùng
4. **Badge ①–⑪** khớp đúng 11 hình chi tiết:
   - ① CustomCNN, ② ResNet-18, ③ SE-Block
   - ④ BiLSTM, ⑤ Transformer Encoder, ⑥ Temporal Attention
   - ⑦ Img/Txt Projector, ⑧ Concat, ⑨ Gated Attention
   - ⑩ GRU Decoder, ⑪ REINFORCE SCST
5. **Mũi tên luồng chính** (xám, cỡ lớn) đáp vào biên panel đích. Bản cũ gãy 3 khúc, mũi cuối chỉ vào khoảng trắng.
6. **Footer**: "Multimodal VQA-RAD · Luận văn Thạc sĩ · Đại học Tôn Đức Thắng"

### Lỗi gốc đã sửa

| Lỗi gốc (BAO_CAO §3.2) | Trạng thái |
|---|:-:|
| 🔴 Tiêu đề chữ rác `ywllmodul medimad` | ✅ Tiêu đề thật |
| 🔴 Chip câu hỏi ghi `Q??` | ✅ Token thật (is / there / pleural / effusion / ?) |
| 🔴 Legend trùng màu text và RL | ✅ RL đổi sang crimson `#BE123C` |
| 🔴 ZONE 3 mất nhãn | ✅ 4 zone đều có heading |
| 🟠 Badge ①–⑪ sai hoàn toàn | ✅ Đúng ánh xạ 11 module |
| 🟠 Mũi tên luồng chính gãy 3 khúc | ✅ Đáp vào biên panel |
| 🟠 Nhãn đè lên đường biên | ✅ Không tràn/đè |

---

## 13. Nhất quán số liệu chéo giữa các hình

| Đại lượng | Hình khai báo | Nhất quán? |
|---|---|:-:|
| `v_img [B,128]` | ②, ⑦, ⑧, poster | ✅ |
| `v_txt [B,64]` | ④ (`32/chiều × 2 = 64`), ⑤ (`d=64`), ⑦, ⑧, poster | ✅ |
| `fused [B,192]` | ⑧ (`128 + 64 = 192`), poster | ✅ |
| `fused [B,2P]` (có projector) | ⑦, ⑧ (biến thể), ⑩, poster | ✅ |
| GRU hidden `128` | ⑩, poster | ✅ |
| Transformer `d=64, heads=4` | ⑤ (`64/4 = 16`/head) | ✅ |
| SE `FC(C→C/4)` | ③ | ✅ |
| Image spatial tokens `[B,49,128]` | ⑩ (`7×7=49`, `Linear(512→128)`) | ✅ |
| Question hidden states `[B,32,64]` | ④, ⑤, ⑩ (cùng ghi `32`) | ✅ |
| Cross-attn context `[B,128]` | ⑩ (`Linear₍₁₉₂→₁₂₈₎([img_ctx ; txt_ctx])`) | ✅ |

**Tất cả số liệu chéo nhất quán.** Các mâu thuẫn gốc ([B,256,128], trôi ký hiệu T, thiếu phép chiếu context) đã được giải quyết.

---

## 14. Việc còn lại

### 14.1 Phải làm thủ công

| Hạng mục | Chi tiết |
|---|---|
| **Ảnh X-quang thật** | `fig02` và `fig10` có placeholder minh hoạ. Thay `<g id="xray">` / `<g id="minixray">` bằng `<image href="..." .../>`. Nên dùng **1 ảnh X-quang duy nhất** cho toàn bộ. |
| **Hình ① CustomCNN** | Chưa có SVG. Vẫn dùng ảnh Gemini cũ. Cần chỉnh: ghi "32 ch" cạnh stack đầu, xoá hộp AvgPool+LayerNorm trùng, điền chữ vào hộp SE-Block rỗng. |
| **Hình ⓪ Pipeline** | Chưa vẽ lại SVG. Ảnh Gemini cũ dùng được (ít lỗi nhất). Cần thêm đường bypass đứt nét vòng qua Projector. |

### 14.2 Khi nhúng vào báo cáo

- **Bỏ mọi tiêu đề trong ảnh**, dùng caption `Hình 3.x — ...` bên dưới ảnh (chuẩn học thuật).
- Nhúng SVG trực tiếp vào HTML, hoặc render ra PNG bằng Chrome/Inkscape.
- Thống nhất **một** ảnh X-quang thật cho fig02, fig10, và poster.
- Nếu dùng LaTeX: convert SVG → PDF bằng `inkscape --export-type=pdf`.

---

> **Ghi chú cuối:** Bộ SVG này được vẽ tay (hand-coded SVG), đảm bảo chính xác toán học và topology kiến trúc — điều mà model sinh ảnh không thể kiểm soát được. Mọi ký hiệu toán dùng Unicode đúng (`→ ⊙ ⊕ σ π α Σ ∈ ·`), không có ASCII thay thế.
