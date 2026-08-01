# SPEAKER NOTE BÁO CÁO MEDICAL VQA (SLIDE 3 ĐẾN SLIDE 17)

Bản Speaker Note được cô đọng ngắn gọn, đi thẳng vào sơ đồ/flow trên slide và nêu rõ Ý nghĩa cốt lõi của từng thiết kế kỹ thuật. Dạng văn bản thuần (Plain Text), sẵn sàng Copy-Paste trực tiếp vào PowerPoint.

---

### Số trang: Slide 3

Slide hiển thị:
- Sơ đồ Flow: Input (Ảnh 224x224 và Câu hỏi) -> Mô hình Đa phương thức -> Output.
- Khung tím bên phải: Phân loại Câu hỏi đóng (Yes/No, chiếm 50% dữ liệu, dễ đoán bừa 50%) và Câu hỏi mở (Sinh tên cơ quan, vị trí tổn thương, bệnh lý).

Ý nghĩa cốt lõi:
Tách riêng câu đóng và câu mở để tránh việc điểm câu đóng ngẫu nhiên che lấp sự yếu kém thực sự ở câu mở. Toàn bộ báo cáo tập trung hoàn toàn vào khả năng sinh câu tự do (Generative) cho câu hỏi mở.

Speaker note:
Slide 3 đặt bài toán Medical VQA dạng sinh câu tự do. Đầu vào gồm Ảnh 224x224 và Câu hỏi đi qua Mô hình đa phương thức để sinh câu trả lời. 

Điểm mấu chốt là ô khung tím bên phải phân loại dữ liệu: Câu hỏi đóng Yes/No dễ đoán bừa 50%, còn Câu hỏi mở mới là trọng tâm đánh giá năng lực sinh từ tự do. Ý nghĩa của việc tách này là tránh để điểm câu đóng ngẫu nhiên che lấp sự yếu kém thực sự ở câu mở — bài toán tập trung hoàn toàn vào câu hỏi mở.

---

### Số trang: Slide 4

Slide hiển thị:
- Định nghĩa Học sâu Đa phương thức tạo Không gian biểu diễn chung (Shared Representation Space).
- 4 ô chức năng: Tính dị thể, Căn chỉnh (Alignment), Hợp nhất (Fusion), Bổ sung thông tin.

Ý nghĩa cốt lõi:
Giải quyết sự khác biệt bản chất giữa lưới điểm ảnh liên tục và chuỗi token rời rạc, giúp hai nhánh bổ trợ thông tin cho nhau để suy luận.

Speaker note:
Slide 4 trình bày nền tảng Đa phương thức nhằm xây dựng Không gian biểu diễn chung. 

4 khung ô giải quyết 4 bài toán: Tính dị thể quy đổi ảnh pixel và chữ token về chung không gian; Căn chỉnh nối từ ngữ với đúng vùng ảnh; Hợp nhất trộn đặc trưng; và Bổ sung thông tin bù trừ lẫn nhau. Ý nghĩa cốt lõi là giúp PubMedCLIP và PubMedBERT kết hợp thông tin thị giác và ngữ nghĩa để cùng suy luận ra đáp án.

---

### Số trang: Slide 5

Slide hiển thị:
- Sơ đồ Flow trái: Vector ảnh v_img 128 chiều (teal) + Vector chữ v_txt 64 chiều (cam) -> Khối Concat -> Vector fused 192 chiều -> GRU Decoder.
- Biểu tượng cảnh báo đỏ: "KHÔNG phải phép cộng" (đây là phép NỐI chuỗi tổng 192 chiều).
- Khung nét đứt: Biến thể Projector đưa về cùng chiều P -> [B, 2P].

Ý nghĩa cốt lõi:
Intermediate Concat là giải pháp hợp nhất ghép nối đơn giản, chi phí tính toán rẻ, giữ nguyên ngữ nghĩa gốc của từng nhánh trước khi nâng cấp lên Cross-Attention.

Speaker note:
Slide 5 minh họa bước Hợp nhất đặc trưng bằng Concatenation. Vector ảnh 128 chiều màu xanh ghép nối trực tiếp với vector chữ 64 chiều màu cam tạo thành vector hợp nhất 192 chiều đi vào GRU Decoder. 

Lưu ý cảnh báo đỏ trên slide: Đây là phép NỐI chuỗi (tổng độ dài 192) chứ không phải phép cộng matrix. Ý nghĩa cốt lõi là giữ nguyên ngữ nghĩa gốc của từng nhánh với chi phí rẻ, làm tiền đề đối chiếu với khối Q-Former nâng cấp sau này.

---

### Số trang: Slide 6

Slide hiển thị:
- Sơ đồ Top: v_img (128d) -> Img Projector -> z_img (Pd); v_txt (64d) -> Txt Projector -> z_txt (Pd) -> Concat [B, 2P] -> GRU Decoder.
- Sơ đồ Bottom (Vì sao cần Projector): Không gian gốc R^128 khác R^64 (các điểm lệch nhau) -> Projector -> Không gian chung R^P (các cặp ảnh-chữ kéo sát cạnh nhau).

Ý nghĩa cốt lõi:
Đồng bộ số chiều và không gian hình học để các điểm đặc trưng tương ứng của ảnh và chữ nằm sát cạnh nhau, giúp Fusion và Cross-Attention đạt hiệu quả tối đa.

Speaker note:
Slide 6 giải thích về Bộ chiếu Projector. Sơ đồ phía trên cho thấy 2 khối Img và Txt Projector riêng biệt chiếu v_img 128 chiều và v_txt 64 chiều về cùng chiều P màu tím. 

Sơ đồ bên dưới minh họa ý nghĩa cốt lõi: Trước bộ chiếu, ảnh (R^128) và chữ (R^64) khác số chiều nên lệch nhau; sau bộ chiếu, hai nguồn được kéo về chung không gian R^P, các cặp ảnh-văn bản tương ứng nằm sát cạnh nhau, giúp fusion và cross-attention diễn ra chính xác.

---

### Số trang: Slide 7

Slide hiển thị:
- 3 khung ô: Self-attention (nội bộ 1 chuỗi), Cross-attention (Query chữ "fracture?" truy vấn Key/Value ảnh xương gãy), Vai trò & Q-Former (16 query học rút 17 visual token).

Ý nghĩa cốt lõi:
Cross-Attention tạo cơ chế định vị "mắt nhìn vào đâu khi đọc từ này", cho phép token câu hỏi trích xuất đúng vùng điểm ảnh tổn thương thay vì nén phẳng bức ảnh.

Speaker note:
Slide 7 so sánh Self-Attention và Cross-Attention. Ở ô Cross-Attention bên phải, Query là token câu hỏi ('fracture?') bắn truy vấn vào Key/Value ảnh để định vị vùng xương gãy. 

Ý nghĩa cốt lõi của cơ chế này là tạo liên kết định vị 'từ nào ứng với vùng ảnh nào'. Đây là nền tảng để Q-Former rút ra 17 visual token giàu chi tiết thay vì nén phẳng toàn bộ bức ảnh thành 1 vector.

---

### Số trang: Slide 8

Slide hiển thị:
- Top Flow -> Biểu đồ cột trọng số alpha: `effusion` (0.42 cao nhất), `pleural` (0.31), `is` (0.08), 2 vị trí `<pad>` bị mask score = -infinity -> alpha = 0 (cột đỏ gạch chéo).
- Output: context = tổng của (alpha_t * h_t), tổng alpha = 1.00.

Ý nghĩa cốt lõi:
Dồn trọng số vào từ khóa lâm sàng quan trọng, loại bỏ hoàn toàn nhiễu từ token đệm PAD, tạo tính minh bạch có thể giải thích được cho mô hình.

Speaker note:
Slide 8 trực quan hóa Temporal Attention bằng biểu đồ cột. Mô hình tự chấm điểm từ khóa y khoa: 'effusion' nhận alpha cao nhất 0.42, 'pleural' nhận 0.31; trong khi hai token đệm <pad> bị mask score = -infinity nên alpha = 0 (cột đỏ gạch chéo). 

Ý nghĩa cốt lõi là giúp mô hình dồn trọng số vào từ khóa lâm sàng chính và triệt tiêu hoàn toàn nhiễu từ token đệm.

---

### Số trang: Slide 9

Slide hiển thị:
- Khung Nguyên lý: Trượt Kernel / Pooling nén đặc trưng phân cấp (cạnh -> texture -> bộ phận -> đối tượng).
- Khung Đánh giá: Ưu/Nhược/Vai trò & 3 nhánh thực nghiệm (CustomCNN, ResNet-18 frozen, PubMedCLIP).

Ý nghĩa cốt lõi:
Tận dụng tính bất biến tịnh tiến của CNN để trích xuất đặc trưng hình học phân cấp từ mức thấp đến mức cao trên phim X-quang.

Speaker note:
Slide 9 tóm tắt họ mạng tích chập CNN. Nguyên lý trượt Kernel và Pooling nén giúp trích đặc trưng phân cấp từ đường nét đến cơ quan giải phẫu. 

Ý nghĩa cốt lõi là tận dụng tính bất biến tịnh tiến của CNN để nắm bắt cấu trúc hình học ảnh y khoa. Đồ án khảo sát 3 cấp độ: CustomCNN tự viết, ResNet-18 đóng băng và PubMedCLIP.

---

### Số trang: Slide 10

Slide hiển thị:
- Sơ đồ Flow Top: Ảnh 224x224 -> Block 1 (32ch 112x112) -> Block 2 (64ch 56x56) -> Block 3 (128ch 28x28) -> SE-Block (nét đứt) -> AdaptiveAvgPool + LayerNorm -> v_img [B,128].
- Khung công thức & 3 ô lý giải (Vì sao xếp tầng, BatchNorm, nén vector).

Ý nghĩa cốt lõi:
Thiết kế mạng tích chập thu nhỏ 3 khối gọn nhẹ nhưng vẫn đảm bảo kích thước chuẩn 128 chiều, hoàn toàn tương thích với đầu ra của ResNet-18.

Speaker note:
Slide 10 chi tiết hóa mạng CustomCNN qua sơ đồ luồng 3 khối: Kênh tăng dần từ 32 -> 64 -> 128, còn kích thước nén dần từ 112 -> 56 -> 28x28. Tầng AdaptiveAvgPool cuối nén về vector v_img 128 chiều. 

Ý nghĩa cốt lõi của thiết kế này là tạo ra một encoder ảnh tự phát triển cực kỳ gọn nhẹ nhưng vẫn đảm bảo tương thích 128 chiều với ResNet-18.

---

### Số trang: Slide 11

Slide hiển thị:
- 5 bước Top: Input F -> Squeeze GAP -> Excite FC-Sigmoid -> Scale -> Output F'.
- Visual Flow Bottom: 6 kênh c1-c6 ban đầu màu xanh nhạt -> Trọng số (c4=0.95, c5=0.10) -> Output F' (c4 xanh đậm rực rỡ khuếch đại, c5 mờ đục bị dập tắt).

Ý nghĩa cốt lõi:
Tự động tái phân bổ tầm quan trọng theo kênh — khuếch đại các kênh đặc trưng tổn thương quan trọng và dập tắt kênh nhiễu nền mà không làm tăng kích thước tensor.

Speaker note:
Slide 11 minh họa cơ chế SE-Block trên 6 kênh c1-c6. Qua 3 bước Squeeze-Excite-Scale, mô hình tính ra trọng số: kênh c4 đạt 0.95 được tô xanh đậm khuếch đại tối đa, kênh c5 chỉ đạt 0.10 bị mờ dập tắt. 

Ý nghĩa cốt lõi là giúp mô hình tự lọc ra kênh đặc trưng tổn thương quan trọng và loại bỏ nhiễu nền mà không tốn thêm chi phí bộ nhớ.

---

### Số trang: Slide 12

Slide hiển thị:
- Nguyên lý BiLSTM (3 cổng + Cell state, đọc 2 chiều xuôi/ngược).
- Flow: Tokens [B,32] -> Embedding -> BiLSTM -> (Temporal Attention / Concat last hidden) -> v_txt [B,64].

Ý nghĩa cốt lõi:
Đọc ngữ cảnh câu hỏi hai chiều để từ ở đầu câu vẫn nắm được ngữ cảnh từ ở cuối câu, cô đọng câu hỏi 32 token thành vector ngữ nghĩa 64 chiều.

Speaker note:
Slide 12 thể hiện nhánh mã hóa câu hỏi bằng BiLSTM. Nhờ cơ chế đọc hai chiều xuôi và ngược, các từ ở đầu câu vẫn nắm trọn ngữ cảnh từ ở cuối câu. 

Ý nghĩa cốt lõi là cô đọng câu hỏi 32 token thành một vector ngữ nghĩa v_txt 64 chiều chuẩn xác để chuẩn bị cho bước hợp nhất.

---

### Số trang: Slide 13

Slide hiển thị:
- Khối E Encoder (HIỂU, 2 chiều, PubMedBERT 768->64) vs Khối D Decoder (SINH, Masked Causal, GPT-2 124M 12 layers, greedy argmax, dừng ở eos).

Ý nghĩa cốt lõi:
Phân định rạch ròi hai vai trò trong mô hình Generative VQA — Encoder chịu trách nhiệm thấu hiểu đầu vào, Decoder chịu trách nhiệm sinh văn bản tự hồi quy.

Speaker note:
Slide 13 phân biệt Encoder và Decoder trong Transformer. Encoder bên trái dùng Self-Attention 2 chiều để thấu hiểu câu hỏi (PubMedBERT chiếu về 64d). Decoder bên phải dùng Causal Masked Attention để sinh từng từ tự hồi quy (GPT-2 124M). 

Ý nghĩa cốt lõi là phân định rõ vai trò: Encoder để HIỂU, Decoder để SINH.

---

### Số trang: Slide 14

Slide hiển thị:
- Khái niệm Seq2Seq (Encoder mã hóa -> Decoder sinh qua Cross-Attention/Prefix tokens).
- Bảng so sánh 4 hàng (Dữ liệu, Quan hệ tầm xa, Song song hóa, Điểm yếu) giữa CNN, LSTM và Transformer.

Ý nghĩa cốt lõi:
Lý giải sự tiến hóa kiến trúc — Transformer khắc phục điểm yếu tuần tự của LSTM và điểm yếu cục bộ của CNN, trở thành khung Seq2Seq tối ưu cho VLM.

Speaker note:
Slide 14 tổng hợp kiến trúc Seq2Seq và bảng đối chiếu 3 họ mô hình. Bảng so sánh cho thấy CNN mạnh về ảnh nhưng thiếu toàn cục; LSTM giỏi chuỗi ngắn nhưng bị nghẽn tuần tự; Transformer áp đảo về khả năng bắt quan hệ xa và tính toán song song. 

Ý nghĩa cốt lõi là khẳng định khung Encoder-Decoder dựa trên Transformer là cấu trúc tối ưu nhất cho các hệ thống VQA.

---

### Số trang: Slide 15

Slide hiển thị:
- Sơ đồ toàn cảnh 4 khối A-B-C-D mã hóa màu với 11 nút bấm (Nodes 1-11):
  + A - BỘ MÃ HOÁ (Node 1 CustomCNN/Node 2 ResNet-18/Node 3 SE-Block -> v_img 128d; Node 4 BiLSTM/Node 5 TransEnc/Node 6 TempAttn -> v_txt 64d).
  + B - CĂN CHỈNH & HỢP NHẤT (Node 7 Projector 128/64->P; Node 8 Concat 192d/2P; Node 9 Gated Attn).
  + C - SINH ĐÁP ÁN (Node 10 GRU Decoder 128 hidden + Cross-Attn -> logits -> greedy -> "right lung").
  + D - TINH CHỈNH RL (Node 11 Model SFT = Policy pi -> Sample vs Greedy Baseline -> Advantage -> Policy Gradient Loss).

Ý nghĩa cốt lõi:
Đóng khung toàn bộ không gian thiết kế thực nghiệm (Design Space), cho phép tiến hành các bài đánh giá Ablation Study bóc tách hiệu quả của từng khối thành phần.

Speaker note:
Slide 15 là sơ đồ toàn cảnh hệ thống được chia làm 4 khối A-B-C-D với 11 nút bấm chức năng. Khối A mã hóa Ảnh và Chữ; Khối B Hợp nhất qua Projector và Concat; Khối C Giải mã GRU sinh ra đáp án 'right lung'; Khối D Tinh chỉnh RL với SCST. 

Ý nghĩa cốt lõi của sơ đồ này là đóng khung toàn bộ không gian thiết kế, giúp tiến hành các thực nghiệm Ablation Study bóc tách hiệu quả của từng khối.

---

### Số trang: Slide 16

Slide hiển thị:
- Khối S - SFT (Cross-Entropy, Teacher Forcing, ổn định, hội tụ nhanh, áp dụng cho mọi mô hình).
- Khối R - RL (SCST Policy Gradient, tối ưu trực tiếp EM/F1/Semantic, phương sai lớn, bất ổn định trên dữ liệu nhỏ).

Ý nghĩa cốt lõi:
Đối chiếu hai triết lý học — SFT học bắt chước đáp án cứng ổn định; RL học tối ưu trực tiếp chỉ số đánh giá nhưng nhạy cảm với nhiễu trên dữ liệu nhỏ.

Speaker note:
Slide 16 so sánh hai phương pháp huấn luyện: SFT bên trái học có giám sát theo Cross-Entropy cực kỳ ổn định và là chuẩn bắt buộc. RL bên phải dùng SCST để tối ưu trực tiếp các chỉ số EM, F1. 

Ý nghĩa cốt lõi là đối chiếu 2 triết lý: SFT ổn định làm nền tảng, còn RL dù lý thuyết tối ưu trực tiếp thước đo nhưng thực nghiệm lại bất ổn trên tập dữ liệu y khoa nhỏ.

---

### Số trang: Slide 17

Slide hiển thị:
- Top Flowchart: Model SFT -> Logits -> 2 nhánh: Sample (r_sample) vs Greedy Baseline (r_greedy) -> Advantage A = r_sample - r_greedy -> Policy Gradient Loss.
- Middle Chart (Biểu đồ cột 2 trường hợp): 
  + Trường hợp A: Sample 0.9 > Greedy 0.5 -> A = +0.4 -> Mũi tên xanh LÊN (tăng xác suất).
  + Trường hợp B: Sample 0.1 < Greedy 0.9 -> A = -0.8 -> Mũi tên đỏ XUỐNG (giảm xác suất).

Ý nghĩa cốt lõi:
Sử dụng chính câu Greedy làm đường cơ sở tự phê bình (Self-Critical Baseline) mà không cần mạng Critic, tự điều chỉnh xác suất chuỗi sinh dựa trên chênh lệch Advantage.

Speaker note:
Slide 17 phân tích vòng lặp SCST. Mô hình sinh hai câu: câu Sample và câu Greedy (đóng vai trò Baseline tự thân). Nhìn vào biểu đồ cột ở giữa: Trường hợp A, câu Sample tốt hơn Greedy (A = +0.4), mũi tên xanh hướng lên tăng xác suất. Trường hợp B, câu Sample tệ hơn (A = -0.8), mũi tên đỏ hướng xuống giảm xác suất. 

Ý nghĩa cốt lõi của SCST là tự dùng bản thể Greedy của mình làm thước đo tự phê bình để định hướng cập nhật policy.

---

### Số trang: Slide 46

Slide hiển thị:
- Biểu đồ cột so sánh Gemma 2B (xanh lá) vs Gemma 4B (xanh dương):
  + EM Overall VQA-RAD: Gemma 2B (37.92%) vs Gemma 4B (41.46%).
  + EM Overall RadImageNet (tập sạch): Gemma 2B (30.20%) vs Gemma 4B (25.00%).
  + EM Open VQA-RAD: Gemma 2B (20.00%) vs Gemma 4B (14.50%).
  + sem_open x100: Gemma 2B (51.12) vs Gemma 4B (43.67).
- 2 ô thông tin bên phải: Gemma 4B thắng nội miền (EM Closed 62.95%); Gemma 2B thắng tập sạch (áp đảo ở mọi cột).
- Dòng chú thích: Gemma 4B có 41.2M tham số LoRA dư thừa biến thành trí nhớ học vẹt ảnh VQA-RAD chứ không thành năng lực tổng quát.

Ý nghĩa cốt lõi:
Phát hiện nghịch lý quá khớp: Mô hình lớn hơn (Gemma 4B) chứa tham số dư thừa nên học vẹt và ghi nhớ tập VQA-RAD nội miền, dẫn đến sụt giảm nghiêm trọng năng lực tổng quát khi đánh giá trên tập sạch RadImageNet-VQA. Gemma 2B nhỏ hơn nhưng tổng quát hóa vượt trội.

Speaker note:
Slide 46 thể hiện phát hiện đắt giá nhất: Nghịch lý mô hình lớn hơn lại tổng quát hóa kém hơn. 

Quan sát biểu đồ: Gemma 4B có vẻ dẫn trước ở tập nội miền VQA-RAD (EM 41.46%), nhưng lợi thế đó 100% đến từ câu đóng Yes/No. Ngay khi đưa sang tập sạch RadImageNet-VQA, Gemma 4B bị tụt xuống 25.00%, trong khi bản nhỏ Gemma 2B lại dẫn trước áp đảo với 30.20% EM tổng thể và 20% EM câu mở. 

Ý nghĩa cốt lõi là trên tập train nhỏ 1.793 mẫu, dung lượng tham số lớn của Gemma 4B chủ yếu bị mô hình biến thành trí nhớ học vẹt ảnh cũ chứ không tạo ra năng lực tổng quát cho bệnh nhân mới.

---

### Số trang: Slide 47

Slide hiển thị:
- Bảng tổng hợp kết quả 11 mô hình trên tập VQA-RAD (nội miền, rò rỉ 99.51% ảnh).
- Kết quả chính:
  + Baseline P_pubmed_gpt2: EM Overall 34.15%, Open 9.00%, sem_open 0.3480.
  + Best MLP P_resnet_trans: EM Overall 38.80%, Clinical AUC 0.6745 (cao nhất AUC).
  + gemma_2_lora_E2b: EM Open 20.00%, F1 Open 29.58%, sem_open 0.5112 (tất cả cao nhất câu mở).
  + gemma_4_lora_E4b: EM Overall 41.46%, EM Closed 62.95% (cao nhất câu đóng).
- Ghi chú chân slide: Cột đáng tin nhất để phân biệt mô hình là EM Open và sem_open - nơi Gemma bỏ xa toàn bộ nhánh GPT-2.

Ý nghĩa cốt lõi:
Tập VQA-RAD có tỉ lệ rò rỉ ảnh test cao (99.51%), nên điểm EM Overall bị câu hỏi đóng làm nhiễu. Thước đo giá trị nhất là EM Open và sem_open, chứng minh kiến trúc Gemma vượt trội hoàn toàn so với các baseline GPT-2.

Speaker note:
Slide 47 bảng tổng hợp toàn bộ kết quả trên tập VQA-RAD nội miền. 

Nếu chỉ nhìn điểm EM Overall tổng thể, Gemma 4B đứng đầu với 41.46%. Tuy nhiên, do tập VQA-RAD bị rò rỉ 99.51% ảnh test và ranh giới tin cậy là cộng trừ 4.5 điểm, cột đáng tin cậy nhất để phân biệt năng lực mô hình là EM Open và sem_open. Tại đây, Gemma 2B thiết lập kỷ lục tuyệt đối với 20% EM Open và điểm tương đồng ngữ nghĩa sem_open 0.5112, bỏ xa toàn bộ nhánh GPT-2. 

Ý nghĩa cốt lõi là Gemma 2B mới là mô hình xử lý câu mở tốt nhất trên dữ liệu nội miền.

---

### Số trang: Slide 48

Slide hiển thị:
- Bảng tổng hợp kết quả trên tập sạch RadImageNet-VQA (0% rò rỉ ảnh):
  + GPT-2 Baseline P_pubmed_gpt2: EM Overall 25.60%, EM Open 0.40% (tụt thảm hại!).
  + Q_qformer & D_semantic: EM Overall 23.60% (bị tụt sâu).
  + gemma_2_lora_E2b: EM Overall 30.20%, EM Closed 53.60%, EM Open 6.80%, sem_open 0.3986, Clinical AUC 0.6684 (áp đảo toàn bộ bảng!).
  + gemma_4_lora_E4b: EM Overall tụt về 25.00%, Open 3.60%, Clinical AUC 0.4770 (bị phạt nặng).
- 3 con số Highlight bên dưới: 30.20% (Kỷ lục tập sạch) | ~0% (EM Open của toàn bộ nhánh GPT-2 bị liệt) | 0.3986 (sem_open cao nhất).

Ý nghĩa cốt lõi:
Tập test sạch RadImageNet-VQA phơi bày sự thật: Toàn bộ nhánh GPT-2 bị tê liệt ở câu mở (EM Open gần 0%), trong khi Gemma 2B thể hiện khả năng tổng quát hóa vượt trội, xác lập kỷ lục mới 30.20% EM Overall và 0.3986 sem_open.

Speaker note:
Slide 48 đánh giá các mô hình trên tập test sạch RadImageNet-VQA hoàn toàn không rò rỉ ảnh. 

Ba con số highlight bên dưới lột tả bức tranh thực tế:
Thứ nhất, kỷ lục mới 30.20% EM Overall thuộc về Gemma 2B.
Thứ hai, con số xấp xỉ 0% phản ánh việc toàn bộ nhánh GPT-2 bị tê liệt hoàn toàn ở khả năng trả lời câu hỏi mở khi gặp ảnh mới.
Thứ ba, điểm sem_open 0.3986 của Gemma 2B vượt xa mức tối đa 0.2306 của GPT-2. 

Ý nghĩa cốt lõi là tập sạch chứng minh Gemma 2B có năng lực tổng quát hóa thực sự trên bệnh nhân mới, trong khi GPT-2 chỉ biết học thuộc lòng tập train.

---

### Số trang: Slide 49

Slide hiển thị:
- Biểu đồ cột đôi so sánh Mức tụt giảm từ VQA-RAD (xám) sang RadImageNet (xanh):
  + P_resnet_trans: 38.80% -> 27.00% (tụt -11.80%).
  + Q_qformer: 35.25% -> 23.60% (tụt -11.65%).
  + gemma_2 (2B): 37.92% -> 30.20% (chỉ tụt -7.72% - tốt nhất nhóm mạnh!).
  + gemma_4 (4B): 41.46% -> 25.00% (tụt nặng nhất -16.46%!).
- Khung bên phải: Bảng xếp hạng mức tụt giảm khi ra tập sạch.
- Chú thích chân slide: Ba giai đoạn thí nghiệm cùng chỉ về 1 kết luận: thêm dung lượng mô hình chủ yếu làm tăng khả năng ghi nhớ ảnh chứ không tăng năng lực suy luận trên bệnh nhân mới.

Ý nghĩa cốt lõi:
Thông điệp đắt giá nhất báo cáo "Dung lượng đổi lấy trí nhớ, không đổi lấy năng lực": Mô hình càng lớn (Gemma 4B) càng dễ học vẹt và tụt điểm thảm hại nhất khi gặp dữ liệu mới. Gemma 2B chính quy hóa tốt là cấu hình tối ưu nhất.

Speaker note:
Slide 49 đúc kết thông điệp quan trọng nhất đề tài: Dung lượng đổi lấy trí nhớ, không đổi lấy năng lực. 

Nhìn vào biểu đồ và khung bên phải: Mô hình lớn nhất Gemma 4B là mô hình chịu mức tụt điểm thảm hại nhất (-16.46%) khi chuyển từ tập rò rỉ sang tập sạch. Trái lại, Gemma 2B có mức tụt thấp hơn nhiều (-7.72%) và về đích với điểm số cao nhất trên tập sạch. 

Ý nghĩa cốt lõi đúc kết từ cả 3 giai đoạn thí nghiệm: Trên tập train nhỏ 1.793 mẫu, việc tăng dung lượng tham số chỉ làm mô hình tăng khả năng ghi nhớ ảnh cũ chứ không tăng năng lực suy luận y khoa cho bệnh nhân mới. Gemma 2B chính là cấu hình tối ưu nhất cho bài toán này.

