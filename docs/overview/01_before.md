Nếu loại bỏ dạng câu hỏi trắc nghiệm (Multiple Choice) và chỉ tập trung vào việc **tăng độ chính xác (Accuracy - Exact Match & F1)** cho 2 dạng câu hỏi cốt lõi là **Open-ended** (Câu hỏi mở chuyên môn) và **Closed-ended** (Câu hỏi Yes/No), dưới đây là các hướng tiếp cận kỹ thuật hiệu quả nhất:

---

### Hướng 1: Áp dụng LoRA vào cả phần Vision Encoder (Cực kỳ hiệu quả cho ảnh Y khoa)
*   **Lý do:** Các mô hình VLM lớn (Gemma 4, PaliGemma) được tiền huấn luyện phần ảnh (Vision Tower - SigLIP) trên các bức ảnh tự nhiên (chó, mèo, phong cảnh, đồ vật). Do đó, mắt nhìn của nó đối với ảnh X-quang, MRI hay CT là rất hạn chế.
*   **Cách làm:** Trong file cấu hình LoRA (Block 2 của notebook), hiện tại bạn chỉ đang target LoRA vào các lớp text (Attention & MLP). Bạn hãy **mở khóa (unfreeze) hoặc áp dụng LoRA vào cả các lớp của Vision Encoder**.
    *   *Cách cấu hình trong code:*
        ```python
        # Thêm các module của Vision Tower vào target_modules, ví dụ:
        # "patch_projection", "self_attn", "mlp" của ViT
        ```
*   **Hiệu quả:** Cách này giúp Vision Encoder "học cách nhìn" các đặc trưng ảnh y khoa (vết nứt xương, khối u, dịch phổi) tốt hơn, tăng mạnh khả năng trả lời đúng câu hỏi mở.

---

### Hướng 2: Ràng buộc từ vựng đầu ra khi sinh (Constrained Decoding)
*   **Lý do:** Đối với câu hỏi **Yes/No (Closed)**, đôi khi mô hình hiểu đúng nhưng lại sinh ra các từ đồng nghĩa như: `"positive"`, `"negative"`, `"not present"`, hoặc `"normal"` thay vì đáp án chuẩn `"yes"` hoặc `"no"`. Điều này khiến điểm **Exact Match (EM) bị bằng 0** một cách đáng tiếc.
*   **Cách làm:** Sử dụng **Logits Processor** trong Hugging Face để ép mô hình chỉ được chọn từ trong nhóm token cho phép.
    *   *Ví dụ trong lúc chạy benchmark:* Nếu phát hiện câu hỏi thuộc nhóm `closed`, bạn cấu hình hàm `generate` chỉ được chọn giữa token ID của từ `"yes"` và `"no"`.
    *   *Code minh họa:*
        ```python
        # Force model to output only "yes" or "no" for closed questions
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1, 
            allowed_token_ids=[yes_token_id, no_token_id] # Chỉ cho phép 2 token này
        )
        ```
*   **Hiệu quả:** Tăng vọt điểm EM của các câu hỏi Yes/No lên mức tối đa vì loại bỏ hoàn toàn các câu trả lời lệch định dạng.

---

### Hướng 3: Ensemble Routing (Kết hợp thế mạnh của 2B và 4B)
*   **Lý do:** Qua bảng phân tích kết quả của bạn:
    *   Bản **2B** làm tốt hơn rõ rệt ở câu hỏi mở (**Open**: `23.00%` vs `20.00%` của 4B).
    *   Bản **4B** làm tốt hơn rõ rệt ở câu hỏi đóng (**Closed**: `70.92%` vs `68.53%` của 2B).
*   **Cách làm:** Xây dựng một bộ điều hướng (Router) khi chạy thử nghiệm:
    *   Nếu gặp câu hỏi dạng **Open** -> Gọi mô hình **Fine-tuned 2B** để trả lời.
    *   Nếu gặp câu hỏi dạng **Closed** -> Gọi mô hình **Fine-tuned 4B** để trả lời.
*   **Hiệu quả:** Đây là phương pháp "lấy ngắn nuôi dài", tận dụng tối đa thế mạnh của từng kích thước mô hình để đẩy điểm Overall EM tổng thể lên mức cao nhất mà không cần train lại.

---

### Hướng 4: Tiền xử lý ảnh chuyên dụng cho Y khoa (Medical Image Enhancement)
*   **Lý do:** Ảnh chụp X-quang và MRI thường có độ tương phản thấp hoặc nhiễu sáng, gây khó khăn cho mô hình khi nhận diện tổn thương nhỏ.
*   **Cách làm:** Áp dụng thuật toán tăng cường độ tương phản **CLAHE** (Contrast Limited Adaptive Histogram Equalization) lên ảnh trước khi đưa vào mô hình.
    *   *Thực hiện qua thư viện OpenCV/PIL:*
        ```python
        import cv2
        # Áp dụng CLAHE để làm rõ nét các chi tiết xương và mô mềm
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(raw_image)
        ```
*   **Hiệu quả:** Ảnh sắc nét hơn giúp Vision Encoder trích xuất đặc trưng chính xác hơn, trực tiếp nâng cao độ chính xác của câu trả lời.

---

### Hướng 5: Tiếp cận RAG Đa phương thức (Multimodal Retrieval-Augmented Generation)
*   **Lý do:** Mô hình có thể thiếu các kiến thức y khoa chuyên sâu, hiếm gặp hoặc khó đưa ra thuật ngữ chẩn đoán chuẩn xác nếu chỉ dựa vào trọng số tĩnh sau khi huấn luyện.
*   **Cách làm:**
    1.  **Xây dựng Vector Database:** Lưu trữ một thư viện các ca lâm sàng lịch sử (gồm ảnh X-quang/MRI và báo cáo chẩn đoán chi tiết của bác sĩ).
    2.  **Truy vấn thông tin (Retrieval):** Khi nhận ảnh và câu hỏi mới, sử dụng một mô hình embedding đa phương thức (như CLIP/SigLIP) để tìm ra 3 ca lâm sàng có hình ảnh và bệnh lý tương tự nhất trong cơ sở dữ liệu.
    3.  **Bổ sung ngữ cảnh (Augmentation):** Đính kèm các báo cáo chẩn đoán tương tự đó vào Prompt dưới dạng Text Context làm tài liệu tham khảo cho Gemma 4 trước khi trả lời.
*   **Hiệu quả:** Mô hình có nguồn tham chiếu trực tiếp để đối chiếu giải phẫu học, giúp tăng cực mạnh độ chính xác cho câu hỏi mở (Open-ended).

---

### Hướng 6: Trích xuất đặc trưng và Chuẩn hóa dựa trên Lược đồ xám (Histogram-based Processing)
*   **Lý do:** Ảnh y khoa có dải cường độ sáng (Pixel Intensity Distribution) rất hẹp. Các chi tiết tổn thương quan trọng thường bị che lấp do độ tương phản thấp của lược đồ xám thô.
*   **Cách làm:**
    1.  **Histogram Equalization (Cân bằng lược đồ xám):** Áp dụng cân bằng lược đồ xám toàn cục hoặc thích ứng cục bộ (CLAHE) để kéo giãn phân bố cường độ sáng của pixel, làm nổi bật biên giới hạn của mô mềm và cấu trúc xương.
    2.  **Explicit Statistical Features (Nhúng đặc trưng thống kê):** Tính toán các đặc trưng thống kê của lược đồ xám (như Mean, Variance, Skewness, Kurtosis, Entropy) của các vùng nghi vấn và chuyển thành dạng mô tả text hoặc vector phụ trợ đưa vào mạng để mô hình nhận diện mật độ mô (ví dụ: mật độ dịch phổi, độ dày thành tim).
*   **Hiệu quả:** Giúp Vision Encoder của mô hình dễ dàng tách biệt tổn thương khỏi nền ảnh xám mờ nhạt, nâng cao độ chính xác phân loại của cả câu hỏi mở và đóng.

---