# 📋 Hướng Dẫn Gán Nhãn Dữ Liệu Ground Truth (Annotation Guide)

> **Dự án:** Multi-Agent Cross-Domain Video Highlight Extractor  
> **Mục tiêu:** Xây dựng tập dữ liệu chuẩn (In-Domain Ground Truth) gồm 10 video ($\le 30$ phút) để đo lường và đánh giá chất lượng mô hình theo các độ đo **Temporal IoU**, **Hit@3** và **F1-Score**.

---

## 🎯 1. Mục Tiêu & Nguyên Tắc Chung

Mỗi thành viên khi xem video sẽ đóng vai trò là một **Chuyên gia Biên tập Video Ngắn (Shorts/Reels/TikTok Creator)** để chọn ra những khoảnh khắc đắt giá nhất.

### ⚠️ 4 Quy Tắc Cốt Lõi (BẮT BUỘC)

1. **Số lượng clip:** Mỗi video chọn chính xác **3 đến 5 đoạn highlight** (khuyến nghị: 3 đoạn tốt nhất).
2. **Thời lượng chuẩn:** Mỗi đoạn BẮT BUỘC có độ dài từ **30 đến 90 giây** ($30.0s \le \text{end\_time} - \text{start\_time} \le 90.0s$).
   * *Lý tưởng nhất:* **45s – 60s** (vừa đủ truyền tải trọn vẹn 1 ý).
   * *Không chọn clip < 30s:* Nếu đoạn đắt giá chỉ dài 15s, hãy lấy thêm phần dẫn dắt (*setup*) phía trước hoặc phần đúc kết (*payoff*) phía sau.
3. **Canh biên tự nhiên:** Điểm bắt đầu (`start_time`) và điểm kết thúc (`end_time`) phải **trùng khớp với ranh giới câu nói tự nhiên** (dấu chấm câu, khoảng lặng ngắt nghỉ), tuyệt đối không cắt giữa chừng từ ngữ hoặc câu văn.
4. **Tính độc lập (Standalone):** Đoạn trích xuất phải có nghĩa hoàn chỉnh, người xem lướt mạng xã hội có thể hiểu ngay mà không cần xem toàn bộ video gốc.

---

## 🎬 2. Tiêu Chí Chọn Highlight Theo Từng Miền (Domain)

### 🎓 A. Bài Giảng (Lecture)
* **Khái niệm cốt lõi:** Đoạn định nghĩa hoặc giải thích bản chất vấn đề một cách trực quan, dễ hiểu.
* **Khoảnh khắc "Aha!":** Đoạn gỡ nút thắt tư duy, giải thích lý do vì sao một công thức hay hiện tượng lại hoạt động.
* **Mẹo / Kỹ năng thực hành:** Hướng dẫn giải quyết một bài toán cụ thể, mẹo ghi nhớ nhanh.
* **Cấu trúc clip chuẩn:**
  * *0s – 10s (Hook):* Đặt vấn đề / câu hỏi gây tò mò (*"Tại sao mạng nơ-ron lại cần đạo hàm?"*).
  * *10s – 45s (Body):* Minh họa trực quan, dẫn dắt logic ngắn gọn.
  * *45s – 60s (Takeaway):* Kết luận / bài học cốt lõi.

### 🎙️ B. Trò Chuyện & Phỏng Vấn (Podcast)
* **Quan điểm phản trực giác / gây tranh luận:** Ý kiến đi ngược số đông hoặc góc nhìn mới lạ từ chuyên gia.
* **Câu chuyện cá nhân đắt giá:** Trải nghiệm thất bại, bước ngoặt cuộc đời, bài học đắt giá.
* **Lời khuyên thực chiến:** Bí quyết thành công, phương pháp học tập/làm việc đã được chứng minh.
* **Cấu trúc clip chuẩn:**
  * *0s – 5s (Hook):* Phát biểu gây sốc hoặc câu hỏi hấp dẫn từ người dẫn/khách mời.
  * *5s – 45s (Story/Argument):* Diễn giải chi tiết câu chuyện hoặc luận điểm.
  * *45s – 60s (Punchline/Lesson):* Câu nói đúc kết mang tính truyền cảm hứng.

### 🎤 C. Hài Độc Thoại (Stand-up Comedy - Tuỳ chọn)
* **Cấu trúc Joke chuẩn:** Setup (dẫn chuyện) $\to$ Punchline (cú lật bất ngờ) $\to$ Tiếng cười của khán giả.

---

## 📊 3. Thang Điểm Đánh Giá (`importance_score`)

Chấm điểm từ **1 đến 5** cho từng đoạn highlight được chọn:

| Điểm | Mức độ | Ý nghĩa |
|:---:|:---|:---|
| **5** | 🔥 **Viral / Must-watch** | Nội dung xuất sắc, cực kỳ cuốn hút, chắc chắn giữ chân người xem đến giây cuối cùng. |
| **4** | ⭐ **Rất hay** | Nội dung giá trị cao, lập luận sắc bén hoặc mang lại kiến thức bổ ích. |
| **3** | 👍 **Đạt chuẩn** | Nội dung ổn, truyền tải trọn vẹn 1 ý nhưng chưa tạo ấn tượng quá mạnh. |
| **2** | ⚠️ **Trung bình** | Đoạn nói hơi dài dòng hoặc thiếu điểm nhấn (hạn chế chọn). |
| **1** | ❌ **Không đạt** | Rời rạc, thiếu ngữ cảnh, không phù hợp làm video ngắn. |

---

## 🗂️ 4. Danh Sách 10 Video Cần Gán Nhãn ($\le 30$ phút)

### 🎓 5 Video Bài Giảng (Lecture)
1. **L1 (`WUvTyaaNkzM`):** *The Essence of Calculus* (3Blue1Brown - 17.1 phút)  
   Link: `https://www.youtube.com/watch?v=WUvTyaaNkzM`
2. **L2 (`aircAruvnKk`):** *But What is a Neural Network?* (3Blue1Brown - 18.7 phút)  
   Link: `https://www.youtube.com/watch?v=aircAruvnKk`
3. **L3 (`IHZwWFHWa-w`):** *Gradient Descent, How Neural Networks Learn* (3Blue1Brown - 20.6 phút)  
   Link: `https://www.youtube.com/watch?v=IHZwWFHWa-w`
4. **L4 (`wjZofJX0v4M`):** *Transformers, The Tech Behind LLMs* (3Blue1Brown - 27.2 phút)  
   Link: `https://www.youtube.com/watch?v=wjZofJX0v4M`
5. **L5 (`g2-_pnmhO4A`):** *Learn 97% of Harvard's CS50 in 25 Minutes* (Fireship - 25.4 phút)  
   Link: `https://www.youtube.com/watch?v=g2-_pnmhO4A`

### 🎙️ 5 Video Podcast / Phỏng Vấn (Podcast)
1. **P1 (`DNQDqq4mWSY`):** *Sam Altman on GPT-5 & Future of AGI* (Lex Fridman Clips - 11.5 phút)  
   Link: `https://www.youtube.com/watch?v=DNQDqq4mWSY`
2. **P2 (`waLjtcUq5Mc`):** *Tucker Carlson Reflects on Putin Interview* (Lex Fridman Clips - 16.6 phút)  
   Link: `https://www.youtube.com/watch?v=waLjtcUq5Mc`
3. **P3 (`1bszFX_XcbU`):** *The Top Study Habits to Improve Learning* (Huberman Lab Clips - 14.4 phút)  
   Link: `https://www.youtube.com/watch?v=1bszFX_XcbU`
4. **P4 (`-cRswJf8OnI`):** *Billionaire Reveals BRUTAL Truth About Money* (Diary Of A CEO Clips - 21.9 phút)  
   Link: `https://www.youtube.com/watch?v=-cRswJf8OnI`
5. **P5 (`u36A-YTxiOw`):** *The Best Way To Launch Your Startup* (Y Combinator - 21.1 phút)  
   Link: `https://www.youtube.com/watch?v=u36A-YTxiOw`

---

## 📝 5. Định Dạng File Nhãn Đầu Ra (JSON Format)

Mỗi video sau khi gán nhãn sẽ được lưu thành 1 file JSON độc lập trong thư mục `docs/ground_truth/<video_id>.json`.

### Schema Định Dạng:
```json
{
  "video_id": "string (Mã ID của video trên YouTube)",
  "title": "string (Tên video)",
  "domain": "lecture | podcast | standup",
  "annotator": "string (Tên người gán nhãn)",
  "duration": "float (Tổng độ dài video tính bằng giây)",
  "highlights": [
    {
      "highlight_id": "hl_01",
      "start_time": 120.5,
      "end_time": 175.0,
      "importance_score": 5,
      "summary": "Tóm tắt ngắn gọn nội dung đoạn clip (1 câu)",
      "reason": "Lý do vì sao đoạn này hay và đáng xem"
    }
  ]
}
```

---

## 💡 6. Ví Dụ File Nhãn Hoàn Chỉnh Mẫu

### Ví dụ 1: File `docs/ground_truth/DNQDqq4mWSY.json` (Podcast)

```json
{
  "video_id": "DNQDqq4mWSY",
  "title": "Sam Altman on GPT-5 | Lex Fridman Podcast",
  "domain": "podcast",
  "annotator": "ThaoAnh",
  "duration": 690.0,
  "highlights": [
    {
      "highlight_id": "hl_01",
      "start_time": 45.2,
      "end_time": 98.6,
      "importance_score": 5,
      "summary": "Sam Altman thừa nhận GPT-4 còn nhiều điểm hạn chế và kỳ vọng vào bước nhảy vọt của GPT-5.",
      "reason": "Câu mở đầu thẳng thắn 'GPT-4 kind of sucks' tạo hook cực mạnh; sau đó là giải thích về tầm nhìn của các mô hình kế tiếp, rất thu hút người quan tâm công nghệ."
    },
    {
      "highlight_id": "hl_02",
      "start_time": 210.0,
      "end_time": 265.4,
      "importance_score": 4,
      "summary": "Bàn luận về khả năng suy luận và đột phá khoa học của AGI trong tương lai.",
      "reason": "Đoạn đối thoại có chiều sâu triết học, thảo luận về việc liệu AI có thể tự khám phá ra các định luật vật lý mới hay không."
    },
    {
      "highlight_id": "hl_03",
      "start_time": 420.1,
      "end_time": 472.8,
      "importance_score": 4,
      "summary": "Góc nhìn về chi phí tính toán và nhu cầu năng lượng khổng lồ của AI thế hệ mới.",
      "reason": "Cung cấp insight thực tế về rào cản phần cứng và năng lượng mà ngành công nghiệp AI đang phải đối mặt."
    }
  ]
}
```

---

## 🛠️ 7. Quy Trình Thực Hiện Từng Bước Cho Nhóm

1. **Bước 1 — Xem video và mở transcript:**
   * Bạn có thể chạy lệnh `python -m scripts.run_backend "URL"` để hệ thống tải video và tạo sẵn file `transcript.json`.
   * Mở video trên YouTube hoặc mở `output/<video_id>/transcript.json` để đối chiếu mốc giây chính xác.
2. **Bước 2 — Chọn 3 đoạn hay nhất:**
   * Xác định mốc `start_time` và `end_time` (chú ý: phải cách nhau từ 30s đến 90s).
3. **Bước 3 — Tạo file nhãn JSON:**
   * Tạo file tương ứng `docs/ground_truth/<video_id>.json` theo mẫu ở Mục 6.
4. **Bước 4 — Kiểm tra chất lượng:**
   * Kiểm tra xem các clip có bị đè thời gian lên nhau không.
   * Đảm bảo `end_time - start_time` nằm trong khoảng $[30, 90]$ giây.
