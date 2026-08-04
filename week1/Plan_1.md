## Xử lí cấu trúc của từng node(Nguyên, Thảo Anh)
1. Thảo Anh tìm video phân bố theo từng loại tín hiệu
2. Nguyên xử lí dạng đầu vào của video: 
    - URL
    - Local File
3. Nguyên transcript, tách audito
    - Output của transcript: text, timestamp
    - Render 9:16
4. Thảo Anh:
    - Nhận output từ 3
    - Tạo Prompt chặt chẽ nhất có thể để gửi vào LLM, cụ thể hơn: 
        - Yêu cầu đọc toàn bộ transcript chọn 5 khoảng khắc chú ý nhất có độ dài 60s
        - Output: Trả về cấu trúc dễ đọc để máy tính xử lí: khoảng thời gian lựa chọn, lý do chọn khoảng,... (tìm loại định dạng file luôn nhé).

##  Tìm hiểu kĩ phần trích xuất đặc trưng theo từng tầng tín hiệu ở phần 2.1 trong proposal(Khánh Vân)
Trong tuần một tạm thời chưa cần tích hợp vào pipeline nhưng tuần 2 sẽ cần.
- Nêu sự khác biệt của từng tầng tín hiệu, và trích xuất đặc trưng theo từng tầng
(phần phân công ở đây có thể sai vì em chưa tìm hiểu kĩ phần này)


## UI Cơ bản(Nguyên Anh)
- Input: Nhận URL/local file
- Excution: nút xử lí video
- Output: Phần hiện thị 5 đoạn clip được cắt

## Xây dựng luồng xử lí dữ liệu và thiết kế mô hình: 
- Nhận kết quả từ Nguyên, Thảo Anh và Nguyên Anh để hoàn thiện baseline cơ bản



