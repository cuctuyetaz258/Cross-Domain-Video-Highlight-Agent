# Dùng môi trường Python 3.11 chuẩn của dự án
FROM python:3.11-slim

# Cài đặt ffmpeg (RẤT QUAN TRỌNG cho xử lý video/audio) và git
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc trong Docker
WORKDIR /app

# Copy file requirements.txt vào và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Lệnh mặc định: chạy backend với video từ argument
ENTRYPOINT ["python", "-m"]
CMD ["scripts.run_backend", "--help"]
