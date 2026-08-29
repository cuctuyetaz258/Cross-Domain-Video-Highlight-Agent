$ErrorActionPreference = "Stop"

# Chuyển đến thư mục gốc của dự án
cd "f:\MLIoT2026\MLIoT2026-Final Project\Cross-Domai1n-Video-Highlight-Agent"

# Thêm tất cả thay đổi
Write-Host "Đang stage tất cả các thay đổi..."
git add .

# Thực hiện commit
Write-Host "Đang thực hiện commit..."
$commitMessage = @"
docs: Cập nhật context dự án và tài liệu implementation record 2026-08-29

- Tổng hợp tiến độ triển khai LTR-required pipeline
- Cập nhật các file markdown trong thư mục docs
"@
git commit -m $commitMessage

# Push lên origin
# Giả sử bạn đang làm việc trên branch mặc định hoặc branch hiện tại đã có upstream
Write-Host "Đang push lên GitHub..."
git push origin HEAD

Write-Host "Hoàn thành lệnh commit và push thành công!"
