$ErrorActionPreference = "Stop"

# Chuyển đến thư mục gốc của dự án
cd "f:\MLIoT2026\MLIoT2026-Final Project\Cross-Domai1n-Video-Highlight-Agent"

# Thêm tất cả thay đổi
Write-Host "Đang stage tất cả các thay đổi..."
git add .

# Loại bỏ thư mục docs khỏi danh sách chuẩn bị commit (unstage docs/)
Write-Host "Đang loại bỏ thư mục docs/ ra khỏi commit..."
git reset -- docs/

# Thực hiện commit
Write-Host "Đang thực hiện commit..."
$commitMessage = "Cập nhật mã nguồn hệ thống (LTR-Required Pipeline), bỏ qua thư mục docs"
git commit -m $commitMessage

# Push lên origin
Write-Host "Đang push lên GitHub..."
git push origin HEAD

Write-Host "Hoàn thành lệnh commit (đã bỏ qua docs) và push thành công!"
