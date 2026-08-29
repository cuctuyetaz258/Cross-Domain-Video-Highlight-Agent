# Hướng dẫn giải phóng RAM và GPU trên Windows

Tài liệu này dành cho project `Cross-Domain-Video-Highlight-Agent` khi chạy bằng
Conda `MLIoT` và Streamlit trong VS Code.

## 1. Khi nào cần giải phóng bộ nhớ?

Các dấu hiệu thường gặp:

- Chạy lại Streamlit nhiều lần nhưng terminal cũ chưa dừng.
- Cổng `8501` vẫn đang được sử dụng.
- Task Manager còn nhiều tiến trình `python.exe` dùng hàng trăm MB RAM.
- PyTorch báo `CUDA out of memory` dù pipeline hiện tại không chạy.
- UI vẫn hiển thị phiên bản code cũ sau khi restart.

Mỗi Streamlit process có thể giữ model, CUDA context và cache riêng. Vì vậy nhiều
process chạy song song sẽ làm RAM/GPU tăng nhanh.

## 2. Cách an toàn nhất: dừng từ terminal

Trong từng terminal đang chạy Streamlit, nhấn:

```text
Ctrl+C
```

Đợi terminal trở về prompt rồi mới chạy lại UI. Chỉ nên giữ **một** Streamlit
process cho project.

## 3. Kiểm tra các cổng UI

Chạy trong PowerShell:

```powershell
netstat -ano | Select-String ':8501|:8502|:8765|:8766|:8767'
```

Dòng có trạng thái `LISTENING` cho biết PID đang giữ cổng. Nếu không có
`LISTENING`, Streamlit đã dừng. Các kết nối `SYN_SENT` hoặc `CLOSE_WAIT` từ trình
duyệt thường tự biến mất sau một lúc và không giữ model trong RAM/GPU.

## 4. Liệt kê đúng Streamlit process của project

Không dùng `taskkill /IM python.exe`, vì lệnh đó có thể dừng Jupyter, VS Code,
training job hoặc project Python khác.

Mở PowerShell và chạy lệnh chỉ-đọc sau:

```powershell
$projectProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'streamlit\s+run\s+frontend[\\/]app\.py'
    }

$projectProcesses |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
    Format-List
```

Kiểm tra từng `CommandLine` trước khi dừng. Nếu PowerShell báo `Access denied`,
mở PowerShell bằng quyền Administrator rồi chạy lại bước kiểm tra.

Xem lượng RAM của các PID vừa tìm được:

```powershell
$projectProcesses.ProcessId | ForEach-Object {
    Get-Process -Id $_ -ErrorAction SilentlyContinue |
        Select-Object Id, ProcessName,
            @{Name='RAM_MB'; Expression={[math]::Round($_.WorkingSet64 / 1MB, 1)}}
}
```

## 5. Dừng đúng các PID đã xác minh

Chỉ chạy bước này sau khi danh sách ở bước 4 đúng là Streamlit của project:

```powershell
$projectProcesses.ProcessId | ForEach-Object {
    taskkill.exe /PID $_ /F
}
```

Sau đó xác nhận không còn process:

```powershell
$remaining = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'streamlit\s+run\s+frontend[\\/]app\.py'
    }

"Remaining project Streamlit processes: $(@($remaining).Count)"
```

Kết quả mong đợi:

```text
Remaining project Streamlit processes: 0
```

## 6. Kiểm tra GPU

Nếu máy có NVIDIA GPU:

```powershell
nvidia-smi
```

Hoặc chỉ liệt kê compute process:

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Khi Streamlit đã dừng, Python của project không còn xuất hiện. Trên Windows WDDM,
`used_memory` đôi khi hiện `[N/A]`; PID vẫn là thông tin hữu ích để xác định process.

`torch.cuda.empty_cache()` chỉ giải phóng cache bên trong **process đang sống**.
Nó không thể giải phóng CUDA context của một Streamlit process khác. Muốn thu hồi
toàn bộ RAM/GPU của process cũ, cần dừng process đó.

## 7. Khởi động lại đúng một UI

```powershell
conda activate MLIoT
cd "F:\MLIoT2026\MLIoT2026-Final Project\Cross-Domai1n-Video-Highlight-Agent"
python -m streamlit run frontend\app.py --server.port 8501
```

Nếu `8501` thực sự đang được ứng dụng khác sử dụng, chọn một cổng khác:

```powershell
python -m streamlit run frontend\app.py --server.port 8502
```

Không chạy lại cùng lệnh trong terminal thứ hai khi UI thứ nhất vẫn còn hoạt động.

## 8. RAM, GPU và dung lượng ổ đĩa là ba vấn đề khác nhau

- Dừng Python/Streamlit giải phóng RAM và GPU.
- Xóa `output/`, feature cache hoặc model chỉ giải phóng dung lượng ổ đĩa.
- Xóa file không làm giảm RAM của process đang chạy.
- Không xóa `output/<video_id>/analysis/ltr_analysis_snapshot.json` nếu còn muốn
  rerender LTR hoặc thử model OpenAI khác mà không chạy lại feature extraction.

## Checklist nhanh

1. Nhấn `Ctrl+C` trong terminal Streamlit.
2. Kiểm tra cổng bằng `netstat`.
3. Liệt kê và đọc `CommandLine` của từng PID.
4. Chỉ dừng PID khớp `streamlit run frontend/app.py`.
5. Kiểm tra lại bằng `nvidia-smi`.
6. Khởi động đúng một UI.
