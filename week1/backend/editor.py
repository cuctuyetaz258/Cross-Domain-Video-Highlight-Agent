import os
import subprocess

def render_short_9_16(source_video: str, start_time: int, end_time: int, output_path: str):
    """
    Cắt video từ mốc start_time đến end_time và crop tâm khung hình từ 16:9 sang 9:16.
    """
    print(f"[Backend] Đang render clip ngắn 9:16: {output_path} ({start_time}s -> {end_time}s)...")
    
    # Bộ lọc ffmpeg:
    # 1. crop=ih*(9/16):ih -> Cắt chiều rộng bằng chiều cao * 9/16, giữ chiều cao (ih), tự động canh giữa
    # 2. scale=1080:1920   -> Chuẩn hóa độ phân giải về Full HD dọc
    vf_filter = "crop=ih*(9/16):ih,scale=1080:1920"
    
    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),     # Mốc bắt đầu
        "-to", str(end_time),       # Mốc kết thúc
        "-i", source_video,
        "-vf", vf_filter,
        "-c:v", "libx264",          # Codec H.264 phổ biến
        "-preset", "fast",          # Render nhanh cho baseline
        "-c:a", "aac",              # Codec audio AAC
        "-b:a", "128k",
        output_path
    ]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[Backend] Hoàn tất render: {output_path}")

def process_all_highlights(source_video: str, highlights: list, workspace_dir: str) -> list:
    """
    Nhận danh sách highlights từ Node 'decide', render toàn bộ thành file MP4.
    Trả về danh sách đường dẫn các file thành phẩm.
    """
    output_files = []
    shorts_dir = os.path.join(workspace_dir, "shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    
    for idx, hl in enumerate(highlights):
        out_name = f"highlight_{idx+1}.mp4"
        out_path = os.path.join(shorts_dir, out_name)
        
        render_short_9_16(
            source_video=source_video,
            start_time=hl["start_time"],
            end_time=hl["end_time"],
            output_path=out_path
        )
        output_files.append(out_path)
        
    return output_files