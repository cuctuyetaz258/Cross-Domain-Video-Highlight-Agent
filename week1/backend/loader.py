import os
import json
import subprocess 
from datetime import datetime, timedelta
import time
import yt_dlp
import wave
from faster_whisper import WhisperModel

def download_video(url: str, workspace_dir: str) -> str:
    """
    Chỉ tải Video/Audio từ YouTube, hoàn toàn KHÔNG tải phụ đề 
    để tránh triệt để lỗi HTTP 429.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    video_path = os.path.join(workspace_dir, "source_video.mp4")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_path,
        'writesubtitles': False,       # Tắt tải phụ đề YouTube
        'writeautomaticsub': False,    # Tắt tải sub tự động
        'quiet': True,
        'no_warnings': True,
        # "cookiesfrombrowser": ("chrome",),
    }
    
    print(f"[Backend - yt-dlp] Đang tải video từ: {url} ...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return video_path


def extract_audio_16k(video_path: str, output_audio_path: str) -> str:
    """
    Bóc tách audio chuẩn WAV 16kHz mono phục vụ Whisper và librosa.
    """
    print(f"[Backend - ffmpeg] Đang bóc tách audio 16kHz mono...")
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_audio_path


def transcribe_audio_whisper(audio_path: str, workspace_dir: str, model_size="small.en") -> str:
    """
    Chạy faster-whisper (mặc định small.en) để sinh bản dịch transcript.json
    gồm chính xác thời gian start, end, text của từng câu thoại.
    """
    print(f"[Backend - Whisper] Đang nạp mô hình faster-whisper '{model_size}'...")
    
    # Sử dụng CPU với int8 cho baseline nhẹ máy; nếu có GPU CUDA thì đổi:
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print(f"[Backend - Whisper] Đang nhận dạng giọng nói từ: {audio_path} ...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    transcript_data = []
    print(f" -> Ngôn ngữ nhận diện: '{info.language}' (Độ tin cậy: {round(info.language_probability * 100, 1)}%)")
    
    for segment in segments:
        transcript_data.append({
            "id": segment.id,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        
    # Lưu ra file json trong workspace để Node Observe và Explain dùng
    transcript_path = os.path.join(workspace_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
    print(f"[Backend - Whisper] Đã sinh xong bản dịch với {len(transcript_data)} câu thoại -> {transcript_path}")
    return transcript_path


def prepare_media_workspace(video_input: str, workspace_dir: str) -> dict:
    """
    Hàm tổng hợp cho Node 'observe': Tải video -> Bóc Audio -> Chạy Whisper small.en
    """
    os.makedirs(workspace_dir, exist_ok=True)
    audio_path = os.path.join(workspace_dir, "audio.wav")
    
    # 1. Tải video gốc
    if "youtube.com" in video_input or "youtu.be" in video_input:
        source_video = download_video(video_input, workspace_dir)
    else:
        source_video = video_input
        
    # 2. Bóc tách âm thanh 16kHz
    extract_audio_16k(source_video, audio_path)
    
    # 3. Sinh transcript bằng faster-whisper small.en
    transcript_path = transcribe_audio_whisper(audio_path, workspace_dir, model_size="small.en")
    
    return {
        "source_video": source_video,
        "audio_path": audio_path,
        "transcript_path": transcript_path
    }
    
def get_audio_duration(audio_path: str) -> float:
    """Đọc độ dài (giây) của file audio WAV không cần cài thêm thư viện."""
    with wave.open(audio_path, 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def transcribe_audio_whisper(audio_path: str, workspace_dir: str, model_size="small.en") -> str:
    """
    Chạy faster-whisper kèm thanh tiến độ % hiển thị theo thời gian thực.
    """
    # 1. ĐO ĐỘ DÀI VIDEO & ƯỚC TÍNH THỜI GIAN
    audio_duration = get_audio_duration(audio_path)
    # Cập nhật hệ số 0.8 (80% thời lượng) chuẩn với tốc độ CPU của bạn
    est_seconds = int(audio_duration * 0.8) + 15  
    
    start_time = datetime.now()
    expected_finish_time = start_time + timedelta(seconds=est_seconds)
    
    print("\n" + "="*60)
    print(f"  THÔNG TIN XỬ LÝ VIDEO:")
    print(f"   • Thời lượng video  : {int(audio_duration // 60)} phút {int(audio_duration % 60)} giây ({audio_duration:.1f}s)")
    print(f"   • Thời gian dự kiến: ~{est_seconds} giây")
    print(f"   • DỰ KIẾN HOÀN THÀNH : {expected_finish_time.strftime('%H:%M:%S')} (Giờ hệ thống)")
    print("="*60 + "\n")

    # 2. CHẠY WHISPER
    print(f"[Backend - Whisper] Đang nạp mô hình faster-whisper '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print(f"[Backend - Whisper] Đang nhận dạng giọng nói từ: {audio_path} ...")
    t0 = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    total_duration = info.duration
    print(f" -> Ngôn ngữ nhận diện: '{info.language}' (Độ tin cậy: {round(info.language_probability * 100, 1)}%)")
    print(f" -> Tổng thời lượng Audio: {total_duration:.1f}s | Bắt đầu xử lý từng đoạn thoại...\n")
    
    transcript_data = []
    
    # 3. VÒNG LẶP DỊCH THOẠI KÈM THANH TIẾN ĐỘ (%)
    for segment in segments:
        transcript_data.append({
            "id": segment.id,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        
        # --- TÍNH TOÁN VÀ IN PHẦN TRĂM (%) ---
        percent = min(100, int((segment.end / total_duration) * 100))
        
        # Tạo thanh trượt đồ họa dài 25 ký tự: [█████████----------------]
        bar_length = 25
        filled_length = int(bar_length * percent // 100)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        # Dùng \r và end="" để in ghi đè liên tục lên cùng 1 dòng (không bị trôi log)
        print(
            f"\r    Tiến độ dịch: [{bar}] {percent}% | Đã xử lý: {segment.end:.1f}s/{total_duration:.1f}s | Câu #{segment.id}", 
            end="\n", 
            flush=True
        )
        
    print() # Xuống dòng mới sau khi chạy xong 100%
        
    transcript_path = os.path.join(workspace_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
    actual_time = round(time.time() - t0, 1)
    print(f"\n[Backend - Whisper] Đã dịch xong {len(transcript_data)} câu thoại (Thực tế mất: {actual_time} giây) -> {transcript_path}")
    return transcript_path