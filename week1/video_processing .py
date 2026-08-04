
import yt_dlp
import os
import json
import subprocess

def is_youtube_url(path: str) -> bool:
    """Kiểm tra input là URL YouTube hay file local."""
    return "youtube.com" in path or "youtu.be" in path

def download_youtube_media(url: str, output_dir: str):
    """
    Tải video và transcript/subtitles từ YouTube bằng yt-dlp.
    Trả về đường dẫn video đã tải và danh sách transcript (nếu có).
    """
    video_path = os.path.join(output_dir, "video.mp4")
    transcript_path = os.path.join(output_dir, "transcript.json")
    
    # Cấu hình yt-dlp tải video chất lượng vừa đủ (tránh file quá nặng) + auto subtitle
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': video_path,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'vi'],
        'subtitlesformat': 'json3',
        'quiet': True,
        'overwrites': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        
        # Lấy chapter nếu có[cite: 1]
        chapters = info.get('chapters', [])
        
        # Xử lý lưu transcript và chapters vào transcript.json
        metadata_to_save = {
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "chapters": chapters,
            "subtitles_available": bool(info.get("requested_subtitles"))
        }
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)
            
    return video_path, transcript_path

def extract_audio_16k_mono(video_path: str, audio_output_path: str):
    """
    Dùng ffmpeg tách track audio ra định dạng .wav 16kHz mono[cite: 1].
    """
    command = [
        "ffmpeg",
        "-y",                   # Ghi đè nếu file đã tồn tại
        "-i", video_path,       # File đầu vào
        "-ar", "16000",         # Sample rate 16kHz[cite: 1]
        "-ac", "1",             # Audio channels: 1 (mono)[cite: 1]
        "-c:a", "pcm_s16le",    # Codec chuẩn cho WAV
        audio_output_path
    ]
    # Chạy lệnh ffmpeg dưới dạng subprocess
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_output_path