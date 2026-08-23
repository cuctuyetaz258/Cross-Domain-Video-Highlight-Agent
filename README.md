# Cross-Domain Video Highlight Agent

Hệ thống tự động trích xuất 3–5 highlight dọc 9:16 từ video bài giảng,
podcast và hài độc thoại, kèm transcript và lý do lựa chọn.

## Trạng thái

Repository đang trong giai đoạn chuẩn hóa nền tảng sau prototype Sprint 1.
Source cũ được giữ trong `week1/` để tham khảo; implementation mới nằm trong
`highlight_agent/`.

## Yêu cầu hệ thống

- Python 3.10–3.12; khuyến nghị Python 3.11 cho môi trường chung của nhóm.
- `ffmpeg`, `ffprobe` có trong `PATH`.
- Git.

### Cài ffmpeg

**macOS:**

```bash
brew install ffmpeg
```

**Windows:**

```bash
# Tải từ https://www.gyan.dev/ffmpeg/builds/ (bản essentials)
# Giải nén và thêm thư mục bin/ vào biến PATH hệ thống
# Kiểm tra:
ffmpeg -version
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Khởi tạo môi trường

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` là nguồn dependency duy nhất của project. File này bao
gồm cả dependency Sprint 1, feature extraction Sprint 2 và công cụ test.

## Biến môi trường

```bash
cp .env.example .env
```

Điền API key cần sử dụng vào `.env`. Không commit `.env` lên GitHub.

| Biến | Mô tả | Bắt buộc |
|---|---|---|
| `GROQ_API_KEY` | API key từ [console.groq.com](https://console.groq.com) | ✅ Có |
| `GOOGLE_API_KEY` | API key từ [aistudio.google.com](https://aistudio.google.com) | Tùy chọn |
| `OPENROUTER_API_KEY` | API key từ [openrouter.ai](https://openrouter.ai) | Tùy chọn |
| `HF_TOKEN` | HuggingFace token | Tùy chọn |
| `YTDLP_COOKIES_BROWSER` | Browser đang đăng nhập YouTube (VD: `chrome`) | Tùy chọn |

## Cấu trúc chính

```text
highlight_agent/
├── agent/       # LangGraph state, nodes và graph
├── features/    # Năm tầng tín hiệu đa miền
├── llm/         # LLM clients, prompts và extractor
├── media/       # Input, audio, transcript, render, thumbnail
└── schemas/     # Pydantic data contracts dùng chung

frontend/        # Streamlit UI
scripts/         # Script hỗ trợ phát triển/demo
tests/           # Automated tests
docs/            # Tài liệu kỹ thuật và danh sách video mẫu
week1/           # Prototype cũ, không phải package chính
output/          # Artifact sinh ra, không được commit
```

Data contract phiên bản hiện tại: [docs/schema_for_all.md](docs/schema_for_all.md).

## Chạy kiểm tra

```bash
pytest
ruff check .
```

## Chạy Backend Sprint 1

Chỉ chuẩn bị video, audio và transcript:

```bash
python -m scripts.run_backend sample.mp4
```

Với YouTube:

```bash
python -m scripts.run_backend "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-browser chrome
```

Nếu đã có candidate JSON từ LLM/scoring step, render 3–5 clip:

```bash
python -m scripts.run_backend sample.mp4 --candidates candidates.json
```

## Trích xuất Highlight bằng AI (Prompt Engineering)

Sau khi đã có file `transcript.json`, dùng script sau để gọi AI chọn highlight:

```bash
python -m scripts.extract_highlights "output/VIDEO_ID/transcript.json" --output "output/VIDEO_ID/candidates.json"
```

Script sẽ:
1. Đọc transcript từ file JSON.
2. Gửi cho Groq AI (Llama 3.3 70B) kèm prompt đa miền.
3. Lưu kết quả thành file `candidates.json` chuẩn Pydantic.

Yêu cầu: `GROQ_API_KEY` đã được điền trong file `.env`.

## Chạy Agent đầy đủ

Chạy đầy đủ năm pha LangGraph với naive baseline:

```bash
python -m scripts.run_agent sample.mp4 --domain lecture --highlight-count 3
```

Ép dùng Whisper và tắt subtitle khi cần test riêng:

```bash
python -m scripts.run_agent sample.mp4 \
  --domain lecture \
  --transcript-source whisper \
  --no-subtitles
```

`--transcript-source` nhận `auto`, `youtube` hoặc `whisper`. Chế độ `auto`
ưu tiên caption YouTube và fallback sang Whisper.

`Analyze` sẽ dùng candidate bên ngoài nếu truyền `--candidates`; nếu không,
nó tạo baseline giả lập có seed ổn định để demo Sprint 1.

## Chạy bằng Docker

### Build image

```bash
docker build -t highlight-agent .
```

### Chạy backend với YouTube URL

```bash
docker run --rm --env-file .env -v ./output:/app/output highlight-agent scripts.run_backend "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Chạy extract highlights

```bash
docker run --rm --env-file .env -v ./output:/app/output highlight-agent scripts.extract_highlights "output/VIDEO_ID/transcript.json" --output "output/VIDEO_ID/candidates.json"
```

### Chạy agent đầy đủ

```bash
docker run --rm --env-file .env -v ./output:/app/output highlight-agent scripts.run_agent "https://www.youtube.com/watch?v=VIDEO_ID" --domain lecture
```

> **Lưu ý:** Flag `--env-file .env` truyền API key vào container.
> Flag `-v ./output:/app/output` mount thư mục output ra máy host để lấy kết quả.

## Sprint 2: Audio và interaction features

Node `Analyze` luôn trích xuất RMS Energy, pitch và silence từ audio 16 kHz
mono bằng Librosa. Với `--domain podcast`, nó chạy thêm Pyannote speaker
diarization và tính số lần đổi speaker (`turn_count`).

Chạy riêng acoustic extractor trên audio có sẵn:

```bash
conda run -n video-highlight python -c '
from highlight_agent.features import extract_acoustic_features
print(extract_acoustic_features("output/VIDEO_ID/audio.wav"))
'
```

Chạy đầy đủ feature timeline 30 giây từ audio đã có, không tạo highlight:

```bash
conda run -n video-highlight python -m scripts.run_features \
  output/pbyQhbZJhwI/audio.wav \
  --domain podcast \
  --known-speaker-count 2
```

Lệnh ghi output đầy đủ vào `output/{video_id}/features/features.json`.

Tạo trang review trực quan từ kết quả đó:

```bash
conda run -n video-highlight python -m scripts.build_diarization_review \
  output/pbyQhbZJhwI/features/features.json
```

Trang `output/{video_id}/features/review.html` có video player, timeline speaker
và các ô 30 giây có thể bấm để vừa xem vừa nghe đối chiếu. Mở file bằng Chrome
hoặc Safari; file chỉ đọc output đã có, không chạy model lại.

Pyannote cần `HF_TOKEN` trong `.env` và tài khoản Hugging Face phải chấp nhận
điều khoản của model
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Thiết bị được chọn theo thứ tự CUDA, Apple MPS, rồi CPU. Vì vậy Podcast vẫn
chạy được trên CPU local; chỉ gọi là demo GPU khi `torch.cuda.is_available()`
trả về `True`.


## Git workflow

- `main` luôn ở trạng thái chạy được.
- Tạo branch ngắn theo task, ví dụ `feature/backend-ingestion`.
- Push branch, mở Pull Request vào `main`, review và merge.
- Không commit API key, video, audio, model weights hoặc `output/`.
