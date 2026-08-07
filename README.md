# Cross-Domain Video Highlight Agent

Hệ thống tự động trích xuất 3–5 highlight dọc 9:16 từ video bài giảng,
podcast và hài độc thoại, kèm transcript và lý do lựa chọn.

## Trạng thái

Repository đang trong giai đoạn chuẩn hóa nền tảng sau prototype Sprint 1.
Source cũ được giữ trong `week1/` để tham khảo; implementation mới nằm trong
`highlight_agent/`.

## Yêu cầu hệ thống

- Python 3.10–3.12; khuyến nghị Python 3.11 cho môi trường chung của nhóm.
- `ffmpeg`, `ffprobe` và Deno 2.3+ có trong `PATH`.
- Git.

Trên macOS, có thể cài ffmpeg bằng Homebrew:

```bash
brew install ffmpeg
brew install deno
```

## Khởi tạo môi trường

```bash
python3.11 -m venv .venv
source .venv/bin/activate
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

Nếu YouTube yêu cầu xác minh đăng nhập, đặt browser đang đăng nhập YouTube:

```text
YTDLP_COOKIES_BROWSER=chrome
```

## Cấu trúc chính

```text
highlight_agent/
├── agent/       # LangGraph state, nodes và graph
├── features/    # Năm tầng tín hiệu đa miền
├── llm/         # LLM clients và prompts
├── media/       # Input, audio, transcript, render, thumbnail
└── schemas/     # Pydantic data contracts dùng chung

frontend/        # Streamlit UI
scripts/         # Script hỗ trợ phát triển/demo
tests/           # Automated tests
docs/            # Đề cương và tài liệu kỹ thuật
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

## Git workflow

- `main` luôn ở trạng thái chạy được.
- Tạo branch ngắn theo task, ví dụ `feature/backend-ingestion`.
- Push branch, mở Pull Request vào `main`, review và merge.
- Không commit API key, video, audio, model weights hoặc `output/`.
