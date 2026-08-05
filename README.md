# Cross-Domain Video Highlight Agent

Hệ thống tự động trích xuất 3–5 highlight dọc 9:16 từ video bài giảng,
podcast và hài độc thoại, kèm transcript và lý do lựa chọn.

## Trạng thái

Repository đang trong giai đoạn chuẩn hóa nền tảng sau prototype Sprint 1.
Source cũ được giữ trong `week1/` để tham khảo; implementation mới nằm trong
`highlight_agent/`.

## Yêu cầu hệ thống

- Python 3.10–3.12; khuyến nghị Python 3.11 cho môi trường chung của nhóm.
- `ffmpeg` và `ffprobe` có trong `PATH`.
- Git.

Trên macOS, có thể cài ffmpeg bằng Homebrew:

```bash
brew install ffmpeg
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

Data contract phiên bản hiện tại: [docs/data_contract_v1.md](docs/data_contract_v1.md).

## Chạy kiểm tra

```bash
pytest
ruff check .
```

## Git workflow

- `main` luôn ở trạng thái chạy được.
- Tạo branch ngắn theo task, ví dụ `feature/backend-ingestion`.
- Push branch, mở Pull Request vào `main`, review và merge.
- Không commit API key, video, audio, model weights hoặc `output/`.
