# Cross-Domain Video Highlight Agent

Hệ thống tự động trích xuất 3–5 highlight dọc 9:16 từ video bài giảng,
podcast và hài độc thoại, kèm transcript và lý do lựa chọn.

## Trạng thái

Repository đang trong giai đoạn chuẩn hóa nền tảng sau prototype Sprint 1.
Source cũ được giữ trong `week1/` để tham khảo; implementation mới nằm trong
`highlight_agent/`.

## Yêu cầu hệ thống

- Python 3.10–3.12; khuyến nghị Python 3.11 cho môi trường chung của nhóm.
- Khuyến nghị `ffmpeg`, `ffprobe` có trong `PATH`; `imageio-ffmpeg` và PyAV là fallback.
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

### Cài checkpoint production

Sau khi cài dependency, chạy đúng một lệnh; không cần tài khoản hay Kaggle CLI:

```bash
python -m scripts.download_ltr_checkpoint
```

Script tải artifact `ltr-scorer-v1.1`, kiểm tra kích thước và SHA-256, chạy LTR preflight trên CPU,
sau đó mới atomic install vào `data/models/ltr_scorer.pt`. Nếu checkpoint hợp lệ đã tồn tại, lệnh
trả `already_installed` và không tải lại. Không dùng `--force` trừ khi chủ động muốn thay một file
local không khớp manifest.

## Biến môi trường

```bash
cp .env.example .env
```

Điền API key cần sử dụng vào `.env`. Không commit `.env` lên GitHub.

| Biến | Mô tả | Bắt buộc |
|---|---|---|
| `GROQ_API_KEY` | API key Groq khi dùng `--llm-provider groq` hoặc extractor cũ | Tùy chọn |
| `OPENAI_API_KEY` | API key khi dùng `--llm-provider openai` | Tùy chọn |
| `OPENAI_BASE_URL` | OpenAI-compatible URL thay thế cho provider OpenAI | Tùy chọn |
| `HIGHLIGHT_LLM_API_KEY` | API key khi dùng `--llm-provider custom` | Tùy chọn |
| `HIGHLIGHT_LLM_BASE_URL` | Base URL bắt buộc cho provider custom | Tùy chọn |
| `HIGHLIGHT_LLM_MODEL` | Model mặc định nếu không truyền `--llm-model` | Tùy chọn |
| `GOOGLE_API_KEY` | API key từ [aistudio.google.com](https://aistudio.google.com) | Tùy chọn |
| `OPENROUTER_API_KEY` | API key từ [openrouter.ai](https://openrouter.ai) | Tùy chọn |
| `HF_TOKEN` | HuggingFace token, bắt buộc cho Pyannote khi domain là `podcast` | Theo domain |
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

Pipeline production có sáu pha và bắt buộc checkpoint LTR hợp lệ:

```bash
python -m scripts.run_agent sample.mp4 \
  --domain lecture \
  --highlight-count 3
```

CLI mặc định đọc `data/models/ltr_scorer.pt`, nên không cần truyền `--ltr-model-path` sau khi đã
chạy downloader ở bước cài đặt.

Ép dùng Whisper và tắt subtitle khi cần test riêng:

```bash
python -m scripts.run_agent sample.mp4 \
  --domain lecture \
  --transcript-source whisper \
  --no-subtitles
```

`--transcript-source` nhận `auto`, `youtube` hoặc `whisper`. Chế độ `auto`
ưu tiên caption YouTube và fallback sang Whisper.

Luồng chấm điểm duy nhất là:

```text
SceneDetect + MediaPipe + audio + transcript + interaction
→ matrix 7 channel ở 10 Hz
→ checkpoint LTR
→ dense score + deterministic NMS
→ LLM rerank tùy chọn
→ boundary refinement + render
```

Graph preflight checkpoint trước khi download hoặc Whisper. Path trống/không tồn tại, schema sai,
feature lỗi, model lỗi hoặc không đủ candidate đều làm pipeline dừng với mã `LTR_*`; hệ thống không
đổi sang PixelDiff, RAFT, weighted fusion hoặc random baseline. Thành công dùng
`features.mode=ltr_required`.

### LTR + LLM semantic reranking (Model 1.1)

LLM là tầng tùy chọn chạy sau candidate generator. Agent chỉ gửi transcript cục bộ
`BEFORE/CORE/AFTER` của Top-M candidate, nhận structured assessment, kết hợp điểm theo công thức
bootstrap `0.60 * normalized_ltr + 0.40 * semantic_quality`, rồi mới chọn 3–5 clip để render.

Ví dụ OpenAI:

```bash
python -m scripts.run_agent sample.mp4 \
  --domain lecture \
  --ltr-model-path data/models/ltr_scorer.pt \
  --llm-provider openai \
  --llm-model gpt-4.1-mini \
  --llm-top-m 10
```

Ví dụ Groq:

```bash
python -m scripts.run_agent sample.mp4 \
  --domain podcast \
  --ltr-model-path data/models/ltr_scorer.pt \
  --llm-provider groq \
  --llm-top-m 10
```

Không truyền API key qua tham số dòng lệnh. Provider đọc key từ environment. Nếu thiếu key,
timeout hoặc response sai schema, agent giữ nguyên thứ tự LTR và ghi lý do vào `llm_run.fallback_reason`.
Assessment được cache trong `output/{video_id}/llm/` theo hash của context, provider, model, prompt
version và fingerprint checkpoint. Cache không lưu các block `BEFORE/CORE/AFTER` hay toàn bộ raw
transcript, nhưng có lưu
`evidence` ngắn do LLM trích hoặc diễn giải. `features.mode=ltr_llm_rerank` chỉ xuất hiện khi LLM
thực sự được áp dụng.

OpenAI dùng strict JSON Schema. Groq và endpoint custom dùng JSON mode tương thích rồi được Pydantic
validate. Mọi boundary do LLM đề xuất phải khớp timestamp transcript thật, nằm trong video, dài
30–90 giây và không lệch candidate quá 15 giây; nếu sai, hệ thống giữ boundary LTR rồi dùng bộ canh
biên xác định hiện tại.

### Checkpoint chia sẻ trên Kaggle

Checkpoint production được phát hành tại:

- <https://www.kaggle.com/datasets/cuctuyetaz258/cross-validation-checkpoint>

Người dùng thông thường chỉ cần:

```bash
python -m scripts.download_ltr_checkpoint
```

Không cần tải ZIP thủ công, giải nén, đổi tên hay nhập checkpoint path trên CLI. Danh tính artifact
được khóa trong `artifacts/manifests/ltr_scorer_v1_1.json`:

- artifact `ltr-scorer-v1.1`, feature schema `1.1`;
- best epoch `3`, validation AP `0.840564`;
- SHA-256 `059038c7dd9113a48a3fc6c2e8167f7ee40ccfeaa48952a91c84cd614beb3596`;
- 7 channel theo đúng thứ tự `rms`, `pitch`, `silence`, `text_score`, `scene_change`, `gesture`,
  `turn_rate`.

Nếu downloader báo HTTP `401`, `403` hoặc `404`, maintainer cần kiểm tra Kaggle dataset đã được
publish với visibility **Public**. Nếu file local sai nhưng thực sự muốn thay thế, chạy:

```bash
python -m scripts.download_ltr_checkpoint --force
```

Pipeline chuẩn bị TVSum và lệnh tạo smoke manifest/cache nằm trong `docs/tvsum_setup.md`.

### Train LTR offline

Trainer yêu cầu mỗi cache có hai file:

```text
data/features_cache/VIDEO_ID/
├── feature_matrix.npy   # float32, shape (7, T), sample rate 10 Hz
└── metadata.json        # schema_version, video_id, sample_rate, channel_order, shape, dtype
```

Chạy training với QVHighlights train/validation annotations:

```bash
python -m highlight_agent.models.train_offline \
  --qvhighlights data/raw/qvhighlights/highlight_train_release.jsonl \
  --val-qvhighlights data/raw/qvhighlights/highlight_val_release.jsonl \
  --cache-dir data/features_cache \
  --output data/models/ltr_scorer.pt \
  --training-log data/models/training_log.json \
  --seed 42
```

Checkpoint được chọn theo Average Precision trên validation windows và chứa feature schema,
`L_ref`, epoch, AP, dataset fingerprint và training config. `training_log.json` ghi riêng
margin loss, temporal smoothness loss và total loss theo epoch.

### Đánh giá Full LTR, channel ablation và LLM

Evaluator mới luôn tách bốn nhóm: Full LTR 7-channel, từng channel bị zero-out, LTR+LLM thành
công và LLM failure giữ ranking LTR. Không có profile-weight hoặc random fallback ngầm. Chạy trên
cùng manifest/cache và xuất JSON, CSV cùng bảng Markdown:

```bash
python -m evaluation.evaluate_ltr_variants \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --cache-dir data/features_cache \
  --checkpoint data/models/ltr_scorer.pt \
  --split val \
  --device auto
```

Để report thêm các lần chạy LLM, truyền mỗi artifact production bằng một flag riêng:

```bash
python -m evaluation.evaluate_ltr_variants \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --cache-dir data/features_cache \
  --checkpoint data/models/ltr_scorer.pt \
  --split val \
  --run-metadata output/VIDEO_LLM_OK/metadata.json \
  --run-metadata output/VIDEO_LLM_FAILED/metadata.json
```

`metadata.json` mới lưu `pipeline.llm_run`. Run có `applied=true` chỉ được tính vào
`ltr_llm_rerank`; provider lỗi với `mode=ltr_required` và `fallback_reason` được tính riêng vào
`ltr_llm_failure`. Nếu chưa cung cấp artifact, hai nhóm có trạng thái `not_run`, không được gán AP
hoặc sinh prediction ngẫu nhiên. Các variant `ltr_without_<channel>` là channel-sensitivity
diagnostic dùng checkpoint full; ablation chính thức vẫn cần retrain checkpoint riêng cho từng
channel subset.

Trên validation TVSum 4 video/915 windows, checkpoint schema 1.1 đạt AP `0.840564`, Kendall tau
`0.244332`, Spearman rho `0.393953`, window-F1 `0.716814`; NMS tạo đủ 5 candidate cho cả bốn video.
F1 trong report là diagnostic theo window, không phải shot-level F-score chính thức TVSum/SumMe.

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
  --min-speaker-count 1 \
  --max-speaker-count 3
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

Manifest media được commit bằng path tương đối dùng `/`, nên chạy được trên
macOS và Windows. Loader vẫn chấp nhận manifest cũ dùng `\\`. Pyannote nhận
waveform đã được Librosa decode thay vì tự mở file bằng TorchCodec, tránh lỗi
ABI TorchCodec/FFmpeg khác nhau giữa các máy.


## Git workflow

- `main` luôn ở trạng thái chạy được.
- Tạo branch ngắn theo task, ví dụ `feature/backend-ingestion`.
- Push branch, mở Pull Request vào `main`, review và merge.
- Không commit API key, video, audio, model weights hoặc `output/`.
