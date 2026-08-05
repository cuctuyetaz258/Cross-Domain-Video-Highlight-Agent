# Data Contract v1.0

## Mục đích

Tài liệu này định nghĩa dữ liệu trao đổi giữa Backend, LLM baseline,
LangGraph, feature extraction và Web UI của dự án Cross-Domain Video
Highlight Agent.

Mục tiêu là để các bước phía sau không cần rẽ nhánh theo nguồn video và
không phụ thuộc trực tiếp vào implementation của nhau.

## Quy ước chung

- Thời gian được biểu diễn bằng số giây (`float`) tính từ đầu video.
- Đường dẫn trong JSON là đường dẫn tương đối tính từ repository root.
- Video trong Sprint 1 sử dụng tiếng Anh.
- `schema_version` của contract hiện tại là `1.0`.
- Các bước phải giữ nguyên `video_id` xuyên suốt pipeline.
- Backend ưu tiên caption từ YouTube; chỉ dùng `faster-whisper small.en`
  khi caption không tồn tại hoặc không sử dụng được.

## Cấu trúc workspace

Mỗi video được xử lý trong một workspace riêng:

```text
output/{video_id}/
├── source_video.mp4
├── audio.wav
├── transcript.json
├── metadata.json
├── shorts/
│   ├── highlight_01.mp4
│   └── highlight_02.mp4
└── thumbnails/
    ├── highlight_01.jpg
    └── highlight_02.jpg
```

Với YouTube URL, backend tải video về `source_video.mp4`. Với local file,
backend có thể dùng trực tiếp file gốc trong Sprint 1 để tránh sao chép file
lớn. `source_video_path` luôn phải trỏ đến file local mà `ffmpeg` đọc được.

## MediaWorkspace

`MediaWorkspace` là output thống nhất của bước chuẩn bị media cho cả URL
YouTube và local file.

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `schema_version` | `str` | Có | Phiên bản contract, hiện là `1.0` |
| `video_id` | `str` | Có | ID ổn định của video/workspace |
| `source_type` | `youtube \| local` | Có | Loại input ban đầu |
| `original_input` | `str` | Có | URL hoặc local path người dùng cung cấp |
| `source_video_path` | `str` | Có | Đường dẫn file video local đã chuẩn hóa |
| `audio_path` | `str` | Có | Đường dẫn WAV 16 kHz mono |
| `transcript_path` | `str` | Có | Đường dẫn `transcript.json` |
| `has_source_transcript` | `bool` | Có | Có dùng được caption từ nguồn hay không |

Ví dụ:

```json
{
  "schema_version": "1.0",
  "video_id": "jbL9kl4KPZI",
  "source_type": "youtube",
  "original_input": "https://www.youtube.com/watch?v=jbL9kl4KPZI",
  "source_video_path": "output/jbL9kl4KPZI/source_video.mp4",
  "audio_path": "output/jbL9kl4KPZI/audio.wav",
  "transcript_path": "output/jbL9kl4KPZI/transcript.json",
  "has_source_transcript": true
}
```

## TranscriptDocument

`transcript.json` là input trực tiếp cho LLM baseline, feature extraction và
bước canh biên highlight.

### Cấu trúc

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `schema_version` | `str` | Có | Phiên bản contract |
| `video_id` | `str` | Có | Phải trùng với workspace |
| `language` | `str` | Có | Mã ngôn ngữ, Sprint 1 là `en` |
| `source` | `youtube_caption \| whisper` | Có | Nguồn transcript thực tế |
| `duration` | `float` | Có | Thời lượng video/audio tính bằng giây |
| `segments` | `list[TranscriptSegment]` | Có | Các đoạn thoại theo thứ tự thời gian |
| `chapters` | `list[Chapter]` | Có | Chapter YouTube; dùng danh sách rỗng nếu không có |

### TranscriptSegment

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `id` | `int` | Có | Số thứ tự segment |
| `start` | `float` | Có | Thời điểm bắt đầu |
| `end` | `float` | Có | Thời điểm kết thúc |
| `text` | `str` | Có | Nội dung đã loại khoảng trắng thừa |
| `words` | `list[TranscriptWord]` | Có | Word timestamps; rỗng nếu nguồn không cung cấp |

### TranscriptWord

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `start` | `float` | Có | Thời điểm bắt đầu từ |
| `end` | `float` | Có | Thời điểm kết thúc từ |
| `text` | `str` | Có | Nội dung từ/token |

### Chapter

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `title` | `str` | Có | Tên chapter |
| `start` | `float` | Có | Thời điểm bắt đầu |
| `end` | `float` | Có | Thời điểm kết thúc |

Ví dụ:

```json
{
  "schema_version": "1.0",
  "video_id": "jbL9kl4KPZI",
  "language": "en",
  "source": "youtube_caption",
  "duration": 620.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Welcome to the lecture.",
      "words": [
        {
          "start": 0.0,
          "end": 0.5,
          "text": "Welcome"
        }
      ]
    }
  ],
  "chapters": [
    {
      "title": "Introduction",
      "start": 0.0,
      "end": 80.0
    }
  ]
}
```

## HighlightCandidate

`HighlightCandidate` là output từ LLM baseline hoặc scoring engine. Backend
dùng các mốc thời gian này để canh biên và render.

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `candidate_id` | `str` | Có | ID duy nhất trong video |
| `start_time` | `float` | Có | Thời điểm bắt đầu đề xuất |
| `end_time` | `float` | Có | Thời điểm kết thúc đề xuất |
| `score` | `float` | Có | Điểm xếp hạng ứng viên |
| `reason` | `str` | Có | Lý do ứng viên được chọn |
| `signals` | `dict[str, float]` | Có | Điểm theo tín hiệu; có thể chỉ chứa LLM ở Sprint 1 |

Ví dụ Sprint 1:

```json
{
  "candidate_id": "candidate_01",
  "start_time": 120.5,
  "end_time": 180.5,
  "score": 4.7,
  "reason": "The speaker gives a concise and memorable explanation.",
  "signals": {
    "llm_relevance": 4.7
  }
}
```

Ở Sprint 2, `signals` có thể được mở rộng mà không đổi schema:

```json
{
  "acoustic": 0.7,
  "paralinguistic": 0.3,
  "linguistic": 0.9,
  "structural": 0.6,
  "interaction": 0.0
}
```

## RenderedHighlight

`RenderedHighlight` là kết quả backend bàn giao cho UI và bước đóng gói
metadata.

| Field | Type | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `candidate_id` | `str` | Có | Liên kết về candidate ban đầu |
| `video_path` | `str` | Có | Clip MP4 dọc 9:16 |
| `thumbnail_path` | `str \| null` | Có | Thumbnail; có thể là `null` trước khi bước 9 chạy |
| `start_time` | `float` | Có | Boundary thực tế sau khi canh biên |
| `end_time` | `float` | Có | Boundary thực tế sau khi canh biên |
| `reason` | `str` | Có | Lý do chọn highlight |

Ví dụ:

```json
{
  "candidate_id": "candidate_01",
  "video_path": "output/jbL9kl4KPZI/shorts/highlight_01.mp4",
  "thumbnail_path": "output/jbL9kl4KPZI/thumbnails/highlight_01.jpg",
  "start_time": 120.5,
  "end_time": 180.5,
  "reason": "The speaker gives a concise and memorable explanation."
}
```

## Validation rules

- `video_id` không được rỗng và không chứa path separator.
- Mọi đường dẫn backend trả về phải là đường dẫn local hợp lệ.
- `start`, `end`, `start_time`, `end_time` không được âm.
- `end` phải lớn hơn `start`.
- Các segment và word phải được sắp xếp theo thời gian.
- Highlight MVP phải dài từ 30 đến 90 giây.
- Highlight không được vượt quá `duration` của video.
- Backend tạo thư mục output cần thiết nhưng không được âm thầm ghi đè input local.
- API key và thông tin bí mật không được xuất hiện trong bất kỳ JSON output nào.

## Ranh giới trách nhiệm

- Backend: tạo `MediaWorkspace`, `TranscriptDocument`, render clip,
  thumbnail và metadata.
- Prompt/LLM: nhận `TranscriptDocument`, trả `HighlightCandidate`.
- Model Lead/LangGraph: đưa các contract vào state và điều phối node.
- Feature extraction: bổ sung điểm vào `signals` từ Sprint 2.
- Web UI: đọc `RenderedHighlight` và reasoning để hiển thị/tải xuống.

Mọi thay đổi không tương thích với các field bắt buộc phải tăng phiên bản
contract và được các module tiêu thụ dữ liệu xác nhận.
