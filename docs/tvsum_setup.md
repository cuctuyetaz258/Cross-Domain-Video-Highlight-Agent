# Chuẩn bị TVSum cho LTR

Tài liệu này hướng dẫn tái lập smoke dataset TVSum dùng cho LTR Highlight Engine
Raw media, annotation, transcript, feature cache, report và checkpoint đều là artifact local, không commit lên Git

## Yêu cầu

- Conda environment `video-highlight`
- `ffmpeg` và `ffprobe` có trong `PATH`
- TVSum raw videos và annotation chính thức `ydata-tvsum50.mat`
- Dependencies trong `requirements.txt`, bao gồm `h5py` để đọc annotation MATLAB v7.3

## Cấu trúc dữ liệu local

Đặt dữ liệu đã tải bên ngoài Git theo cấu trúc sau

```text
data/raw/tvsum/
├── video/
│   ├── 0tmA_C6XwfM.mp4
│   └── ...
└── matlab/
    └── ydata-tvsum50.mat
```

Tên file video phải có stem trùng `video_id` trong annotation, ví dụ `0tmA_C6XwfM.mp4`
Không dùng riêng benchmark feature file `.h5`, vì pipeline cần raw video để trích xuất đủ bảy channels

## Tạo smoke manifest và derived media

Lệnh dưới đây chọn 20 video theo category với seed cố định, tạo audio 16 kHz mono và transcript Whisper `small.en`, sau đó chia theo video thành 16 train và 4 validation

```bash
conda run -n video-highlight python scripts/prepare_tvsum.py \
  --annotations data/raw/tvsum/matlab/ydata-tvsum50.mat \
  --video-dir data/raw/tvsum/video \
  --processed-dir data/raw/tvsum/processed \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --limit 20 \
  --train-count 16 \
  --val-count 4 \
  --seed 42 \
  --whisper-model-size small.en
```

Video không có speech tiếng Anh đủ dùng sẽ bị bỏ qua và script chọn video khác để vẫn đủ số lượng yêu cầu
Chỉ khi chuẩn bị đủ 20 video thì script mới ghi manifest

## Validate và build feature cache

```bash
conda run -n video-highlight python scripts/validate_training_data.py \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --project-root . \
  --minimum-videos 20

conda run -n video-highlight python scripts/build_feature_cache.py \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --output-dir data/features_cache \
  --project-root . \
  --report data/reports/tvsum_cache_build.json
```

Warning về chỉ có domain `benchmark` là bình thường với TVSum-only smoke run
TVSum giữ nguyên category gốc như `BK`, `BT`, `DS` hoặc `VT`, không map sang `lecture`, `podcast` hay `standup`
`turn_rate` có thể bằng zero vì TVSum smoke pipeline không chạy speaker diarization
Trên một số máy macOS, MediaPipe FaceMesh có thể không tạo được OpenGL context và metadata cache sẽ ghi `gesture_status: facemesh_initialization_failed` thay vì hiểu nhầm là video không có khuôn mặt

Sau khi cập nhật code để bổ sung metadata gesture, có thể refresh observation mà không cần build lại toàn bộ feature matrix

```bash
conda run -n video-highlight python scripts/build_feature_cache.py \
  --manifest data/manifests/tvsum_smoke.jsonl \
  --output-dir data/features_cache \
  --project-root . \
  --refresh-gesture-observation
```

## Phạm vi Git

Commit script, code, test và tài liệu này
Không commit các thư mục `data/raw/`, `data/features_cache/`, `data/models/`, `data/reports/`, file media hoặc archive dataset
