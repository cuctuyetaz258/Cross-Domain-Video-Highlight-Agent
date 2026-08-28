# Cập nhật tiến độ — Top-K Highlight LTR Engine

**Ngày đối chiếu:** 2026-08-28
**Repo:** `Cross-Domai1n-Video-Highlight-Agent`
**Nhánh kiểm tra:** `feature/ltr-model`
**HEAD code baseline:** `12b90d3` — `fix: normalize LTR peak scores and stabilize fallback tests`

Tài liệu này cập nhật trạng thái thực tế của codebase dựa trên `implementation_plan.md`,
`project_context.md`, `task_assignment.md`, lịch sử Git, source code và kết quả kiểm thử.
Các tài liệu ban đầu vẫn là baseline thiết kế; khi thông tin trạng thái mâu thuẫn, tài liệu này
được ưu tiên cho câu hỏi “đã làm gì” và “làm gì tiếp theo”.

## 1. Tóm tắt điều hành

Top-K Highlight LTR Engine đã hoàn thành **phần khung triển khai và kiểm thử tự động bằng dữ liệu
mock/toy** cho cả bốn nhóm A–D. Toàn bộ module mới trong kế hoạch đã có trong codebase, nhánh LTR
đã được gắn vào `analyze()` và pipeline cũ vẫn được giữ làm fallback.

Hệ thống **đã bắt đầu Phase 3 trên media thật**: có manifest smoke, validation report và một cache
7-channel hợp lệ. Tuy nhiên cache validation thứ hai bị dừng theo yêu cầu, Phase 4 chưa train nên
chưa có checkpoint/đánh giá mô hình. Vì vậy chưa nên đánh dấu toàn bộ LTR pipeline là production-ready.

### Trạng thái kiểm chứng

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Module mới theo kế hoạch | ✅ Đã tạo đủ | `visual_new.py`, `alignment.py`, `sliding_window.py`, `overlap_blender.py`, `nms_topk.py`, `ltr_scorer.py`, `train_offline.py` |
| Tích hợp state/node/export | ✅ Đã có code | `AgentState`, `features/__init__.py`, nhánh LTR trong `analyze()` |
| Dependencies visual mới | ✅ Đã thêm | `scenedetect>=0.6.4`, `mediapipe==0.10.14`, `protobuf>=4.25,<5` |
| Toàn bộ pytest | ✅ 116 passed | Chạy bằng Conda env `MLIoT` sau Phase 1–2 |
| Dependencies extraction | 🟡 Đã cài | `librosa`, `scenedetect`, `mediapipe`; import pass, nhưng protobuf 4.x xung đột constraint với một số package Google/Streamlit |
| CUDA cho `extract_windows` | ✅ Test pass | Test CUDA không nằm trong danh sách skip của lần chạy trên |
| Dataset thật và feature cache | 🟡 Một phần | Manifest 2 video lecture pass; cache `27hFBBkickA` có shape `(7, 9925)` |
| Checkpoint `ltr_scorer.pt` | ❌ Chưa có | Chưa train trên dữ liệu thật |
| E2E bằng video + checkpoint thật | ❌ Chưa xác minh | Integration hiện dùng mock/toy data |
| Đánh giá so với baseline | ❌ Chưa có | Chưa có Hit Rate@K, Kendall's τ, F1@K hoặc val_AP thật |
| Ruff | ⚠️ Chưa xác minh lại | `ruff` không có trong Conda env `MLIoT` hiện tại |

## 2. Những gì đã làm

### Thành viên A — Feature Extraction

- Đã triển khai `extract_scene_changes()` bằng PySceneDetect.
- Đã triển khai `extract_gesture_signal()` bằng MediaPipe Face Mesh, sampling 2 Hz và fallback về
  vector 0 khi không đọc được video/không phát hiện khuôn mặt.
- Đã triển khai `build_feature_matrix()` tạo ma trận 7 kênh trên lưới 10 Hz, gồm nội suy,
  scene Gaussian smoothing và normalization về `[0, 1]`.
- Đã thêm dependencies cần thiết và export public API.
- Đã có 6 unit tests trực tiếp cho `visual_new.py` và `alignment.py`.

**Chưa hoàn tất của A:** đã có batch cache builder và chạy thật trên một video lecture, nhưng chưa có
cache validation thứ hai và chưa có video đại diện podcast/standup.

### Thành viên B — LTR Inference

- Đã triển khai dense sliding window bằng `torch.unfold`, mặc định W=50 frame và hop=10 frame.
- Đã triển khai mean pooling thành vector 7 chiều cho từng cửa sổ.
- Đã triển khai overlap-average blending về timeline 10 Hz.
- Đã triển khai local peak detection và distance-based temporal NMS, sinh `HighlightCandidate` 30 giây.
- Đã có 10 unit tests trực tiếp cho sliding window, blender và NMS; test CUDA đã pass trong lần chạy hiện tại.

**Chưa hoàn tất của B:** chưa kiểm chứng chất lượng peak/NMS trên score timeline của video thật và
chưa đánh giá trường hợp không đủ K local peaks.

### Thành viên C — Model & Training

- Đã triển khai `AdditiveAttentionScorer` 7→32→1, Xavier initialization và save/load checkpoint.
- Đã triển khai loader cho TVSum, SumMe và QVHighlights.
- Đã có cache contract nghiêm ngặt `(7,T)`, window examples có provenance, pairwise dataset và
  training loop dùng margin loss cùng temporal smoothness theo từng video.
- Validation dùng Average Precision thật; checkpoint và `training_log.json` chứa schema, metric,
  fingerprint, config và loss theo epoch.

**Chưa hoàn tất của C:**

- Chưa tải dataset vào `data/raw/` và chưa quyết định/chốt QVHighlights subset thực tế.
- Đã build 1/2 cache smoke; cache validation bị dừng trước khi hoàn tất.
- Chưa chạy training subset hoặc full training; chưa có `ltr_scorer.pt`.
- Chưa chạy training trên cache/dataset thật nên chưa có checkpoint và learning curve thực nghiệm.

### Thành viên D — Integration & Tests

- Đã thêm `ltr_model_path` và `scene_mediapipe` vào `AgentState`.
- Đã export các hàm mới qua `features/__init__.py`.
- Đã gắn nhánh LTR vào `analyze()` khi checkpoint tồn tại; có fallback khi path rỗng, file không tồn tại
  hoặc nhánh LTR phát sinh exception.
- Đã sửa `source_video_path`, đồng bộ device CPU/CUDA và load checkpoint/metadata một lần.
- CLI/frontend đã nhận `ltr_model_path` và `scene_mediapipe`; execution mode được hiển thị rõ.
- Integration tests dùng `MediaWorkspace` thật, checkpoint toy và kiểm tra LTR/fallback wiring.
- Các thay đổi model/training/integration nằm trong commit `baf526e`; phần chuẩn hóa NMS và ổn định
  fallback tests nằm trong commit `12b90d3`.

**Chưa hoàn tất của D:**

- Integration test chưa chạy full agent với video và checkpoint thật.
- Chưa có E2E bằng video thật với checkpoint thật; frontend chưa được browser-test thủ công.

## 3. Đối chiếu thứ tự triển khai ban đầu

| # | Task trong kế hoạch | Trạng thái hiện tại |
|---:|---|---|
| 1 | `visual_new.py` | ✅ Đã implement + unit test mock |
| 2 | `alignment.py` | ✅ Đã implement + unit test |
| 3 | Chuẩn bị labels/data | 🟡 Manifest smoke 2 video pass; nhãn là pseudo-label, chỉ có lecture |
| 4 | `models/ltr_scorer.py` | ✅ Đã implement + unit test |
| 5 | `sliding_window.py` | ✅ Đã implement + CPU/CUDA test |
| 6 | `overlap_blender.py` | ✅ Đã implement + unit test |
| 7 | `nms_topk.py` | ✅ Đã implement + unit test |
| 8 | `models/train_offline.py` | ✅ Cache contract, smoothness, AP, metadata và logging đã implement/test |
| 9 | Chạy training | ⏸️ Chưa thực hiện; chờ cache validation thứ hai |
| 10 | Tích hợp `nodes.py` + `state.py` | ✅ Runtime/CLI/frontend wiring đã sửa; còn E2E thật ở Phase 5 |
| 11 | Unit/integration tests | 🟡 116 pass; thiếu real-data E2E, 1 acoustic test skip |

## 4. Việc sẽ thực hiện tiếp

Thứ tự ưu tiên đề xuất:

1. **Tiếp tục cache smoke:** chạy lại cache builder không có `--force`; cache train hợp lệ sẽ được skip
   và chỉ video validation `ud99NN7Tntw` được xử lý.
2. **Smoke train:** sau khi đủ hai cache, chạy tối đa 10 epoch để xuất best/last checkpoint, config,
   training log và evaluation snapshot; kết quả chỉ là kiểm chứng pipeline do dùng pseudo-label.
3. **Chuẩn bị dữ liệu chuẩn:** tải/chốt TVSum, SumMe và QVHighlights subset; bảo đảm tối thiểu 20 video
   và có lecture/podcast/standup trước khi gọi Phase 3 hoàn thành.
4. **Train theo hai giai đoạn:** smoke train 10–20 video, sau đó full/subset train; xuất
   `data/models/ltr_scorer.pt` và `training_log.json`.
5. **E2E thật:** chạy agent trên ít nhất một video của mỗi domain với checkpoint, xác minh K=3–5,
   duration 30–90 giây, khoảng cách NMS, render và fallback.
6. **Đánh giá định lượng:** so LTR với `PROFILE_WEIGHTS` bằng Hit Rate@K, Kendall's τ, F1@K và val_AP;
   chỉ sau đó mới quyết định Mean+Std hoặc W=10s/Hop=2s.
7. **Hoàn tất hygiene/release:** cài/chạy Ruff, xử lý acoustic dependency để không còn test skip,
   cập nhật README và đồng bộ Git remote.

## 5. Trạng thái Git cần lưu ý

- Nhánh hiện tại: `feature/ltr-model`, track `origin/feature/ltr-model`.
- Hai commit kỹ thuật chính: `baf526e` và `12b90d3`.
- Có file chưa track: `uv.lock`.
- Trước khi merge/push cần xác nhận upstream branch đúng và quyết định có commit `uv.lock` hay không.

## 6. Điều kiện để đánh dấu hoàn tất

LTR Engine chỉ nên chuyển sang **Done** khi đồng thời thỏa các điều kiện sau:

- Có dataset/cache/checkpoint thật và training log tái lập được.
- Smoothness loss và val_AP đúng với thiết kế, không còn placeholder.
- Full LTR path chạy trên CPU và CUDA mà không rơi về fallback ngoài ý muốn.
- CLI/backend có thể bật LTR bằng cấu hình công khai.
- E2E video thật của ba domain pass; pytest và Ruff pass, không còn skip do môi trường.
- Báo cáo định lượng chứng minh hoặc bác bỏ cải thiện so với baseline tĩnh.
