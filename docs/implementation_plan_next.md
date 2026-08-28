# Implementation Plan — Giai đoạn tiếp theo của Top-K Highlight LTR Engine

**Ngày lập:** 2026-08-28
**Baseline:** nhánh `feature/ltr-model`, commit `12b90d3`  
**Tài liệu trạng thái:** `docs/project_status_2026-08-28.md`

## 1. Mục tiêu giai đoạn

Chuyển LTR Engine từ trạng thái **đã có module + test mock/toy** sang trạng thái **chạy được với
checkpoint thật, có thể bật từ CLI/frontend, được đánh giá định lượng và tái lập được**.

Giai đoạn này không thay kiến trúc 7 channels, W=5s/Hop=1s, overlap-average hoặc distance-based NMS.
Các thay đổi kiến trúc như Mean+Std hay W=10s chỉ được xem xét sau khi có kết quả baseline.

### Điều kiện hoàn thành cấp cao

- Nhánh LTR chạy thành công trên cả CPU và CUDA, không rơi về fallback vì lỗi wiring/device.
- Có pipeline dữ liệu tái lập được: raw/metadata → feature cache `(7, T)` → train/validation.
- Có checkpoint thật, metadata và training log.
- `val_AP` và temporal smoothness loss được tính đúng, không còn placeholder.
- CLI và frontend có thể truyền `ltr_model_path` một cách rõ ràng.
- Có E2E test trên video thật và báo cáo so sánh LTR với baseline `PROFILE_WEIGHTS`.
- Pytest và Ruff pass trong môi trường được tài liệu hóa.

## 2. Baseline issues và trạng thái xử lý

| ID | Vấn đề | Mức độ | Trạng thái sau Phase 1–2 |
|---|---|---:|---|
| B-01 | `nodes.py` dùng `workspace.video_path`, nhưng schema chỉ có `source_video_path` | P0 | ✅ Đã sửa và có regression test |
| B-02 | `extract_windows()` có thể trả tensor CUDA, còn `LTRScorer.load()` trả model CPU | P0 | ✅ Model/window dùng chung device |
| B-03 | CLI/frontend chưa đưa `ltr_model_path` vào `AgentState` | P0 | ✅ CLI/frontend đã expose cấu hình |
| B-04 | Integration tests dùng `MagicMock` cho workspace và ép window tensor về CPU | P0 | ✅ Dùng `MediaWorkspace` thật và checkpoint toy |
| B-05 | `smooth_loss` luôn bằng 0 | P1 | ✅ Đã dùng temporal smoothness theo từng video |
| B-06 | `val_ap = -total_loss` | P1 | ✅ Đã dùng Average Precision thật |
| B-07 | Chưa có data/cache/checkpoint/log | P1 | 🟡 Có manifest smoke, validator và 1 cache thật; chưa có validation cache/checkpoint |
| B-08 | Downloader hiện tạo `.h5` benchmark, trong khi trainer đọc `.mat`/`.jsonl` và cache 7 channels | P1 | 🟡 Đã có manifest/cache builder; chưa tải bộ dữ liệu benchmark training |
| B-09 | Ruff chưa có trong env `MLIoT`; acoustic test bị skip vì `librosa` | P2 | 🟡 Còn mở cho Phase 6 |

## 3. Luồng đích

```text
Dataset manifest + raw media/annotations
        │
        ▼
scripts/build_feature_cache.py
        │  feature_matrix.npy: float32 (7, T), 10 Hz
        │  metadata.json: video_id, duration, domain, source, schema_version
        ▼
models/train_offline.py
        │  margin ranking + temporal smoothness
        │  validation Average Precision
        ▼
data/models/ltr_scorer.pt + training_log.json
        │
        ▼
CLI / Frontend → AgentState.ltr_model_path
        │
        ▼
analyze(): align → unfold → model → blend → NMS
        │
        ▼
decide() → boundary snapping → render → evaluation report
```

## 4. Kế hoạch triển khai theo phase

## Phase 1 — Sửa runtime và expose cấu hình LTR

**Trạng thái:** ✅ Hoàn thành trong working tree ngày 2026-08-28; test Phase 1 pass.  
**Ưu tiên:** P0  
**Ước lượng:** 1–2 ngày
**Chủ trì:** D; phối hợp B/C

### R1. Sửa media path và device placement

**Files:**

- `highlight_agent/agent/nodes.py`
- `highlight_agent/models/ltr_scorer.py` nếu cần mở rộng API load
- `tests/test_integration_ltr.py`

**Thực hiện:**

1. Thay `workspace.video_path` bằng `workspace.source_video_path` trong nhánh LTR.
2. Chọn device một lần tại đầu nhánh LTR.
3. Đưa window tensor và model về cùng device trước `forward()`.
4. Load checkpoint một lần, dùng cùng object/metadata thay vì đọc checkpoint lần thứ hai chỉ để lấy `L_ref`.
5. Validate checkpoint có `in_features=7`; lỗi checkpoint phải emit lý do rõ ràng trước khi fallback.
6. Giữ pipeline cũ nguyên vẹn khi `ltr_model_path=None` hoặc file không tồn tại.

**Tiêu chí nghiệm thu:**

- Test dùng `MediaWorkspace` thật phát hiện được tên field sai nếu tái xuất hiện.
- CPU inference pass.
- CUDA inference pass khi CUDA available; test được skip có lý do khi không có CUDA.
- `features.mode == "ltr_dense_overlap"` khi LTR thành công và phản ánh đúng trong progress event.

### R2. Expose LTR qua CLI và frontend

**Files:**

- `scripts/run_agent.py`
- `frontend/agent_runner.py`
- file UI đang khai báo lựa chọn visual/model trong `frontend/`
- `README.md`

**Thực hiện:**

1. Thêm `--ltr-model-path PATH` cho CLI và truyền vào state.
2. Mở lựa chọn `scene_mediapipe` trong `--visual-method`.
3. Cho frontend nhận model path từ cấu hình/session state; không hard-code checkpoint.
4. Hiển thị rõ mode cuối cùng là LTR hay fallback.
5. Validate path sớm, nhưng vẫn giữ fallback theo contract hiện tại.

**Tiêu chí nghiệm thu:**

- `python -m scripts.run_agent --help` hiển thị hai cấu hình mới.
- Unit test xác nhận CLI → state truyền đúng `ltr_model_path`.
- Frontend không lỗi khi model path trống.

### R3. Strengthen integration tests

1. Thay `MagicMock` workspace bằng `MediaWorkspace` thật trong test wiring.
2. Tạo checkpoint toy bằng `AdditiveAttentionScorer.save()` trong `tmp_path`.
3. Mock riêng extractor media nặng, nhưng chạy thật alignment → unfold → model → blend → NMS.
4. Thêm case video ngắn hơn 5 giây, checkpoint sai dimension, timeline không đủ K peaks và CUDA device.

**Gate Phase 1:** chỉ chuyển sang training/data khi LTR branch tạo candidate thật từ checkpoint toy và
không fallback trên CPU/CUDA.

## Phase 2 — Hoàn thiện training đúng thiết kế

**Trạng thái:** ✅ Hoàn thành trong working tree ngày 2026-08-28; test cache/loss/AP/checkpoint pass.  
**Ưu tiên:** P1  
**Ước lượng:** 2–3 ngày
**Chủ trì:** C; phối hợp B

### T1. Chốt cache contract

Canonical contract:

```text
feature_matrix.npy
  dtype: float32
  shape: (7, T)
  sample_rate: 10 Hz
  channel order:
    rms, pitch, silence, text_score, scene_change, gesture, turn_rate
```

Trainer phải reject file sai shape/dtype hoặc metadata không khớp thay vì tự đoán `(T, 7)` và bỏ qua im lặng.
Nếu cần hỗ trợ cache cũ `(T, 7)`, việc transpose phải được điều khiển bởi `schema_version`, không heuristic.

### T2. Refactor training examples

**Files:**

- `highlight_agent/models/train_offline.py`
- có thể thêm `highlight_agent/models/training_data.py` để tách loader/dataset khỏi training loop

Mỗi window example cần giữ tối thiểu:

- `video_id`, `domain`, `window_index`, `start`, `end`
- vector 7 chiều
- binary label và raw importance score

Pairwise samples dùng cho margin loss; ordered window sequences dùng riêng cho smoothness loss.

### T3. Implement loss thật

```text
L_total = L_margin + lambda_smooth * L_smooth
L_margin = mean(max(0, gamma - (s_pos - s_neg)))
L_smooth = mean((s[k+1] - s[k])^2) theo từng video, sorted by time
```

Không nối hai window thuộc hai video khác nhau khi tính smoothness. Log riêng `train_margin_loss`,
`train_smooth_loss` và `train_total_loss` theo epoch.

### T4. Implement validation Average Precision

1. Chạy scorer trên toàn bộ non-ignored validation windows.
2. Tính `average_precision_score(y_true, y_score)`.
3. Chọn best checkpoint theo `val_ap` cao nhất.
4. Nếu validation chỉ có một class, fail với thông báo dữ liệu không hợp lệ thay vì dùng `-loss`.
5. Lưu metadata: schema version, channel order, sample rate, W/H, hidden dim, gamma, lambda,
   dataset manifest hash, `L_ref`, best epoch và best val_AP.

### T5. Reproducibility và logging

- Thêm seed cho Python/NumPy/PyTorch và deterministic config phù hợp.
- Ghi `training_log.json` theo epoch.
- Lưu config train cùng checkpoint.
- Tạo output directory an toàn nếu chưa tồn tại.
- Resume chỉ được thêm sau khi smoke training ổn định; không nằm trên critical path.

### Test bắt buộc Phase 2

- Smoothness loss > 0 với sequence dao động và = 0 với sequence hằng.
- Không tính smoothness xuyên video.
- AP trên toy labels khớp giá trị kỳ vọng từ scikit-learn.
- Best checkpoint đúng epoch có AP cao nhất.
- Cache `(7, T)` hợp lệ; shape sai bị reject rõ ràng.
- Checkpoint round-trip giữ nguyên predictions và metadata.

**Gate Phase 2:** không còn chuỗi `mock val_ap` hoặc `smooth_loss = 0` trong training path.

## Phase 3 — Dữ liệu và feature cache

**Trạng thái:** 🟡 Đang thực hiện; đã dừng theo yêu cầu sau khi hoàn tất cache train 1/2 video smoke.  
**Ưu tiên:** P1
**Ước lượng:** 3–5 ngày, phụ thuộc tải dữ liệu
**Chủ trì:** A+C

### D1. Tách rõ training data và benchmark data

- `data/benchmark/*.h5`: dùng cho benchmark/evaluation hiện có.
- `data/raw/{tvsum,summe,qvhighlights}`: annotations và media phục vụ 7-channel extraction.
- `data/manifests/*.jsonl`: nguồn chuẩn ánh xạ `video_id`, media path, annotation path, domain, split.
- `data/features_cache/{video_id}/`: cache 7-channel.
- `data/models/`: checkpoint và log local, không commit binary lớn.

`scripts/download_benchmark.py` hiện chỉ giải quyết `.h5`; cần đổi tên/mô tả rõ hoặc mở rộng bằng các
download command riêng. Thêm dependency `gdown` nếu tiếp tục dùng script hiện tại.

### D2. Data manifest và validation

**Đã thực hiện:**

- Thêm `data/manifests/smoke_local.jsonl` gồm 2 video local, split theo video: 1 train và 1 val.
- Gắn nguồn nhãn là `custom_pseudo`/`pipeline_baseline_bootstrap_smoke_only`; không coi đây là human ground truth.
- Thêm `scripts/validate_training_data.py`: kiểm tra media/transcript/path/timeline/annotation,
  duplicate/leakage, phân bố split/domain/label và `L_ref`.
- Manifest smoke pass: `L_ref=60s`; train 148 positive/829 negative; val 180 positive/833 negative.
- Cảnh báo còn lại: cả hai video đều thuộc domain `lecture`, chưa đạt cross-domain gate.

Tạo script `scripts/validate_training_data.py`:

- kiểm tra file tồn tại và đọc được;
- kiểm tra duration, fps, annotation range và split;
- phát hiện duplicate video ID/data leakage;
- xuất thống kê domain, positive/negative windows và median highlight duration `L_ref`;
- fail nếu một split thiếu positive hoặc negative sample.

### D3. Batch feature cache

**Đã thực hiện một phần:**

- Thêm `scripts/build_feature_cache.py`, atomic write, strict reload validation, skip cache hợp lệ,
  `--force`, `--limit`, `--split`, `--device` và report theo stage.
- Tối ưu MediaPipe gesture từ random seek sang sequential decode và thêm regression test.
- Đã build cache thật cho `27hFBBkickA`: float32 `(7, 9925)`, 10 Hz, đủ 7 channel.
- Thời gian đo được: acoustic 125.371s, semantic 0.008s, scene 195.684s, gesture 181.246s;
  tổng 502.357s, cache 278,028 bytes.
- Cache video validation `ud99NN7Tntw` đang extract thì dừng theo yêu cầu; không có file cache dở dang.

**Lệnh tiếp tục (không dùng `--force` để tự skip cache train đã hợp lệ):**

```powershell
python scripts/build_feature_cache.py --manifest data/manifests/smoke_local.jsonl `
  --project-root . --output-dir data/features_cache `
  --report data/reports/smoke_cache_build.json
```

Tạo `scripts/build_feature_cache.py` sử dụng chính extractor production:

1. Đọc manifest.
2. Extract acoustic/transcript/interaction/scene/gesture.
3. Gọi `build_feature_matrix()`.
4. Atomic write `feature_matrix.npy` và `metadata.json`.
5. Skip cache hợp lệ; hỗ trợ `--force`, `--limit`, `--device` và log lỗi theo video.

**Smoke scope:** 2 video/domain nếu dữ liệu cho phép.
**Integration scope:** tối thiểu 20 video.
**Full scope:** TVSum + SumMe + QVHighlights subset đã chốt.

### D4. Quyết định QVHighlights subset

Chạy estimate trên 10–20 clip đầu để đo thời gian, GPU/CPU memory và dung lượng cache. Chọn subset
200–300 clip stratified nếu full extraction vượt ngân sách thời gian. Ghi selection seed và manifest
để tái lập; không chọn thủ công theo kết quả model.

**Gate Phase 3:** ít nhất 20 cache hợp lệ, validation script pass và trainer đọc được toàn bộ cache
không transpose/skip ngầm.

## Phase 4 — Smoke training và full training

**Trạng thái:** ⏸️ Chưa chạy training vì cache validation chưa hoàn tất. Trainer đã được mở rộng để đọc
manifest split, lưu best/last checkpoint, resolved config và evaluation snapshot.  
**Ưu tiên:** P1
**Ước lượng:** 1–2 ngày chạy + thời gian compute
**Chủ trì:** C

### TR1. Smoke training

- Dùng 10–20 video, tối đa 5–10 epoch.
- Xác minh loss hữu hạn, AP tính được, checkpoint load/inference được.
- Kiểm tra overfit một tập toy nhỏ; nếu không overfit được thì chưa chạy full training.

### TR2. Full/subset training

- Split theo video, không split theo window.
- Giữ validation độc lập; ghi phân bố domain.
- Early stopping theo val_AP, patience=15.
- Lưu best checkpoint, last checkpoint và log.

### TR3. Artifact đầu ra

```text
data/models/ltr_scorer.pt
data/models/training_log.json
data/models/training_config.json
data/models/evaluation_snapshot.json
```

Binary model/data vẫn nằm ngoài Git; commit manifest, config mẫu và tài liệu tái lập.

**Gate Phase 4:** checkpoint thật chạy được qua `scripts.run_agent` và metadata có `L_ref`, `val_ap`,
`epoch`, feature schema và training config.

**Lệnh dự kiến sau khi đủ cache train/val:**

```powershell
python -m highlight_agent.models.train_offline `
  --manifest data/manifests/smoke_local.jsonl `
  --cache-dir data/features_cache `
  --output data/models/ltr_scorer.pt `
  --last-output data/models/ltr_scorer_last.pt `
  --training-log data/models/training_log.json `
  --training-config data/models/training_config.json `
  --evaluation-snapshot data/models/evaluation_snapshot.json `
  --max-epochs 10 --patience 5
```

## Phase 5 — E2E và đánh giá định lượng

**Ưu tiên:** P1
**Ước lượng:** 2–3 ngày
**Chủ trì:** D; phối hợp toàn nhóm

### E1. E2E video thật

Chọn ít nhất một video đại diện cho lecture, podcast và standup. Với mỗi video, chạy:

- baseline `PROFILE_WEIGHTS`;
- LTR checkpoint trên CPU;
- LTR checkpoint trên CUDA nếu available;
- fallback với path rỗng và path không tồn tại.

Kiểm tra:

- 3–5 candidates và highlights;
- duration 30–90 giây sau boundary refinement;
- score hữu hạn, thứ hạng ổn định;
- khoảng cách peak đúng NMS contract;
- clips render được;
- mode/report phản ánh đúng LTR hay fallback.

### E2. Mở rộng evaluation

**Files:**

- `evaluation/evaluate_benchmarks.py`
- `evaluation/metrics.py`
- có thể thêm `evaluation/evaluate_ltr.py`

Thêm method `ltr` và `profile_weights`, tái sử dụng:

- Average Precision trên labeled windows;
- Kendall's τ/Spearman's ρ cho importance ranking;
- F1 theo protocol TVSum/SumMe;
- Hit@K/temporal IoU cho in-domain highlights;
- latency và peak memory theo video.

Xuất JSON/CSV machine-readable và bảng Markdown cho báo cáo.

### E3. Decision gate cho ablation

Chỉ thử Mean+Std hoặc W=10s/Hop=2s khi:

- baseline 7-channel W=5s đã có kết quả hợp lệ;
- underperformance được phân tách theo domain;
- cùng split, seed và metric được giữ nguyên.

**Gate Phase 5:** có bảng so sánh baseline/LTR, kết quả E2E ba domain và danh sách lỗi còn lại được phân loại.

## Phase 6 — Quality, tài liệu và release

**Ưu tiên:** P2
**Ước lượng:** 1 ngày
**Chủ trì:** D

1. Đồng bộ dependencies để env `MLIoT` có `pytest`, `ruff`, `librosa` và các visual dependencies.
2. Chạy toàn bộ pytest không còn skip do thiếu dependency.
3. Chạy Ruff và sửa lỗi trong phạm vi files đã thay đổi.
4. Cập nhật README với setup, cache build, train, inference và evaluation commands.
5. Xác nhận upstream branch; quyết định policy cho `uv.lock` trước khi commit.
6. Không commit raw dataset, video, cache hoặc `.pt`; chỉ commit manifest/config/docs cần tái lập.

## 5. Phân công đề xuất

| Thành viên | Workstream chính | Deliverable |
|---|---|---|
| A | Dữ liệu feature và cache | `build_feature_cache.py`, cache contract validation, real-video feature QA |
| B | Runtime scoring/NMS | device-safe inference, edge cases window/blending/NMS, CUDA regression tests |
| C | Training | loss thật, val_AP, dataset abstraction, checkpoint/log, smoke/full training |
| D | Integration/evaluation | CLI/frontend wiring, real E2E, benchmark report, README/quality gate |

### Điểm đồng bộ

| Gate | Người publish | Người nhận | Điều kiện |
|---|---|---|---|
| G1 — Runtime contract | B+D | C/A | CPU/CUDA checkpoint toy pass |
| G2 — Cache schema | A+C | B+D | `(7,T)` + metadata schema được test |
| G3 — Training contract | C | D | Checkpoint metadata và CLI load ổn định |
| G4 — First real checkpoint | C | B+D | Smoke checkpoint + training log |
| G5 — E2E report | D | Cả nhóm | Ba domain + baseline comparison |

## 6. Danh sách file dự kiến

| Loại | File | Thay đổi dự kiến |
|---|---|---|
| Update | `highlight_agent/agent/nodes.py` | source path, device, checkpoint metadata, mode/fallback |
| Update | `highlight_agent/models/ltr_scorer.py` | device-aware load hoặc checkpoint helper |
| Refactor | `highlight_agent/models/train_offline.py` | loss, AP, logging, deterministic training |
| Optional new | `highlight_agent/models/training_data.py` | cache/record/window dataset contract |
| Update | `scripts/run_agent.py` | `--ltr-model-path`, `scene_mediapipe` |
| Update | `frontend/agent_runner.py` và UI config | truyền model path/mode |
| New | `scripts/validate_training_data.py` | validate manifest/annotations/splits |
| New | `scripts/build_feature_cache.py` | batch extract `(7,T)` cache |
| Update | `scripts/download_benchmark.py` | làm rõ benchmark vs training data, dependency |
| New/Update | `evaluation/evaluate_ltr.py` hoặc `evaluate_benchmarks.py` | LTR/baseline metrics |
| Update | `tests/test_integration_ltr.py` | real schema + CPU/CUDA checkpoint path |
| Update/New | `tests/test_train_offline.py`, data/cache/evaluation tests | loss/AP/cache/report |
| Update | `requirements.txt`, `README.md` | reproducible environment và hướng dẫn |

## 7. Verification matrix

| Layer | Test | Pass condition |
|---|---|---|
| Unit | model/device/checkpoint | prediction round-trip, CPU/CUDA đồng nhất trong tolerance |
| Unit | training loss | margin và smoothness đúng trên dữ liệu tính tay |
| Unit | validation AP | khớp scikit-learn trên toy labels |
| Unit | cache schema | `(7,T)`, float32, 10 Hz, metadata hợp lệ |
| Integration | LTR branch | `MediaWorkspace` thật + toy checkpoint → LTR candidates |
| Integration | fallback | None/missing/corrupt checkpoint → baseline và emit lý do |
| Integration | CLI/frontend | config truyền đúng vào AgentState |
| E2E | real video | ba domain render thành công, K/duration/NMS hợp lệ |
| Evaluation | baseline comparison | JSON/CSV/Markdown tái lập được |
| Quality | pytest + Ruff | zero failures; zero dependency-related skips |

## 8. Rủi ro và phương án xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Không có raw media cho public datasets | Chỉ có annotations hoặc `.h5` features | Tách benchmark evaluation khỏi 7-channel training; dùng subset có media hợp lệ hoặc in-domain set |
| QVHighlights quá lớn | Cache time/storage vượt ngân sách | Estimate trước, chọn manifest subset stratified với seed cố định |
| Positive windows quá thưa | AP thấp, nhiều video không có positive pair | Báo thống kê trước train; sau baseline mới thử W=10/Hop=2 |
| NMS không trả đủ K | Timeline ít local peaks hoặc clip ở biên <30s | Test policy rõ: relax peak threshold có kiểm soát hoặc fallback bổ sung, không im lặng |
| Optional extractor fail | MediaPipe/Pyannote lỗi theo máy | Ghi channel availability trong metadata; log fallback và đo impact |
| Data leakage | Window cùng video xuất hiện ở train và val | Split theo video ID và validate duplicate trước cache/train |
| Checkpoint/config lệch nhau | Model load được nhưng feature order sai | Schema version + channel order bắt buộc trong metadata |

## 9. Thứ tự thực hiện ngắn gọn

```text
Phase 1: Runtime P0
    ↓
Phase 2: Training correctness ─┐
                               ├─→ Phase 4: Train
Phase 3: Data + cache ─────────┘
    ↓
Phase 5: E2E + evaluation
    ↓
Phase 6: Quality + docs + release
```

**Ước lượng tổng:** 10–16 ngày làm việc, không tính thời gian chờ tải dữ liệu và full feature extraction.
Phase 2 và Phase 3 có thể chạy song song sau khi Gate G1 hoàn tất.

## 10. Definition of Done

Giai đoạn được đánh dấu hoàn tất khi:

- [ ] B-01 đến B-09 đã đóng hoặc có waiver được ghi rõ.
- [ ] LTR path chạy thật trên CPU và CUDA.
- [ ] CLI/frontend bật được LTR và hiển thị đúng execution mode.
- [ ] Có ít nhất 20 cache thực để integration và bộ cache train/val đã chốt.
- [ ] Checkpoint thật có metadata đầy đủ và training log.
- [ ] Smoothness loss và val_AP không còn placeholder.
- [ ] E2E pass trên lecture, podcast và standup.
- [ ] Có báo cáo baseline vs LTR bằng AP/τ/F1/Hit@K phù hợp từng dataset.
- [ ] Toàn bộ pytest và Ruff pass, không còn skip do thiếu dependency.
- [ ] README đủ để thành viên khác tái lập cache → train → inference → evaluation.
