# Training plan: LTR pretraining, target fine-tuning, and LTR–LLM fusion

## 1. Mục tiêu

Kế hoạch này xây dựng checkpoint phục vụ hai domain sản phẩm chính:

- `lecture`;
- `podcast`.

Quá trình training được tách thành ba bài toán độc lập:

1. Pretrain LTR trên TVSum và SumMe để học ranking highlight tổng quát.
2. Fine-tune LTR trên dữ liệu `lecture` và `podcast` do dự án tự gán nhãn.
3. Học lớp fusion kết hợp `ltr_score` và một `llm_score` duy nhất.

LLM không được fine-tune trong phạm vi kế hoạch này. Ta chỉ cache đánh giá của
OpenAI và học cách kết hợp score đó với LTR.

## Trạng thái triển khai

Đã hoàn thiện trong code:

- loader SumMe với `domain=benchmark`;
- split TVSum 40/5/5 và SumMe 15/5/5 theo video;
- ghép/validate manifest và kiểm tra leakage;
- adapter CSV importance 1–5 và generator 5-fold custom manifest;
- source/video-balanced pair sampler và giới hạn pair mỗi video;
- khởi tạo fine-tuning từ pretrained LTR checkpoint;
- `overall_quality` duy nhất cho LLM ranking;
- percentile/rank normalization và grid search global `alpha` bằng
  `macro_nDCG@3`;
- fusion dataset builder và checkpoint-bound fusion calibrator;
- UI không còn slider weight hoặc năm component score.

Trạng thái chạy thực tế ngày 2026-08-30:

- TVSum 40/5/5 và SumMe 15/5/5 đã được chuẩn bị, validate và build đủ 75/75
  cache theo feature contract 7-channel, extractor version 1.1;
- pretrained checkpoint đã hoàn tất bằng CUDA, best epoch 1 với
  `macro_source_average_precision=0.628726`; test full-LTR AP là `0.693276`;
- lịch sử training CSV, biểu đồ SVG, best/last checkpoint và test ablation report
  đã được lưu trong `data/models/` và `data/reports/`;
- 10/10 CSV mục tiêu (5 lecture, 5 podcast) đã điền đủ importance và vượt qua
  annotation validator; 10/10 boundary JSON cũng đã có;
- custom media mới đủ 6/10 video. Còn thiếu `source_video.mp4`, `audio.wav` và
  `transcript.json` cho bốn lecture: `aircAruvnKk`, `g2-_pnmhO4A`,
  `wjZofJX0v4M`, `WUvTyaaNkzM`;
- sáu custom cache cũ đang ở extractor version 1.0 và phải được build lại bằng
  extractor 1.1 sau khi đủ media, trước khi tạo five-fold manifest/fine-tune;
- OpenAI validation runs và release fusion calibrator vẫn chờ custom folds.

## 2. Kiến trúc score mục tiêu

### 2.1 LTR score

LTR tiếp tục dùng bảy feature channel chuẩn:

1. `rms`;
2. `pitch`;
3. `silence`;
4. `text_score`;
5. `scene_change`;
6. `gesture`;
7. `turn_rate`.

Checkpoint LTR chịu trách nhiệm chấm tín hiệu vật lý, cảm xúc, chuyển cảnh,
tương tác và semantic feature cục bộ.

### 2.2 LLM score tạm thời

Hiện tại `semantic_quality` là tổng có trọng số của:

```text
0.30 × semantic_relevance
+ 0.20 × standalone_value
+ 0.25 × completeness
+ 0.10 × hook_strength
+ 0.15 × shareability
```

Trong phiên bản training tiếp theo, năm trọng số trên chưa được dùng để quyết
định ranking. LLM sẽ cung cấp một trường duy nhất:

```text
overall_quality ∈ [0, 1]
```

và:

```text
llm_score = overall_quality
```

Các trường chi tiết chỉ được giữ trong assessment cache phục vụ phân tích lỗi
và ablation về sau. Chúng không tham gia trực tiếp vào công thức score và không
được hiển thị trên UI sản phẩm để tránh tạo cảm giác đây là các trọng số đã
được huấn luyện hoặc có thể cấu hình.

UI sản phẩm chỉ hiển thị:

- `ltr_score` đã normalize;
- `llm_score` (`overall_quality`);
- `final_score`;
- tên/fingerprint của fusion calibrator đang dùng.

`alpha` được đọc từ fusion artifact, không cho người dùng chỉnh bằng slider
trong chế độ chạy chuẩn. Component score và trọng số chi tiết chỉ có thể xuất
hiện trong report nghiên cứu hoặc developer diagnostics.

`risk_flags` và `suggested_start_time`/`suggested_end_time` vẫn là metadata kiểm
tra rủi ro và boundary, không được gộp vào `llm_score`.

### 2.3 Fusion score

Phiên bản đầu chỉ học một tham số độc lập `alpha`:

```text
final_score = alpha × calibrated_ltr_score
            + (1 - alpha) × calibrated_llm_score
```

Trong đó:

- `alpha = 1`: LTR-only;
- `alpha = 0`: LLM-only;
- `0 < alpha < 1`: kết hợp hai nguồn score.

Không cần hai tham số tự do `w_ltr` và `w_llm` nếu bắt buộc tổng trọng số bằng
1. Một tham số `alpha` tránh nghiệm tương đương do thay đổi scale.

## 3. Dataset và nguyên tắc chia dữ liệu

### 3.1 Vai trò của từng nguồn

| Dataset | Vai trò | Domain trong manifest |
|---|---|---|
| TVSum | Pretraining và benchmark tổng quát | `benchmark` |
| SumMe | Pretraining và benchmark tổng quát | `benchmark` |
| Custom lecture | Fine-tuning và đánh giá sản phẩm | `lecture` |
| Custom podcast | Fine-tuning và đánh giá sản phẩm | `podcast` |

SumMe không được gán thành `standup`. QVHighlights chưa được dùng trong vòng
training này vì annotation phụ thuộc query, trong khi pipeline inference hiện
không nhận query.

### 3.2 Chống data leakage

Split phải được thực hiện theo `video_id`, không theo dòng CSV, frame hoặc
sliding window. Toàn bộ artifact của một video phải ở cùng một split:

```text
video
├── media/audio/transcript
├── annotation 2 giây
├── feature cache
├── sliding windows
└── positive-negative pairs
```

Manifest phải khóa `seed`, danh sách video và fingerprint dataset. Không được
đổi test split sau khi đã xem kết quả.

### 3.3 TVSum và SumMe

Split khởi đầu đề xuất:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| TVSum | 40 | 5 | 5 |
| SumMe | 15 | 5 | 5 |

Việc chia cần được stratify theo category nếu metadata cho phép. Pretraining
sampler khởi đầu với tỷ lệ:

```text
TVSum: 60%
SumMe: 40%
```

Đây là xác suất chọn source khi tạo training batch, không phải tỷ lệ chia
train/validation/test và cũng chưa phải hyperparameter đã được chứng minh tối
ưu. TVSum có 50 video, SumMe có khoảng 25 video; sampling theo kích thước thô
sẽ xấp xỉ 67/33. Tỷ lệ 60/40 chủ động tăng nhẹ tần suất SumMe nhưng vẫn giữ
TVSum là nguồn chính. Cần so sánh tối thiểu `67/33`, `60/40` và `50/50` trên
macro validation metric; chỉ giữ `60/40` nếu kết quả validation xác nhận.

Sampling phải cân bằng theo dataset và video, không theo tổng số window. Số
positive-negative pair của mỗi video cần được giới hạn theo epoch để video dài
không chi phối loss.

### 3.4 Custom lecture và podcast

Trước khi fine-tune cần hoàn thành năm video lecture và năm video podcast. Với
mỗi domain có năm video, sử dụng 5-fold cross-validation theo video.

Mỗi fold:

| Split | Lecture | Podcast | Tổng |
|---|---:|---:|---:|
| Train | 3 | 3 | 6 |
| Validation | 1 | 1 | 2 |
| Test | 1 | 1 | 2 |

Qua năm fold, mỗi video được dùng làm test đúng một lần. Báo cáo mean và
standard deviation giữa các fold.

### 3.5 Chuyển custom CSV thành supervision

Adapter cần đọc importance score 1–5 mỗi hai giây và tạo canonical window
score. Baseline nhị phân:

```text
1–2 → negative
3   → ignored
4–5 → positive
```

Với pairwise ranking, ưu tiên giữ tính thứ bậc và chỉ tạo pair khi chênh lệch
importance tối thiểu 2. Pair chỉ được tạo giữa các window của cùng một video.

## 4. Phase 0 — Chuẩn bị dữ liệu

- [x] Hoàn thành label năm lecture và năm podcast.
- [x] Chạy validator cho 10/10 custom CSV mục tiêu.
- [x] Sửa loader SumMe dùng `domain="benchmark"`.
- [x] Xây adapter custom CSV importance 1–5.
- [x] Chuẩn hóa TVSum và SumMe về canonical manifest.
- [ ] Chuẩn hóa custom manifest sau khi bổ sung bốn lecture media còn thiếu.
- [x] Tạo benchmark split manifest theo `video_id` với seed cố định.
- [x] Kiểm tra benchmark media hoặc `video_id` không bị lặp giữa các split.
- [x] Build 75/75 benchmark feature cache bằng feature contract 7-channel v1.1.
- [ ] Rebuild 10/10 custom cache bằng extractor v1.1 sau khi đủ media.
- [x] Lưu benchmark report phân bố source, domain, score và positive/negative pair.

### Điều kiện hoàn thành

- Mọi record có `video_id`, `source`, `domain`, `split`, duration và annotation
  metadata hợp lệ.
- Mỗi train/validation split có cả positive và negative window.
- Feature cache có cùng sample rate, channel order và extractor version.

## 5. Phase 1 — Pretrain LTR trên TVSum và SumMe

- [x] Train `AdditiveAttentionScorer` bằng pairwise ranking loss.
- [x] Dùng source-balanced và video-balanced sampler.
- [x] Chọn epoch bằng macro validation AP của TVSum và SumMe.
- [ ] Báo cáo AP, nDCG, Kendall tau và Spearman theo từng dataset.
- [x] Chạy full 7-channel và zero-one-channel sensitivity.
- [x] Lưu best và last checkpoint.
- [x] Tự lưu lịch sử từng epoch thành CSV và biểu đồ SVG phục vụ báo cáo.

Artifact chính:

```text
data/models/ltr_pretrained_tvsum_summe.pt
data/reports/pretraining_history.csv
data/reports/pretraining_curves.svg
```

Checkpoint metadata tối thiểu:

- feature contract và channel order;
- dataset/split fingerprint;
- random seed;
- `L_ref`;
- epoch tốt nhất;
- selection metric;
- metric riêng TVSum/SumMe và macro metric.

## 6. Phase 2 — Fine-tune LTR trên lecture và podcast

- [ ] Khởi tạo từ `ltr_pretrained_tvsum_summe.pt`.
- [ ] Fine-tune từng fold bằng custom train split.
- [ ] Cân bằng lecture/podcast theo tỷ lệ 50/50.
- [ ] Chọn epoch bằng macro metric trên validation của fold.
- [ ] Chỉ chạy test một lần sau khi khóa cấu hình fold.
- [ ] Tổng hợp mean/std qua năm fold.
- [ ] Sau khi chốt hyperparameter, train release checkpoint trên toàn bộ custom
      trainable data theo protocol đã khóa.

Artifact dự kiến:

```text
data/models/ltr_target_lecture_podcast.pt
data/reports/custom_fold{0..4}_history.csv
data/reports/custom_fold{0..4}_curves.svg
```

Giữ cả JSON training log, CSV và SVG của từng fold. CSV là nguồn số liệu gốc để tổng hợp
mean/std hoặc vẽ lại; SVG dùng trực tiếp trong báo cáo và đánh dấu best epoch. Các file được ghi
lại sau mỗi epoch để vẫn bảo toàn phần training đã hoàn thành nếu early stopping xảy ra.

## 7. Phase 3 — Đơn giản hóa và cache LLM assessment

- [ ] Bổ sung `overall_quality` vào structured output của OpenAI.
- [ ] Dùng `overall_quality` làm `llm_score` duy nhất.
- [ ] Giữ năm tiêu chí chi tiết trong cache/report nghiên cứu, không hiển thị
      trên UI sản phẩm.
- [ ] Bỏ slider LTR/LLM weight khỏi chế độ chạy chuẩn; UI đọc `alpha` từ fusion
      calibrator.
- [ ] Version lại prompt và schema.
- [ ] Cache assessment theo candidate, transcript fingerprint, LTR checkpoint
      fingerprint, OpenAI model và prompt version.
- [ ] Không gọi lại OpenAI khi chỉ thay đổi phương pháp fusion.
- [ ] Nếu thiếu transcript hoặc OpenAI lỗi, ghi `llm_failure`; không tạo random
      score và không giả lập `llm_score=0`.

## 8. Phase 4 — Tạo fusion training dataset

Mỗi candidate cần một record tương tự:

```json
{
  "video_id": "example",
  "domain": "lecture",
  "candidate_id": "candidate_01",
  "ltr_score_raw": 5.72,
  "ltr_score_normalized": 0.81,
  "llm_score": 0.74,
  "target_importance": 4.3,
  "split": "train",
  "ltr_checkpoint_fingerprint": "...",
  "llm_model": "gpt-4o-mini",
  "prompt_version": "ltr-semantic-rerank-v2"
}
```

`target_importance` được tính từ các interval annotation hai giây giao với
candidate. Cần khóa trước quy tắc aggregation, ví dụ weighted mean theo thời
lượng overlap.

Candidate cho validation/test phải được sinh bởi LTR checkpoint không train
trên video đó. Điều này ngăn leakage từ candidate generation sang fusion model.

## 9. Phase 5 — Percentile/rank normalization và grid search alpha

### 9.1 Baseline cần giữ để so sánh

Giữ công thức hiện tại làm baseline:

```text
0.60 × ltr_score + 0.40 × llm_score
```

Ngoài fixed fusion, luôn báo cáo LTR-only và LLM-only.

### 9.2 Percentile/rank normalization

Phương án release được khóa là percentile/rank normalization. Với từng video,
xếp hạng các candidate theo từng nguồn score rồi ánh xạ rank về `[0, 1]`:

```text
rank_score = (N - rank) / (N - 1)
```

Candidate tốt nhất nhận 1, candidate cuối nhận 0. Nếu chỉ có một candidate,
score được đặt 0.5. Tie phải dùng average rank và quy tắc tie phải được khóa
trong metadata.

Normalization được thực hiện riêng cho LTR và LLM trong cùng candidate set:

```text
ltr_rank = percentile_rank(ltr_score)
llm_rank = percentile_rank(llm_score)
```

Cách này phù hợp hơn min-max khi hai model có scale và độ phân tán khác nhau.
Nó tối ưu thứ tự candidate, không diễn giải score như xác suất.

### 9.3 Grid search global alpha

Sau rank normalization, thử `alpha` từ 0 đến 1 với bước 0.05:

```text
final_score = alpha × ltr_rank + (1 - alpha) × llm_rank
```

Chọn `alpha` bằng macro-nDCG trên validation. Macro-AP được dùng làm metric phụ
và tie-breaker. Nếu nhiều `alpha` có kết quả tương đương trong sai số cho phép,
chọn giá trị lớn hơn để ưu tiên checkpoint LTR ổn định và giảm phụ thuộc API.

Đây là phương án chính của Phase 5 vì ít tham số, dễ giải thích và phù hợp dữ
liệu nhỏ.

### 9.4 Các phương án nghiên cứu sau release

Các phương án dưới đây không thuộc đường chạy release đầu tiên:

#### Trainable linear fusion

```text
final_score = sigmoid(b + w_ltr × ltr_score + w_llm × llm_score)
```

Thử logistic regression, pairwise logistic loss hoặc RankNet loss. Bắt buộc có
regularization và cross-validation.

#### Domain-specific alpha

```text
alpha_lecture
alpha_podcast
```

Chỉ thử sau khi có đủ dữ liệu hoàn chỉnh cho cả hai domain và phải so sánh với
global alpha. Không dùng nếu cải thiện không ổn định qua các fold.

#### Small fusion network

Một MLP nhỏ có thể dùng `ltr_score`, `llm_score`, duration, domain, transcript
coverage và risk count. Đây là phương án dài hạn; chưa dùng cho release đầu do
nguy cơ overfit và khó giải thích.

## 10. Phase 6 — Evaluation và ablation

Báo cáo tối thiểu các variant sau:

| Variant | Mục đích |
|---|---|
| LTR-only | Baseline vật lý/đa phương thức |
| LLM-only | Baseline ngữ nghĩa |
| Fixed 0.60/0.40 | So sánh với implementation hiện tại |
| Learned global alpha | Fusion ít tham số |
| Calibrated global alpha | Kiểm tra ảnh hưởng scale score |
| Trainable linear fusion | So sánh mô hình học hai trọng số |
| Domain-specific alpha | Chỉ dùng khi dữ liệu đủ |
| LLM failure | Xác nhận ranking giữ nguyên từ LTR |

Metric cần báo cáo:

- AP;
- nDCG;
- Kendall tau;
- Spearman correlation;
- positive Hit@K;
- metric riêng lecture/podcast;
- macro-average;
- mean/std qua cross-validation folds.

### 10.1 Định nghĩa metric riêng lecture và podcast

Metric không được tính bằng cách gộp tất cả window của một domain thành một
danh sách lớn, vì video dài sẽ chi phối kết quả. Quy trình đúng:

1. Tính metric riêng trên từng video test.
2. Lấy trung bình các video lecture để có `lecture_<metric>`.
3. Lấy trung bình các video podcast để có `podcast_<metric>`.
4. Tính macro domain metric bằng trung bình không trọng số của hai domain.

Ví dụ:

```text
lecture_nDCG = mean(nDCG của các lecture video)
podcast_nDCG = mean(nDCG của các podcast video)
macro_nDCG = (lecture_nDCG + podcast_nDCG) / 2
```

Các metric theo domain gồm:

- `AP`: khả năng xếp positive window cao hơn negative window;
- `nDCG@K`: chất lượng thứ tự top-K và hỗ trợ relevance nhiều mức 1–5;
- `Kendall tau`: tương quan thứ tự giữa score dự đoán và importance;
- `Spearman rho`: tương quan hạng giữa prediction và ground truth;
- `positive Hit@K`: top-K có chứa positive window hay không.

`nDCG@K` là selection metric chính cho fusion vì custom annotation có relevance
1–5. AP là metric phụ sau khi quy đổi thành positive/negative. Temporal IoU
không phải metric chính cho CSV hai giây; nó chỉ dùng khi có ground-truth clip
boundary 30–90 giây được chuẩn hóa riêng.

Không được fallback ngầm sang random. LLM failure phải được báo cáo riêng và
`ranking_source` phải vẫn là LTR.

## 11. Phase 7 — Đóng gói artifact

LTR checkpoint và fusion parameter phải là hai artifact độc lập:

```text
data/models/
├── ltr_target_lecture_podcast.pt
└── fusion_calibrator.json
```

Ví dụ `fusion_calibrator.json`:

```json
{
  "schema_version": "1.0",
  "method": "percentile_rank_global_alpha",
  "ltr_weight": 0.72,
  "llm_weight": 0.28,
  "ltr_checkpoint_fingerprint": "...",
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "prompt_version": "ltr-semantic-rerank-v2",
  "training_dataset_fingerprint": "...",
  "normalization": "per_video_average_percentile_rank",
  "selection_metric": "macro_ndcg",
  "selection_score": 0.81
}
```

Fusion calibrator phải được đánh giá hoặc train lại khi thay đổi:

- LTR checkpoint;
- OpenAI model;
- prompt version;
- định nghĩa `overall_quality`;
- phương pháp normalize/calibrate score;
- target dataset hoặc label protocol.

## 12. Thứ tự thực hiện

1. Hoàn thành và validate custom lecture labels.
2. Chuẩn hóa TVSum/SumMe/custom manifest và cache.
3. Pretrain LTR trên TVSum + SumMe.
4. Fine-tune LTR trên lecture + podcast bằng 5-fold cross-validation.
5. Thêm một `overall_quality` duy nhất cho LLM ranking.
6. Cache LTR score, LLM score và target importance theo candidate.
7. Percentile/rank-normalize LTR và LLM score theo candidate set.
8. Grid search global `alpha` bằng validation macro-nDCG.
9. Chỉ thử domain-specific hoặc neural fusion khi dữ liệu đủ.
10. Đóng gói checkpoint và fusion calibrator riêng biệt.

## 13. Tiêu chí nghiệm thu

- [ ] Dataset split không leakage theo video.
- [ ] TVSum và SumMe được báo cáo riêng, không che khuất domain target.
- [ ] Custom lecture và podcast được cân bằng khi fine-tune.
- [ ] LTR release checkpoint có metadata và fingerprint đầy đủ.
- [ ] LLM ranking chỉ dùng một `overall_quality` trong phiên bản đầu.
- [ ] UI sản phẩm không hiển thị năm component weight/score hoặc slider `alpha`.
- [ ] Learned fusion tốt hơn hoặc ít nhất ổn định hơn fixed 0.60/0.40 qua các
      fold; nếu không, giữ baseline đơn giản hơn.
- [ ] LLM failure không làm thay đổi ranking LTR và không sinh random output.
- [ ] Mọi report có metric theo domain, macro metric và mean/std.
- [ ] Thay OpenAI model/prompt hoặc LTR checkpoint làm invalid fusion cache theo
      đúng fingerprint.
