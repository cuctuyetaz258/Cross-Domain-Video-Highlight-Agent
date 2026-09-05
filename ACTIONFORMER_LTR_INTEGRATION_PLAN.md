# ActionFormer + LTR Integration Plan

## 1. Mục tiêu

Tích hợp ActionFormer vào pipeline hiện tại để học biên highlight có độ dài biến thiên, thay cho cách lấy peak LTR rồi tạo clip cố định 30 giây.

Kiến trúc đích:

```text
7-channel temporal features
        -> ActionFormer temporal encoder
        -> highlight proposal + boundary regression
        -> proposal-level LTR
        -> duration filter + Soft-NMS
        -> sentence/silence boundary refinement
        -> optional LLM reranking
        -> rendered highlights
```

ActionFormer chịu trách nhiệm trả lời đoạn bắt đầu và kết thúc ở đâu. LTR chịu trách nhiệm xếp hạng các proposal hợp lệ. LLM reranker tiếp tục là bước tùy chọn sau LTR.

## 2. Phạm vi

### Trong phạm vi

- Tái sử dụng feature matrix 7 kênh, 10 Hz hiện có.
- Huấn luyện class-agnostic temporal localization với một lớp `highlight`.
- Sinh proposal dài 30–90 giây.
- Xếp hạng proposal bằng pairwise LTR trong cùng video.
- Soft-NMS và boundary refinement theo transcript/silence.
- 5-fold cross-validation không leakage theo video.
- Giữ pipeline LTR hiện tại làm baseline và fallback có chủ đích.

### Ngoài phạm vi phiên bản đầu

- Thay feature extractor 7 kênh bằng backbone video lớn.
- Query-dependent localization.
- Huấn luyện end-to-end từ frame RGB/audio waveform.
- Neural fusion mới giữa LTR và LLM.
- Tự động tải hoặc phân phối lại source video không có quyền sử dụng.

## 3. Hiện trạng và khoảng trống

- Runtime đang dùng feature matrix `(7, T)` tại 10 Hz.
- LTR hiện tại mean-pool cửa sổ 5 giây, hop 1 giây.
- Candidate hiện được tạo quanh peak với độ dài cố định 30 giây.
- Schema và boundary refinement đã chấp nhận clip 30–90 giây.
- Có 18 boundary JSON, 66 highlight, gồm 10 lecture và 8 podcast.
- Ground truth hiện có thời lượng khoảng 29,8–91,46 giây; cần chuẩn hóa về 30–90 giây theo quy tắc được version hóa.
- Chỉ 9/18 video có đủ media và feature cache tại thời điểm audit.
- Manifest 5-fold hiện tại chỉ chứa 10 video với `6 train / 2 val / 2 test`; chưa phù hợp bộ 18 video.
- Luồng URL/`yt-dlp` đã được kiểm tra E2E với URL công khai. Các video cần đăng nhập vẫn cần test riêng với cookie.

## 4. Nguyên tắc tích hợp

1. Không phá checkpoint LTR v1 hoặc luồng inference hiện tại.
2. Tách localization, proposal ranking và post-processing thành các module độc lập.
3. Mọi split phải theo `video_id`; không chia window hoặc proposal của cùng video sang nhiều split.
4. Validation/test proposal phải được sinh bởi checkpoint không train trên video đó.
5. Không mặc định ActionFormer là tốt hơn; chỉ chuyển default sau khi vượt baseline theo protocol đã khóa.
6. Mọi heuristic như duration clamp, TopKMean và sentence snapping phải có ablation.

## 5. Kiến trúc kỹ thuật

### 5.1 Input adapter

Input chuẩn:

```text
feature_matrix: float32 [B, 7, T]
padding_mask:   bool    [B, T]
sample_rate:    10 Hz
```

Adapter đề xuất:

- `Conv1d(7, d_model)` để projection.
- Temporal downsampling stride 5, đưa 10 Hz xuống 2 Hz.
- Giữ padding mask sau downsampling.
- Khởi đầu với `d_model=128`; chỉ tăng lên 256 nếu validation cho thấy thiếu capacity.

Độ phân giải 0,5 giây đủ cho proposal 30–90 giây và giảm chi phí attention trên video dài.

### 5.2 ActionFormer backbone

- Local self-attention thay vì global attention.
- Temporal feature pyramid 4–5 mức, stride ×2.
- Một classification head cho xác suất foreground `highlight`.
- Một regression head dự đoán khoảng cách trái/phải tới boundary.
- Center sampling và regression range riêng cho từng pyramid level.

Output nội bộ:

```text
level_logits:  list[Tensor[B, T_l, 1]]
level_offsets: list[Tensor[B, T_l, 2]]
level_masks:   list[Tensor[B, T_l]]
level_feats:   list[Tensor[B, T_l, D]]
```

### 5.3 Proposal decoder

Mỗi temporal point giải mã thành:

```text
start = center - left_offset
end   = center + right_offset
```

Decoder thực hiện:

1. Loại padding và score dưới threshold.
2. Clip proposal vào phạm vi video.
3. Loại proposal ngoài 30–90 giây.
4. Pre-NMS top-K theo highlight confidence.
5. Chuyển proposal sang representation cho LTR.

### 5.4 Proposal-level LTR

Không sửa `AdditiveAttentionScorer`; tạo `ProposalLTRHead` mới.

Representation proposal gồm:

- attention-pooled feature bên trong proposal;
- pooled left context;
- pooled right context;
- ActionFormer confidence/logit;
- normalized duration;
- pyramid level embedding, nếu ablation chứng minh có ích.

LTR head trả một scalar score cho mỗi proposal. Pair chỉ được tạo trong cùng video.

Utility từ importance annotation 2 giây:

```text
utility(P) = eta * mean_importance(P)
           + (1 - eta) * top_k_mean_importance(P)
```

Chỉ tạo pair khi chênh lệch utility lớn hơn hoặc bằng `pair_delta`, tránh học từ các cặp gần hòa.

### 5.5 Loss

```text
total_loss = focal_loss
           + lambda_reg * diou_loss
           + lambda_rank * pairwise_ranking_loss
           + lambda_smooth * masked_smoothness_loss
```

- Focal loss: foreground/background imbalance.
- DIoU: boundary regression.
- Pairwise hinge/ranking loss: proposal priority.
- Smoothness: chỉ áp dụng ngoài vùng boundary mask để không làm mờ chuyển tiếp thật.

### 5.6 Post-processing

Thứ tự cố định:

1. Duration filter.
2. LTR ranking.
3. Soft-NMS theo temporal IoU.
4. Chọn candidate pool.
5. Sentence/silence boundary refinement.
6. Optional LLM reranking.
7. Chọn top-K và render.

Boundary refinement không được làm clip ngắn hơn 30 giây, dài hơn 90 giây hoặc vượt khỏi video.

## 6. Thay đổi code dự kiến

### File mới

```text
configs/actionformer_ltr.yaml
highlight_agent/models/actionformer/__init__.py
highlight_agent/models/actionformer/backbone.py
highlight_agent/models/actionformer/neck.py
highlight_agent/models/actionformer/heads.py
highlight_agent/models/actionformer/assignment.py
highlight_agent/models/actionformer/losses.py
highlight_agent/models/actionformer/decoder.py
highlight_agent/models/actionformer/model.py
highlight_agent/models/proposal_ltr.py
highlight_agent/features/proposal_pooling.py
highlight_agent/features/soft_nms.py
highlight_agent/models/train_actionformer_ltr.py
scripts/train_actionformer_ltr.py
evaluation/evaluate_actionformer_ltr.py
```

### File cần chỉnh sửa

```text
highlight_agent/agent/nodes.py
highlight_agent/agent/state.py
highlight_agent/features/ltr_contract.py
highlight_agent/schemas/highlight.py
scripts/prepare_custom_manifest.py
scripts/run_agent.py
README.md
TRAINING_PLAN.md
```

### Feature flag

Runtime hỗ trợ:

```text
--scorer-type legacy-ltr
--scorer-type actionformer-ltr
```

Trong giai đoạn rollout, default vẫn là `legacy-ltr`.

## 7. Data plan

### 7.1 Hoàn thiện artifact

Mỗi video cần đủ:

```text
source_video.mp4
audio.wav
transcript.json
feature_matrix.npy
feature metadata.json
importance CSV
boundary JSON
```

Thêm data-audit command kiểm tra ID, duration, coverage và schema giữa toàn bộ artifact.

### 7.2 Chuẩn hóa boundary

- Lưu `original_start/end` và `normalized_start/end`.
- Boundary ngắn hơn 30 giây: mở rộng quanh tâm trong phạm vi video.
- Boundary dài hơn 90 giây: cắt quanh vùng utility cao nhất, sau đó sentence-snap.
- Không sửa trực tiếp annotation gốc.
- Version hóa normalization policy trong manifest.

### 7.3 Cross-validation

Tạo stratified group 5-fold trên 18 video:

```text
test sizes: 4, 4, 4, 3, 3
validation: fold kế tiếp
training: ba fold còn lại
```

Mục tiêu:

- Mỗi video xuất hiện đúng một lần trong test.
- Lecture/podcast được phân bổ cân bằng nhất có thể.
- Split fingerprint được lưu trong checkpoint và report.

## 8. Training roadmap

### Stage A — Localization-only

- Train classification + boundary regression.
- Chọn checkpoint bằng validation mAP/Recall, không dùng ranking metric.
- Dùng ground-truth boundary đã chuẩn hóa.

Điều kiện hoàn thành: ActionFormer-only vượt fixed-30-second baseline về Recall@3 và mean tIoU.

### Stage B — LTR bằng GT và jitter proposal

- Freeze ActionFormer.
- Tạo proposal từ GT với translation/scale jitter.
- Train proposal LTR bằng pair trong cùng video.
- Ablate inside-only và contextual pooling.

### Stage C — LTR bằng predicted proposal

- Sinh out-of-fold proposal cho train/validation.
- Train LTR trên phân bố proposal thật.
- Không sinh validation/test proposal bằng model đã thấy video đó.

### Stage D — Joint fine-tuning

- Unfreeze phần cuối backbone.
- Learning rate backbone thấp hơn head 5–10 lần.
- Early stopping theo metric tổng hợp đã khóa.
- Bỏ stage này nếu validation không cải thiện ổn định.

### Stage E — Release checkpoint

- Khóa hyperparameter sau cross-validation.
- Train checkpoint release trên toàn bộ trainable in-domain data.
- Không dùng test fold để chọn epoch hoặc hyperparameter.

## 9. Evaluation plan

### Baseline và ablation

| Variant | Mục đích |
|---|---|
| Legacy LTR + fixed 30 s | Baseline production |
| ActionFormer-only | Đo localization |
| ActionFormer + LTR | Đo proposal ranking |
| ActionFormer + contextual LTR | Đo context pooling |
| Không smoothness | Đo regularization |
| Hard NMS và Soft-NMS | Đo overlap handling |
| Mean, TopKMean, mixed utility | Đo label aggregation |

### Metric

Localization:

- mAP tại tIoU 0.3, 0.5 và 0.7;
- Recall@1, Recall@3 và Recall@5;
- mean tIoU;
- start/end absolute boundary error.

Ranking:

- nDCG@3;
- Average Precision;
- Kendall tau;
- Spearman rho.

Product:

- duration-valid rate;
- overlap/diversity trong top-K;
- tỷ lệ sentence snapping thành công;
- latency, peak RAM/VRAM;
- tỷ lệ fallback/error.

Báo cáo mean/std theo fold, metric theo domain và bootstrap confidence interval.

## 10. Checkpoint contract

Checkpoint mới phải tách khỏi LTR v1:

```json
{
  "model_family": "actionformer_ltr",
  "checkpoint_version": "2.0",
  "feature_schema_version": "1.1",
  "input_sample_rate": 10,
  "model_sample_rate": 2,
  "channel_order": [],
  "duration_range": [30, 90],
  "architecture_config": {},
  "dataset_fingerprint": "...",
  "split_fingerprint": "...",
  "normalization_policy_version": "...",
  "selection_metric": "...",
  "selection_score": 0.0
}
```

Preflight phải fail-fast nếu checkpoint sai family, version, channel order, sample rate hoặc architecture.

## 11. Test plan

```text
tests/test_actionformer_shapes.py
tests/test_actionformer_masks.py
tests/test_actionformer_assignment.py
tests/test_actionformer_losses.py
tests/test_actionformer_decoder.py
tests/test_proposal_pooling.py
tests/test_proposal_ranking.py
tests/test_soft_nms.py
tests/test_actionformer_checkpoint.py
tests/test_actionformer_cv_protocol.py
tests/test_actionformer_integration.py
```

Edge cases bắt buộc:

- video dưới 30 giây;
- padding trong batch có độ dài khác nhau;
- proposal vượt đầu/cuối video;
- proposal đúng 30 hoặc 90 giây;
- không có proposal vượt threshold;
- nhiều proposal trùng nhau;
- score chứa NaN/Inf;
- boundary snapping làm vi phạm duration;
- checkpoint sai contract;
- URL YouTube có tracking/playlist;
- `yt-dlp` thiếu format, cookie bị khóa hoặc cần đăng nhập.

## 12. Milestone và tiêu chí nghiệm thu

### M0 — Data ready

- 18/18 video đủ artifact.
- Data audit pass.
- 5-fold split không leakage.

### M1 — Localization ready

- Unit test model/loss/decoder pass.
- ActionFormer-only vượt localization baseline.

### M2 — Ranking ready

- Proposal LTR train được bằng out-of-fold proposal.
- nDCG@3 hoặc AP cải thiện so với ActionFormer confidence-only.

### M3 — Runtime ready

- `actionformer-ltr` chạy E2E từ URL và local file.
- Soft-NMS, sentence snapping, LLM reranking và render hoạt động.
- Legacy LTR vẫn chạy không regression.

### M4 — Release ready

- 5-fold report hoàn chỉnh.
- Checkpoint contract/preflight pass.
- Latency và memory nằm trong ngân sách được chấp nhận.
- ActionFormer chỉ trở thành default nếu cải thiện có tính ổn định; nếu không giữ legacy LTR.

## 13. Thứ tự thực hiện đề xuất

1. Hoàn thiện 18/18 media, transcript và feature cache.
2. Sửa manifest và khóa normalization policy.
3. Implement input adapter, backbone, heads, assignment và loss.
4. Train/evaluate localization-only.
5. Implement proposal pooling và LTR.
6. Train bằng GT/jitter rồi predicted proposal.
7. Implement Soft-NMS và runtime feature flag.
8. Chạy E2E và regression tests.
9. Chạy 5-fold, ablation và tổng hợp report.
10. Đóng gói checkpoint release; quyết định có đổi default hay không.

## 14. Ước lượng

Không tính thời gian tải media và chạy training dài:

| Hạng mục | Ước lượng |
|---|---:|
| Data audit, normalization, split | 1–2 ngày |
| ActionFormer localization | 2–3 ngày |
| Proposal LTR và losses | 1–2 ngày |
| Runtime integration và checkpoint | 1–2 ngày |
| Test, 5-fold scripts và report | 2–3 ngày |

Tổng phần implementation dự kiến: 7–12 ngày làm việc. Dataset nhỏ là rủi ro chính; ưu tiên mô hình gọn, regularization và đánh giá out-of-fold thay vì tăng capacity sớm.
