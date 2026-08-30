# In-Domain LTR 5-Fold Fine-Tuning Report

## 1. Trạng thái thí nghiệm

| Trường | Giá trị |
| --- | --- |
| Status | Hoàn thành trên macOS/CPU ngày 2026-08-30. Đây là in-domain pilot, không phải bằng chứng generalization. |
| Mục tiêu | Đo tác động của fine-tune có giám sát trên 10 video lecture/podcast đã gán nhãn, khởi tạo từ checkpoint TVSum + SumMe. |
| Baseline | Frozen `data/models/ltr_pretrained_tvsum_summe.pt`. |
| Fine-tuned model | Seven-channel LTR scorer, mỗi fold khởi tạo lại từ checkpoint baseline. |
| Scope | Chỉ LTR. LLM assessment/rerank/fusion không chạy và không được đưa vào metric này. |
| Source revision | `89d9e7f89f1aeb68eecef718d7fe268d6e720eb2`. |

Kết quả chỉ áp dụng cho 10 video, nhãn importance hiện có, feature extractor v1.1 và split cố định được mô tả dưới đây. Không được diễn giải là model đã tổng quát hóa sang video lecture, podcast hoặc domain khác nói chung.

## 2. Câu hỏi nghiên cứu

1. Fine-tune từ checkpoint TVSum + SumMe có cải thiện khả năng đưa cửa sổ importance cao lên trên các test video nội bộ hay không?
2. Mức cải thiện có xuất hiện ở cả lecture và podcast, hay bị chi phối bởi một vài video?
3. Kết quả binary retrieval (AP) có nhất quán với thứ tự ordinal toàn cục (Kendall tau) hay không?

So sánh chính là **frozen pretrained checkpoint** với **TVSum+SumMe-initialized fine-tuned checkpoint** trên cùng held-out video. TVSum/SumMe pretraining score không được so sánh trực tiếp với in-domain score vì tập dữ liệu và nhãn khác nhau.

## 3. Dữ liệu, cache và nhãn

| Mục | Giá trị |
| --- | --- |
| Videos | 10: 5 lecture, 5 podcast |
| Nguồn nhãn | CSV importance ordinal 1--5 theo interval 2 giây |
| Feature cache | `data/features_cache/<video_id>.npz` |
| Cache build report | `data/reports/custom_cache_build_v1_1.json` (5 rebuilt, 5 valid cache reused, 0 failed; 900.165 s) |
| Contract | schema `1.1`, extractor `1.1`, `float32`, 7 x T, 10 Hz, min-max per video |
| Kênh | `rms`, `pitch`, `silence`, `text_score`, `scene_change`, `gesture`, `turn_rate` |

Nhãn 2 giây được quy đổi time-weighted sang cửa sổ 5 giây, hop 1 giây. Mean importance `>= 4` là positive, `<= 2` là negative, và xấp xỉ score 3 là ignored cho AP/F1. Kendall tau và Spearman dùng continuous window score. Feature matrix được mean-pool trong từng cửa sổ; model không nhận chuỗi 50 timestep trực tiếp.

## 4. Split và chống leakage

Năm manifest `data/manifests/custom_fold{0..4}.jsonl` giữ toàn bộ cửa sổ của một video trong đúng một split. Mỗi fold gồm 6 train, 2 validation và 2 test video (mỗi domain có một validation và một test video). Mỗi video xuất hiện đúng một lần ở test qua năm fold.

Validation chỉ được dùng để chọn epoch của fine-tuned checkpoint. Test split chỉ chạy sau khi checkpoint của fold đã được chọn. Headline metric là macro theo **video** (10 test-video entries), không phải AP tính trên toàn bộ cửa sổ gộp lại.

## 5. Mô hình và cấu hình

`AdditiveAttentionScorer` là MLP nhỏ: `Linear(7, 32) -> Tanh -> Linear(32, 1)`. Dù tên class có chữ “attention”, phiên bản này không thực hiện temporal attention trong một cửa sổ; nó score vector 7 chiều sau mean pooling.

| Hyperparameter | Giá trị |
| --- | --- |
| Init checkpoint | `ltr_pretrained_tvsum_summe.pt` |
| Init SHA-256 | `c5243ab35347a5790f6a288e172e114189e065986884477b2db875693966902c` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Pairwise margin gamma | `1.0` |
| Temporal smoothness lambda | `0.01` |
| Batch size | `32` |
| Max epochs / patience | `50 / 15` |
| Seed | `42` |
| Device | CPU |

Mỗi epoch tạo pair positive/negative trong cùng video, cân bằng theo source `custom_scores=1.0`; tối đa 2,048 pairs/video/epoch. Chi tiết raw được lưu trong `data/reports/custom_fold*_log.json`, history CSV và SVG curves (các artifact này bị Git ignore có chủ đích).

## 6. Run ledger

| Fold | Best epoch | Validation AP chọn checkpoint | Epoch đã chạy | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 0 | 2 | 0.770586 | 17 | `56d43aca1efc93f2ee82c7db2dbb6ffe37245678f7e3317b15b6afedbbf22255` |
| 1 | 48 | 0.934891 | 50 | `c62448ba43e46e2d62e1c4a521cabbf006e541fbf698991bb913feafeb4743f7` |
| 2 | 3 | 0.779584 | 18 | `4cd0b14bc8aae8d8bc054db41b354551bf474daf5c7d9dd4352776e43790c2fc` |
| 3 | 44 | 0.796511 | 50 | `99d176389efec52011d663b12b005fef527d1cab18ef1452ff6f099130bcecfc` |
| 4 | 40 | 0.608752 | 50 | `e33971c9ff23881cc75d4a512f250889f8ceb5349f965d13ca370176f956a9ee` |

Best epoch có độ phân tán lớn (2, 48, 3, 44, 40), vì vậy không nên kết luận một epoch tối ưu ổn định từ 10 video này.

## 7. Held-out test results

### 7.1 Macro theo 10 held-out video

| Metric | Frozen mean +/- SD | Fine-tuned mean +/- SD | Delta mean |
| --- | ---: | ---: | ---: |
| Average Precision | 0.6345 +/- 0.1959 | 0.7175 +/- 0.2017 | +0.0830 |
| Kendall tau | -0.0299 +/- 0.1334 | 0.0430 +/- 0.0857 | +0.0728 |
| Spearman rho | -0.0368 +/- 0.1704 | 0.0562 +/- 0.1112 | +0.0930 |
| Window F1 at positive count | 0.6315 +/- 0.2175 | 0.7012 +/- 0.2044 | +0.0697 |
| Positive Hit@5 | 0.3800 +/- 0.3938 | 0.5400 +/- 0.4812 | +0.1600 |

AP là metric chính: binary AP trên positive/negative window, bỏ ignored. F1 và Hit@5 là diagnostic ở mức window, **không** phải shot-level summary F-score của TVSum/SumMe.

### 7.2 AP theo domain và video

| Test video | Domain | Frozen AP | Fine-tuned AP | Delta |
| --- | --- | ---: | ---: | ---: |
| `IHZwWFHWa-w` | lecture | 0.5117 | 0.5267 | +0.0150 |
| `waLjtcUq5Mc` | podcast | 0.5065 | 0.5665 | +0.0599 |
| `DNQDqq4mWSY` | podcast | 0.6601 | 0.8932 | +0.2331 |
| `aircAruvnKk` | lecture | 0.7837 | 0.6884 | -0.0953 |
| `u36A-YTxiOw` | podcast | 0.9310 | 0.9766 | +0.0456 |
| `wjZofJX0v4M` | lecture | 0.7810 | 0.8386 | +0.0575 |
| `1bszFX_XcbU` | podcast | 0.7720 | 0.7745 | +0.0024 |
| `g2-_pnmhO4A` | lecture | 0.6897 | 0.7949 | +0.1052 |
| `-cRswJf8OnI` | podcast | 0.3150 | 0.3000 | -0.0150 |
| `WUvTyaaNkzM` | lecture | 0.3943 | 0.8160 | +0.4217 |
| Macro lecture (5 videos) | lecture | 0.6321 | 0.7329 | +0.1008 |
| Macro podcast (5 videos) | podcast | 0.6369 | 0.7021 | +0.0652 |

Fine-tuned AP tăng ở 8/10 held-out video. Đây là tín hiệu tích cực, nhưng không phải kiểm định ý nghĩa thống kê; video `WUvTyaaNkzM` đóng góp lớn cho mức tăng macro, và hai video giảm AP cần được review định tính.

## 8. Operational checkpoint

Sau cross-validation, tạo `data/manifests/custom_all_train.jsonl` với cả 10 video có `split: train`. Script tái lập manifest là `scripts/build_all_train_manifest.py`.

Epoch phát hành được khóa bằng median best epoch của 5 fold: `median(2, 48, 3, 44, 40) = 40`. Checkpoint vận hành train đúng 40 epochs, không có validation/test split:

| Artifact | Giá trị |
| --- | --- |
| Release checkpoint | `data/models/ltr_target_lecture_podcast.pt` |
| SHA-256 | `6355b8c39ad3038db5fde803b6cc8e90fb4044af8cbddfe7e567b463af76cb87` |
| Preflight | Valid trên CPU; full feature contract 1.1; epoch 40; `L_ref=68.0` |
| Training-set AP tại epoch 40 | 0.857773 |
| Best-train checkpoint lưu audit | `data/models/ltr_target_lecture_podcast_best_train.pt`, epoch 13, không dùng làm release |

Training AP của operational checkpoint không phải held-out metric và không được so sánh với bảng ở Mục 7. Bản release dùng tất cả video nên không còn test set độc lập cho chính nó; nó chỉ phục vụ demo/vận hành sau khi CV đã hoàn tất.

## 9. Kết luận và giới hạn

Fine-tune LTR-only từ checkpoint TVSum + SumMe cải thiện macro held-out AP từ 0.6345 lên 0.7175 (+0.0830) và correlation macro từ âm nhẹ lên dương nhẹ. Tuy nhiên, độ lệch chuẩn cao, best epoch không ổn định và vẫn có 2/10 video giảm AP. Kết quả này là bằng chứng pilot có hướng tích cực, chưa đủ để khẳng định generalization hoặc superiority thống kê.

Các giới hạn chính:

- Chỉ 10 video, một split schedule cố định và chưa có inter-annotator agreement.
- Feature được min-max theo từng video; score giữa video không phải thang tuyệt đối.
- Không đánh giá LLM rerank/fusion ở đây; không suy diễn kết quả LTR sang pipeline fusion.
- Auto diarization/YouTube ingest có phụ thuộc môi trường và network, dù cache hiện tại đã được build/validate.
- Cần review failure cases của `aircAruvnKk` và `-cRswJf8OnI`, rồi chạy ablation theo nhóm acoustic/text/visual/interaction trước khi đưa claim mạnh hơn.

## 10. Tái lập

Từ repository root, sau khi có media, transcript và cache v1.1:

```bash
# Build all-data operational manifest.
python scripts/build_all_train_manifest.py \
  --input data/manifests/custom_fold0.jsonl \
  --output data/manifests/custom_all_train.jsonl

# Train one CV fold; đổi số fold để chạy đủ 0..4.
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold0.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold0.pt \
  --max-epochs 50 --patience 15 --lr 1e-4 --seed 42

# Evaluate only after fold checkpoint được chọn bằng validation.
python -m evaluation.evaluate_ltr \
  --manifest data/manifests/custom_fold0.jsonl \
  --cache-dir data/features_cache \
  --checkpoint data/models/custom_fold0.pt \
  --split test --device cpu --top-k 5 \
  --output-json data/reports/custom_fold0_test.json
```

Generated media, caches, models and raw reports are ignored by Git. Commit manifest generator, all-data manifest and report; publish checkpoint separately through the team artifact store when approved.
