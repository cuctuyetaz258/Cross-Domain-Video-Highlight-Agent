# In-Domain LTR Fine-Tuning Pilot Report

## 1. Experiment status

| Field | Value |
| --- | --- |
| Status | Completed pilot on 2026-08-30; do not interpret as a final cross-domain result. |
| Objective | Measure whether fine-tuning the released TVSum LTR scorer improves ranking of manually annotated project videos. |
| Starting checkpoint | `data/models/ltr_scorer.pt` (`ltr-scorer-v1.1`) |
| Target model | Seven-channel `AdditiveAttentionScorer` LTR scorer |
| Unit of annotation | Non-overlapping 2-second intervals, importance score from 1 to 5 |
| Evaluation unit | Five-second windows with one-second hop; this differs from the annotation interval and is recorded explicitly below. |
| Source-control revision | Record `git rev-parse HEAD` immediately before each run. |

This is a small in-domain pilot. It must not be described as proof of generalization to all lecture, podcast, or standup videos. Results are conditional on the labelled videos, the selected split protocol, and the current feature extractors.

## 2. Research questions and hypotheses

1. Does a checkpoint fine-tuned from TVSum rank high-importance internal video regions better than the unmodified TVSum checkpoint?
2. Is any observed improvement consistent across held-out videos rather than driven by one long podcast?
3. Which channels and failure modes appear responsible for a gain or regression?

The primary comparison is **TVSum checkpoint (frozen baseline)** versus **TVSum-initialized fine-tuned checkpoint**. Heuristic domain-weight scoring is reported only as an additional operational baseline, not as a replacement for the learned scorer.

## 3. Dataset inventory and provenance

### 3.1 Raw annotation snapshot

Annotation location: `data/annotations/raw/`. The filename stem is the YouTube video ID and identifies the intended source as `https://www.youtube.com/watch?v=<video_id>`. A reproducible manifest must store that canonical URL as well as local relative paths; an ID alone does not guarantee that a video will remain accessible or unchanged.

| Item | Current count |
| --- | ---: |
| CSV files | 13 |
| Annotation rows | 7,242 |
| Rows with an `importance` value | 3,183 |
| Rows without an `importance` value | 4,059 |
| Fully scored videos | 6 |
| Fully scored podcast videos | 5 |
| Fully scored lecture videos | 1 |
| Fully scored standup videos | 0 |

The seven partially initialized files are excluded from all training, validation, and test metrics until every interval has an `importance` score. Their timestamps and transcript hints may be retained as source preparation metadata but are not labels.

`IHZwWFHWa-w` annotation template ends at 1236s while the downloaded source media and Whisper transcript end at 1233s. The generated training manifest uses the probed 1233s duration and excludes the final 3s annotation tail; this is recorded as a source-template discrepancy rather than padding media or labels.

### 3.2 Fully scored annotation files

| Video ID | Domain | Scored intervals |
| --- | --- | ---: |
| `-cRswJf8OnI` | podcast | 657 |
| `1bszFX_XcbU` | podcast | 432 |
| `DNQDqq4mWSY` | podcast | 345 |
| `IHZwWFHWa-w` | lecture | 618 |
| `u36A-YTxiOw` | podcast | 633 |
| `waLjtcUq5Mc` | podcast | 498 |

The repository validator reports timestamp continuity and non-empty labels for these six files. It warns that some videos have a high proportion of scores 4-5; this is a data-quality observation, not a reason to silently remove labels. The exact validator output must be saved with the run artifacts.

### 3.3 Media and access record (must be completed before feature extraction)

For each included video, record the following in a committed, portable manifest using project-relative paths only:

| Required field | Purpose |
| --- | --- |
| `video_id`, canonical YouTube URL, domain | Provenance and lookup |
| `video_path`, `audio_path`, `transcript_path` | Rebuild the seven-channel cache |
| duration, FPS, media SHA-256 | Detect changed, incomplete, or wrong media |
| transcript source/model/language | Reproduce `text_score` and boundary alignment |
| annotation CSV SHA-256 | Tie results to the exact labels |
| acquisition date and license/access note | Responsible reporting |

Do not commit downloaded media, API keys, cookies, generated caches, or model binaries unless the team explicitly agrees to version them through an appropriate artifact store.

## 4. Data handling and label construction

1. Validate every raw CSV with `python -m evaluation.validate_annotations data/annotations/raw`.
2. Include only the six completed files listed above; log exclusions and their reason.
3. Download or locate the matching source media from the canonical YouTube URL, then verify duration against the annotation end time.
4. Prepare audio and a Whisper transcript with word timestamps. The runtime backend currently forces Whisper to keep transcript timing consistent for boundary refinement.
5. Extract and min-max normalize each feature per video to `[0, 1]`, at 10 Hz. Channel order is fixed as `rms`, `pitch`, `silence`, `text_score`, `scene_change`, `gesture`, `turn_rate`.
6. Convert 2-second ordinal importance labels to the trainer's 5-second, one-second-hop labels by time-weighted mean over the five-second interval. A mean `>= 4` is positive, `<= 2` is negative, and score 3 is ignored for AP/F1 but retained for ordinal ranking pairs.
7. Train only pairs from the same video whose aggregated scores differ by at least 1.0. Cap each video at 10,000 deterministically sampled pairs (seed 42) so long podcasts do not dominate the loss; scale the pairwise margin by score gap / 4.
8. Build caches only after recording extractor versions and any non-fatal fallback (for example unavailable diarization).

Implementation note: the offline trainer includes an `in_domain_ordinal` adapter for the project CSV `importance` column. The adapter and its weighted window aggregation were exercised in this pilot.

## 5. Split, anti-leakage, and model-selection protocol

All windows from a video remain in exactly one fold. Randomly splitting individual windows is prohibited because adjacent windows share audio, transcript, and visual context.

With only six labelled videos, the preferred report protocol is leave-one-video-out cross-validation (LOVO):

- One video is the test video in each of six folds.
- Of the remaining five, choose one validation video deterministically for early stopping and use four for fine-tuning.
- Rotate validation assignment across a nested or predefined schedule and publish the exact fold table and seed.
- Report every fold, macro-average across videos, and dispersion (standard deviation or bootstrap confidence interval). Do not pool all windows as the only headline metric.

Because five of six labelled videos are podcasts, report lecture performance separately and state that no standup conclusion is possible. Never tune on a test video.

## 6. Model and fine-tuning configuration

### 6.1 Frozen baseline

Evaluate the released checkpoint without any update:

- Artifact: `ltr-scorer-v1.1`
- Schema: 1.1
- Channels: seven, in the order specified above
- TVSum validation AP stored in checkpoint: 0.840564
- Checkpoint selection epoch: 3

The TVSum AP is not comparable directly with in-domain metrics because data, labels, and split are different.

### 6.2 Fine-tuned model

Fine-tuning means loading the TVSum checkpoint weights into an architecture-compatible scorer, then continuing optimization using only the training videos of a fold. The training code must record:

| Parameter | Value used |
| --- | --- |
| Initialization checkpoint SHA-256 | `059038c7dd9113a48a3fc6c2e8167f7ee40ccfeaa48952a91c84cd614beb3596` |
| Hidden dimension | 32 unless changed |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Pairwise margin `gamma` | `1.0` |
| Temporal smoothness `lambda_smooth` | `0.01` |
| Batch size | `32` |
| Maximum epochs / patience | `20 / 5` |
| Random seed(s) | `42` |
| Device, PyTorch, Python, OS | CPU, PyTorch environment, Python 3.11.15, macOS 26.3 arm64 |

The trainer exposes `--init-checkpoint` and validates its checkpoint version, feature contract, seven-channel order, and architecture before loading weights. Optimizer and scheduler state are deliberately new for each fold; training from scratch must not be mislabeled as fine-tuning.

## 6.3 Reproduction commands

Run commands from the repository root. Generated media, caches, checkpoints, and JSON logs are local artifacts; commit the catalog/report/code but record artifact hashes in the run ledger.

```bash
# 1. Download once, use Chrome cookies locally, and force Whisper timestamps.
python -m scripts.run_in_domain_pilot prepare --cookies-browser chrome

# 2. Materialize the six deterministic 4/1/1 fold manifests and validate/cache media.
python -m scripts.run_in_domain_pilot write-folds
python -m scripts.run_in_domain_pilot cache

# 3. For every fold: frozen TVSum evaluation, fine-tune, and held-out evaluation.
python -m scripts.run_in_domain_pilot run-folds \
  --init-checkpoint data/models/ltr_scorer.pt

# 4. Create the non-evaluated operational model after CV. Its epoch count is
# the median selected fold epoch; never use its training score as test evidence.
python -m scripts.run_in_domain_pilot train-operational \
  --init-checkpoint data/models/ltr_scorer.pt

# 5. Smoke the full LTR-only render path with one podcast and one lecture.
python -m scripts.run_agent data/raw/in_domain_pilot/-cRswJf8OnI/source_video.mp4 \
  --domain podcast --min-speaker-count 1 --max-speaker-count 3 \
  --ltr-model-path data/models/in_domain_pilot/all_data_operational.pt \
  --llm-provider disabled --no-subtitles --output-dir output/in_domain_smoke_podcast
python -m scripts.run_agent data/raw/in_domain_pilot/IHZwWFHWa-w/source_video.mp4 \
  --domain lecture --ltr-model-path data/models/in_domain_pilot/all_data_operational.pt \
  --llm-provider disabled --no-subtitles --output-dir output/in_domain_smoke_lecture
```

Before step 3, verify that `data/models/ltr_scorer.pt` SHA-256 is `059038c7dd9113a48a3fc6c2e8167f7ee40ccfeaa48952a91c84cd614beb3596`.

## 7. Metrics and reporting rules

Primary metric: Average Precision (AP) over non-ignored five-second windows in held-out videos.

Secondary metrics:

- Kendall tau between continuous predicted score and ordinal annotation-derived score;
- window-level F1 and Hit@K as diagnostics only;
- per-video and per-domain metrics, macro average, and uncertainty across folds;
- ranking comparison of frozen baseline versus fine-tuned scorer for the same test windows;
- qualitative top-highlight review with timestamps and annotation scores.

State in every table that these are **window-level ranking diagnostics**, not official TVSum/SumMe shot-level summary F-scores. LLM reranking is excluded from the primary LTR training evaluation; if separately evaluated, report successful LLM reranks and safe LTR fallbacks as distinct conditions.

## 8. Run ledger

Create one row per fold and append only observed values.

| Run ID | Git SHA | Fold train/val/test IDs | Cache fingerprint | Init checkpoint SHA | Best epoch | Val AP | Test AP | Kendall tau | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-08-30-fold-01 | `bdfdc6f` | `DNQDqq4mWSY,IHZwWFHWa-w,u36A-YTxiOw,waLjtcUq5Mc` / `1bszFX_XcbU` / `-cRswJf8OnI` | canonical seven-channel cache | `059038c7...` | 3 | 0.7414 | 0.3861 | 0.0067 | Fine-tune AP lower than frozen for this test video. |
| 2026-08-30-fold-02 | `bdfdc6f` | `-cRswJf8OnI,IHZwWFHWa-w,u36A-YTxiOw,waLjtcUq5Mc` / `DNQDqq4mWSY` / `1bszFX_XcbU` | canonical seven-channel cache | `059038c7...` | 8 | 0.8465 | 0.7138 | 0.0241 | Fine-tune AP lower than frozen for this test video. |
| 2026-08-30-fold-03 | `bdfdc6f` | `-cRswJf8OnI,1bszFX_XcbU,u36A-YTxiOw,waLjtcUq5Mc` / `IHZwWFHWa-w` / `DNQDqq4mWSY` | canonical seven-channel cache | `059038c7...` | 1 | 0.5756 | 0.6073 | 0.0165 | Fine-tune AP higher than frozen. |
| 2026-08-30-fold-04 | `bdfdc6f` | `-cRswJf8OnI,1bszFX_XcbU,DNQDqq4mWSY,waLjtcUq5Mc` / `u36A-YTxiOw` / `IHZwWFHWa-w` | canonical seven-channel cache | `059038c7...` | 20 | 0.9998 | 0.6203 | 0.0190 | Only lecture test fold; do not generalize. |
| 2026-08-30-fold-05 | `bdfdc6f` | `-cRswJf8OnI,1bszFX_XcbU,DNQDqq4mWSY,IHZwWFHWa-w` / `waLjtcUq5Mc` / `u36A-YTxiOw` | canonical seven-channel cache | `059038c7...` | 20 | 0.5902 | 0.9991 | 0.1435 | Fine-tune AP higher than frozen. |
| 2026-08-30-fold-06 | `bdfdc6f` | `1bszFX_XcbU,DNQDqq4mWSY,IHZwWFHWa-w,u36A-YTxiOw` / `-cRswJf8OnI` / `waLjtcUq5Mc` | canonical seven-channel cache | `059038c7...` | 12 | 0.4816 | 0.6659 | 0.0856 | Fine-tune AP higher than frozen. |

Store linked artifacts outside Git when large: resolved manifest, validation report, cache metadata, training log, last and selected checkpoint, evaluation JSON/CSV, environment export, and command log. Record SHA-256 for each artifact in this document or an adjacent machine-readable ledger.

## 9. Results and findings

The pilot completed from saved artifacts. The aggregate artifact SHA-256 is `f0db73a7384ea9818ca5d6675daeb9b0a35845b14dd1d49196c1809d29592337`; the post-CV operational checkpoint SHA-256 is `8c32e2e5def0f5736c7abbb75fed8d2c7cee8f2f366fd8cf33be617eac6daa96`.

### 9.1 Quantitative results

| Condition | Test AP macro mean | Test AP dispersion | Kendall tau | F1 diagnostic | Hit@K | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TVSum frozen baseline | 0.6303 | 0.2096 SD | 0.0923 | 0.6453 | 0.2222 | Reference model has lower AP but higher rank correlation on this small pilot. |
| TVSum-initialized fine-tune | 0.6654 | 0.1986 SD | 0.0493 | 0.6529 | 0.6667 | AP improves by 0.0351 macro, but lower Kendall tau means the result is mixed. |

### 9.2 Qualitative findings

- Fine-tuning improves held-out AP in folds 03, 05, and 06 but regresses in folds 01, 02, and 04. The six-fold AP macro gain is therefore modest and not a basis for a broad quality claim.
- The lecture-only fold regresses from 0.7224 to 0.6203 AP, so this training set does not establish lecture improvement.
- The final operational model is trained for 10 epochs, the median selected fold epoch. It is an operational artifact, not an independently evaluated model.
- End-to-end smoke runs using Faster Whisper and LLM disabled produced three clips each for one podcast and one lecture under `output/in_domain_smoke_podcast/` and `output/in_domain_smoke_lecture/`.

## 10. Limitations and next actions

- The present completed data is small and domain-imbalanced; it cannot support a strong cross-domain claim.
- Annotation scores are currently provided by one annotator per video. Inter-annotator agreement cannot yet be measured.
- YouTube media may change or become inaccessible; hashes, acquisition dates, and provenance are mandatory.
- The trainer's current binary pairwise objective requires an explicit, reviewed conversion from ordinal 1-5 labels.
- Complete the seven pending CSVs, especially standup, before a final model-selection run.
- Run the exact same protocol on Windows and macOS once the manifest uses only portable relative paths; record dependency differences and any extractor fallbacks.
