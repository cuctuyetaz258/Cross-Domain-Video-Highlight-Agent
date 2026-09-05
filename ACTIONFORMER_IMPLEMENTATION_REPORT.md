# ActionFormer-LTR Implementation Report

> Audit correction (2026-09-05): the recorded five-fold experiment uses a shared OOF cache with possible indirect outer-test leakage. Validation nDCG is computed on predicted + GT + grid candidates. Results below are preserved as exploratory artifacts, not evidence of clean generalization or of localization being the sole bottleneck. Nested OOF and predicted-only validation remain required. Historical statements that no commit/push was performed describe the original training session; repository publication is a separate step.

## Status

Implementation and partial-data training were completed locally on 2026-09-04. The legacy LTR scorer remains the default. No commit or push was performed.

All 18 annotated videos now have the required media and cached features. The model is not release-ready because fold-0 localization has not surpassed the baseline acceptance criterion.

## Implemented scope

- Seven-channel, 10 Hz input adapter with temporal downsampling to 2 Hz.
- Local-attention temporal pyramid, foreground head, and left/right boundary regression.
- Center-sampled assignment, focal classification loss, 1D DIoU loss, and masked smoothness loss.
- 30–90 second proposal decoder and temporal Soft-NMS.
- Contextual proposal pooling and within-video pairwise LTR.
- SetRank-style contextual proposal scorer with two IMSAB blocks, 16 learned inducing points per block, padding masks, and optional ActionFormer ordinal-rank embeddings.
- Versioned `proposal_utility_v2` target combining coverage mean, top-20-percent importance, and maximum temporal IoU.
- Utility-weighted RankNet as the default pairwise objective; unweighted RankNet, Delta-NDCG-weighted RankNet, and the legacy margin loss remain selectable ablations.
- Proposal-LTR training lists now include ActionFormer-predicted candidates in addition to ground-truth/grid candidates to reduce train-inference shift.
- Versioned ActionFormer-LTR checkpoint contract with fail-fast schema, channel order, sample-rate, duration, model-family, and version validation.
- Runtime `legacy-ltr` / `actionformer-ltr` feature flag in the agent CLI.
- Snapshot persistence for both scorer families.
- Dataset audit, non-destructive boundary normalization, and five video-disjoint manifests.
- Evaluation command for mAP at tIoU 0.3/0.5/0.7, Recall@1/3/5, mean IoU, boundary error, duration validity, and latency.
- Incremental JSON, CSV, and SVG training artifacts written after every epoch.

## Data audit

Audit file: `data/reports/actionformer_data_audit.json`

- Annotated videos: 18
- Ready videos: 18
- Highlights: 66
- Domains: 10 lecture, 8 podcast
- Normalization: 62 unchanged, 3 cropped to maximum duration, 1 expanded to minimum duration
- Fold test sizes across all annotated videos: 4, 4, 4, 3, 3
- Fold 0 ready subset: 10 train, 4 validation, 4 test

All 18 videos now have media, audio, source-caption transcripts, and strictly validated seven-channel feature caches.

### Cache completion run

- Downloaded and normalized 9/9 missing public YouTube videos without errors.
- Built and strictly validated 6/6 missing lecture caches: `4q1dgn_C0AU`, `8KkKuTCFvzI`, `bBC-nXj3Ng4`, `iCvmsMzlF7o`, `rrkrvAUbU9Y`, and `UF8uR6Z6KLc`.
- Built and strictly validated 3/3 missing podcast caches with Pyannote 3.1 diarization: `hp6n1qwo1Ws`, `Ks-_Mh1QhMc`, and `z6X5oEIg6Ak`.
- Media report: `data/reports/actionformer_media_prepare.json`.
- Lecture cache report: `data/reports/actionformer_missing_lecture_cache_build.json`.
- Podcast cache report: `data/reports/actionformer_missing_podcast_cache_build.json` (548.558 seconds, zero failures).
- Final audit: 18/18 ready, 66 highlights, zero issues.

## Recorded runs

### Pipeline smoke run

Two training videos and one validation video, one epoch per stage:

- Localization validation loss: 0.240913
- Localization Recall@3, tIoU 0.3: 0.0
- Proposal-LTR validation nDCG@3: 0.655692
- Logs: `data/reports/actionformer_smoke/`
- Checkpoints: `data/models/actionformer_smoke_*.pt`

This run verifies artifact persistence and checkpoint handoff only.

### Fold-0 partial-data run

Six training videos, two validation videos, and one test video using `d_model=32` on CUDA:

- Localization stopped after epoch 25 by patience; best epoch: 17
- Best validation loss: 0.931171
- Best validation Recall@3, tIoU 0.3: 0.0
- Proposal-LTR stopped after epoch 13 by patience; best epoch: 5
- Best proposal-LTR validation nDCG@3: 0.618714
- Test mAP@0.3: 0.066667
- Test mAP@0.5: 0.066667
- Test mAP@0.7: 0.0
- Test Recall@5, tIoU 0.3: 0.333333
- Test mean IoU: 0.121241
- Duration-valid rate: 1.0
- Evaluation latency: 0.473 seconds for one cached-feature video on CUDA

Artifacts:

- `data/reports/actionformer_fold0/localization_log.json`
- `data/reports/actionformer_fold0/localization_history.csv`
- `data/reports/actionformer_fold0/localization_curves.svg`
- `data/reports/actionformer_fold0/proposal_ltr_log.json`
- `data/reports/actionformer_fold0/proposal_ltr_history.csv`
- `data/reports/actionformer_fold0/proposal_ltr_curves.svg`
- `data/reports/actionformer_fold0/evaluation_test_log.json`
- `data/models/actionformer_fold0_localization.pt`
- `data/models/actionformer_fold0_ltr.pt`

The generated model/report directories are intentionally ignored by Git. They remain available in the local workspace for the final report.

### IMSAB proposal-LTR smoke run

One training video, one validation video, and one epoch on CPU using the target configuration (`d_model=128`, two IMSAB blocks, 16 inducing points, two attention heads, FFN 256, dropout 0.3, ordinal rank signal, utility-weighted RankNet):

- Training pairwise loss: 0.663230
- Validation pairwise loss: 0.662249
- Validation nDCG@3: 0.368283
- Full fold-0 test duration-valid rate: 1.0
- Test mAP@0.3 / 0.5 / 0.7: 0.072261 / 0.072261 / 0.006993
- Test Recall@1 / 3 / 5 at tIoU 0.3: 0.25 / 0.416667 / 0.416667
- Test mean IoU: 0.119144
- Evaluation latency: 0.1643 seconds per cached-feature video on CPU

Artifacts:

- `data/reports/actionformer_fold0/imsab_smoke_log.json`
- `data/reports/actionformer_fold0/imsab_smoke_history.csv`
- `data/reports/actionformer_fold0/imsab_smoke_curves.svg`
- `data/reports/actionformer_fold0/imsab_smoke_evaluation_test.json`
- `data/models/actionformer_fold0_imsab_smoke.pt`
- `data/models/actionformer_fold0_imsab_smoke_last.pt`

This run validates training, checkpoint reconstruction, logging, and held-out inference only. With one training video and one epoch, its metrics must not be used as evidence that the model improves on the baseline.

### Five-fold OOF IMSAB run

All five localization folds were trained with the same locked configuration. Their held-out test predictions were combined into `actionformer_oof_v1`, a leakage-checked cache covering all 18 videos with 6,537 proposals. Each IMSAB scorer was then trained from this shared OOF candidate cache using utility-weighted RankNet.

Cross-fold mean and population standard deviation:

- Localization validation Recall@3, tIoU 0.3: 0.2894 ± 0.1225
- Proposal-LTR validation nDCG@3: 0.7567 ± 0.0662
- Test mAP@0.3: 0.0343 ± 0.0292
- Test mAP@0.5: 0.0158 ± 0.0098
- Test mAP@0.7: 0.0050 ± 0.0065
- Test Recall@3, tIoU 0.3: 0.1692 ± 0.0796
- Test Recall@5, tIoU 0.3: 0.2125 ± 0.1281
- Test mean IoU: 0.0900 ± 0.0435
- CPU evaluation latency: 0.1917 ± 0.0215 seconds per cached-feature video

The scorer learns a useful validation ranking signal, but localization remains the bottleneck. End-to-end mAP and IoU do not pass the production gate, so IMSAB remains experimental. Full proposal lists caused CUDA OOM on the 6 GB RTX 4050; the recorded five-fold runs were completed on CPU without changing the locked model configuration.

Aggregate artifacts:

- `data/proposals/actionformer_oof_v1.json`
- `data/reports/actionformer_cv/oof_proposal_stats.csv`
- `data/reports/actionformer_cv/cv_summary.json`
- `data/reports/actionformer_cv/cv_fold_metrics.csv`
- `data/reports/actionformer_cv/cv_metrics.svg`
- `ACTIONFORMER_CV_REPORT.md`

Per-fold JSON logs, CSV histories, SVG training curves, best/last checkpoints, and test evaluation reports are stored under `data/reports/actionformer_cv/` and `data/models/`.

## Verification

- Full project suite: 241 tests passed with 3 dependency deprecation warnings.
- ActionFormer checkpoint/evaluator/runtime tests are included in that suite.
- IMSAB permutation-equivariance, padding invariance, inducing-point gradient, within-video pairing, and pairwise-loss direction tests are included in that suite.
- Ruff passed across `highlight_agent`, `scripts`, `evaluation`, and `tests`.
- YouTube URL ingestion regression tests remain in the suite, including canonicalization of watch, short, Shorts, Music, and privacy-enhanced URLs.

## Remaining release gates

1. Compare ActionFormer confidence, legacy MLP + margin, MLP + utility-weighted RankNet, and IMSAB on the exact same OOF candidate cache.
2. Add GT translation/scale jitter, contextual-pooling ablation, hard/Soft-NMS ablation, and utility aggregation ablation.
3. Add a deterministic proposal shortlist or gradient-accumulation policy before retrying IMSAB training on the 6 GB GPU.
4. Run full ActionFormer URL-to-render E2E only after a checkpoint meets localization acceptance thresholds.
5. Promote ActionFormer to the default only if improvement is stable across folds, seeds, and domains.
