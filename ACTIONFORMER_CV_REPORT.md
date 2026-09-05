# ActionFormer + IMSAB OOF: 5-fold Training Report

> Audit correction (2026-09-05): these results are exploratory, not a clean held-out estimate. The shared OOF cache excludes each predicted video from its own generator's training but permits indirect outer-test leakage through other generators. Validation nDCG includes GT/grid candidates. Nested OOF and predicted-only validation must be implemented before using these results for model selection claims or attributing errors solely to localization. Original numbers below are preserved for audit.

## Configuration

- ActionFormer localization: `d_model=32`, attention window 64, seed 42.
- Proposal scorer: two IMSAB blocks, 16 inducing points, `d_model=128`, two heads.
- Objective: utility-weighted RankNet using `proposal_utility_v2`.
- Candidate source: leakage-checked OOF proposal cache covering all 18 videos.
- Execution device: CPU after CUDA OOM on the full proposal lists.

## Per-fold results

| Fold | Loc. val Recall@3 | LTR val nDCG@3 | Test mAP@0.3 | Test Recall@3 | Test mean IoU |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2250 | 0.8031 | 0.0182 | 0.2500 | 0.0547 |
| 1 | 0.3833 | 0.6696 | 0.0379 | 0.1958 | 0.1301 |
| 2 | 0.0833 | 0.6826 | 0.0893 | 0.2500 | 0.1547 |
| 3 | 0.4222 | 0.8211 | 0.0071 | 0.0833 | 0.0589 |
| 4 | 0.3333 | 0.8072 | 0.0192 | 0.0667 | 0.0517 |

## Cross-validation summary

Values are the unweighted mean and population standard deviation across five folds.

- Localization validation Recall@3: `0.2894 ± 0.1225`.
- LTR validation nDCG@3: `0.7567 ± 0.0662`.
- Test mAP@0.3: `0.0343 ± 0.0292`.
- Test mAP@0.5: `0.0158 ± 0.0098`.
- Test Recall@3: `0.1692 ± 0.0796`.
- Test mean IoU: `0.0900 ± 0.0435`.

## Interpretation

The scorer learns the validation ranking signal, but localization remains the bottleneck: test mAP and IoU are low and vary substantially by fold. These results do not pass the production gate. The next experiment should compare the same OOF candidates against ActionFormer confidence and the legacy MLP baseline before changing the IMSAB architecture.
