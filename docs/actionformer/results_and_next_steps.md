# ActionFormer Result And Next Steps

## Locked result

The completed run is `nested_cv_seed42_pretrained_frozen_v2`. It uses the
existing 7-channel, 10 Hz cache, benchmark-pretrained frozen backbone, nested
outer five-fold protocol, and predicted-only evaluation. Its artifacts are
backed up in the private Kaggle dataset `nguyentrann0703/actionformer-artifacts-seed42`.

| Metric | ActionFormer confidence | ActionFormer + IMSAB |
| --- | ---: | ---: |
| mAP@0.3, five-fold mean | 0.1423 | 0.0710 |
| Recall@3 at IoU 0.3, five-fold mean | 0.2156 | 0.1167 |

IMSAB is therefore not a release candidate. `legacy-ltr` remains the runtime
default. This is a valid negative result: its confidence baseline and IMSAB
use the same outer localization generator and predicted test candidates.

## Relation To The Historical LTR Report

`docs/in_domain_ltr_5fold_report.md` cannot be used as a numeric baseline for
the table above. It evaluates 10 custom lecture/podcast videos with 5-second
importance windows and reports window AP/Kendall tau. ActionFormer evaluates
18 annotated videos as temporal proposal localization and reports mAP at IoU
and proposal recall. The task, label unit, test population, and metrics differ.

The defensible comparison is qualitative only: the historical seven-channel
MLP was a positive in-domain ranking pilot; the current ActionFormer proposal
pipeline is reproducible but has not yet shown a ranking gain over its own
confidence baseline.

## Next Experiment

Do not retrain localization. Reuse each fixed outer localization checkpoint
and `nested_proposals.json`, then train only proposal rankers with identical
candidate caches and held-out evaluation:

1. MLP plus margin/hinge loss, unweighted pairs.
2. MLP plus utility-weighted RankNet.
3. IMSAB plus utility-weighted RankNet with ordinal rank signal disabled.

The new `scripts/run_actionformer_rerank_ablation.py` records resumable
`best.pt` and `last.pt` per fold, plus a fold-level evaluation and CV summary.
Only run multi-seed replication if a configuration clears the predeclared
release gate in `experiment_protocol.md`.

## Example Commands

Run these from the project root on the GPU VM. Each command is cheap relative
to the completed nested run because it reuses localization and proposals.

```bash
python scripts/run_actionformer_rerank_ablation.py \
  --source-run runs/actionformer/nested_cv_seed42_pretrained_frozen_v2 \
  --output-dir runs/actionformer/ablation_mlp_margin_seed42 \
  --architecture mlp --loss margin --pair-weighting none \
  --device cuda

python scripts/run_actionformer_rerank_ablation.py \
  --source-run runs/actionformer/nested_cv_seed42_pretrained_frozen_v2 \
  --output-dir runs/actionformer/ablation_mlp_ranknet_seed42 \
  --architecture mlp --loss ranknet --pair-weighting utility \
  --device cuda

python scripts/run_actionformer_rerank_ablation.py \
  --source-run runs/actionformer/nested_cv_seed42_pretrained_frozen_v2 \
  --output-dir runs/actionformer/ablation_imsab_no_ordinal_seed42 \
  --architecture setrank_imsab --loss ranknet --pair-weighting utility \
  --rank-signal none --device cuda
```
