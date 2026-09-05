# Clean ActionFormer Experiment Protocol

## Stage 0: Audit and Lock

Create a new run ID and manifest lock. Historical ActionFormer OOF metrics are audit artifacts only: the old shared cache can indirectly expose outer-test labels, and old validation ranking mixed predicted, ground-truth, and grid candidates.

## Stage 1: Benchmark Importance Pretraining

Train an ActionFormer backbone plus a temporary scalar importance head on the locked TVSum/SumMe train split. Use timestamp-aligned source-specific importance normalization, within-video pairwise ranking, and early stopping only on the benchmark validation split.

- Do not train boundary offsets from benchmark importance labels.
- Keep the benchmark test split untouched until the configuration is locked.
- Save a transfer checkpoint with architecture, feature schema, label policy, split IDs, content hashes, optimizer/scheduler state, and RNG state.
- Report benchmark ranking metrics separately from custom localization metrics.

## Stage 2: Custom Localization

Compare two otherwise identical localization configurations on the custom data:

1. ActionFormer trained from scratch.
2. ActionFormer initialized from the benchmark-pretrained backbone.

For the first transfer experiment, freeze the shared pretrained backbone. This ensures all generator and LTR folds pool representations from the same feature space. Any unfreezing experiment requires a new cross-fitting design and is not part of the baseline.

## Stage 3: Nested Outer Five-Fold CV

For each outer fold:

1. Lock outer train, validation, and test video IDs.
2. Split only outer-train videos into fixed inner folds.
3. Train inner localization generators and create predictions only for their inner held-out partitions.
4. Build the outer-fold LTR training cache from those inner held-out predicted proposals. Record every generator's training and selection IDs.
5. Train an outer localization generator using only permitted outer-train data; use outer validation for epoch selection.
6. Evaluate outer test only after model selection is fixed. Neither the test video nor its labels may occur in any ancestor generator's training/selection IDs.

The validator must reject both direct and indirect lineage leakage. The primary validation and test candidate lists contain predicted proposals only; GT/grid candidates are permitted solely in a separately labeled oracle diagnostic.

## Stage 4: Proposal Ranking Ablations

On the same outer-fold generator, backbone, and predicted candidate cache, compare:

1. ActionFormer confidence only.
2. Proposal MLP with hinge ranking.
3. Proposal MLP with utility-weighted RankNet.
4. IMSAB with utility-weighted RankNet.
5. IMSAB ordinal-rank signal on versus off.

Keep candidate caps, utility version, pair sampling seed, and duration filter identical across the relevant comparison. GT/jitter/grid candidates, if used for training augmentation, must be source-labeled and excluded from the primary predicted-only metric.

## Metrics and Release Gate

Primary reports include predicted nDCG@3/5, mAP at temporal IoU 0.3/0.5/0.7, Recall@1/3/5, mean IoU, boundary error, duration-valid rate, latency, peak VRAM, and per-domain results. Report fold and seed variance explicitly.

Run seed 42 first. Once validation selects a configuration, rerun two predeclared additional seeds without changing the test-based decision rule.

The default scorer remains `legacy-ltr` until the clean protocol shows all of the following:

- mean predicted nDCG@3 improves by at least 0.02 against the locked MLP-hinge baseline;
- the improvement does not regress on more than two of five folds;
- mAP@0.3 and Recall@3 do not decline; and
- either mAP@0.5 or mean IoU improves.

Do not change these gates after seeing held-out test results.
