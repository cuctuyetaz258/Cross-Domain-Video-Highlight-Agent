# V2 Experiment Protocol

## Dataset lock

Use exactly the 10 videos for which media, transcript, labels, and schema-1.1
feature caches are available at experiment start. Create and commit a
`v2_10video` manifest listing video ID, domain, source paths, duration, cache
fingerprint, and assigned split. Do not add/remove a video after inspecting
test metrics.

## Fair comparison

1. Rerun V1 MLP on the locked manifest and fixed video-disjoint five-fold
   schedule.
2. Pretrain V2 TCN on TVSum + SumMe using an independently versioned V2
   checkpoint.
3. Train each V2 custom fold from that V2 checkpoint.
4. Select epoch and hyperparameters only from the fold validation videos.
5. Evaluate the selected checkpoint once on its held-out test videos.

The benchmark pretraining command is valid only after every TVSum/SumMe record
has a compatible seven-channel cache. The local custom 10-video cache is
sufficient for V2 fine-tuning and the CPU smoke test, but it does not by itself
make TVSum+SumMe pretraining reproducible.

Each fold retains whole videos in a single split. Pair construction and TCN
context stay inside the current video. Every custom video appears in test once
across the five folds.

## Metrics

Primary and diagnostic metrics are calculated per held-out video, then reported
as macro mean +/- standard deviation over all 10 held-out entries:

| Role | Metric |
| --- | --- |
| Primary | Average Precision over positive/negative windows |
| Ranking diagnostic | Kendall tau using continuous ordinal window targets |
| Additional diagnostics | Spearman rho, window F1 at positive count, Positive Hit@5 |

Ignored windows remain excluded from AP/F1 but participate in ordinal metrics
where applicable, matching V1. LLM rerank/fusion remains disabled.

## Required artifacts

- Locked manifest and split manifests, with SHA-256 fingerprints.
- Pretraining and fold configs, random seeds, dependency/device details.
- One checkpoint per fold plus checkpoint SHA-256 and selected epoch.
- Train/validation history CSV and curve SVG for every run.
- Frozen/V1 and V2 raw test JSON, per-video table, and aggregate report.
- Parameter count and wall-clock cost for V1 and V2.
- At least one qualitative review for every AP regression.

## Decision rule

Apply the V2 promotion gate in [`../roadmap.md`](../roadmap.md) without
changing it after test results are known. If V2 passes, an all-data model is
explicitly labelled operational/demo-only and is not reported as an
independent held-out result.
