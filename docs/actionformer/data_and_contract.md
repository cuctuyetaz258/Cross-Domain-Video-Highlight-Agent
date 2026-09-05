# Data and Feature Contract

## Input Artifacts

Create immutable, versioned input bundles before opening the rented GPU instance. Training consumes cached artifacts only.

| Bundle | Required contents | Not required |
| --- | --- | --- |
| `benchmark-inputs` | TVSum/SumMe feature caches, cache metadata, importance labels, locked 55/10/10 manifest | MP4, WAV, frames, transcripts |
| `custom-inputs` | 18 custom feature caches, cache metadata, boundary JSON, importance CSV, five fold manifests | MP4, WAV, frames, transcripts |

The bundles may be stored as private Kaggle datasets or archived outputs, but each run must record the exact dataset/version and SHA-256 hashes of all consumed files.

## Model Input

```text
features:   float32 [B, 7, T]
valid_mask: bool    [B, T]
sample_rate: 10 Hz
channel order:
  rms, pitch, silence, text_score, scene_change, gesture, turn_rate
```

`valid_mask` marks padded time positions. The seven channels must use the existing feature schema and normalization policy. No semantic modality mask is introduced in this milestone because SBERT/CLIP are explicitly out of scope.

ActionFormer uses a stem with stride 5, therefore its first temporal pyramid level operates at 2 Hz. Timestamps are inferred from the 10 Hz cache index; do not silently resample a cache or labels without recording the conversion rule.

## Preflight Audit

Before every smoke or GPU run, the runner must fail fast unless all checks pass:

1. Every manifest record resolves to an existing `feature_matrix.npy` and metadata file.
2. The matrix is `float32`, two-dimensional, and has exactly seven channels in canonical order.
3. The cache sample rate, extractor version, and normalization match the locked contract.
4. Benchmark train/validation/test IDs are disjoint and contain 55/10/10 videos.
5. Custom fold train/validation/test IDs are disjoint by `video_id`; all 18 videos occur in the fold inventory.
6. Importance timestamps use their source frame/time coordinates and are not stretched after media trimming.
7. A manifest lock records IDs, source artifact versions, feature/label hashes, code revision, and decoder configuration.

## Output Contract

Every run writes to an empty, unique `runs/actionformer/<run-id>/` directory. At minimum it contains:

```text
config.json              environment.json          manifest_lock.json
lineage.json             events.jsonl              history.csv
train_log.json           best.pt                   last.pt
evaluation_val.json      evaluation_test.json      report.md
```

Write `last.pt` and the log state after every epoch. `last.pt` must include model, optimizer, scheduler, RNG state, and run status so an interrupted rented instance can resume exactly. Copy the completed run directory to durable local storage before shutting down the instance.
