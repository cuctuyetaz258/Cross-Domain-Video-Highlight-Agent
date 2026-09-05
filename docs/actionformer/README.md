# ActionFormer Execution Docs

This folder is the canonical execution record for the seven-channel ActionFormer + IMSAB work. It supersedes neither code nor the historical [Kaggle integration plan](../ACTIONFORMER_KAGGLE_INTEGRATION_PLAN.md); that document preserves the original Kaggle-oriented proposal.

## Locked scope

- Input remains the existing handcrafted feature matrix `float32[B, 7, T]` at 10 Hz. The model downsamples its first temporal level to 2 Hz.
- This milestone excludes SBERT, CLIP, CLAP, new feature extraction, and TCN changes.
- TVSum and SumMe caches are reused. Raw video, audio, transcript extraction, MediaPipe, and Hugging Face access are not required for training.
- Kaggle is the private source of immutable input artifacts. A rented Linux RTX 4090 instance runs CPU smoke tests and GPU training.
- Success means reproducible, leakage-safe nested five-fold CV with reloadable checkpoints. Historical shared-OOF results remain exploratory.

## Pipeline

```text
TVSum + SumMe cached 7 x T features + importance labels
    -> ActionFormer backbone + temporary importance head
    -> benchmark-pretrained backbone
    -> custom ActionFormer localization (nested outer folds)
    -> outer-fold-specific predicted proposal caches
    -> IMSAB proposal LTR + Soft-NMS
    -> predicted-only held-out evaluation
```

The temporary benchmark importance head is not a 30--90 second boundary head. It transfers temporal representations only; custom boundary regression is trained and evaluated on the annotated 18-video dataset.

## Reading Order

1. [Data and contract](data_and_contract.md) defines what is downloaded and validated.
2. [Linux runbook](linux_runbook.md) defines the portable smoke/train environment.
3. [Experiment protocol](experiment_protocol.md) defines the only acceptable CV and reporting procedure.

## Current Implementation Status

- ActionFormer localization, proposal decoding, IMSAB ranking, nested-OOF utilities, and predicted-only validation support exist on `feature/actionformer-nested-pretrain`.
- The dedicated benchmark importance-pretraining runner and the Linux packaging/run scripts are still implementation work.
- Do not change the runtime default from `legacy-ltr` until the protocol in this folder passes its release gate.
