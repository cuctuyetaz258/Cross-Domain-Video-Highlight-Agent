# Linux RTX 4090 Runbook

## Goal

Run the same Python CLI for CPU smoke tests and GPU training. Do not put model or training logic inside a notebook; Kaggle remains an artifact source, not the execution environment for this milestone.

## Instance Baseline

- Ubuntu 22.04 or newer, one RTX 4090 (24 GB VRAM), persistent project/input/output volume, and network access to private Kaggle artifacts.
- Clone `feature/actionformer-nested-pretrain` into a dedicated working directory. Never train directly on `main`.
- Create a pinned Python environment compatible with the repository's PyTorch/CUDA requirements. Record `python --version`, PyTorch version, CUDA runtime, NVIDIA driver, GPU name, and `nvidia-smi` output in `environment.json`.
- Install the Kaggle CLI only to download private input bundles. No Kaggle secret, Hugging Face token, Whisper, MediaPipe, or FFmpeg is needed while training cached seven-channel features.

## Bring Inputs from Kaggle

1. Download the immutable benchmark and custom input bundles to a local `inputs/` directory on the persistent volume.
2. Verify archive checksums before extracting.
3. Run the preflight audit from [data_and_contract.md](data_and_contract.md).
4. Keep `inputs/` read-only for the run; generated caches, logs, and checkpoints go only under `runs/actionformer/`.

## Mandatory CPU Smoke

Run this before consuming GPU time, on the rented Linux instance itself:

1. Select one ready training video and one case covering an unavailable source modality recorded in cache metadata.
2. Load caches and labels through the real manifest loader.
3. Execute one forward/backward/optimizer step for the applicable stage.
4. Save `last.pt`, reload it in a fresh process, and run one evaluation pass with finite metrics.
5. Produce `history.csv`, a chart, and the complete run metadata.

The smoke passes only when it exits successfully and all required artifacts exist. It is a functional check, not a benchmark result.

## GPU Execution and Recovery

- Switch only the device/configuration from CPU to CUDA; preserve the same code path, manifest lock, and run schema.
- Log peak allocated/reserved VRAM, video ID, temporal length, proposal count, wall-clock time, and CUDA errors by stage.
- On interruption, leave inputs intact, mark the run `interrupted`, archive logs/checkpoints in a `finally` path, then resume from `last.pt`.
- Do not solve OOM by silently dropping candidates. Profile first; any shortlist, crop, or checkpointing policy must be explicit in `config.json` and applied consistently to train/validation/test.
- After each completed stage, copy the run directory to local durable storage. Optionally publish it as a new private Kaggle dataset version for backup; never overwrite the input dataset.
