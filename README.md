# Cross-Domain Video Highlight Agent

An end-to-end system that ranks, refines, and renders short highlights from long-form lecture and podcast videos. It combines seven interpretable audio, text, visual, and interaction signals with a lightweight Learning-to-Rank (LTR) scorer. An experimental ActionFormer-LTR path adds variable 30–90 second temporal proposals while the legacy LTR path remains the default.

The project is an editor-assistance workflow: it proposes ranked candidates and rendered clips, while people retain the final editorial decision.

## What It Does

- Processes a local video or YouTube URL into audio, transcript, and visual evidence.
- Extracts `rms`, `pitch`, `silence`, `text_score`, `scene_change`, `gesture`, and `turn_rate`.
- Scores overlapping five-second windows with an LTR model, applies deterministic NMS, refines boundaries, and renders highlights.
- Supports optional post-LTR semantic assessment with an LLM; LLM reranking is not part of the reported LTR evaluation.
- Provides a Streamlit console for analysis, candidate review, and rendering.

```mermaid
flowchart LR
    A[Video or YouTube URL] --> B[Media and transcript preparation]
    B --> C[7-channel feature timeline]
    C --> D[LTR window scoring]
    D --> E[Candidate pool and NMS]
    E --> F[Optional LLM rerank]
    F --> G[Boundary refinement and MP4 render]
```

## Quick Start

### 1. Create the environment

Python 3.11 and `ffmpeg` are recommended. Install `ffmpeg` first:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Create the Python environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure optional credentials

```bash
cp .env.example .env
```

| Variable | Needed when |
| --- | --- |
| `HF_TOKEN` | Running podcast diarization with Pyannote; accept the model terms on Hugging Face first. |
| `OPENAI_API_KEY` | Enabling the optional OpenAI reranker. |
| `GROQ_API_KEY` | Enabling the optional Groq reranker. |
| `YTDLP_COOKIES_BROWSER` | A YouTube download needs an authenticated browser session, for example `chrome`. |

Do not commit `.env` or browser cookies.

### 3. Install the shared LTR checkpoint

The public Kaggle artifact is verified by size, SHA-256, and feature-contract preflight before installation:

```bash
python -m scripts.download_ltr_checkpoint
```

It installs the current operational model at `data/models/ltr_target_lecture_podcast.pt`.
The checkpoint is also available on [Kaggle](https://www.kaggle.com/datasets/nguyentrann0703/in-domain-ltr-5fold-results).

### 4. Launch the Streamlit console

```bash
streamlit run frontend/app.py --server.fileWatcherType none
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. A new UI session defaults to the shared checkpoint.

## Run From The CLI

Run an LTR-only analysis with explicit Whisper timestamps:

```bash
python -m scripts.run_agent "https://www.youtube.com/watch?v=VIDEO_ID" \
  --domain lecture \
  --highlight-count 3 \
  --transcript-source whisper
```

Use `--domain podcast` for conversational videos. If the speaker count is known, pass `--known-speaker-count 2`; otherwise use `--min-speaker-count 1 --max-speaker-count 3` to constrain diarization.

Generated media, feature caches, and run metadata are written below `output/` and intentionally excluded from Git.

### Experimental ActionFormer-LTR

The current shared OOF proposal cache does not isolate all upstream training from each outer test fold, and LTR validation nDCG includes ground-truth candidates. Recorded CV results are exploratory until nested OOF and predicted-only validation are implemented. The legacy scorer remains the runtime default.

Pretrain the ActionFormer temporal backbone on cached TVSum + SumMe frame-importance labels before custom localization. This uses the locked 55/10/10 benchmark split and does not require the 18 custom caches:

```bash
python -m scripts.pretrain_actionformer_backbone \
  --manifest data/manifests/tvsum_summe.jsonl \
  --cache-dir data/features_cache \
  --output data/models/actionformer_benchmark_pretrained.pt \
  --report data/reports/actionformer_benchmark_pretraining.json \
  --device cuda
```

After all 18 custom caches are available, pass that checkpoint to localization training. Its architecture and seven-channel feature contract are verified before fine-tuning:

```bash
python -m scripts.train_actionformer_ltr \
  --manifest data/manifests/actionformer_fold0.jsonl \
  --stage localization \
  --init-checkpoint data/models/actionformer_benchmark_pretrained.pt \
  --output data/models/actionformer_fold0_localization.pt
```

Audit the 18-video annotation set and rebuild the five video-disjoint manifests:

```bash
python -m scripts.prepare_actionformer_data
```

Prepare missing YouTube media with a resumable per-video report, then build caches by domain:

```bash
python -m scripts.prepare_actionformer_media
python -m scripts.build_feature_cache \
  --manifest data/manifests/actionformer_fold0.jsonl \
  --domain lecture \
  --report data/reports/actionformer_lecture_cache_build.json
```

Podcast caches require `pyannote.audio>=3.3,<4`, a Hugging Face token (via `HF_TOKEN`
or `hf auth login`), and accepted access to both
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. The legacy 3.1
pipeline is intentional: Community-1 requires Pyannote 4 and a newer Torch stack
that conflicts with the project's pinned MediaPipe runtime.

Train localization, then the proposal-level LTR head. JSON is updated after every epoch; CSV history and SVG curves are written beside it.

```bash
python -m scripts.train_actionformer_ltr \
  --manifest data/manifests/actionformer_fold0.jsonl \
  --stage localization \
  --output data/models/actionformer_fold0_localization.pt \
  --last-output data/models/actionformer_fold0_localization_last.pt \
  --log data/reports/actionformer_fold0/localization_log.json \
  --history-csv data/reports/actionformer_fold0/localization_history.csv \
  --curves data/reports/actionformer_fold0/localization_curves.svg

python -m scripts.train_actionformer_ltr \
  --manifest data/manifests/actionformer_fold0.jsonl \
  --stage ltr \
  --init-checkpoint data/models/actionformer_fold0_localization.pt \
  --output data/models/actionformer_fold0_ltr.pt \
  --log data/reports/actionformer_fold0/proposal_ltr_log.json
```

Evaluate the combined checkpoint and opt into it at runtime:

```bash
python -m evaluation.evaluate_actionformer_ltr \
  --checkpoint data/models/actionformer_fold0_ltr.pt \
  --manifest data/manifests/actionformer_fold0.jsonl \
  --split test \
  --output data/reports/actionformer_fold0/evaluation_test_log.json

python -m scripts.run_agent "https://www.youtube.com/watch?v=VIDEO_ID" \
  --domain lecture \
  --scorer-type actionformer-ltr \
  --actionformer-model-path data/models/actionformer_fold0_ltr.pt
```

Models and raw training reports under `data/models/` and `data/reports/` are local artifacts excluded from Git. Do not promote an experimental checkpoint until the five-fold acceptance criteria are met.

## Model And Evaluation

The operational scorer is a compact `7 -> 32 -> 1` MLP. Each five-second window is mean-pooled from the 10 Hz feature timeline before scoring; the current model does not use temporal attention inside a window.

The checkpoint was initialized from a TVSum + SumMe LTR checkpoint and fine-tuned on 10 internally annotated videos (5 lecture, 5 podcast). In video-disjoint 5-fold evaluation, fine-tuning improved macro held-out AP:

| Model | Macro AP | Kendall tau |
| --- | ---: | ---: |
| Frozen TVSum + SumMe initialization | 0.6345 | -0.0299 |
| In-domain fine-tuned LTR | 0.7175 | 0.0430 |

This is an in-domain pilot, not evidence of generalization. The sample is small, two videos regress in AP, and LLM fusion was not evaluated. See the [full 5-fold report](docs/in_domain_ltr_5fold_report.md) for per-video results, protocol, hashes, and limitations.

## Repository Map

- `frontend/`: Streamlit application and UI components.
- `highlight_agent/`: media, feature, LTR, LLM, and agent implementation.
- `scripts/`: runnable utilities and entrypoints.
- `evaluation/`: LTR evaluation and metric utilities.
- `data/manifests/`: reproducible train/validation/test split definitions.
- `docs/`: reports, presentation material, and reference guides.

## Documentation

- [In-domain LTR 5-fold report](docs/in_domain_ltr_5fold_report.md): evaluation protocol, results, artifact hashes, and limitations.
- [Training handover guide](TRAINING_HANDOVER_GUIDE.md): feature-cache reuse, cross-validation, release training, and artifact handling.
- [Training plan](TRAINING_PLAN.md): data policy, LTR/LLM roadmap, evaluation, and ablation plan.
- [ActionFormer-LTR integration plan](ACTIONFORMER_LTR_INTEGRATION_PLAN.md): architecture, rollout, tests, and acceptance criteria.
- [ActionFormer-LTR implementation report](ACTIONFORMER_IMPLEMENTATION_REPORT.md): implemented scope, smoke/full-fold-partial logs, and current blockers.
- [Sample videos](docs/sample_videos.md): lecture, podcast, and stand-up source inventory.
- [Docker advanced setup](docs/docker.md): backend-oriented container workflow.

## Troubleshooting

| Symptom | Recommended action |
| --- | --- |
| `LTR_CHECKPOINT_SCHEMA_MISMATCH` | Reinstall the shared model with `python -m scripts.download_ltr_checkpoint --force`. |
| Podcast diarization cannot start | Set `HF_TOKEN` in `.env` or run `hf auth login`, then accept both Pyannote 3.1 model terms. |
| `yt-dlp is not installed` | Activate `.venv`, then run `python -m pip install -U "yt-dlp[default]"`. |
| YouTube returns HTTP 403 or asks to sign in | Update `yt-dlp`; if required, set `YTDLP_COOKIES_BROWSER=chrome`, close Chrome so its cookie database is unlocked, and retry. |
| YouTube warns that no JavaScript runtime is available | Install Deno 2.3+ and ensure `deno` is on `PATH`; current YouTube extraction may expose fewer formats without it. |
| Streamlit keeps rerunning | Stop duplicate Streamlit processes and launch one instance with `--server.fileWatcherType none`. |

## Quality Checks

```bash
python -m pytest
python -m ruff check highlight_agent scripts frontend tests
```

## Scope

The repository does not currently publish a license. Do not redistribute source media, cookies, API keys, or generated video artifacts without confirming the relevant permissions.
