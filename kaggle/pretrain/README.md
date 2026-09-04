# V2 Benchmark Pretraining on Kaggle

This private Kaggle Script adapts the mounted private benchmark-media Dataset
without changing it. Its authoritative labels and splits remain in the private
`video-highlight-v2-inputs` Dataset's `tvsum_summe.jsonl`.

`machine_shape: "NvidiaTeslaT4"` requests Kaggle's **GPU T4 x2** on every
push. Verify the assigned accelerator in the Kaggle run details; the backend
can still reject the request when that quota is unavailable.

Run three published versions, not one long job:

1. `RUN_MODE = "validate"`: confirm all 75 manifest IDs map to public media.
2. `RUN_MODE = "materialize"`: extract audio, make Whisper transcripts, build
   schema-1.1 caches, and download the successful output archive. Publish that
   archive as private dataset `video-highlight-v2-pretrain-cache`.
3. Attach the cache Dataset, set `RUN_MODE = "train"`, and save the output
   archive. Publish the resulting checkpoint as private dataset
   `video-highlight-v2-pretrain` before attaching it to `kaggle/v2` folds.

Internet is enabled only because `faster-whisper` may need to download the
`small.en` model and the Kaggle base image omits `faster-whisper`,
`scenedetect`, and MediaPipe. Clips with no audio stream receive a duration-
matched silent WAV and an empty transcript; their audio/text channels are zero,
rather than failing the batch. Do not expose credentials or write to the public
source Dataset.

For faster and less rate-limited Hugging Face downloads, add a Kaggle Secret
named `HF_TOKEN` in the kernel UI and grant the kernel access. The runner reads
it only at runtime and never writes the token to logs, artifacts, or Git.
