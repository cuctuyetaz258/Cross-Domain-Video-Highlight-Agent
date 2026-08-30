# Docker Advanced Setup

Docker is an advanced, backend-oriented workflow. The Streamlit UI is best run directly from the local Python environment because video processing and authenticated browser cookies are machine-specific.

## Build

From the repository root:

```bash
docker build -t highlight-agent .
```

The image uses Python 3.11, installs `ffmpeg`, and exposes `python -m` as its entrypoint.

## Run A Backend Command

Mount `output/` to retain generated files on the host. On macOS/Linux:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/output:/app/output" \
  highlight-agent scripts.run_backend sample.mp4
```

For a YouTube URL, pass the URL as the module argument:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/output:/app/output" \
  highlight-agent scripts.run_backend "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Notes

- Download the LTR checkpoint into the container or mount a host `data/models/` directory before running an LTR command.
- Do not place cookies or API keys in the Docker image. Provide them at runtime through `.env` or the deployment secret store.
- Browser-cookie extraction is generally more reliable when media is downloaded on the host before container processing.
