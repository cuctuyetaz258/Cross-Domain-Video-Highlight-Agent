"""Interface chung để tích hợp Backend Sprint 1"""

import json
from pathlib import Path
from typing import Literal

from highlight_agent.media import prepare_media_workspace, render_highlights
from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
)


def prepare_video(
    video_input: str,
    *,
    output_root: str | Path | None = None,
    cookies_browser: str | None = None,
    transcript_source: Literal["auto", "youtube", "whisper"] = "auto",
) -> MediaWorkspace:
    """Chuẩn hóa input, tách audio và tạo transcript ưu tiên caption"""

    return prepare_media_workspace(
        video_input,
        output_root=output_root,
        cookies_browser=cookies_browser,
        transcript_source=transcript_source,
    )


def load_transcript(path: str | Path) -> TranscriptDocument:
    return TranscriptDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_candidates(path: str | Path) -> list[HighlightCandidate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_candidates = payload.get("highlights", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_candidates, list):
        raise TypeError("candidate JSON must be a list or an object containing a 'highlights' list")
    return [HighlightCandidate.model_validate(item) for item in raw_candidates]


def render_candidates(
    workspace: MediaWorkspace,
    candidates: list[HighlightCandidate],
    *,
    burn_subtitles: bool = True,
) -> list[RenderedHighlight]:
    transcript = load_transcript(workspace.transcript_path)
    return render_highlights(
        workspace,
        candidates,
        transcript=transcript,
        burn_subtitles=burn_subtitles,
    )
