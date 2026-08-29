"""Interface chung để tích hợp Backend Sprint 1"""

import json
from pathlib import Path
from typing import Literal

from highlight_agent.boundary import refine_candidate_boundaries
from highlight_agent.features import extract_acoustic_features
from highlight_agent.media import prepare_media_workspace, render_highlights
from highlight_agent.schemas import (
    BoundaryAdjustment,
    FeatureTimeline,
    HighlightCandidate,
    LLMHighlightAssessment,
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


def refine_candidates_for_render(
    workspace: MediaWorkspace,
    candidates: list[HighlightCandidate],
    *,
    transcript: TranscriptDocument | None = None,
) -> tuple[list[HighlightCandidate], list[BoundaryAdjustment]]:
    """Canh biên candidate bằng transcript và silence toàn video trước khi render"""

    document = transcript or load_transcript(workspace.transcript_path)
    feature_path = workspace.audio_path.parent / "features" / "features.json"
    if feature_path.is_file():
        timeline = FeatureTimeline.model_validate_json(feature_path.read_text(encoding="utf-8"))
        if timeline.video_id != workspace.video_id:
            raise ValueError("feature timeline video_id must match workspace video_id")
        silence_intervals = timeline.acoustic.silence_intervals
    else:
        silence_intervals = extract_acoustic_features(workspace.audio_path).silence_intervals
    return refine_candidate_boundaries(
        candidates,
        document,
        silence_intervals,
        video_duration=document.duration,
    )


def render_candidates(
    workspace: MediaWorkspace,
    candidates: list[HighlightCandidate],
    *,
    aspect_ratio: Literal["9:16", "16:9"] = "9:16",
    burn_subtitles: bool = True,
    boundary_adjustments: list[BoundaryAdjustment] | None = None,
    refine_boundaries: bool = True,
    llm_assessments: dict[str, LLMHighlightAssessment] | None = None,
    pipeline_metadata: dict | None = None,
    render_namespace: str | None = None,
) -> list[RenderedHighlight]:
    transcript = load_transcript(workspace.transcript_path)
    if refine_boundaries:
        candidates, boundary_adjustments = refine_candidates_for_render(
            workspace,
            candidates,
            transcript=transcript,
        )
    return render_highlights(
        workspace,
        candidates,
        aspect_ratio=aspect_ratio,
        transcript=transcript,
        burn_subtitles=burn_subtitles,
        boundary_adjustments=boundary_adjustments,
        llm_assessments=llm_assessments,
        pipeline_metadata=pipeline_metadata,
        render_namespace=render_namespace,
    )
