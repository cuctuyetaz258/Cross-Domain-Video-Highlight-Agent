"""State dùng chung cho các pha Agent"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Literal, NotRequired, Required, TypedDict

from highlight_agent.schemas import (
    BoundaryAdjustment,
    HighlightCandidate,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
)

Domain = Literal["lecture", "podcast", "standup"]
SignalProfile = dict[str, float]


# ──────────────────────────────────────────────
# Event callback cho hiển thị tiến độ thời gian thực
# ──────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Sự kiện tiến độ emit từ bên trong các node."""

    node: str              # "observe" | "plan" | "analyze" | "decide" | "explain"
    step: str              # "start" | "visual_window" | "done" | "fallback" | ...
    message: str           # mô tả ngắn
    meta: dict = dc_field(default_factory=dict)  # thông tin phụ (timing, score, ...)


EmitFn = Callable[[ProgressEvent], None] | None


class ReasoningEntry(TypedDict):
    candidate_id: str
    explanation: str


class AgentState(TypedDict, total=False):
    """State tích lũy qua các pha của Agent"""

    video_path: Required[str]
    domain: Required[Domain]
    output_root: NotRequired[str]
    cookies_browser: NotRequired[str | None]
    transcript_source: NotRequired[Literal["auto", "youtube", "whisper"]]
    known_speaker_count: NotRequired[int]
    burn_subtitles: NotRequired[bool]
    highlight_count: NotRequired[int]
    workspace: NotRequired[MediaWorkspace]
    transcript: NotRequired[TranscriptDocument]
    profile: NotRequired[SignalProfile]
    features: NotRequired[dict[str, Any]]
    feature_path: NotRequired[str]
    feature_timeline: NotRequired[dict[str, Any]]
    candidates: NotRequired[list[HighlightCandidate]]
    highlights: NotRequired[list[HighlightCandidate]]
    boundary_adjustments: NotRequired[list[BoundaryAdjustment]]
    rendered_highlights: NotRequired[list[RenderedHighlight]]
    reasoning: NotRequired[list[ReasoningEntry]]

    # ── Visual scoring config ──
    visual_method: NotRequired[Literal["pixel_diff", "raft"]]
    visual_sample_fps: NotRequired[float]

    # ── Progress callback ──
    emit: NotRequired[EmitFn]
