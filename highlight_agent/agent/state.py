"""State dùng chung cho các pha Agent"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Literal, NotRequired, Required, TypedDict

from highlight_agent.schemas import (
    BoundaryAdjustment,
    HighlightCandidate,
    LLMHighlightAssessment,
    LLMRunInfo,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
)

Domain = Literal["lecture", "podcast", "standup"]


# ──────────────────────────────────────────────
# Event callback cho hiển thị tiến độ thời gian thực
# ──────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Sự kiện tiến độ emit từ bên trong các node."""

    node: str              # "preflight" | "observe" | "plan" | "analyze" | "decide" | "explain"
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
    min_speaker_count: NotRequired[int]
    max_speaker_count: NotRequired[int]
    burn_subtitles: NotRequired[bool]
    highlight_count: NotRequired[int]
    aspect_ratio: NotRequired[Literal["9:16", "16:9"]]
    workspace: NotRequired[MediaWorkspace]
    transcript: NotRequired[TranscriptDocument]
    analysis_plan: NotRequired[dict[str, Any]]
    features: NotRequired[dict[str, Any]]
    feature_path: NotRequired[str]
    feature_timeline: NotRequired[dict[str, Any]]
    candidates: NotRequired[list[HighlightCandidate]]
    highlights: NotRequired[list[HighlightCandidate]]
    boundary_adjustments: NotRequired[list[BoundaryAdjustment]]
    rendered_highlights: NotRequired[list[RenderedHighlight]]
    reasoning: NotRequired[list[ReasoningEntry]]
    ltr_checkpoint_info: NotRequired[dict[str, Any]]
    candidate_pool_size: NotRequired[int]
    analysis_snapshot_path: NotRequired[str]
    analysis_id: NotRequired[str]
    render_namespace: NotRequired[str]

    # ── LLM semantic reranking config/output ──
    llm_provider: NotRequired[Literal["auto", "disabled", "openai", "groq", "custom"]]
    llm_model: NotRequired[str | None]
    llm_base_url: NotRequired[str | None]
    llm_top_m: NotRequired[int]
    fusion_calibrator_path: NotRequired[str | None]
    llm_timeout_seconds: NotRequired[float]
    llm_assessments: NotRequired[list[LLMHighlightAssessment]]
    llm_run: NotRequired[LLMRunInfo]

    # ── Required LTR scorer ──
    ltr_model_path: NotRequired[str | None]

    # ── Progress callback ──
    emit: NotRequired[EmitFn]
