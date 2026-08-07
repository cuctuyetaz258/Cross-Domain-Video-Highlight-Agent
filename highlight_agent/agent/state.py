"""State dùng chung cho năm pha Agent Sprint 1"""

from typing import Any, Literal, NotRequired, Required, TypedDict

from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
)

Domain = Literal["lecture", "podcast", "standup"]
SignalProfile = dict[str, float]


class ReasoningEntry(TypedDict):
    candidate_id: str
    explanation: str


class AgentState(TypedDict, total=False):
    """State tích lũy qua năm pha của Agent"""

    video_path: Required[str]
    domain: Required[Domain]
    output_root: NotRequired[str]
    cookies_browser: NotRequired[str | None]
    transcript_source: NotRequired[Literal["auto", "youtube", "whisper"]]
    burn_subtitles: NotRequired[bool]
    highlight_count: NotRequired[int]
    workspace: NotRequired[MediaWorkspace]
    transcript: NotRequired[TranscriptDocument]
    profile: NotRequired[SignalProfile]
    features: NotRequired[dict[str, Any]]
    candidates: NotRequired[list[HighlightCandidate]]
    highlights: NotRequired[list[HighlightCandidate]]
    rendered_highlights: NotRequired[list[RenderedHighlight]]
    reasoning: NotRequired[list[ReasoningEntry]]
