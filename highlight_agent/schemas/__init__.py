"""Thống nhất schema dùng chung trong pipeline"""

from .features import (
    AcousticFeatures,
    FeatureTimeline,
    FeatureWindow,
    InteractionFeatures,
    SemanticFeatures,
    SpeakerTurn,
    TimeInterval,
    VisualFeatures,
)
from .highlight import BoundaryAdjustment, HighlightCandidate, RenderedHighlight
from .llm import (
    CandidateTranscriptContext,
    LLMHighlightAssessment,
    LLMHighlightAssessmentBatch,
    LLMRunInfo,
)
from .media import MediaWorkspace
from .transcript import Chapter, TranscriptDocument, TranscriptSegment, TranscriptWord

__all__ = [
    "AcousticFeatures",
    "BoundaryAdjustment",
    "Chapter",
    "CandidateTranscriptContext",
    "FeatureTimeline",
    "FeatureWindow",
    "HighlightCandidate",
    "InteractionFeatures",
    "LLMHighlightAssessment",
    "LLMHighlightAssessmentBatch",
    "LLMRunInfo",
    "MediaWorkspace",
    "RenderedHighlight",
    "SemanticFeatures",
    "SpeakerTurn",
    "TimeInterval",
    "TranscriptDocument",
    "TranscriptSegment",
    "TranscriptWord",
    "VisualFeatures",
]
