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
from .media import MediaWorkspace
from .transcript import Chapter, TranscriptDocument, TranscriptSegment, TranscriptWord

__all__ = [
    "AcousticFeatures",
    "BoundaryAdjustment",
    "Chapter",
    "FeatureTimeline",
    "FeatureWindow",
    "HighlightCandidate",
    "InteractionFeatures",
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
