"""Thống nhất schema dùng chung trong pipeline"""

from .highlight import HighlightCandidate, RenderedHighlight
from .media import MediaWorkspace
from .transcript import Chapter, TranscriptDocument, TranscriptSegment, TranscriptWord

__all__ = [
    "Chapter",
    "HighlightCandidate",
    "MediaWorkspace",
    "RenderedHighlight",
    "TranscriptDocument",
    "TranscriptSegment",
    "TranscriptWord",
]
