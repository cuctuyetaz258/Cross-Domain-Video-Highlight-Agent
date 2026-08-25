"""Trích xuất năm tầng tín hiệu đa miền"""

from .acoustic import extract_acoustic_features, extract_windowed_acoustic_features
from .interaction import (
    extract_interaction_features,
    interaction_features_from_turns,
    windowed_interaction_features,
)
from .scoring import (
    Domain,
    GridSearchResult,
    PROFILE_WEIGHTS,
    SIGNAL_NAMES,
    WindowScore,
    calculate_total_score,
    grid_search_weights,
    normalize_features,
    score_from_domain,
)
from .timeline import build_feature_timeline, save_feature_timeline
from .visual import (
    WindowVisualScore,
    extract_visual_scores,
    scores_to_array,
)

__all__ = [
    # Acoustic
    "extract_acoustic_features",
    "extract_windowed_acoustic_features",
    # Interaction
    "extract_interaction_features",
    "interaction_features_from_turns",
    "windowed_interaction_features",
    # Timeline
    "build_feature_timeline",
    "save_feature_timeline",
    # Visual
    "WindowVisualScore",
    "extract_visual_scores",
    "scores_to_array",
    # Scoring
    "Domain",
    "PROFILE_WEIGHTS",
    "SIGNAL_NAMES",
    "WindowScore",
    "GridSearchResult",
    "normalize_features",
    "calculate_total_score",
    "score_from_domain",
    "grid_search_weights",
]
