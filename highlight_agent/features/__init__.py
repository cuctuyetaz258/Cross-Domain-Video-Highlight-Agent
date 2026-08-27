"""Trích xuất năm tầng tín hiệu đa miền"""

from .acoustic import extract_acoustic_features, extract_windowed_acoustic_features
from .interaction import (
    extract_interaction_features,
    interaction_features_from_turns,
    windowed_interaction_features,
)
from .scoring import (
    PROFILE_WEIGHTS,
    SIGNAL_NAMES,
    Domain,
    GridSearchResult,
    WindowScore,
    calculate_total_score,
    grid_search_weights,
    normalize_features,
    score_from_domain,
)
from .semantic import SemanticWindowScore, extract_windowed_semantic_features
from .timeline import build_feature_timeline, save_feature_timeline
from .visual import (
    WindowVisualScore,
    extract_visual_scores,
    scores_to_array,
)
from .visual_new import extract_scene_changes, extract_gesture_signal
from .alignment import build_feature_matrix
from .sliding_window import extract_windows
from .overlap_blender import blend_scores
from .nms_topk import extract_topk_nms

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
    # Semantic
    "SemanticWindowScore",
    "extract_windowed_semantic_features",
    "extract_scene_changes",
    "extract_gesture_signal",
    "build_feature_matrix",
    "extract_windows",
    "blend_scores",
    "extract_topk_nms",
]
