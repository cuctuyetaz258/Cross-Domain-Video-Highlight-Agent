"""Trích xuất năm tầng tín hiệu đa miền"""

from .acoustic import extract_acoustic_features, extract_windowed_acoustic_features
from .alignment import build_feature_matrix
from .interaction import (
    extract_interaction_features,
    interaction_features_from_turns,
    windowed_interaction_features,
)
from .ltr_pipeline import LTRFeatureBundle, build_ltr_features
from .nms_topk import extract_topk_nms
from .overlap_blender import blend_scores
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
from .sliding_window import extract_windows
from .timeline import build_feature_timeline, save_feature_timeline
from .visual import (
    WindowVisualScore,
    extract_visual_scores,
    scores_to_array,
)
from .visual_new import (
    extract_gesture_signal,
    extract_scene_changes,
    extract_scene_observation,
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
    # Semantic
    "SemanticWindowScore",
    "extract_windowed_semantic_features",
    "extract_scene_changes",
    "extract_gesture_signal",
    "extract_scene_observation",
    "LTRFeatureBundle",
    "build_ltr_features",
    "build_feature_matrix",
    "extract_windows",
    "blend_scores",
    "extract_topk_nms",
]
