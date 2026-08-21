"""Trích xuất năm tầng tín hiệu đa miền"""

from highlight_agent.features.scoring import (
    GridSearchResult,
    WindowScore,
    calculate_total_score,
    grid_search_weights,
    normalize_features,
    score_from_domain,
)
from highlight_agent.features.visual import (
    WindowVisualScore,
    extract_visual_scores,
    scores_to_array,
)

__all__ = [
    # Visual
    "WindowVisualScore",
    "extract_visual_scores",
    "scores_to_array",
    # Scoring / Normalization
    "WindowScore",
    "GridSearchResult",
    "normalize_features",
    "calculate_total_score",
    "score_from_domain",
    "grid_search_weights",
]
