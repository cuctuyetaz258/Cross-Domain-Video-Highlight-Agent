"""Tạo và lưu timeline feature thô của Sprint 2"""

from pathlib import Path

from highlight_agent.schemas import (
    AcousticFeatures,
    FeatureTimeline,
    FeatureWindow,
    InteractionFeatures,
    SemanticFeatures,
    VisualFeatures,
)


def build_feature_timeline(
    *,
    video_id: str,
    domain: str,
    duration: float,
    window_seconds: float,
    hop_seconds: float,
    acoustic: AcousticFeatures,
    acoustic_windows: list[FeatureWindow],
    interaction: InteractionFeatures | None = None,
    interaction_windows: list[InteractionFeatures] | None = None,
    semantic_windows: list[SemanticFeatures] | None = None,
    visual_windows: list[VisualFeatures] | None = None,
) -> FeatureTimeline:
    """Ghép output các layer mà chưa chấm điểm hay chuẩn hóa giá trị"""

    if interaction_windows is not None and len(interaction_windows) != len(acoustic_windows):
        raise ValueError("acoustic and interaction window counts must match")
    if semantic_windows is not None and len(semantic_windows) != len(acoustic_windows):
        raise ValueError("acoustic and semantic window counts must match")
    if visual_windows is not None and len(visual_windows) != len(acoustic_windows):
        raise ValueError("acoustic and visual window counts must match")
    windows = [
        acoustic_window.model_copy(
            update={
                "interaction": interaction_windows[index] if interaction_windows else None,
                "semantic": semantic_windows[index] if semantic_windows else None,
                "visual": visual_windows[index] if visual_windows else None,
            }
        )
        for index, acoustic_window in enumerate(acoustic_windows)
    ]
    return FeatureTimeline(
        video_id=video_id,
        domain=domain,
        duration=duration,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        acoustic=acoustic,
        interaction=interaction,
        windows=windows,
    )


def save_feature_timeline(timeline: FeatureTimeline, output_path: str | Path) -> Path:
    """Ghi nguyên tử dữ liệu feature thô để chấm điểm, hiển thị và tái lập kết quả"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path
