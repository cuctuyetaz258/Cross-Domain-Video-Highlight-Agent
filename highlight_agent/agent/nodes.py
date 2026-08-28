"""Logic năm node LangGraph tích hợp đa tầng tín hiệu"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from highlight_agent.backend import (
    load_transcript,
    prepare_video,
    refine_candidates_for_render,
    render_candidates,
)
from highlight_agent.features import (
    PROFILE_WEIGHTS,
    WindowVisualScore,
    build_feature_timeline,
    calculate_total_score,
    extract_interaction_features,
    extract_visual_scores,
    extract_windowed_acoustic_features,
    extract_windowed_semantic_features,
    normalize_features,
    save_feature_timeline,
    windowed_interaction_features,
)
from highlight_agent.schemas import HighlightCandidate, VisualFeatures

from .state import AgentState, ProgressEvent, ReasoningEntry

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helper: Emit progress events
# ──────────────────────────────────────────────

def _emit(state: AgentState, node: str, step: str, message: str, **meta) -> None:
    """Gửi sự kiện tiến độ ra ngoài nếu có emit callback được cung cấp."""
    emit_fn = state.get("emit")
    if emit_fn:
        try:
            emit_fn(ProgressEvent(node=node, step=step, message=message, meta=meta))
        except Exception:
            pass


# ──────────────────────────────────────────────
# Node 1: Observe
# ──────────────────────────────────────────────

def observe(state: AgentState) -> dict:
    """Chuẩn hóa media và đưa transcript vào state"""

    _emit(state, "observe", "start", f"Chuẩn hóa video: {state['video_path']}")
    workspace = prepare_video(
        state["video_path"],
        output_root=state.get("output_root"),
        cookies_browser=state.get("cookies_browser"),
        transcript_source=state.get("transcript_source", "auto"),
    )
    transcript = load_transcript(workspace.transcript_path)
    _emit(state, "observe", "done", f"Đã chuẩn hóa video (duration={transcript.duration:.1f}s)")
    return {
        "workspace": workspace,
        "transcript": transcript,
    }


# ──────────────────────────────────────────────
# Node 2: Plan
# ──────────────────────────────────────────────

def plan(state: AgentState) -> dict:
    """Chọn profile tín hiệu theo domain"""

    _emit(state, "plan", "start", f"Lập kế hoạch cho domain={state['domain']}")
    domain = state["domain"]
    if domain not in PROFILE_WEIGHTS:
        raise ValueError(f"unsupported domain: {domain}")
    profile = dict(PROFILE_WEIGHTS[domain])
    _emit(state, "plan", "done", f"Profile weights: {profile}")
    return {"profile": profile}


# ──────────────────────────────────────────────
# Visual feature extraction
# ──────────────────────────────────────────────

def _extract_visual_windows(
    state: AgentState, window_seconds: float, duration: float
) -> list[WindowVisualScore]:
    """Trích xuất visual motion cùng cửa sổ với các tầng feature khác"""
    workspace = state.get("workspace")
    transcript = state.get("transcript")
    if workspace is None or transcript is None:
        raise ValueError("Analyze requires workspace and transcript from Observe")
    if transcript.duration < 30:
        raise ValueError("video must be at least 30 seconds to create an MVP highlight")

    visual_method = state.get("visual_method", "pixel_diff")
    sample_fps = state.get("visual_sample_fps", 1.0)
    video_path = workspace.source_video_path

    _emit(
        state,
        "analyze",
        "visual_start",
        f"Trích xuất visual motion ({visual_method}, {sample_fps} fps)...",
    )

    t0 = time.perf_counter()

    def on_window(score: WindowVisualScore) -> None:
        _emit(
            state,
            "analyze",
            "visual_window",
            f"[{score.start:.0f}s–{score.end:.0f}s] motion={score.motion_score:.3f}",
            start=score.start,
            end=score.end,
            score=score.motion_score,
        )

    raw_scores = extract_visual_scores(
        video_path=video_path,
        window_size=window_seconds,
        sample_fps=sample_fps,
        method=visual_method,
        on_window=on_window,
        duration=duration,
    )
    elapsed = time.perf_counter() - t0

    if not raw_scores:
        raise ValueError("Không trích xuất được visual score nào")

    _emit(
        state,
        "analyze",
        "visual_done",
        f"Trích xuất {len(raw_scores)} visual windows ({elapsed:.1f}s)",
        count=len(raw_scores),
        elapsed=elapsed,
    )

    return raw_scores


def _acoustic_window_scores(acoustic_windows) -> list[float]:
    """Quy đổi feature âm học thô thành một signal emphasis theo cửa sổ"""

    return [
        float(
            0.55 * window.acoustic.rms_p95
            + 0.25 * min((window.acoustic.pitch_std_hz or 0.0) / 100.0, 1.0)
            + 0.15 * window.acoustic.voiced_ratio
            + 0.05 * (1.0 - window.acoustic.silence_ratio)
        )
        for window in acoustic_windows
    ]


def _interaction_window_scores(interaction_windows) -> list[float]:
    """Quy đổi turn-taking thành signal tương tác theo cửa sổ"""

    if interaction_windows is None:
        return []
    return [
        float(
            0.50 * min(window.turn_rate_per_minute / 8.0, 1.0)
            + 0.30 * window.speech_ratio
            + 0.20 * min(window.speaker_count / 2.0, 1.0)
        )
        for window in interaction_windows
    ]


def _multimodal_candidates(state: AgentState, timeline) -> list[HighlightCandidate]:
    """Chuẩn hóa và fusion các signal để tạo candidate có evidence"""

    raw_signals: dict[str, list[float]] = {
        "semantic": [window.semantic.raw_score if window.semantic else 0.0 for window in timeline.windows],
        "acoustic": _acoustic_window_scores(timeline.windows),
        "visual": [window.visual.motion_score if window.visual else 0.0 for window in timeline.windows],
    }
    if state["domain"] == "podcast":
        raw_signals["interaction"] = _interaction_window_scores(
            [window.interaction for window in timeline.windows]
        )

    normalized = normalize_features(raw_signals, scaler_type="robust")
    scores = calculate_total_score(
        normalized,
        state["profile"],
        window_starts=[window.start for window in timeline.windows],
        window_ends=[window.end for window in timeline.windows],
    )
    candidates: list[HighlightCandidate] = []
    for score in scores:
        if score.end - score.start < 30.0:
            continue
        semantic = timeline.windows[score.window_idx].semantic
        evidence = ", ".join(semantic.cue_phrases) if semantic and semantic.cue_phrases else "không có cue phrase"
        candidates.append(
            HighlightCandidate(
                candidate_id=f"fusion_{score.window_idx + 1:02d}",
                start_time=score.start,
                end_time=score.end,
                score=round(score.total_score * 10.0, 3),
                reason=(
                    f"Multimodal fusion score={score.total_score:.3f}; "
                    f"semantic cue: {evidence}"
                ),
                signals={
                    **{name: round(value, 6) for name, value in score.signals_normalized.items()},
                    **{f"{name}_raw": round(raw_signals[name][score.window_idx], 6) for name in score.signals_normalized},
                },
            )
        )
    if not candidates:
        raise ValueError("Không có cửa sổ đủ 30 giây để tạo candidate")
    return candidates


# ──────────────────────────────────────────────
# Naive baseline (fallback)
# ──────────────────────────────────────────────

def _naive_candidates(state: AgentState, count: int = 5) -> list[HighlightCandidate]:
    transcript = state.get("transcript")
    workspace = state.get("workspace")
    if transcript is None or workspace is None:
        raise ValueError("Analyze requires workspace and transcript from Observe")
    if transcript.duration < 30:
        raise ValueError("video must be at least 30 seconds to create an MVP highlight")

    clip_duration = min(60.0, transcript.duration)
    max_start = max(0.0, transcript.duration - clip_duration)
    randomizer = random.Random(workspace.video_id)
    candidates = []
    for index in range(count):
        start = round(randomizer.uniform(0, max_start), 3) if max_start else 0.0
        end = round(start + clip_duration, 3)
        score = round(randomizer.uniform(2.0, 5.0), 3)
        candidates.append(
            HighlightCandidate(
                candidate_id=f"baseline_{index + 1:02d}",
                start_time=start,
                end_time=end,
                score=score,
                reason="Sprint 1 deterministic naive baseline; replace with LLM/features later.",
                signals={"baseline_random": round(score / 5, 3)},
            )
        )
    return candidates


# ──────────────────────────────────────────────
# Node 3: Analyze
# ──────────────────────────────────────────────

def analyze(state: AgentState) -> dict:
    """Trích xuất features đa tầng, fusion để tạo candidate hoặc fallback an toàn"""

    _emit(state, "analyze", "start", "Bắt đầu phân tích đặc trưng...")
    workspace = state.get("workspace")
    if workspace is None:
        raise ValueError("Analyze requires workspace from Observe")

    window_seconds = 30.0
    hop_seconds = 30.0

    # 1. Acoustic features
    acoustic, acoustic_windows = extract_windowed_acoustic_features(
        workspace.audio_path,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    features: dict[str, object] = {
        "acoustic": acoustic.model_dump(mode="json"),
        "profile": state.get("profile", {}),
    }

    # 2. Interaction features
    if state["domain"] == "podcast":
        interaction = extract_interaction_features(
            workspace.audio_path,
            num_speakers=state.get("known_speaker_count"),
        )
        features["interaction"] = interaction.model_dump(mode="json")
        interaction_windows = windowed_interaction_features(
            interaction,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
    else:
        interaction = None
        interaction_windows = None

    # 3. Semantic và visual dùng cùng schedule của acoustic để fusion từng cửa sổ
    supplied_candidates = state.get("candidates")
    try:
        transcript = state.get("transcript")
        if transcript is None:
            raise ValueError("Analyze requires transcript from Observe")
        _emit(state, "analyze", "semantic_start", "Trích xuất semantic evidence từ transcript...")
        semantic_scores = extract_windowed_semantic_features(
            transcript,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            duration=acoustic.duration,
        )
        semantic_windows = [score.features for score in semantic_scores]
        _emit(state, "analyze", "semantic_done", f"Trích xuất {len(semantic_windows)} semantic windows", count=len(semantic_windows))

        visual_scores = _extract_visual_windows(state, window_seconds, acoustic.duration)
        visual_windows = [
            VisualFeatures(
                motion_score=score.motion_score,
                method=score.method,
                frame_count=score.frame_count,
            )
            for score in visual_scores
        ]
        timeline = build_feature_timeline(
            video_id=workspace.video_id,
            domain=state["domain"],
            duration=acoustic.duration,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            acoustic=acoustic,
            acoustic_windows=acoustic_windows,
            interaction=interaction,
            interaction_windows=interaction_windows,
            semantic_windows=semantic_windows,
            visual_windows=visual_windows,
        )
        if supplied_candidates:
            candidates = [HighlightCandidate.model_validate(item) for item in supplied_candidates]
            mode = "external_candidates"
        else:
            _emit(state, "analyze", "fusion_start", "Chuẩn hóa các signal và chấm điểm fusion...")
            candidates = _multimodal_candidates(state, timeline)
            mode = "multimodal_fusion"
            _emit(state, "analyze", "fusion_done", f"Đã tạo {len(candidates)} fusion candidates", count=len(candidates))
    except Exception as exc:
        logger.warning("Semantic/visual fusion thất bại (%s), fallback sang naive baseline.", exc)
        _emit(state, "analyze", "fallback", f"Semantic/visual fusion thất bại ({exc}), dùng naive baseline")
        timeline = build_feature_timeline(
            video_id=workspace.video_id,
            domain=state["domain"],
            duration=acoustic.duration,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            acoustic=acoustic,
            acoustic_windows=acoustic_windows,
            interaction=interaction,
            interaction_windows=interaction_windows,
        )
        candidates = _naive_candidates(state)
        mode = "naive_baseline"

    feature_path = save_feature_timeline(
        timeline,
        workspace.audio_path.parent / "features" / "features.json",
    )
    features.update(
        {
            "feature_path": str(feature_path),
            "window_count": len(timeline.windows),
            "window_seconds": window_seconds,
            "visual_method": state.get("visual_method", "pixel_diff"),
        }
    )

    # ── LTR Dense Overlap branch (if model path provided) ──
    _ltr_model_path = state.get("ltr_model_path")
    if _ltr_model_path and Path(_ltr_model_path).exists():
        _emit(state, "analyze", "ltr_start", "Running LTR dense scoring...")
        try:
            import torch as _torch

            from highlight_agent.features.alignment import build_feature_matrix as _build_feature_matrix
            from highlight_agent.features.nms_topk import extract_topk_nms as _extract_topk_nms
            from highlight_agent.features.overlap_blender import blend_scores as _blend_scores
            from highlight_agent.features.semantic import (
                transcript_tfidf_density_scores as _transcript_tfidf_density_scores,
            )
            from highlight_agent.features.sliding_window import extract_windows as _extract_windows
            from highlight_agent.features.visual_new import (
                extract_gesture_signal as _extract_gesture_signal,
            )
            from highlight_agent.features.visual_new import (
                extract_scene_changes as _extract_scene_changes,
            )
            from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer as _LTRScorer

            _scene_times = _extract_scene_changes(workspace.source_video_path, acoustic.duration)
            _gesture_sparse = _extract_gesture_signal(workspace.source_video_path, acoustic.duration)

            _transcript = state.get("transcript")
            _word_scores = (
                _transcript_tfidf_density_scores(_transcript) if _transcript is not None else []
            )

            _feature_matrix = _build_feature_matrix(
                acoustic, acoustic_windows, _scene_times, _gesture_sparse,
                _word_scores, interaction, acoustic.duration
            )
            _ltr_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            _window_tensor = _extract_windows(_feature_matrix, device=_ltr_device)
            _ltr_model, _checkpoint_metadata = _LTRScorer.load_checkpoint(
                _ltr_model_path,
                device=_ltr_device,
                expected_in_features=7,
            )
            with _torch.no_grad():
                _raw_scores = _ltr_model(_window_tensor).squeeze(-1)
            _window_scores = _raw_scores.cpu().numpy()
            _timeline_score = _blend_scores(_window_scores, T=_feature_matrix.shape[1])
            _l_ref = _checkpoint_metadata.get("L_ref", 40.0)
            _k = max(3, min(5, state.get("highlight_count", 3)))
            _ltr_candidates = _extract_topk_nms(_timeline_score, k=_k, reference_duration=float(_l_ref))
            if len(_ltr_candidates) >= _k:
                candidates = _ltr_candidates
                mode = "ltr_dense_overlap"
                _emit(
                    state,
                    "analyze",
                    "ltr_done",
                    f"LTR produced {len(_ltr_candidates)} candidates on {_ltr_device.type}",
                    device=_ltr_device.type,
                    count=len(_ltr_candidates),
                )
            else:
                _emit(
                    state,
                    "analyze",
                    "ltr_fallback",
                    f"LTR produced {len(_ltr_candidates)}/{_k} candidates; using baseline",
                    count=len(_ltr_candidates),
                    required=_k,
                )
        except Exception as _ltr_exc:  # noqa: BLE001
            logger.warning("LTR scoring failed, fallback to baseline: %s", _ltr_exc)
            _emit(state, "analyze", "ltr_fallback", f"LTR error, using baseline: {_ltr_exc}")

    _emit(state, "analyze", "done", f"mode={mode} | {len(candidates)} candidates", mode=mode, count=len(candidates))

    return {
        "features": {"mode": mode, "candidate_count": len(candidates), **features},
        "feature_path": str(feature_path),
        "feature_timeline": timeline.model_dump(mode="json"),
        "candidates": candidates,
    }


# ──────────────────────────────────────────────
# Node 4: Decide
# ──────────────────────────────────────────────

def decide(state: AgentState) -> dict:
    """Xếp hạng, canh biên và gọi Backend render"""

    _emit(state, "decide", "start", "Xếp hạng và canh biên highlights...")
    workspace = state.get("workspace")
    if workspace is None:
        raise ValueError("Decide requires workspace from Observe")
    candidates = state.get("candidates") or []
    highlight_count = state.get("highlight_count", 3)
    if not 3 <= highlight_count <= 5:
        raise ValueError("highlight_count must be between 3 and 5")
    if len(candidates) < highlight_count:
        raise ValueError("not enough candidates to satisfy highlight_count")

    selected = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:highlight_count]
    _emit(state, "decide", "ranking", f"Đã chọn top {highlight_count} highlights. Tiến hành canh biên...")

    highlights, boundary_adjustments = refine_candidates_for_render(workspace, selected)

    _emit(state, "decide", "rendering", f"Bắt đầu render {len(highlights)} clips...")
    rendered = render_candidates(
        workspace,
        highlights,
        burn_subtitles=state.get("burn_subtitles", True),
        boundary_adjustments=boundary_adjustments,
        refine_boundaries=False,
    )
    _emit(state, "decide", "done", f"Render xong {len(rendered)} clips.", rendered_count=len(rendered))

    return {
        "highlights": highlights,
        "boundary_adjustments": boundary_adjustments,
        "rendered_highlights": rendered,
    }


# ──────────────────────────────────────────────
# Node 5: Explain
# ──────────────────────────────────────────────

def explain(state: AgentState) -> dict:
    """Tạo reasoning minh bạch cho kết quả highlight"""

    domain = state["domain"]
    profile = state.get("profile", {})
    mode = state.get("features", {}).get("mode", "unknown")
    reasoning: list[ReasoningEntry] = []
    for rank, candidate in enumerate(state.get("highlights", []), start=1):
        reasoning.append(
            {
                "candidate_id": candidate.candidate_id,
                "explanation": (
                    f"Rank #{rank}: {candidate.start_time:.2f}s–{candidate.end_time:.2f}s, "
                    f"score={candidate.score:.3f}, domain={domain}, mode={mode}. "
                    f"Reason: {candidate.reason} Active profile: {profile}."
                ),
            }
        )
    _emit(state, "explain", "done", f"Pipeline hoàn tất! {len(reasoning)} reasoning entries.", count=len(reasoning))
    return {"reasoning": reasoning}
