"""Logic năm node LangGraph tích hợp đa tầng tín hiệu"""

from __future__ import annotations

import logging
import random
import time

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
    extract_interaction_features,
    extract_visual_scores,
    extract_windowed_acoustic_features,
    normalize_features,
    save_feature_timeline,
    scores_to_array,
    windowed_interaction_features,
)
from highlight_agent.schemas import HighlightCandidate

from .state import AgentState, Domain, ProgressEvent, ReasoningEntry, SignalProfile

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
# Visual candidate generator
# ──────────────────────────────────────────────

def _visual_candidates(state: AgentState) -> list[HighlightCandidate]:
    """Tạo highlight candidates dựa trên visual motion scoring."""
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
        window_size=30.0,
        sample_fps=sample_fps,
        method=visual_method,
        on_window=on_window,
    )
    elapsed = time.perf_counter() - t0

    if not raw_scores:
        raise ValueError("Không trích xuất được visual score nào từ video.")

    _emit(state, "analyze", "visual_normalize", "Chuẩn hóa và tính điểm...")
    motion_arr = scores_to_array(raw_scores)
    normed = normalize_features({"visual": motion_arr})
    norm_motion = normed["visual"]

    candidates: list[HighlightCandidate] = []
    for idx, window in enumerate(raw_scores):
        norm_score = float(norm_motion[idx])
        score_10 = round(norm_score * 10.0, 2)
        candidates.append(
            HighlightCandidate(
                candidate_id=f"vis_{idx + 1:02d}",
                start_time=window.start,
                end_time=window.end,
                score=score_10,
                reason=(
                    f"Visual motion score ({visual_method}): "
                    f"raw={window.motion_score:.4f}, normalized={norm_score:.4f}. "
                    f"Cửa sổ {window.start:.1f}s–{window.end:.1f}s."
                ),
                signals={
                    "visual_motion_raw": round(window.motion_score, 4),
                    "visual_motion_norm": round(norm_score, 4),
                    **{k: float(v) for k, v in window.extra.items() if isinstance(v, (int, float))},
                },
            )
        )

    _emit(
        state,
        "analyze",
        "visual_done",
        f"Trích xuất {len(candidates)} visual windows ({elapsed:.1f}s)",
        count=len(candidates),
        elapsed=elapsed,
    )

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
    """Trích xuất features đa tầng (acoustic, interaction, visual), xây dựng timeline và tạo candidates"""

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

    # 3. Timeline
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

    # 4. Candidates: Supplied > Visual/Multi-modal Scoring > Naive Baseline fallback
    supplied_candidates = state.get("candidates")
    if supplied_candidates:
        candidates = [HighlightCandidate.model_validate(item) for item in supplied_candidates]
        mode = "external_candidates"
        _emit(state, "analyze", "done", f"Dùng {len(candidates)} external candidates.")
    else:
        try:
            candidates = _visual_candidates(state)
            visual_method = state.get("visual_method", "pixel_diff")
            mode = f"visual_{visual_method}"
        except Exception as exc:
            logger.warning("Visual scoring thất bại (%s), fallback sang naive baseline.", exc)
            _emit(state, "analyze", "fallback", f"⚠️ Visual scoring thất bại ({exc}), dùng naive baseline.")
            candidates = _naive_candidates(state)
            mode = "naive_baseline"

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
