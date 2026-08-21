"""Logic năm node LangGraph — Sprint 2: Visual Scoring + Live Progress"""

import logging
import random
import time

from highlight_agent.backend import load_transcript, prepare_video, render_candidates
from highlight_agent.features.scoring import normalize_features
from highlight_agent.features.visual import extract_visual_scores, scores_to_array
from highlight_agent.schemas import HighlightCandidate

from .state import AgentState, Domain, EmitFn, ProgressEvent, ReasoningEntry, SignalProfile

logger = logging.getLogger(__name__)

PROFILE_WEIGHTS: dict[Domain, SignalProfile] = {
    "lecture": {
        "acoustic": 0.30,
        "paralinguistic": 0.00,
        "linguistic": 0.50,
        "structural": 0.20,
        "interaction": 0.00,
    },
    "podcast": {
        "acoustic": 0.20,
        "paralinguistic": 0.10,
        "linguistic": 0.30,
        "structural": 0.10,
        "interaction": 0.30,
    },
    "standup": {
        "acoustic": 0.35,
        "paralinguistic": 0.35,
        "linguistic": 0.20,
        "structural": 0.10,
        "interaction": 0.00,
    },
}


# ──────────────────────────────────────────────
# Helper: emit tiến độ an toàn
# ──────────────────────────────────────────────

def _emit(state: AgentState, node: str, step: str, message: str, **meta):
    """Gọi emit callback nếu có, an toàn (không raise)."""
    emit = state.get("emit")
    if emit:
        try:
            emit(ProgressEvent(node=node, step=step, message=message, meta=meta))
        except Exception:
            pass  # emit lỗi không được crash pipeline


# ──────────────────────────────────────────────
# Node 1: Observe
# ──────────────────────────────────────────────

def observe(state: AgentState) -> dict:
    """Chuẩn hóa media và đưa transcript vào state"""

    _emit(state, "observe", "start",
          "Đang chuẩn bị video và trích xuất transcript...")

    workspace = prepare_video(
        state["video_path"],
        output_root=state.get("output_root"),
        cookies_browser=state.get("cookies_browser"),
        transcript_source=state.get("transcript_source", "auto"),
    )
    transcript = load_transcript(workspace.transcript_path)

    _emit(state, "observe", "done",
          f"Video: {workspace.video_id} | Transcript: {transcript.duration:.0f}s",
          video_id=workspace.video_id, duration=transcript.duration)

    return {
        "workspace": workspace,
        "transcript": transcript,
    }


# ──────────────────────────────────────────────
# Node 2: Plan
# ──────────────────────────────────────────────

def plan(state: AgentState) -> dict:
    """Chọn profile tín hiệu theo domain"""

    domain = state["domain"]
    if domain not in PROFILE_WEIGHTS:
        raise ValueError(f"unsupported domain: {domain}")
    profile = dict(PROFILE_WEIGHTS[domain])

    active = {k: v for k, v in profile.items() if v > 0}
    _emit(state, "plan", "done",
          f"Domain={domain} | Active signals: {', '.join(active.keys())}",
          domain=domain, profile=profile)

    return {"profile": profile}


# ──────────────────────────────────────────────
# Visual Candidates (thay thế naive baseline)
# ──────────────────────────────────────────────

_CLIP_DURATION = 60.0  # HighlightCandidate yêu cầu 30–90s


def _visual_candidates(state: AgentState, count: int = 5) -> list[HighlightCandidate]:
    """Tạo candidates từ visual motion score (pixel_diff hoặc RAFT).

    Flow:
        1. Chạy extract_visual_scores() theo sliding window 60s.
        2. Chuẩn hoá motion_score về [0, 1] qua normalize_features().
        3. Lấy top-N cửa sổ có điểm cao nhất → HighlightCandidate.
    """
    workspace = state.get("workspace")
    transcript = state.get("transcript")
    if workspace is None or transcript is None:
        raise ValueError("Cần workspace và transcript từ Observe")
    if transcript.duration < 30:
        raise ValueError("Video phải dài ít nhất 30 giây")

    visual_method = state.get("visual_method", "pixel_diff")
    sample_fps = state.get("visual_sample_fps", 1.0)

    _emit(state, "analyze", "visual_start",
          f"Bắt đầu visual scoring ({visual_method}, sample_fps={sample_fps})...",
          method=visual_method, sample_fps=sample_fps)

    t0 = time.time()

    # Callback: emit mỗi khi 1 cửa sổ xử lý xong
    def on_window(w):
        _emit(state, "analyze", "visual_window",
              f"[{visual_method}] {w.start:.0f}s–{w.end:.0f}s → score={w.motion_score:.3f}",
              start=w.start, end=w.end, score=w.motion_score, **w.extra)

    raw_scores = extract_visual_scores(
        video_path=workspace.source_video_path,
        window_size=_CLIP_DURATION,
        step_size=_CLIP_DURATION,
        sample_fps=sample_fps,
        method=visual_method,
        on_window=on_window,
    )

    if not raw_scores:
        raise ValueError("Không trích xuất được visual score nào")

    elapsed = time.time() - t0
    _emit(state, "analyze", "visual_normalize",
          f"Chuẩn hoá {len(raw_scores)} cửa sổ... ({elapsed:.1f}s)")

    # Chuẩn hoá về [0, 1]
    motion_array = scores_to_array(raw_scores).tolist()
    normed = normalize_features({"visual": motion_array})
    normed_scores = normed["visual"].tolist()

    # Sort lấy top-N
    scored_windows = sorted(
        zip(raw_scores, normed_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    candidates = []
    for idx, (window, norm_score) in enumerate(scored_windows[:count]):
        candidates.append(
            HighlightCandidate(
                candidate_id=f"visual_{idx + 1:02d}",
                start_time=window.start,
                end_time=window.end,
                score=round(norm_score * 5, 3),  # scale về [0, 5] tương thích
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

    _emit(state, "analyze", "visual_done",
          f"Top {len(candidates)} candidates (scoring took {elapsed:.1f}s)",
          count=len(candidates), elapsed=elapsed)

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
    """Dùng candidate có sẵn, hoặc chạy visual scoring, fallback naive baseline"""

    _emit(state, "analyze", "start", "Bắt đầu phân tích...")

    supplied_candidates = state.get("candidates")
    if supplied_candidates:
        candidates = [HighlightCandidate.model_validate(item) for item in supplied_candidates]
        mode = "external_candidates"
        _emit(state, "analyze", "done",
              f"Dùng {len(candidates)} external candidates.")
    else:
        # Thử visual scoring trước, fallback naive baseline
        try:
            candidates = _visual_candidates(state)
            visual_method = state.get("visual_method", "pixel_diff")
            mode = f"visual_{visual_method}"
        except Exception as exc:
            logger.warning(
                "Visual scoring thất bại (%s), fallback sang naive baseline.", exc
            )
            _emit(state, "analyze", "fallback",
                  f"⚠️ Visual thất bại ({exc}), dùng naive baseline.")
            candidates = _naive_candidates(state)
            mode = "naive_baseline"

    _emit(state, "analyze", "done",
          f"mode={mode} | {len(candidates)} candidates",
          mode=mode, count=len(candidates))

    return {
        "features": {
            "mode": mode,
            "candidate_count": len(candidates),
            "profile": state.get("profile", {}),
            "visual_method": state.get("visual_method", "pixel_diff"),
        },
        "candidates": candidates,
    }


# ──────────────────────────────────────────────
# Node 4: Decide
# ──────────────────────────────────────────────

def decide(state: AgentState) -> dict:
    """Xếp hạng, lấy top K và gọi Backend render"""

    workspace = state.get("workspace")
    if workspace is None:
        raise ValueError("Decide requires workspace from Observe")
    candidates = state.get("candidates") or []
    highlight_count = state.get("highlight_count", 3)
    if not 3 <= highlight_count <= 5:
        raise ValueError("highlight_count must be between 3 and 5")
    if len(candidates) < highlight_count:
        raise ValueError("not enough candidates to satisfy highlight_count")

    highlights = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:highlight_count]

    _emit(state, "decide", "ranking",
          f"Top {highlight_count} highlights chọn xong. Đang render video...",
          count=highlight_count)

    rendered = render_candidates(
        workspace,
        highlights,
        burn_subtitles=state.get("burn_subtitles", True),
    )

    _emit(state, "decide", "done",
          f"Render xong {len(rendered)} clips.",
          rendered_count=len(rendered))

    return {
        "highlights": highlights,
        "rendered_highlights": rendered,
    }


# ──────────────────────────────────────────────
# Node 5: Explain
# ──────────────────────────────────────────────

def explain(state: AgentState) -> dict:
    """Tạo reasoning minh bạch cho baseline hiện tại"""

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

    _emit(state, "explain", "done",
          f"Pipeline hoàn tất! {len(reasoning)} reasoning entries.",
          count=len(reasoning))

    return {"reasoning": reasoning}
