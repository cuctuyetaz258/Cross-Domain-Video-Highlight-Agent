"""Logic năm node LangGraph cho baseline Sprint 1"""

import random

from highlight_agent.backend import load_transcript, prepare_video, render_candidates
from highlight_agent.schemas import HighlightCandidate

from .state import AgentState, Domain, ReasoningEntry, SignalProfile

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


def observe(state: AgentState) -> dict:
    """Chuẩn hóa media và đưa transcript vào state"""

    workspace = prepare_video(
        state["video_path"],
        output_root=state.get("output_root"),
        cookies_browser=state.get("cookies_browser"),
        transcript_source=state.get("transcript_source", "auto"),
    )
    return {
        "workspace": workspace,
        "transcript": load_transcript(workspace.transcript_path),
    }


def plan(state: AgentState) -> dict:
    """Chọn profile tín hiệu theo domain"""

    domain = state["domain"]
    if domain not in PROFILE_WEIGHTS:
        raise ValueError(f"unsupported domain: {domain}")
    return {"profile": dict(PROFILE_WEIGHTS[domain])}


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


def analyze(state: AgentState) -> dict:
    """Dùng candidate có sẵn hoặc tạo naive baseline ổn định"""

    supplied_candidates = state.get("candidates")
    if supplied_candidates:
        candidates = [HighlightCandidate.model_validate(item) for item in supplied_candidates]
        mode = "external_candidates"
    else:
        candidates = _naive_candidates(state)
        mode = "naive_baseline"
    return {
        "features": {
            "mode": mode,
            "candidate_count": len(candidates),
            "profile": state.get("profile", {}),
        },
        "candidates": candidates,
    }


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
    rendered = render_candidates(
        workspace,
        highlights,
        burn_subtitles=state.get("burn_subtitles", True),
    )
    return {
        "highlights": highlights,
        "rendered_highlights": rendered,
    }


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
    return {"reasoning": reasoning}
