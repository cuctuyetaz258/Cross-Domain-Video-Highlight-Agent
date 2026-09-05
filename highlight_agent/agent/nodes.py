"""Logic năm node LangGraph tích hợp đa tầng tín hiệu"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace

import numpy as np
import torch

from highlight_agent.backend import (
    load_transcript,
    prepare_video,
    refine_candidates_for_render,
    render_candidates,
)
from highlight_agent.features import (
    build_feature_timeline,
    save_feature_timeline,
    windowed_interaction_features,
)
from highlight_agent.features.ltr_contract import (
    LTR_HOP_SIZE,
    LTR_WINDOW_SIZE,
    LTRPipelineError,
)
from highlight_agent.features.ltr_pipeline import build_ltr_features
from highlight_agent.features.nms_topk import extract_topk_nms
from highlight_agent.features.overlap_blender import blend_scores
from highlight_agent.features.sliding_window import extract_windows
from highlight_agent.llm import (
    FUSION_METHOD,
    PROMPT_VERSION,
    FusionCalibrator,
    LLMClientConfig,
    OpenAICompatibleAssessmentClient,
    apply_validated_boundaries,
    rerank_candidates,
)
from highlight_agent.models.actionformer import (
    actionformer_checkpoint_info,
    decode_proposals,
    load_actionformer_checkpoint,
    soft_nms,
)
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.proposal_ltr import ProposalLTRConfig, build_proposal_ltr
from highlight_agent.schemas import (
    HighlightCandidate,
    LLMHighlightAssessment,
    LLMRunInfo,
)

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


def preflight(state: AgentState) -> dict:
    """Validate the required checkpoint before any media download or transcription."""

    scorer_type = state.get("scorer_type", "legacy-ltr")
    _emit(state, "preflight", "start", f"Validating required {scorer_type} checkpoint...")
    if scorer_type == "actionformer-ltr":
        path = state.get("actionformer_model_path")
        if not path:
            raise LTRPipelineError(
                "ACTIONFORMER_CHECKPOINT_REQUIRED",
                "provide --actionformer-model-path when scorer-type is actionformer-ltr",
            )
        try:
            info = actionformer_checkpoint_info(path)
        except Exception as exc:
            raise LTRPipelineError("ACTIONFORMER_CHECKPOINT_LOAD_FAILED", str(exc)) from exc
        if not info["has_proposal_ltr"]:
            raise LTRPipelineError(
                "ACTIONFORMER_LTR_HEAD_MISSING",
                "checkpoint contains localization weights but no proposal LTR head",
            )
        info["device"] = (
            "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        )
    elif scorer_type == "legacy-ltr":
        info = AdditiveAttentionScorer.preflight(state.get("ltr_model_path"))
    else:
        raise ValueError(f"unsupported scorer_type: {scorer_type}")
    _emit(
        state,
        "preflight",
        "done",
        (
            f"{scorer_type} checkpoint valid | device={info['device']} | "
            f"sha256={info['fingerprint'][:12]}"
        ),
        device=info["device"],
        fingerprint=info["fingerprint"],
    )
    return {"ltr_checkpoint_info": info}


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
    """Declare extractor behavior; the domain never selects scorer weights."""

    _emit(state, "plan", "start", f"Lập kế hoạch cho domain={state['domain']}")
    domain = state["domain"]
    if domain not in {"lecture", "podcast", "standup"}:
        raise ValueError(f"unsupported domain: {domain}")
    scorer_type = state.get("scorer_type", "legacy-ltr")
    analysis_plan = {
        "scorer": "actionformer_ltr" if scorer_type == "actionformer-ltr" else "ltr_required",
        "scene_extractor": "scenedetect",
        "gesture_extractor": "mediapipe",
        "interaction_extractor": "pyannote" if domain == "podcast" else "zero_channel",
    }
    _emit(state, "plan", "done", f"LTR extraction plan: {analysis_plan}")
    return {"analysis_plan": analysis_plan}


# ──────────────────────────────────────────────
# Node 3: Analyze
# ──────────────────────────────────────────────

def analyze(state: AgentState) -> dict:
    """Run unified features through the selected deterministic scorer."""

    _emit(state, "analyze", "start", "Building unified seven-channel LTR features...")
    workspace = state.get("workspace")
    transcript = state.get("transcript")
    if workspace is None or transcript is None:
        raise LTRPipelineError(
            "LTR_STATE_INCOMPLETE",
            "Analyze requires workspace and transcript from Observe",
        )
    if transcript.duration < 30.0:
        raise LTRPipelineError(
            "LTR_VIDEO_TOO_SHORT",
            "video must be at least 30 seconds to render a highlight",
        )

    scorer_type = state.get("scorer_type", "legacy-ltr")
    if scorer_type == "actionformer-ltr":
        checkpoint_info = actionformer_checkpoint_info(state.get("actionformer_model_path") or "")
        checkpoint_info["device"] = (
            "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        )
    else:
        checkpoint_info = AdditiveAttentionScorer.preflight(state.get("ltr_model_path"))
    prior_info = state.get("ltr_checkpoint_info")
    if prior_info and prior_info.get("fingerprint") != checkpoint_info["fingerprint"]:
        raise LTRPipelineError(
            "LTR_CHECKPOINT_CHANGED",
            "checkpoint changed after preflight; restart the pipeline",
        )

    bundle = build_ltr_features(
        video_path=workspace.source_video_path,
        audio_path=workspace.audio_path,
        transcript=transcript,
        domain=state["domain"],
        duration=transcript.duration,
        known_speaker_count=state.get("known_speaker_count"),
        min_speaker_count=state.get("min_speaker_count"),
        max_speaker_count=state.get("max_speaker_count"),
        device=checkpoint_info["device"],
    )
    feature_dir = workspace.audio_path.parent / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = feature_dir / "feature_matrix.npy"
    matrix_tmp = feature_dir / "feature_matrix.npy.tmp"
    with matrix_tmp.open("wb") as handle:
        np.save(handle, bundle.matrix, allow_pickle=False)
    matrix_tmp.replace(matrix_path)
    report_path = feature_dir / "ltr_features.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(json.dumps(bundle.metadata, indent=2, sort_keys=True), encoding="utf-8")
    report_tmp.replace(report_path)

    interaction_windows = (
        windowed_interaction_features(
            bundle.interaction,
            acoustic_windows=bundle.acoustic_windows,
        )
        if bundle.interaction is not None
        else None
    )
    timeline = build_feature_timeline(
        video_id=workspace.video_id,
        domain=state["domain"],
        duration=bundle.acoustic.duration,
        window_seconds=30.0,
        hop_seconds=30.0,
        acoustic=bundle.acoustic,
        acoustic_windows=bundle.acoustic_windows,
        interaction=bundle.interaction,
        interaction_windows=interaction_windows,
    )
    feature_path = save_feature_timeline(
        timeline,
        feature_dir / "features.json",
    )

    device = torch.device(checkpoint_info["device"])
    try:
        if scorer_type == "actionformer-ltr":
            model, checkpoint_metadata, proposal_state = load_actionformer_checkpoint(
                checkpoint_info["path"],
                device=device,
            )
            if proposal_state is None:
                raise LTRPipelineError(
                    "ACTIONFORMER_LTR_HEAD_MISSING",
                    "checkpoint does not contain proposal LTR weights",
                )
            feature_tensor = torch.as_tensor(
                bundle.matrix,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                outputs = model(feature_tensor)
                proposals = decode_proposals(
                    outputs,
                    model.config,
                    video_durations=[transcript.duration],
                )
                scorer_payload = checkpoint_metadata.get("proposal_ltr_config")
                scorer_config = (
                    ProposalLTRConfig.from_dict(scorer_payload)
                    if isinstance(scorer_payload, dict)
                    else ProposalLTRConfig(architecture="mlp", d_model=128, dropout=0.1)
                )
                proposal_ltr = build_proposal_ltr(model.config.d_model, scorer_config).to(device)
                proposal_ltr.load_state_dict(proposal_state)
                proposal_ltr.eval()
                raw_scores, provenance = proposal_ltr(
                    outputs["features"][0],
                    [proposals],
                    stride_seconds=model.config.base_stride_seconds,
                )
            rank_probabilities = torch.sigmoid(raw_scores).tolist()
            for score, (_, proposal_index) in zip(rank_probabilities, provenance):
                proposals[proposal_index] = replace(proposals[proposal_index], rank_score=float(score))
            ranked_proposals = soft_nms(proposals, top_k=12)
            if ranked_proposals:
                raw = np.asarray([item.score for item in ranked_proposals], dtype=np.float32)
                score_min, score_max = float(raw.min()), float(raw.max())
                normalized = (raw - score_min) / (score_max - score_min) if score_max > score_min else np.ones_like(raw)
            else:
                normalized = np.asarray([], dtype=np.float32)
            candidates = [
                HighlightCandidate(
                    candidate_id=f"actionformer_{index + 1:02d}",
                    start_time=round(proposal.start, 3),
                    end_time=round(proposal.end, 3),
                    score=float(normalized[index]),
                    reason=(
                        f"ActionFormer proposal level={proposal.level}, "
                        f"confidence={proposal.confidence:.4f}, ltr={proposal.score:.4f}"
                    ),
                    signals={
                        "actionformer_confidence": proposal.confidence,
                        "ltr_score_raw": proposal.score,
                        "proposal_duration": proposal.duration,
                        "pyramid_level": float(proposal.level),
                    },
                )
                for index, proposal in enumerate(ranked_proposals)
            ]
        else:
            windows = extract_windows(
                bundle.matrix,
                window_size=LTR_WINDOW_SIZE,
                hop_size=LTR_HOP_SIZE,
                device=device,
            )
            model, checkpoint_metadata = AdditiveAttentionScorer.load_checkpoint(
                checkpoint_info["path"],
                device=device,
                expected_in_features=7,
            )
            with torch.no_grad():
                window_scores = model(windows).squeeze(-1).cpu().numpy()
            timeline_score = blend_scores(
                window_scores,
                T=bundle.matrix.shape[1],
                window_size=LTR_WINDOW_SIZE,
                hop_size=LTR_HOP_SIZE,
            )
            if not np.isfinite(timeline_score).all():
                raise LTRPipelineError("LTR_SCORE_NON_FINITE", "model produced NaN or Inf scores")
    except LTRPipelineError:
        raise
    except Exception as exc:
        raise LTRPipelineError("LTR_SCORING_FAILED", str(exc)) from exc

    highlight_count = state.get("highlight_count", 3)
    llm_enabled = state.get("llm_provider", "disabled") != "disabled"
    requested_pool = state.get("candidate_pool_size")
    if requested_pool is None:
        requested_pool = state.get("llm_top_m", 10) if llm_enabled else highlight_count
    pool_size = max(highlight_count, min(12, requested_pool))
    if scorer_type == "legacy-ltr":
        candidates = extract_topk_nms(
            timeline_score,
            k=pool_size,
            reference_duration=float(checkpoint_metadata["L_ref"]),
        )
    else:
        candidates = candidates[:pool_size]
    if len(candidates) < highlight_count:
        raise LTRPipelineError(
            "LTR_NOT_ENOUGH_CANDIDATES",
            f"deterministic NMS produced {len(candidates)} candidates; need {highlight_count}",
        )

    mode = "actionformer_ltr" if scorer_type == "actionformer-ltr" else "ltr_required"
    features = {
        "mode": mode,
        "candidate_count": len(candidates),
        "requested_pool_size": pool_size,
        "feature_path": str(feature_path),
        "feature_matrix_path": str(matrix_path),
        "feature_report_path": str(report_path),
        "feature_contract": bundle.metadata["feature_contract"],
        "extractor": bundle.metadata["extractor"],
        "observations": bundle.metadata["observations"],
        "channel_stats": bundle.metadata["channel_stats"],
        "checkpoint": checkpoint_info,
        "analysis_plan": state.get("analysis_plan", {}),
    }
    _emit(
        state,
        "analyze",
        "done",
        f"mode={mode} | {len(candidates)} candidates | device={device.type}",
        mode=mode,
        count=len(candidates),
        device=device.type,
    )
    return {
        "features": features,
        "feature_path": str(feature_path),
        "feature_timeline": timeline.model_dump(mode="json"),
        "candidates": candidates,
        "ltr_checkpoint_info": checkpoint_info,
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

    base_mode = state.get("features", {}).get("mode", "unknown")
    raw_provider = state.get("llm_provider", "auto")

    # Automatically resolve provider if set to 'auto' based on available environment keys
    if raw_provider == "auto":
        if os.environ.get("GROQ_API_KEY", "").strip():
            llm_provider = "groq"
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            llm_provider = "openai"
        elif os.environ.get("HIGHLIGHT_LLM_API_KEY", "").strip():
            llm_provider = "custom"
        else:
            llm_provider = "disabled"
    else:
        llm_provider = raw_provider

    llm_assessments = []
    llm_run = LLMRunInfo(enabled=False, applied=False)
    features = dict(state.get("features", {}))
    ranked_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    if llm_provider != "disabled":
        top_m = state.get("llm_top_m", 10)
        if not 3 <= top_m <= 12:
            raise ValueError("llm_top_m must be between 3 and 12")
        pool = ranked_candidates[: max(highlight_count, top_m)]
        _emit(
            state,
            "decide",
            "llm_start",
            f"LLM đang đánh giá ngữ nghĩa cho {len(pool)} candidates...",
            provider=llm_provider,
            count=len(pool),
        )
        try:
            config = LLMClientConfig.from_env(
                provider=llm_provider,
                model=state.get("llm_model"),
                base_url=state.get("llm_base_url"),
                timeout_seconds=state.get("llm_timeout_seconds", 45.0),
            )
            client = OpenAICompatibleAssessmentClient(config)
            calibrator_path = state.get("fusion_calibrator_path")
            if calibrator_path:
                calibrator = FusionCalibrator.load(
                    calibrator_path,
                    expected_checkpoint_fingerprint=state.get(
                        "ltr_checkpoint_info", {}
                    ).get("fingerprint"),
                    expected_llm_model=config.model,
                    expected_prompt_version=PROMPT_VERSION,
                )
            else:
                calibrator = FusionCalibrator(
                    alpha=0.5,
                    selection_metric="untrained_equal_rank_baseline",
                )
            ranked_candidates, llm_assessments, llm_run = rerank_candidates(
                pool,
                state["transcript"],
                domain=state["domain"],
                client=client,
                cache_dir=workspace.transcript_path.parent / "llm",
                ltr_weight=calibrator.alpha,
                checkpoint_fingerprint=state.get("ltr_checkpoint_info", {}).get(
                    "fingerprint", "unknown"
                ),
            )
            llm_run = llm_run.model_copy(
                update={
                    "fusion_method": FUSION_METHOD,
                    "fusion_alpha": calibrator.alpha,
                    "fusion_calibrator_path": calibrator.source_path,
                }
            )
            llm_mode = "ltr_llm_rerank"
            features.update({"base_mode": base_mode, "mode": llm_mode})
            _emit(
                state,
                "decide",
                "llm_done",
                f"LLM rerank hoàn tất ({len(llm_assessments)} assessments).",
                cache_hit=llm_run.cache_hit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM reranking failed, fallback to %s: %s", base_mode, exc)
            llm_run = LLMRunInfo(
                enabled=True,
                applied=False,
                provider=llm_provider,
                model=state.get("llm_model"),
                fallback_reason=str(exc),
            )
            ranked_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
            _emit(
                state,
                "decide",
                "llm_fallback",
                f"LLM không khả dụng; tiếp tục bằng {base_mode}: {exc}",
            )

    selected = ranked_candidates[:highlight_count]
    if llm_run.applied:
        selected, accepted_ids = apply_validated_boundaries(
            selected,
            llm_assessments,
            state["transcript"],
        )
        llm_run = llm_run.model_copy(
            update={"accepted_boundary_candidate_ids": accepted_ids}
        )
    _emit(state, "decide", "ranking", f"Đã chọn top {highlight_count} highlights. Tiến hành canh biên...")

    highlights, boundary_adjustments = refine_candidates_for_render(workspace, selected)

    _emit(state, "decide", "rendering", f"Bắt đầu render {len(highlights)} clips...")
    aspect_ratio = state.get("aspect_ratio", "9:16")
    rendered = render_candidates(
        workspace,
        highlights,
        aspect_ratio=aspect_ratio,
        burn_subtitles=state.get("burn_subtitles", True),
        boundary_adjustments=boundary_adjustments,
        refine_boundaries=False,
        llm_assessments={
            item.candidate_id: item
            for item in llm_assessments
            if item.candidate_id in {candidate.candidate_id for candidate in highlights}
        },
        pipeline_metadata={
            "mode": features.get("mode", base_mode),
            "checkpoint": state.get("ltr_checkpoint_info", {}),
            "feature_contract": features.get("feature_contract", {}),
            "extractor": features.get("extractor", {}),
            "llm_run": llm_run.model_dump(mode="json"),
            "fusion_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "start_time": candidate.start_time,
                    "end_time": candidate.end_time,
                    "ltr_score": candidate.score,
                    "llm_score": assessment.semantic_quality(),
                }
                for candidate in candidates
                for assessment in llm_assessments
                if assessment.candidate_id == candidate.candidate_id
            ],
        },
        render_namespace=state.get("render_namespace"),
    )
    _emit(state, "decide", "done", f"Render xong {len(rendered)} clips.", rendered_count=len(rendered))

    return {
        "highlights": highlights,
        "boundary_adjustments": boundary_adjustments,
        "rendered_highlights": rendered,
        "llm_assessments": llm_assessments,
        "llm_run": llm_run,
        "features": features,
    }


def _format_time_span(start: float, end: float) -> str:
    """Format start and end timestamps into human-readable MM:SS – MM:SS (Duration) format.

    Note for the team: Formatting timestamps as minutes and seconds makes it much easier
    for end users to locate the highlighted moment in the original video player.
    """
    start_m, start_s = divmod(int(start), 60)
    end_m, end_s = divmod(int(end), 60)
    duration = end - start
    return f"{start_m}:{start_s:02d} – {end_m}:{end_s:02d} ({duration:.0f}s)"


def _build_highlight_explanation(
    candidate: HighlightCandidate,
    rank: int,
    domain: str,
    mode: str,
    assessment: LLMHighlightAssessment | None = None,
    llm_run: LLMRunInfo | None = None,
) -> str:
    """Construct a clean, user-facing markdown explanation for a highlight candidate.

    When LLM assessment is present (Scenario A), display the semantic summary, transcript quote
    evidence, and quality dimensions.
    When running offline/pure LTR without LLM (Scenario B), display the rank, exact timing,
    multimodal score, and an informative status notice.
    """
    time_span = _format_time_span(candidate.start_time, candidate.end_time)

    if assessment:
        lines = [
            f"**💡 Key Insight**: {assessment.summary}",
            f'💬 *"{assessment.evidence}"*',
            f"**⏱️ Clip Timing**: {time_span}",
            f"**🎯 LLM Overall Quality**: {assessment.semantic_quality() * 100:.0f}%",
        ]
        return "\n\n".join(lines)

    score_label = (
        f"**📊 LTR Relative Rank Score**: {candidate.score:.1%} "
        "(normalized within this video)"
        if 0.0 <= candidate.score <= 1.0
        else f"**📊 LTR Score**: {candidate.score:.3f}"
    )

    if llm_run and llm_run.fallback_reason:
        reason_notice = f"*ℹ️ LLM semantic assessment failed ({llm_run.fallback_reason}). Displaying multimodal baseline.*"
    else:
        reason_notice = "*ℹ️ Semantic explanation unavailable (LLM API key not provided or LLM disabled).*"

    lines = [
        f"**🎯 Highlight #{rank}**",
        f"**⏱️ Clip Timing**: {time_span}",
        score_label,
        reason_notice,
    ]
    return "\n\n".join(lines)


# ──────────────────────────────────────────────
# Node 5: Explain
# ──────────────────────────────────────────────

def explain(state: AgentState) -> dict:
    """Tạo reasoning minh bạch, thân thiện với người dùng cho kết quả highlight"""

    domain = state["domain"]
    mode = state.get("features", {}).get("mode", "unknown")
    reasoning: list[ReasoningEntry] = []
    assessment_map = {
        item.candidate_id: item for item in state.get("llm_assessments", [])
    }
    llm_run = state.get("llm_run")
    for rank, candidate in enumerate(state.get("highlights", []), start=1):
        assessment = assessment_map.get(candidate.candidate_id)
        explanation = _build_highlight_explanation(
            candidate=candidate,
            rank=rank,
            domain=domain,
            mode=mode,
            assessment=assessment,
            llm_run=llm_run,
        )
        reasoning.append(
            {
                "candidate_id": candidate.candidate_id,
                "explanation": explanation,
            }
        )
    _emit(state, "explain", "done", f"Pipeline hoàn tất! {len(reasoning)} reasoning entries.", count=len(reasoning))
    return {"reasoning": reasoning}
