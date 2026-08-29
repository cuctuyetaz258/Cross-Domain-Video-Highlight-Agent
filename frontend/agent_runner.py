import datetime
import hashlib
import json
import os
import sys
import time
import traceback

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from highlight_agent.agent.graph import build_analysis_graph
from highlight_agent.agent.nodes import decide, explain
from highlight_agent.agent.snapshot import load_analysis_snapshot, save_analysis_snapshot
from highlight_agent.agent.state import ProgressEvent


def _serializable_summary(state: dict) -> dict:
    def dump(value):
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    return {
        "workspace": dump(state.get("workspace")),
        "features": state.get("features"),
        "candidates": [dump(item) for item in state.get("candidates", [])],
        "highlights": [dump(item) for item in state.get("highlights", [])],
        "boundary_adjustments": [dump(item) for item in state.get("boundary_adjustments", [])],
        "rendered_highlights": [dump(item) for item in state.get("rendered_highlights", [])],
        "reasoning": state.get("reasoning", []),
        "llm_assessments": [dump(item) for item in state.get("llm_assessments", [])],
        "llm_run": dump(state.get("llm_run")) if state.get("llm_run") else None,
        "analysis_snapshot_path": state.get("analysis_snapshot_path"),
        "analysis_id": state.get("analysis_id"),
        "variant_id": state.get("render_namespace"),
    }


def _variant_id(state: dict) -> str:
    config = {
        "analysis_id": state["analysis_id"],
        "provider": state["llm_provider"],
        "model": state.get("llm_model"),
        "base_url": state.get("llm_base_url"),
        "top_m": state["llm_top_m"],
        "ltr_weight": state["llm_ltr_weight"],
        "aspect_ratio": state.get("aspect_ratio", "9:16"),
        "burn_subtitles": state.get("burn_subtitles", False),
    }
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    provider = state["llm_provider"]
    model = "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in (state.get("llm_model") or "default")
    ).strip("-")[:35]
    return f"{provider}-{model or 'default'}-{digest}"


def _progress_ui(label: str):
    status_container = st.status(label, expanded=True)
    with status_container:
        st.markdown("#### 📜 Live Execution Logs")
        log_box = st.empty()
    logs: list[str] = []

    def log(message: str, level: str = "INFO"):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        icon = "ℹ️" if level == "INFO" else ("✅" if level == "SUCCESS" else "⚠️")
        logs.append(f"[{now}] {icon} [{level}] {message}")
        log_box.code("\n".join(logs), language="bash")

    def emit(event: ProgressEvent):
        icons = {
            "start": "🔄",
            "done": "✅",
            "ranking": "🏆",
            "rendering": "🎞️",
            "llm_start": "🧠",
            "llm_done": "✅",
            "llm_fallback": "⚠️",
        }
        log(f"{icons.get(event.step, 'ℹ️')} [{event.node}/{event.step}] {event.message}")

    return status_container, log, emit


def run_live_analysis(video_url: str, domain: str, stepper_placeholder):
    """Run Preflight→Analyze once and persist a reusable local snapshot."""

    phases = ["preflight", "observe", "plan", "analyze"]
    status, log, emit = _progress_ui("🎬 Building reusable LTR analysis...")
    ltr_model_path = st.session_state.get("ltr_model_path", "data/models/ltr_scorer.pt")
    known_speaker_count = st.session_state.get("known_speaker_count", None)
    aspect_ratio = st.session_state.get("target_aspect_ratio", "9:16")
    min_speaker_count = st.session_state.get("min_speaker_count", None)
    max_speaker_count = st.session_state.get("max_speaker_count", None)
    state = {
        "video_path": video_url,
        "domain": domain,
        "highlight_count": 3,
        "aspect_ratio": aspect_ratio,
        "candidate_pool_size": st.session_state.get("candidate_pool_size", 12),
        "transcript_source": "auto",
        "cookies_browser": st.session_state.get("cookies_browser"),
        "known_speaker_count": known_speaker_count,
        "min_speaker_count": min_speaker_count,
        "max_speaker_count": max_speaker_count,
        "burn_subtitles": False,
        "ltr_model_path": ltr_model_path,
        "llm_provider": "disabled",
        "emit": emit,
    }
    log(f"Input: {video_url}")
    log(f"Domain: {domain} · checkpoint: {ltr_model_path}")
    log(f"Output aspect ratio: {aspect_ratio}")
    log(f"Reusable candidate pool: {state['candidate_pool_size']}")
    try:
        accumulated_state = dict(state)
        for output in build_analysis_graph().stream(state):
            node_name = next(iter(output))
            accumulated_state.update(output[node_name])
            if node_name in phases:
                with stepper_placeholder:
                    from components.stepper import render_stepper_state

                    render_stepper_state(phases.index(node_name) + 1)
                time.sleep(0.2)

        snapshot_path, analysis_id = save_analysis_snapshot(accumulated_state)
        accumulated_state.update(
            {"analysis_snapshot_path": str(snapshot_path), "analysis_id": analysis_id}
        )
        with stepper_placeholder:
            from components.stepper import render_stepper_state

            render_stepper_state(5)
        log(f"Snapshot saved: {snapshot_path}", "SUCCESS")
        status.update(
            label="✅ LTR analysis is ready for rendering",
            state="complete",
            expanded=False,
        )
        return _serializable_summary(accumulated_state)
    except Exception as exc:  # noqa: BLE001
        _show_failure(status, log, "LTR analysis", exc)
        return None


def run_live_rerank(snapshot_path: str, stepper_placeholder):
    """Resume at Decide/Explain with optional OpenAI semantic reranking."""

    use_openai = st.session_state.get("use_openai_reranking", False)
    status_label = (
        "🧠 Running OpenAI model from saved LTR analysis..."
        if use_openai
        else "🎞️ Rendering directly from saved LTR analysis..."
    )
    status, log, emit = _progress_ui(status_label)
    try:
        state = dict(
            load_analysis_snapshot(
                snapshot_path,
                ltr_model_path=st.session_state.get("ltr_model_path"),
            )
        )
        state.update(
            {
                # LTR is required; OpenAI is an optional semantic reranking layer.
                "llm_provider": "openai" if use_openai else "disabled",
                "llm_model": (
                    st.session_state.get("llm_model") or None
                    if use_openai
                    else None
                ),
                "llm_base_url": None,
                "llm_top_m": st.session_state.get("llm_top_m", 10),
                "llm_ltr_weight": st.session_state.get("llm_ltr_weight", 0.60),
                "burn_subtitles": st.session_state.get("burn_subtitles", False),
                "emit": emit,
            }
        )
        state["render_namespace"] = _variant_id(state)
        run_description = (
            f"openai/{state.get('llm_model') or 'provider default'}"
            if use_openai
            else "ltr-only"
        )
        log(f"Reusing analysis {state['analysis_id'][:12]} · {run_description}")
        state.update(decide(state))
        with stepper_placeholder:
            from components.stepper import render_stepper_state

            render_stepper_state(6)
        state.update(explain(state))
        with stepper_placeholder:
            render_stepper_state(7)

        llm_run = state["llm_run"]
        if not llm_run.enabled:
            log(f"LTR-only render {state['render_namespace']} completed.", "SUCCESS")
            status.update(label="✅ LTR-only render complete", state="complete", expanded=False)
        elif llm_run.applied:
            suffix = " using cached LLM assessments." if llm_run.cache_hit else "."
            log(f"Variant {state['render_namespace']} completed{suffix}", "SUCCESS")
            status.update(label="✅ OpenAI model run complete", state="complete", expanded=False)
        else:
            log(
                f"LLM was skipped or unavailable; explicit LTR-only result: "
                f"{llm_run.fallback_reason}",
                "WARN",
            )
            status.update(
                label="⚠️ LLM skipped/unavailable; LTR-only result rendered",
                state="error",
            )
        return _serializable_summary(state)
    except Exception as exc:  # noqa: BLE001
        _show_failure(status, log, "highlight render", exc)
        return None


def _show_failure(status, log, stage: str, exc: Exception):
    message = f"{stage} failed: {exc}"
    log(message, "ERROR")
    status.update(label=f"❌ {stage} failed", state="error", expanded=True)
    st.error(message)
    with st.expander("🔍 Traceback Details"):
        st.code(traceback.format_exc())
