import datetime
import os
import sys
import time
import traceback

import streamlit as st

# Ensure highlight_agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from highlight_agent.agent.graph import build_agent_graph
from highlight_agent.agent.state import ProgressEvent


def run_live_agent(video_url: str, domain: str, stepper_placeholder):
    """
    Executes the live LangGraph backend and updates the stepper UI & logs dynamically.
    """
    phases = ["preflight", "observe", "plan", "analyze", "decide", "explain"]

    status_container = st.status("🎬 Processing Video Pipeline...", expanded=True)

    with status_container:
        st.markdown("#### 📜 Live Execution Logs")
        log_box = st.empty()

    logs: list[str] = []

    def log(msg: str, level: str = "INFO"):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        icon = "ℹ️" if level == "INFO" else ("✅" if level == "SUCCESS" else "⚠️")
        formatted = f"[{now}] {icon} [{level}] {msg}"
        logs.append(formatted)
        log_box.code("\n".join(logs), language="bash")

    def emit(event: ProgressEvent):
        icon_map = {
            "start": "🔄",
            "visual_start": "🔄",
            "visual_window": "📐",
            "visual_normalize": "⚙️",
            "visual_done": "📊",
            "done": "✅",
            "fallback": "⚠️",
            "ranking": "🏆",
            "rendering": "🎞️",
            "llm_start": "🧠",
            "llm_done": "✅",
            "llm_fallback": "⚠️",
        }
        icon = icon_map.get(event.step, "ℹ️")
        log(f"{icon} [{event.node}/{event.step}] {event.message}")

    log(f"Starting highlight extraction for: {video_url}")
    log(f"Selected Domain: {domain}")

    ltr_model_path = st.session_state.get("ltr_model_path", "data/models/ltr_scorer.pt")
    known_speaker_count = st.session_state.get("known_speaker_count", None)
    aspect_ratio = st.session_state.get("target_aspect_ratio", "9:16")
    log(f"Required LTR checkpoint: {ltr_model_path}")
    log(f"Output Aspect Ratio: {aspect_ratio}")
    llm_provider = st.session_state.get("llm_provider", "disabled")
    log(f"LLM semantic reranker: {llm_provider}")

    state = {
        "video_path": video_url,
        "domain": domain,
        "highlight_count": 3,
        "aspect_ratio": aspect_ratio,
        "transcript_source": "auto",
        "cookies_browser": st.session_state.get("cookies_browser"),
        "known_speaker_count": known_speaker_count,
        "burn_subtitles": False,
        "ltr_model_path": ltr_model_path,
        "llm_provider": llm_provider,
        "llm_model": st.session_state.get("llm_model") or None,
        "llm_base_url": st.session_state.get("llm_base_url") or None,
        "llm_top_m": st.session_state.get("llm_top_m", 10),
        "llm_ltr_weight": st.session_state.get("llm_ltr_weight", 0.60),
        "emit": emit,
    }

    try:
        log("Building LangGraph agent workflow...")
        graph = build_agent_graph()
        accumulated_state = dict(state)

        for output in graph.stream(state):
            node_name = list(output.keys())[0]
            accumulated_state.update(output[node_name])

            if node_name in phases:
                current_phase_idx = phases.index(node_name) + 1
                with stepper_placeholder:
                    from components.stepper import render_stepper_state
                    render_stepper_state(current_phase_idx)
                time.sleep(0.5)

        final_state = accumulated_state

        if final_state:
            execution_mode = final_state.get("features", {}).get("mode", "unknown")
            log(f"Scoring mode: {execution_mode}", "SUCCESS")
            log("Pipeline completed successfully! All highlight clips are ready.", "SUCCESS")
            status_container.update(label="✅ Video Processing Complete!", state="complete", expanded=False)

            summary = {
                "workspace": final_state.get("workspace", {}).model_dump(mode="json") if hasattr(final_state.get("workspace"), "model_dump") else final_state.get("workspace"),
                "features": final_state.get("features"),
                "candidates": [(item.model_dump(mode="json") if hasattr(item, "model_dump") else item) for item in final_state.get("candidates", [])] if final_state.get("candidates") else [],
                "highlights": [item.model_dump(mode="json") for item in final_state.get("highlights", [])] if final_state.get("highlights") else [],
                "boundary_adjustments": [
                    item.model_dump(mode="json") for item in final_state.get("boundary_adjustments", [])
                ] if final_state.get("boundary_adjustments") else [],
                "rendered_highlights": [
                    item.model_dump(mode="json") for item in final_state.get("rendered_highlights", [])
                ] if final_state.get("rendered_highlights") else [],
                "reasoning": final_state.get("reasoning", []),
                "llm_assessments": [
                    item.model_dump(mode="json")
                    for item in final_state.get("llm_assessments", [])
                ],
                "llm_run": (
                    final_state["llm_run"].model_dump(mode="json")
                    if final_state.get("llm_run")
                    else None
                ),
            }
            return summary
        else:
            log("Graph execution returned empty state.", "WARN")
            status_container.update(label="⚠️ Finished with empty state", state="error")
            return None

    except Exception as e:
        error_msg = f"Pipeline Failed: {str(e)}"
        log(error_msg, "ERROR")
        status_container.update(label="❌ Pipeline Execution Failed!", state="error", expanded=True)
        st.error(error_msg)
        with st.expander("🔍 Traceback Details"):
            st.code(traceback.format_exc())
        return None
