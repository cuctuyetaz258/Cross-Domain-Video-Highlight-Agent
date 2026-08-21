import streamlit as st
import traceback
import sys
import os
import datetime
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from highlight_agent.agent.graph import build_agent_graph
from highlight_agent.agent.state import ProgressEvent

def run_live_agent(video_url: str, domain: str, stepper_placeholder):
    """
    Executes the live LangGraph backend, updates stepper UI and displays live logs.
    """
    phases = ["observe", "plan", "analyze", "decide", "explain"]
    
    # 1. Tạo giao diện Status & Log Box trong Streamlit
    status_container = st.status("🎬 Processing Video Pipeline...", expanded=True)
    
    with status_container:
        st.markdown("#### 📜 Live Execution Logs")
        log_box = st.empty()  # Vùng trống dùng để render log liên tục
    
    logs: list[str] = []
    
    def log(msg: str, level: str = "INFO"):
        """Hàm hỗ trợ thêm log kèm timestamp"""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        icon = "ℹ️" if level == "INFO" else ("✅" if level == "SUCCESS" else "⚠️")
        formatted = f"[{now}] {icon} [{level}] {msg}"
        logs.append(formatted)
        # Render lại toàn bộ log trong khung st.code (Terminal style)
        log_box.code("\n".join(logs), language="bash")

    # 2. Emit callback — nhận sự kiện tiến độ từ bên trong nodes
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
        }
        icon = icon_map.get(event.step, "ℹ️")
        log(f"{icon} [{event.node}/{event.step}] {event.message}")

    # 3. Khởi tạo State
    log(f"Starting video highlight process for: {video_url}")
    log(f"Selected Domain: {domain}")
    
    visual_method = st.session_state.get("visual_method", "pixel_diff")
    visual_sample_fps = st.session_state.get("visual_sample_fps", 1.0)
    log(f"Visual method: {visual_method} | Sample FPS: {visual_sample_fps}")

    state = {
        "video_path": video_url,
        "domain": domain,
        "highlight_count": 3,
        "transcript_source": "auto",
        "cookies_browser": st.session_state.get("cookies_browser"),
        "burn_subtitles": False,
        "visual_method": visual_method,
        "visual_sample_fps": visual_sample_fps,
        "emit": emit,
    }
    
    try:
        log("Building LangGraph agent workflow...")
        graph = build_agent_graph()
        accumulated_state = dict(state)
        
        # 4. Lắng nghe từng Node trong LangGraph để đẩy Log thời gian thực
        for output in graph.stream(state):
            node_name = list(output.keys())[0]
            accumulated_state.update(output[node_name])
                
            # Cập nhật thanh Stepper trên UI
            if node_name in phases:
                current_phase_idx = phases.index(node_name) + 2
                with stepper_placeholder:
                    from components.stepper import render_stepper_state
                    render_stepper_state(current_phase_idx)
                
                time.sleep(0.5)
            
        final_state = accumulated_state
            
        if final_state:
            log("Pipeline completed successfully! All highlight clips are ready.", "SUCCESS")
            status_container.update(label="✅ Video Processing Complete!", state="complete", expanded=False)
            
            summary = {
                "workspace": final_state.get("workspace", {}).model_dump(mode="json") if hasattr(final_state.get("workspace"), "model_dump") else final_state.get("workspace"),
                "features": final_state.get("features"),
                "candidates": [(item.model_dump(mode="json") if hasattr(item, "model_dump") else item) for item in final_state.get("candidates", [])] if final_state.get("candidates") else [],
                "highlights": [item.model_dump(mode="json") for item in final_state.get("highlights", [])] if final_state.get("highlights") else [],
                "rendered_highlights": [
                    item.model_dump(mode="json") for item in final_state.get("rendered_highlights", [])
                ] if final_state.get("rendered_highlights") else [],
                "reasoning": final_state.get("reasoning", []),
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