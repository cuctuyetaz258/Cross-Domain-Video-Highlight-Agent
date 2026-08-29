
import streamlit as st
from agent_runner import run_live_analysis, run_live_rerank
from components.sidebar import render_sidebar
from components.stepper import render_stepper
from components.timeline import render_timeline
from components.video_mockup import render_video_mockup

# Premium Web Design Setup
st.set_page_config(
    page_title="Agent Highlight Console",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism and modern fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism for containers (adaptive to light/dark mode) */
    .glass-container {
        background: rgba(128, 128, 128, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    
    /* Hide Streamlit's 'Press enter to apply' tooltip to prevent collision in sidebar */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_phase" not in st.session_state:
    st.session_state["current_phase"] = 0
if "video_id" not in st.session_state:
    st.session_state["video_id"] = None
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "agent_result" not in st.session_state:
    st.session_state["agent_result"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "analysis_snapshot_path" not in st.session_state:
    st.session_state["analysis_snapshot_path"] = ""
if "llm_variant_results" not in st.session_state:
    st.session_state["llm_variant_results"] = []
if "run_mode" not in st.session_state:
    st.session_state["run_mode"] = None


def _variant_label(result: dict) -> str:
    llm_run = result.get("llm_run") or {}
    if not llm_run.get("enabled"):
        return "LTR only"
    status = "OpenAI applied" if llm_run.get("applied") else "OpenAI skipped/unavailable → LTR"
    cache = " · cache" if llm_run.get("cache_hit") else ""
    return (
        f"{llm_run.get('provider') or 'unknown'} / "
        f"{llm_run.get('model') or 'provider default'} · {status}{cache}"
    )


def _render_variant_comparison(variants: list[dict]):
    st.subheader("Highlight Result Comparison")
    rows = []
    for result in variants:
        llm_run = result.get("llm_run") or {}
        rows.append(
            {
                "Mode": (
                    "OpenAI + LTR" if llm_run.get("applied") else "LTR only"
                ),
                "Model": llm_run.get("model") or "—",
                "Applied": bool(llm_run.get("applied")),
                "Cache hit": bool(llm_run.get("cache_hit")),
                "Assessed": llm_run.get("assessed_count", 0),
                "Highlights": len(result.get("rendered_highlights", [])),
                "Failure": llm_run.get("fallback_reason") or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    selected_index = st.selectbox(
        "Result to preview",
        range(len(variants)),
        format_func=lambda index: _variant_label(variants[index]),
    )
    selected = variants[selected_index]
    render_timeline(selected)
    st.subheader("Generated Highlights")
    render_video_mockup(selected)


def main():
    st.title("🎬 Video Highlight Agent Console")

    # Sidebar
    render_sidebar()

    # Main Panel Layout
    st.subheader("Agent Progress")

    stepper_placeholder = st.empty()
    with stepper_placeholder:
        render_stepper()

    if st.session_state["is_running"]:
        target_url = st.session_state.get("target_url", "https://youtu.be/HMOI_lkzW08")
        target_domain = st.session_state.get("target_domain", "lecture")
        run_mode = st.session_state.get("run_mode")
        result = None
        if run_mode == "analysis":
            with st.spinner("⏳ Building LTR analysis snapshot. This is the expensive run..."):
                result = run_live_analysis(target_url, target_domain, stepper_placeholder)
            if result:
                st.session_state["analysis_result"] = result
                st.session_state["analysis_snapshot_path"] = result["analysis_snapshot_path"]
                st.session_state["current_phase"] = 5
        elif run_mode == "rerank":
            spinner_text = (
                "🧠 Reusing LTR candidates with optional OpenAI reranking..."
                if st.session_state.get("use_openai_reranking", False)
                else "🎞️ Rendering directly from saved LTR candidates..."
            )
            with st.spinner(spinner_text):
                result = run_live_rerank(
                    st.session_state["analysis_snapshot_path"], stepper_placeholder
                )
            if result:
                variants = [
                    item
                    for item in st.session_state["llm_variant_results"]
                    if item.get("variant_id") != result.get("variant_id")
                ]
                variants.append(result)
                st.session_state["llm_variant_results"] = variants
                st.session_state["agent_result"] = result
                st.session_state["current_phase"] = 7
        st.session_state["is_running"] = False
        st.session_state["run_mode"] = None
        if result:
            st.rerun()

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        candidate_count = len(analysis_result.get("candidates", []))
        st.success(
            f"LTR analysis ready: {candidate_count} reusable candidates. "
            "Render LTR directly, or optionally enable OpenAI semantic reranking."
        )
        with st.expander("LTR analysis timeline", expanded=False):
            render_timeline(analysis_result)

    variants = st.session_state.get("llm_variant_results", [])
    if variants:
        _render_variant_comparison(variants)
    elif st.session_state.get("analysis_snapshot_path"):
        st.info("Snapshot is ready. Render LTR directly or enable optional OpenAI reranking.")

if __name__ == "__main__":
    main()
