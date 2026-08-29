import streamlit as st
from agent_runner import run_live_agent
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
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Initialize Session State
if "current_phase" not in st.session_state:
    st.session_state["current_phase"] = 0
if "video_id" not in st.session_state:
    st.session_state["video_id"] = None
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "agent_result" not in st.session_state:
    st.session_state["agent_result"] = None


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

        # Run live agent execution
        with st.spinner("⏳ Agent is actively processing... (Downloading and rendering may take several minutes)"):
            result = run_live_agent(target_url, target_domain, stepper_placeholder)

        if result:
            st.session_state["agent_result"] = result
            st.session_state["is_running"] = False
            st.session_state["current_phase"] = 6  # Completed (index 6 in the padded array)
            # The final step is already rendered inside run_live_agent, so no need to render again
        else:
            st.session_state["is_running"] = False

    if st.session_state.get("agent_result"):
        render_timeline(st.session_state["agent_result"])

        st.subheader("Generated Highlights")
        render_video_mockup(st.session_state["agent_result"])


if __name__ == "__main__":
    main()
