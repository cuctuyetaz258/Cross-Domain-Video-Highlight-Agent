import os

import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("### Input Settings")

        input_type = st.radio("Input Source", ["YouTube URL", "Local Video"])

        youtube_url = None
        local_file = None

        if input_type == "YouTube URL":
            youtube_url = st.text_input("YouTube URL", placeholder="https://youtu.be/HMOI_lkzW08")
        else:
            local_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv", "webm"])

        domain = st.selectbox("Content Domain", ["lecture", "podcast", "standup"])

        with st.expander("LTR & Visual Settings"):
            st.selectbox(
                "Visual method",
                ["pixel_diff", "raft", "scene_mediapipe"],
                key="visual_method",
                help="scene_mediapipe is used by the dense LTR feature path.",
            )
            st.number_input(
                "Visual sample FPS",
                min_value=0.25,
                max_value=10.0,
                value=1.0,
                step=0.25,
                key="visual_sample_fps",
            )
            st.text_input(
                "LTR checkpoint path",
                value="",
                key="ltr_model_path",
                placeholder="data/models/ltr_scorer.pt",
                help="Leave empty to use the current weighted-sum pipeline.",
            )

        if st.button("Process Video", type="primary", use_container_width=True):
            if input_type == "YouTube URL" and youtube_url:
                st.session_state["target_url"] = youtube_url
                st.session_state["target_domain"] = domain

                st.session_state["current_phase"] = 1
                st.session_state["is_running"] = True
                st.session_state["agent_result"] = None
                st.rerun()

            elif input_type == "Local Video" and local_file:
                # Save the file to disk so the backend can read it
                upload_dir = os.path.abspath("uploads")
                os.makedirs(upload_dir, exist_ok=True)

                file_path = os.path.join(upload_dir, local_file.name)
                with open(file_path, "wb") as f:
                    f.write(local_file.getbuffer())

                st.session_state["target_url"] = file_path
                st.session_state["target_domain"] = domain

                st.session_state["current_phase"] = 1
                st.session_state["is_running"] = True
                st.session_state["agent_result"] = None
                st.rerun()
            else:
                st.warning(f"Please provide a {'URL' if input_type == 'YouTube URL' else 'file'}.")
