import os

import streamlit as st

from highlight_agent.features.ltr_contract import LTRPipelineError
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer


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

        aspect_ratio_choice = st.radio(
            "Output Video Format",
            ["9:16 — Portrait (Shorts / Reels)", "16:9 — Landscape (YouTube / Standard)"],
            index=0,
            help="Choose the orientation and aspect ratio for rendered highlight clips.",
        )
        aspect_ratio = "9:16" if "9:16" in aspect_ratio_choice else "16:9"

        with st.expander("Required LTR Model", expanded=True):
            checkpoint_path = st.text_input(
                "LTR checkpoint path",
                value="data/models/ltr_scorer.pt",
                key="ltr_model_path",
                help="The pipeline stops before media processing when this checkpoint is invalid.",
            )
            checkpoint_info = None
            try:
                checkpoint_info = AdditiveAttentionScorer.preflight(checkpoint_path)
                st.success(
                    "Valid checkpoint · "
                    f"schema {checkpoint_info['feature_contract']['schema_version']} · "
                    f"{checkpoint_info['device']} · {checkpoint_info['fingerprint'][:12]}"
                )
            except LTRPipelineError as exc:
                st.error(str(exc))

        with st.expander("LLM Semantic Reranking"):
            st.selectbox(
                "LLM provider",
                ["disabled", "openai", "groq", "custom"],
                key="llm_provider",
                help="API key is read from the process environment and is never stored in the result.",
            )
            st.text_input(
                "LLM model",
                value="",
                key="llm_model",
                placeholder="gpt-4.1-mini",
            )
            st.text_input(
                "Custom base URL",
                value="",
                key="llm_base_url",
                placeholder="https://provider.example/v1",
            )
            st.slider("Candidates sent to LLM", 3, 12, 10, key="llm_top_m")
            st.slider("LTR weight", 0.0, 1.0, 0.60, 0.05, key="llm_ltr_weight")

        if st.button(
            "Process Video",
            type="primary",
            use_container_width=True,
            disabled=checkpoint_info is None,
        ):
            if input_type == "YouTube URL" and youtube_url:
                st.session_state["target_url"] = youtube_url
                st.session_state["target_domain"] = domain
                st.session_state["target_aspect_ratio"] = aspect_ratio

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
                st.session_state["target_aspect_ratio"] = aspect_ratio

                st.session_state["current_phase"] = 1
                st.session_state["is_running"] = True
                st.session_state["agent_result"] = None
                st.rerun()
            else:
                st.warning(f"Please provide a {'URL' if input_type == 'YouTube URL' else 'file'}.")
