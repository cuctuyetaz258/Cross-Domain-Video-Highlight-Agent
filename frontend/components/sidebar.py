import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from highlight_agent.features.ltr_contract import LTRPipelineError
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer


def _start_analysis(
    input_type: str,
    youtube_url: str | None,
    local_file,
    domain: str,
    aspect_ratio: str,
):
    if input_type == "YouTube URL" and youtube_url:
        target = youtube_url
    elif input_type == "Local Video" and local_file:
        upload_dir = os.path.abspath("uploads")
        os.makedirs(upload_dir, exist_ok=True)
        target = os.path.join(upload_dir, local_file.name)
        with open(target, "wb") as handle:
            handle.write(local_file.getbuffer())
    else:
        st.warning(f"Please provide a {'URL' if input_type == 'YouTube URL' else 'file'}.")
        return

    st.session_state.update(
        {
            "target_url": target,
            "target_domain": domain,
            "target_aspect_ratio": aspect_ratio,
            "current_phase": 1,
            "run_mode": "analysis",
            "is_running": True,
            "analysis_result": None,
            "agent_result": None,
            "llm_variant_results": [],
        }
    )
    st.rerun()


def render_sidebar():
    load_dotenv()
    with st.sidebar:
        st.markdown("### 1 · Build LTR Analysis")
        st.caption("Download/transcribe the video and run the 7-channel LTR model once.")
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

        if domain == "podcast":
            speaker_mode = st.selectbox(
                "Podcast speaker detection",
                ["Auto", "Auto (1-3 speakers)", "Known speaker count"],
                help="Use a known count when the podcast has a clear host and guest setup.",
            )
            if speaker_mode == "Known speaker count":
                st.session_state["known_speaker_count"] = int(
                    st.number_input("Known speaker count", min_value=1, max_value=10, value=2, step=1)
                )
                st.session_state["min_speaker_count"] = None
                st.session_state["max_speaker_count"] = None
            elif speaker_mode == "Auto (1-3 speakers)":
                st.session_state["known_speaker_count"] = None
                st.session_state["min_speaker_count"] = 1
                st.session_state["max_speaker_count"] = 3
            else:
                st.session_state["known_speaker_count"] = None
                st.session_state["min_speaker_count"] = None
                st.session_state["max_speaker_count"] = None

        checkpoint_path = st.text_input(
            "LTR checkpoint path",
            value="data/models/ltr_scorer.pt",
            key="ltr_model_path",
            help="Verified before media processing and whenever a snapshot is resumed.",
        )
        checkpoint_info = None
        try:
            checkpoint_info = AdditiveAttentionScorer.preflight(checkpoint_path)
            st.success(f"Valid · {checkpoint_info['device']} · {checkpoint_info['fingerprint'][:12]}")
        except LTRPipelineError as exc:
            st.error(str(exc))
        st.slider(
            "Reusable LTR candidate pool",
            3,
            12,
            12,
            key="candidate_pool_size",
            help="A larger pool lets later LLM runs compare more candidates without rerunning LTR.",
        )
        if st.button(
            "Run LTR Analysis Once",
            type="primary",
            use_container_width=True,
            disabled=checkpoint_info is None,
        ):
            _start_analysis(input_type, youtube_url, local_file, domain, aspect_ratio)

        st.divider()
        st.markdown("### 2 · Render Highlights")
        snapshot_path = st.session_state.get("analysis_snapshot_path", "")
        if snapshot_path:
            st.success(f"Snapshot ready\n\n`{snapshot_path}`")
        with st.expander("Load an existing snapshot"):
            manual_snapshot = st.text_input(
                "Snapshot JSON path",
                key="manual_snapshot_path",
                placeholder=".../analysis/ltr_analysis_snapshot.json",
            )
            if st.button("Use this snapshot", use_container_width=True):
                if Path(manual_snapshot).expanduser().is_file():
                    st.session_state["analysis_snapshot_path"] = str(
                        Path(manual_snapshot).expanduser().resolve()
                    )
                    st.session_state["llm_variant_results"] = []
                    st.rerun()
                else:
                    st.error("Snapshot file was not found.")

        use_openai = st.checkbox(
            "Use OpenAI semantic reranking",
            value=False,
            key="use_openai_reranking",
            help="Optional. Leave this off to render directly from the required LTR ranking.",
        )
        st.session_state["llm_provider"] = "openai" if use_openai else "disabled"
        st.session_state.pop("llm_base_url", None)
        openai_key_ready = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        if use_openai:
            st.info("Optional provider: OpenAI · key is read from `OPENAI_API_KEY`.")
            if openai_key_ready:
                st.success("OPENAI_API_KEY loaded")
            else:
                st.error("Missing OPENAI_API_KEY in .env/environment")
            if not st.session_state.get("llm_model"):
                st.session_state["llm_model"] = "gpt-4o-mini"
            st.text_input(
                "OpenAI model",
                key="llm_model",
                help="Change the model name to compare OpenAI models on the same LTR snapshot.",
            )
            st.slider("Candidates sent to OpenAI", 3, 12, 10, key="llm_top_m")
            st.text_input(
                "Fusion calibrator path (optional until trained)",
                key="fusion_calibrator_path",
                placeholder="data/models/fusion_calibrator.json",
                help=(
                    "Loads percentile-rank alpha learned on validation data. "
                    "If blank, the temporary equal-rank baseline is used; no manual weight is exposed."
                ),
            )
            fusion_path = st.session_state.get("fusion_calibrator_path", "").strip()
            fusion_ready = not fusion_path or Path(fusion_path).expanduser().is_file()
            if fusion_path and fusion_ready:
                st.success("Fusion calibrator found; compatibility is checked at runtime.")
            elif fusion_path:
                st.error("Fusion calibrator file was not found.")
            else:
                st.info("Temporary equal-rank fusion baseline; no learned alpha is loaded.")
        else:
            fusion_ready = True
            st.info("LTR-only mode · no API key or LLM request is required.")
        st.checkbox("Burn subtitles into rendered clips", value=False, key="burn_subtitles")
        action_label = (
            "Run / Compare OpenAI Model" if use_openai else "Render LTR Only"
        )
        if st.button(
            action_label,
            type="primary",
            use_container_width=True,
            disabled=(
                not bool(snapshot_path)
                or checkpoint_info is None
                or (use_openai and not openai_key_ready)
                or (use_openai and not fusion_ready)
            ),
        ):
            st.session_state.update(
                {"run_mode": "rerank", "is_running": True, "current_phase": 5}
            )
            st.rerun()
