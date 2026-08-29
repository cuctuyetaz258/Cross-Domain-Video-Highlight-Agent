import os

import streamlit as st


def render_video_mockup(data: dict):
    rendered_highlights = data.get("rendered_highlights", [])
    reasoning = data.get("reasoning", [])

    if not rendered_highlights:
        st.warning("No highlights were rendered.")
        return

    reason_map = {r["candidate_id"]: r["explanation"] for r in reasoning}

    # Create columns for the highlights (max 3 per row)
    cols = st.columns(len(rendered_highlights[:3]))

    for idx, highlight in enumerate(rendered_highlights[:3]):
        with cols[idx]:
            title = highlight.get("title") or f"Highlight {idx + 1}"
            st.markdown(f"### {title}")

            video_path = highlight.get("video_path")
            if video_path and os.path.exists(video_path):
                try:
                    # Streamlit natively serves local files
                    st.video(video_path)
                except Exception:
                    st.error(f"Could not load video: {video_path}")
            else:
                st.error(f"Video file missing: {video_path}")

            exp = reason_map.get(highlight["candidate_id"], highlight.get("reason", "Highly engaging moment based on acoustic and text signals."))

            if highlight.get("summary"):
                st.markdown(highlight["summary"])
            if highlight.get("completeness_score") is not None:
                st.caption(
                    f"Semantic: {highlight.get('semantic_score', 0):.2f} · "
                    f"Completeness: {highlight['completeness_score']:.2f}"
                )
            if highlight.get("llm_risk_flags"):
                st.warning("Risk flags: " + ", ".join(highlight["llm_risk_flags"]))

            with st.expander("Why was this selected?", expanded=True):
                st.markdown(exp)
