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

    cand_map = {c["candidate_id"]: c for c in data.get("highlights", [])}
    bound_map = {b["candidate_id"]: b for b in data.get("boundary_adjustments", [])}
    assess_map = {a["candidate_id"]: a for a in data.get("llm_assessments", [])}

    for idx, highlight in enumerate(rendered_highlights[:3]):
        with cols[idx]:
            title = highlight.get("title") or f"Highlight {idx + 1}"
            ar = highlight.get("aspect_ratio", "9:16")
            ar_label = "9:16 Portrait" if ar == "9:16" else "16:9 Landscape"
            st.markdown(f"### {title}")
            st.caption(f"🎬 {ar_label}")

            video_path = highlight.get("video_path")
            if video_path and os.path.exists(video_path):
                try:
                    # Streamlit natively serves local files
                    st.video(video_path)
                except Exception:
                    st.error(f"Could not load video: {video_path}")
            else:
                st.error(f"Video file missing: {video_path}")

            # Retrieve formatted user-friendly explanation from agent reasoning
            exp = reason_map.get(
                highlight["candidate_id"],
                highlight.get("reason", "Highly engaging moment identified based on multimodal audio and visual signals."),
            )

            # Display risk flags if detected by the semantic evaluation layer
            if highlight.get("llm_risk_flags"):
                st.warning("⚠️ Attention: " + ", ".join(highlight["llm_risk_flags"]))

            # Render the rich markdown explanation inside the expandable container
            with st.expander("Why was this selected?", expanded=True):
                st.markdown(exp)

                # Collapsed sub-expander allowing researchers/developers to inspect raw diagnostic signals
                with st.expander("🔍 View Raw Technical Diagnostics", expanded=False):
                    raw_cand = cand_map.get(highlight["candidate_id"])
                    raw_bound = bound_map.get(highlight["candidate_id"])
                    raw_assess = assess_map.get(highlight["candidate_id"])

                    debug_info = {}
                    if raw_cand:
                        debug_info["candidate_metrics"] = {
                            "candidate_id": raw_cand.get("candidate_id"),
                            "score": raw_cand.get("score"),
                            "internal_reason": raw_cand.get("reason"),
                            "signals": raw_cand.get("signals", {}),
                        }
                    if raw_bound:
                        debug_info["boundary_refinement"] = {
                            "start_source": raw_bound.get("start_source"),
                            "start_reason": raw_bound.get("start_reason"),
                            "end_source": raw_bound.get("end_source"),
                            "end_reason": raw_bound.get("end_reason"),
                        }
                    if raw_assess:
                        debug_info["llm_raw_assessment"] = {
                            "semantic_relevance": raw_assess.get("semantic_relevance"),
                            "standalone_value": raw_assess.get("standalone_value"),
                            "completeness": raw_assess.get("completeness"),
                            "hook_strength": raw_assess.get("hook_strength"),
                            "shareability": raw_assess.get("shareability"),
                            "risk_flags": raw_assess.get("risk_flags", []),
                        }
                    st.json(debug_info if debug_info else highlight)
