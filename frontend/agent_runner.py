import os
import sys
import traceback

import streamlit as st

# Ensure highlight_agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from highlight_agent.agent.graph import build_agent_graph


def run_live_agent(video_url: str, domain: str, stepper_placeholder):
    """
    Executes the live LangGraph backend and updates the stepper UI dynamically.
    """
    phases = ["observe", "plan", "analyze", "decide", "explain"]

    # Initialize the backend State
    state = {
        "video_path": video_url,
        "domain": domain,
        "highlight_count": 3,
        "transcript_source": "auto",
        "burn_subtitles": False,  # Prevent macOS filter error locally
    }

    try:
        graph = build_agent_graph()

        final_state = None

        # Stream the graph execution
        accumulated_state = dict(state)

        for output in graph.stream(state):
            # output is a dict with the node name as key, e.g. {"observe": {...}}
            node_name = list(output.keys())[0]

            # Update the accumulated state with the changes from this node
            accumulated_state.update(output[node_name])

            if node_name in phases:
                # The node just finished, so we move to the next phase
                # +2 because phases is 0-indexed here, but the UI Stepper has a padded "Ready" step at index 0.
                current_phase_idx = phases.index(node_name) + 2
                with stepper_placeholder:
                    from components.stepper import render_stepper_state

                    render_stepper_state(current_phase_idx)

                # Introduce an artificial animation delay so the user can actually read the phase
                # instead of 4 phases flashing by in 0.1s (which looks like a glitch).
                import time

                time.sleep(0.75)

        final_state = accumulated_state

        # Format the final summary exactly like the backend CLI does
        if final_state:
            summary = {
                "workspace": final_state.get("workspace", {}).model_dump(mode="json")
                if hasattr(final_state.get("workspace"), "model_dump")
                else final_state.get("workspace"),
                "features": final_state.get("features"),
                "candidates": [
                    (item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
                    for item in final_state.get("candidates", [])
                ]
                if final_state.get("candidates")
                else [],
                "highlights": [item.model_dump(mode="json") for item in final_state.get("highlights", [])]
                if final_state.get("highlights")
                else [],
                "rendered_highlights": [
                    item.model_dump(mode="json") for item in final_state.get("rendered_highlights", [])
                ]
                if final_state.get("rendered_highlights")
                else [],
                "reasoning": final_state.get("reasoning", []),
            }
            return summary
        else:
            st.error("Graph execution finished but returned empty state.")
            return None

    except Exception as e:
        st.error(f"Pipeline Failed: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None
