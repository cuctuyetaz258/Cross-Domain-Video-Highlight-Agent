import streamlit as st


def render_stepper_state(current: int):
    """
    Renders the Stepper Bar UI natively using Streamlit's st.progress to completely eliminate 
    any DOM teardown flickering (glitching) caused by st.empty() markdown replacements.
    """
    phases = [
        "Ready",
        "Preflight",
        "Observe",
        "Plan",
        "Analyze",
        "Decide",
        "Explain",
        "Completed",
    ]

    html = "<div style='display: flex; justify-content: space-between; align-items: center; width: 100%; margin-top: 10px; margin-bottom: 30px; font-family: \"Inter\", sans-serif;'>"

    for i, phase in enumerate(phases):
        is_completed = i < current
        is_active = i == current

        # Theming logic
        if is_completed:
            circle_bg = "#238636" # Green
            text_color = "inherit"
            content = "✓"
        elif is_active:
            circle_bg = "#58a6ff" # Blue
            text_color = "#58a6ff"
            content = str(i)
        else:
            circle_bg = "rgba(128,128,128,0.2)"
            text_color = "rgba(128,128,128,0.5)"
            content = str(i)

        font_weight = "600" if is_active else "400"

        # Step Node
        html += f"""
        <div style='display: flex; flex-direction: column; align-items: center; z-index: 2;'>
            <div style='width: 32px; height: 32px; border-radius: 50%; background-color: {circle_bg}; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; font-size: 14px; margin-bottom: 8px; transition: background-color 0.3s ease;'>
                {content}
            </div>
            <div style='color: {text_color}; font-weight: {font_weight}; font-size: 13px; text-align: center; white-space: nowrap;'>{phase}</div>
        </div>
        """

        # Connecting Line (skip for the last step)
        if i < len(phases) - 1:
            line_bg = "#238636" if is_completed else "rgba(128,128,128,0.2)"
            html += f"<div style='flex: 1; height: 3px; background-color: {line_bg}; margin-top: -25px; transition: background-color 0.3s ease;'></div>"

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

def render_stepper():
    current = st.session_state.get("current_phase", 0)
    render_stepper_state(current)
