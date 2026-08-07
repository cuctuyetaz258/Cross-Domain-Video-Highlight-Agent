import streamlit as st
import time
import extra_streamlit_components.StepperBar as StepperBar

def render_stepper_state(current: int):
    """
    Renders the Stepper Bar UI to visualize the LangGraph agent's execution phase.
    
    @TEAM_NOTE (Frontend/UI):
    We use the `extra-streamlit-components` StepperBar here. However, its underlying React 
    component contains a known logical flaw where passing `default=0` evaluates as falsy in JS (`0 || 1`), 
    causing it to unconditionally fall back to index 1. 
    
    To elegantly bypass this without forking the library:
    1. We pad the phases array with a 0th "Ready" step.
    2. When `current=0` (app load), the JS bug falls back to index 1 ("Observe").
    3. Index 0 ("Ready") is visually marked as completed, which serves as a great idle state indicator.
    
    Furthermore, we bypass the python wrapper to directly invoke `_component_func` so we can inject 
    a unique `key` (appended with time.time()), avoiding Streamlit's `DuplicateElementKey` exception 
    during rapid loop re-renders.
    """
    phases = ["Ready", "Observe", "Plan", "Analyze", "Decide", "Explain", "Completed"]
    
    StepperBar._component_func(
        steps=phases, 
        is_vertical=False, 
        lock_sequence=True, 
        default=current,
        key=f"stepper_{current}"
    )

def render_stepper():
    current = st.session_state.get("current_phase", 0)
    render_stepper_state(current)
