import streamlit as st
import os

def render_sidebar():
    with st.sidebar:
        st.markdown("### Input Settings")
        
        input_type = st.radio("Input Source", ["YouTube URL", "Local Video"])
        
        youtube_url = None
        local_file = None
        
        if input_type == "YouTube URL":
            youtube_url = st.text_input("YouTube URL", placeholder="https://youtu.be/HMOI_lkzW08")
            cookies_browser = st.selectbox(
                "Browser for Cookies (To bypass YT blocks)", 
                ["chrome", "edge", "firefox", "brave", "safari", "none"],
                index=0
            )
            if cookies_browser == "none":
                cookies_browser = None
        else:
            local_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv", "webm"])
        
        domain = st.selectbox("Content Domain", ["lecture", "podcast", "standup"])
        
        if st.button("Process Video", type="primary", use_container_width=True):
            if input_type == "YouTube URL" and youtube_url:
                st.session_state["target_url"] = youtube_url
                st.session_state["target_domain"] = domain
                st.session_state["cookies_browser"] = cookies_browser
                
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
