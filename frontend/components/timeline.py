import streamlit as st
import plotly.graph_objects as go

def render_timeline(data: dict):
    candidates = data.get("candidates", [])
    highlights = data.get("highlights", [])
    
    if not candidates:
        st.warning("Timeline Debug: No candidates found in the agent payload.")
        return
        
    st.markdown("### 📊 Candidate Signals Timeline")
    st.markdown("Interact with the timeline to see how the LangGraph agent scored and segmented the video. The highlighted red borders indicate the final chosen clips.")
    
    # Sort candidates chronologically for a cleaner Gantt waterfall
    candidates = sorted(candidates, key=lambda x: x["start_time"])
    
    fig = go.Figure()
    
    # Extract IDs of the top K selected highlights for special styling
    highlight_ids = {h["candidate_id"] for h in highlights}
    
    # Extract global max score for the colorscale
    max_score = max((c["score"] for c in candidates), default=1.0)
    
    for idx, c in enumerate(candidates):
        is_highlight = c["candidate_id"] in highlight_ids
        duration = c["end_time"] - c["start_time"]
        
        # Format hover text with HTML
        signals_text = "<br>".join([f"&nbsp;&nbsp;{k}: {v:.3f}" for k, v in c.get("signals", {}).items()])
        hover_text = (
            f"<b>Candidate ID:</b> {c['candidate_id']}<br>"
            f"<b>Score:</b> {c['score']:.3f}<br>"
            f"<b>Range:</b> {c['start_time']:.1f}s - {c['end_time']:.1f}s<br>"
            f"<b>Signals:</b><br>{signals_text}<br>"
            f"<b>Reason:</b> {c.get('reason', '')}"
        )
        
        # Add horizontal bar for this candidate
        fig.add_trace(go.Bar(
            name=c["candidate_id"],
            x=[duration],
            y=[f"C{idx+1}"],
            base=[c["start_time"]],
            orientation='h',
            marker=dict(
                color=[c["score"]],
                colorscale='Viridis',
                cmin=0,
                cmax=max_score,
                line=dict(
                    color='#ff4b4b' if is_highlight else 'rgba(0,0,0,0)',
                    width=3 if is_highlight else 0
                )
            ),
            hoverinfo="text",
            hovertext=[hover_text],
            showlegend=False
        ))
        
    # Dynamically scale height based on number of candidates
    chart_height = max(350, len(candidates) * 35)
    
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Video Time (seconds)",
        yaxis_title="",
        height=chart_height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Modern font for Plotly matching our CSS
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(autorange="reversed")
    )
    
    # Add a dummy scatter trace just to render the unified colorbar on the right
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            cmin=0,
            cmax=max_score,
            showscale=True,
            colorbar=dict(title="Score")
        ),
        showlegend=False,
        hoverinfo='none'
    ))
    
    st.plotly_chart(fig, use_container_width=True)
