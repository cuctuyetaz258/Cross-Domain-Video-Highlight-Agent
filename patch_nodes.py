import re

with open("highlight_agent/agent/nodes.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add _llm_candidates function
llm_candidates_code = """
# ----------------------------------------------
# LLM Extraction (Th?o Anh's Prompt)
# ----------------------------------------------

def _llm_candidates(state: AgentState, count: int = 5) -> list[HighlightCandidate]:
    transcript = state.get("transcript")
    if transcript is None:
        raise ValueError("LLM Extraction requires transcript from Observe")
    if transcript.duration < 30:
        raise ValueError("video must be at least 30 seconds to create an MVP highlight")

    _emit(state, "analyze", "llm_start", "Ðang g?i LLM (Groq) d? phân tích transcript...")
    from highlight_agent.llm.extractor import extract_highlights_from_transcript
    
    # Chu?n b? full text
    full_text = "\n".join(f"[{w.start:.2f} - {w.end:.2f}] {w.word}" for w in transcript.words)
    domain = state.get("domain", "auto")
    
    try:
        candidates = extract_highlights_from_transcript(full_text, domain=domain, highlight_count=count)
        if candidates:
            _emit(state, "analyze", "llm_done", f"LLM tr? v? {len(candidates)} candidates.")
            return candidates
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        _emit(state, "analyze", "llm_error", f"LLM l?i ({exc}), fallback v? naive baseline.")
    
    return _naive_candidates(state, count)

"""

# Insert _llm_candidates before _naive_candidates
content = content.replace("# ----------------------------------------------\n# Naive baseline (fallback)", llm_candidates_code + "\n# ----------------------------------------------\n# Naive baseline (fallback)")

# Modify analyze() to use extraction_mode
analyze_start = """def analyze(state: AgentState) -> dict:
    \"\"\"Trích xu?t features da t?ng, fusion d? t?o candidate ho?c fallback an toàn\"\"\"

    _emit(state, "analyze", "start", "B?t d?u phân tích d?c trung...")
    workspace = state.get("workspace")
    if workspace is None:
        raise ValueError("Analyze requires workspace from Observe")

    window_seconds = 30.0
    hop_seconds = 30.0
    
    extraction_mode = state.get("extraction_mode", "auto")
    highlight_count = state.get("highlight_count", 3)
    
    # -- 1. N?U CH? Ð?NH DÙNG LLM --
    if extraction_mode == "llm":
        _emit(state, "analyze", "mode_llm", "Ch? d? LLM du?c ch?n, b? qua feature extraction.")
        candidates = _llm_candidates(state, highlight_count)
        return {
            "features": {"mode": "llm_extraction", "candidate_count": len(candidates)},
            "feature_path": "",
            "feature_timeline": {},
            "candidates": candidates,
        }
"""

content = re.sub(
    r'def analyze\(state: AgentState\) -> dict:.*?window_seconds = 30\.0\n    hop_seconds = 30\.0',
    analyze_start,
    content,
    flags=re.DOTALL
)

with open("highlight_agent/agent/nodes.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched nodes.py")
