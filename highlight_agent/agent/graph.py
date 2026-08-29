"""Khởi tạo LangGraph mà không tự chạy media khi import"""

from langgraph.graph import END, StateGraph

from .nodes import analyze, decide, explain, observe, plan, preflight
from .state import AgentState


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("preflight", preflight)
    graph.add_node("observe", observe)
    graph.add_node("plan", plan)
    graph.add_node("analyze", analyze)
    graph.add_node("decide", decide)
    graph.add_node("explain", explain)

    graph.set_entry_point("preflight")
    graph.add_edge("preflight", "observe")
    graph.add_edge("observe", "plan")
    graph.add_edge("plan", "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def build_analysis_graph():
    """Run only the expensive deterministic LTR stages.

    The resulting state can be persisted and resumed at Decide for any number of
    LLM provider/model variants.
    """

    graph = StateGraph(AgentState)
    graph.add_node("preflight", preflight)
    graph.add_node("observe", observe)
    graph.add_node("plan", plan)
    graph.add_node("analyze", analyze)
    graph.set_entry_point("preflight")
    graph.add_edge("preflight", "observe")
    graph.add_edge("observe", "plan")
    graph.add_edge("plan", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()
