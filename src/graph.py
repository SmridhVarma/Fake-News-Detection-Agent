"""
graph.py — LangGraph Pipeline Wiring

Imports all nodes from src/nodes/ and wires them into a StateGraph.
This is the only file that knows about the full pipeline topology.
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    ingestion_node,
    training_node,
    ml_classifier_node,
    llm_classifier_node,
    evaluator_node,
    aggregator_node,
)

# ── Build the graph ──────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("ingest", ingestion_node)
builder.add_node("train", training_node)
builder.add_node("ml_classify", ml_classifier_node)
builder.add_node("llm_classify", llm_classifier_node)
builder.add_node("evaluate", evaluator_node)
builder.add_node("aggregate", aggregator_node)

builder.set_entry_point("ingest")
builder.add_edge("ingest", "train")
builder.add_edge("train", "ml_classify")
builder.add_edge("ml_classify", "llm_classify")
builder.add_edge("llm_classify", "evaluate")
builder.add_edge("evaluate", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()


# ── Public entry point ───────────────────────────────────────
def run_agent(article_text: str) -> dict:
    """Run the full detection pipeline on an article."""
    initial_state: AgentState = {
        "article_text": article_text,
        "ml_score": 0.0,
        "ml_label": "",
        "llm_score": 0.0,
        "llm_label": "",
        "llm_reasoning": "",
        "eval_score": 0.0,
        "final_label": "",
        "final_score": 0.0,
        "summary": "",
        "model_trained": False,
    }
    return graph.invoke(initial_state)
