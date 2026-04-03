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
def run_agent(user_input: str, input_type: str = "text") -> dict:
    """Run the full detection pipeline on an article.

    Args:
        user_input: Raw text, URL, or file path from the user.
        input_type: One of "text", "url", or "file".
    """
    initial_state: AgentState = {
        # Raw input
        "input_type": input_type,
        "raw_input": user_input,
        # Preprocessing (set by ingestion node)
        "article_title": None,
        "article_text": "",
        "source_domain": None,
        "word_count": 0,
        # Style signals (set by style_check)
        "caps_ratio": None,
        "exclamation_count": None,
        "amplifier_word_count": None,
        "style_score": None,
        # Source credibility
        "source_credibility": None,
        # Phase 1: ML
        "ml_score": 0.0,
        "ml_label": "",
        # Phase 2: LLM
        "llm_score": 0.0,
        "llm_label": "",
        "llm_reasoning": "",
        # Evaluation
        "eval_agreement": None,
        "eval_confidence_delta": None,
        "eval_score": 0.0,
        # Final output
        "final_label": "",
        "final_score": 0.0,
        "explanation": "",
        "summary": "",
        # Flags
        "model_trained": False,
    }
    return graph.invoke(initial_state)
