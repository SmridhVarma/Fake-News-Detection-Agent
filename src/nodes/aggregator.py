"""
aggregator_node — Combines ML + LLM scores into a final verdict.

Produces the final classification label, score, and human-readable summary.
"""

from src.state import AgentState


def aggregator_node(state: AgentState) -> dict:
    """Aggregate both phase scores and produce a final summary."""
    # TODO: Combine ml_score and llm_score (weighted average, etc.)
    # TODO: Determine final_label based on final_score threshold
    # TODO: Build summary string explaining reasoning
    return {"final_label": "", "final_score": 0.0, "summary": ""}
