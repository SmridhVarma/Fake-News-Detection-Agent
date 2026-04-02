"""
ml_classifier_node — Phase 1: ML model inference.

Loads the trained model and returns a confidence score.
"""

from src.state import AgentState
from src.utils.model_io import load_model


def ml_classifier_node(state: AgentState) -> dict:
    """Run the ML model on the article and return score + label."""
    # TODO: Load model via load_model()
    # TODO: Predict on state["article_text"]
    # TODO: Return {"ml_score": ..., "ml_label": ...}
    return {"ml_score": 0.0, "ml_label": ""}
