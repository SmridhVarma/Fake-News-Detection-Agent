"""
training_node — Trains or loads the ML model.

Uses skills/ for the actual training logic.
Sets model_trained flag in state.
"""

from src.state import AgentState


def training_node(state: AgentState) -> dict:
    """Train the ML model if not already trained, or load existing."""
    # TODO: Check if model exists at MODEL_SAVE_PATH
    # TODO: If not, call skills to train and save
    # TODO: If yes, skip training
    return {"model_trained": True}
