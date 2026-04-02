"""
ingestion_node — Receives raw article text and preprocesses it.

Example node pattern:
  - Takes AgentState as input
  - Returns a dict with ONLY the keys it updates
"""

from src.state import AgentState
from src.utils.preprocessing import clean_text


def ingestion_node(state: AgentState) -> dict:
    """Clean and prepare the article text for downstream nodes."""
    raw_text = state["article_text"]
    cleaned = clean_text(raw_text)
    return {"article_text": cleaned}
