"""
llm_classifier_node — Phase 2: LLM-based fact-checking.

Sends article to OpenAI for analysis with a structured prompt.
"""

from src.state import AgentState
from src.utils.prompts import FACT_CHECK_PROMPT


def llm_classifier_node(state: AgentState) -> dict:
    """Ask the LLM to fact-check the article and return score + reasoning."""
    # TODO: Build prompt using FACT_CHECK_PROMPT + state["article_text"]
    # TODO: Call OpenAI API via LangChain
    # TODO: Parse structured response
    # TODO: Return {"llm_score": ..., "llm_label": ..., "llm_reasoning": ...}
    return {"llm_score": 0.0, "llm_label": "", "llm_reasoning": ""}
