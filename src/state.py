"""
state.py — Shared Agent State

Single source of truth for the data flowing through the LangGraph pipeline.
Every node reads from and writes to this state.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """State shared across all LangGraph nodes."""

    # Input
    article_text: str

    # Phase 1: ML
    ml_score: float
    ml_label: str

    # Phase 2: LLM
    llm_score: float
    llm_label: str
    llm_reasoning: str

    # Evaluation
    eval_score: float

    # Final Output
    final_label: str
    final_score: float
    summary: str

    # Flags
    model_trained: bool
