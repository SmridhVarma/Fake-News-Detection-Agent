"""
evaluator_node — Runs DeepEval metrics on the LLM output.

Evaluates the quality / faithfulness of the LLM's fact-check.
"""

from src.state import AgentState


def evaluator_node(state: AgentState) -> dict:
    """Evaluate the LLM output using DeepEval metrics."""
    # TODO: Build LLMTestCase from state
    # TODO: Run HallucinationMetric / FaithfulnessMetric
    # TODO: Return {"eval_score": ...}
    return {"eval_score": 0.0}
