"""
evaluator_node — Runs DeepEval metrics on the LLM output.

Evaluates the quality / faithfulness of the LLM's fact-check.
"""

from src.state import AgentState
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import os

def evaluator_node(state: AgentState) -> dict:
    """Evaluate the LLM output using DeepEval metrics."""
    print("\n>>> [NODE] Starting Evaluator Node...")
    article_text = state.get("article_text", "")
    llm_label = state.get("llm_label", "")
    llm_reasoning = state.get("llm_reasoning", "")
    
    if not article_text or not llm_reasoning:
        result = {"eval_score": 0.0}
        print(">>> [NODE] Finished Evaluator Node.")
        return result
        
    try:
        # Use a GEval metric to evaluate how well the reasoning aligns with the text
        metric = GEval(
            name="Reasoning Quality",
            criteria="Determine whether the reasoning logically follows from the provided article text and justifies the given label.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model="gpt-4o-mini",
            threshold=0.5,
            model_kwargs={"temperature": 0.0},
        )
        
        test_case = LLMTestCase(
            input=article_text[:4000],  # Truncate to ensure we don't blow up context limit unnecessarily during eval
            actual_output=f"Label: {llm_label}\nReasoning: {llm_reasoning}"
        )
        
        metric.measure(test_case)
        result = {"eval_score": metric.score}
        print(">>> [NODE] Finished Evaluator Node.")
        return result
    except Exception as e:
        print(f"Error during DeepEval execution: {e}")
        result = {"eval_score": 0.0}
        print(">>> [NODE] Finished Evaluator Node.")
        return result
