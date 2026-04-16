---
name: reasoning_evaluation
description: Score the LLM classifier's written reasoning for faithfulness to the article using DeepEval's GEval metric, producing a trust signal surfaced in the Gradio UI.
mode: llm_driven
---

# Reasoning Evaluation Skill

You are the Reasoning Quality Agent. Your job is to measure whether the LLM classifier's `llm_reasoning` logically follows from the article text and justifies the assigned label. This stage is distinct from `evaluate_models` — it evaluates the **LLM's written reasoning**, not the trained ML candidates.

## When to use
- After the LLM Classification stage has emitted a label and written reasoning.
- Before the Aggregator combines ML and LLM verdicts, so the aggregator can surface the reasoning score to the user.
- Only when both the article text and LLM reasoning are non-empty — otherwise short-circuit with `eval_score = 0.0`.

## How to execute
1. **Thought**: Confirm `article_text` and `llm_reasoning` are present; otherwise return 0.0 without calling DeepEval.
2. **Action**:
   - Construct a DeepEval `GEval` metric with criteria: "Determine whether the reasoning logically follows from the provided article text and justifies the given label."
   - Build a `LLMTestCase` where `input` is the article text truncated to ~4000 characters and `actual_output` is `"Label: {llm_label}\nReasoning: {llm_reasoning}"`.
   - Call `metric.measure(test_case)`.
3. **Observation**: Record the numeric `metric.score`.

## Inputs from agent state
- `article_text`: Original article body.
- `llm_label`: REAL/FAKE verdict from the LLM Classification stage.
- `llm_reasoning`: Written justification from the LLM Classification stage.

## Outputs to agent state
- `eval_score`: Float in [0.0, 1.0] representing reasoning faithfulness.

## Output format
```json
{
  "eval_score": 0.0
}
```

## Notes
- Logic lives in `src/nodes/evaluator.py`.
- Article text is truncated to ~4000 characters before being passed to DeepEval to avoid context-length blowups.
- A low `eval_score` does NOT override the verdict — it is a trust signal displayed in the Gradio UI so users can judge the LLM's reliability for this specific article.
- DeepEval failures (missing API key, network error, rate limit) return `eval_score = 0.0` and log the exception rather than crashing the pipeline.
- This skill is **not** the ML model evaluation stage — for model metrics (accuracy, precision, recall, F1, AUC-ROC) see `evaluate_models.md`.
