---
name: aggregation
description: Combine the ML classifier's probability and the LLM classifier's verdict into a single final label, confidence, and user-facing summary for the Gradio UI.
---

# Aggregation Skill

You are the Aggregator Agent. Your job is to merge the ML and LLM signals into one final verdict and produce the business-facing summary and explanation blocks that the Gradio interface renders to the end user.

## When to use
- Once, at the very end of the inference pipeline, after ML Classification, LLM Classification, and Evaluation have all completed.
- Always — this stage is required for producing the UI response.

## How to execute
1. **Thought**: Check whether the ML and LLM labels agree, and how wide the confidence gap is.
2. **Action**:
   - Convert both signals to `P(REAL)` on [0, 1]. `ml_score` is already `P(REAL)`; map `llm_score` via `llm_score if llm_label == "REAL" else 1 - llm_score`.
   - Average the two probabilities with equal weight.
   - Derive the final label (`REAL` if `p_real_final >= 0.5` else `FAKE`) and final confidence.
   - Compose a one-sentence `summary` and a markdown `explanation` block that cites both model outputs, the DeepEval reasoning score, and the LLM's written reasoning.
3. **Observation**: Return label, score, summary, explanation, and agreement flags to state.

## Inputs from agent state
- `ml_score`, `ml_label`: From the ML Classification stage.
- `llm_score`, `llm_label`, `llm_reasoning`: From the LLM Classification stage.
- `eval_score`: From the Evaluation stage.

## Outputs to agent state
- `final_label`: `"REAL"` or `"FAKE"`.
- `final_score`: Confidence in the final label, in [0.0, 1.0].
- `summary`: One-sentence human-readable verdict.
- `explanation`: Markdown block displayed in the Gradio UI.
- `eval_agreement`: Boolean — whether ML and LLM labels matched.
- `eval_confidence_delta`: Absolute difference between `ml_score` and `llm_score`.

## Output format
```json
{
  "final_label": "REAL",
  "final_score": 0.87,
  "summary": "The article is classified as REAL with 0.87 confidence. Both the ML model and the AI agreed on this verdict.",
  "explanation": "### Analysis Results:\n- **ML Model:** REAL (Confidence: 0.91)\n- **AI Classifier:** REAL (Confidence: 0.83)\n- **DeepEval Reasoning Score:** 0.78/1.0\n\n### AI Reasoning:\n...",
  "eval_agreement": true,
  "eval_confidence_delta": 0.08
}
```

## Notes
- Logic lives in `src/nodes/aggregator.py`.
- Do NOT invert `ml_score` when `ml_label == "FAKE"` — it is already `P(REAL)`, not "confidence in the predicted label".
- When ML and LLM disagree, the averaging rule lets whichever signal is more confident dominate, which is the desired behavior for ambiguous articles.
- The `explanation` field is the primary business-facing output consumed by the Gradio UI; keep its markdown structure stable so the UI renders consistently.
