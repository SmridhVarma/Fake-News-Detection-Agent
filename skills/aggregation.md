---
name: aggregation
description: Combine the ML classifier's probability and the LLM classifier's verdict into a single final label, confidence, and user-facing summary for the Gradio UI.
mode: organisational
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
   - Look up `eval_score` from state. Choose weights via a three-tier rule:

     | eval_score   | ml_weight | llm_weight |
     |--------------|-----------|------------|
     | ≥ 0.7        | 0.50      | 0.50       |
     | 0.3 – 0.69   | 0.60      | 0.40       |
     | < 0.3        | 0.70      | 0.30       |

   - Compute `p_real_final = ml_weight · p_real_ml + llm_weight · p_real_llm`.
   - Derive the final label (`REAL` if `p_real_final >= 0.5` else `FAKE`) and final confidence.
   - Compose a one-sentence `summary` (including a weight-tier note) and a markdown `explanation` block that cites both model outputs, the DeepEval reasoning score, and the LLM's written reasoning.
3. **Observation**: Return label, score, summary, explanation, agreement flags, and weights to state.

## Inputs from agent state
- `ml_score`, `ml_label`: From the ML Classification stage.
- `llm_score`, `llm_label`, `llm_reasoning`: From the LLM Classification stage.
- `eval_score`: From the Evaluation stage.

## Outputs to agent state
- `final_label`: `"REAL"` or `"FAKE"`.
- `final_score`: Confidence in the final label, in [0.0, 1.0].
- `summary`: One-sentence human-readable verdict including weight-tier note.
- `explanation`: Markdown block displayed in the Gradio UI.
- `eval_agreement`: Boolean — whether ML and LLM labels matched.
- `eval_confidence_delta`: Absolute difference between `ml_score` and `llm_score`.
- `ml_weight`: Weight applied to the ML signal in the final blend.
- `llm_weight`: Weight applied to the LLM signal in the final blend.

## Output format
```json
{
  "final_label": "REAL",
  "final_score": 0.87,
  "summary": "The article is classified as REAL with 0.87 confidence. LLM reasoning quality was high — equal weight applied.",
  "explanation": "### Analysis Results:\n- **ML Model:** REAL (Confidence: 0.91)\n- **AI Classifier:** REAL (Confidence: 0.83)\n- **DeepEval Reasoning Score:** 0.78/1.0\n\n### AI Reasoning:\n...",
  "eval_agreement": true,
  "eval_confidence_delta": 0.08,
  "ml_weight": 0.50,
  "llm_weight": 0.50
}
```

## Notes
- Logic lives in `src/nodes/aggregator.py`.
- Do NOT invert `ml_score` when `ml_label == "FAKE"` — it is already `P(REAL)`, not "confidence in the predicted label".
- ML `predict_proba` is preferred by default because it is calibrated (especially for logistic regression); LLM confidence is uncalibrated and shifts weight toward ML when LLM reasoning quality is low.
- When ML and LLM disagree, the weighted blend lets the more calibrated signal lead; the weight tiers ensure a confidently wrong LLM with poor reasoning does limited damage.
- The `explanation` field is the primary business-facing output consumed by the Gradio UI; keep its markdown structure stable so the UI renders consistently.
