"""
aggregator_node — Combines ML + LLM scores into a final verdict.

Produces the final classification label, score, and human-readable summary.
"""

from src.state import AgentState


def _compute_weights(eval_score: float) -> tuple[float, float]:
    # ML predict_proba is calibrated; LLM confidence is not.
    # Shift toward LLM only when its reasoning quality (eval_score) justifies it.
    if eval_score >= 0.7:
        return 0.50, 0.50   # strong reasoning → equal weight
    elif eval_score >= 0.3:
        return 0.60, 0.40   # moderate → ML leads
    else:
        return 0.70, 0.30   # poor reasoning → ML dominant


def aggregator_node(state: AgentState) -> dict:
    """Aggregate both phase scores and produce a final summary."""
    print("\n>>> [NODE] Starting Aggregator Node...")
    ml_score = state.get("ml_score", 0.0)
    llm_score = state.get("llm_score", 0.0)
    ml_label = state.get("ml_label", "UNKNOWN").upper()
    llm_label = state.get("llm_label", "UNKNOWN").upper()

    # Check if labels match
    eval_agreement = (ml_label == llm_label) and ml_label != "UNKNOWN"
    eval_confidence_delta = abs(ml_score - llm_score)
    eval_score = state.get("eval_score", 0.0)

    # Convert both signals into P(REAL) on [0, 1] before blending.
    # ML score is already P(REAL) from predict_proba(class=1) — do NOT invert.
    p_real_ml = ml_score

    # LLM score is confidence for its predicted label, so map to P(REAL).
    p_real_llm = llm_score if llm_label == "REAL" else 1.0 - llm_score

    ml_weight, llm_weight = _compute_weights(eval_score)
    p_real_final = ml_weight * p_real_ml + llm_weight * p_real_llm

    final_label = "REAL" if p_real_final >= 0.5 else "FAKE"
    final_score = p_real_final if final_label == "REAL" else 1.0 - p_real_final

    weight_note = (
        "LLM reasoning quality was high — equal weight applied."
        if eval_score >= 0.7
        else "LLM reasoning quality was moderate — ML-led weighting applied."
        if eval_score >= 0.3
        else "LLM reasoning quality was low — ML-dominant weighting applied."
    )
    summary = f"The article is classified as {final_label} with {final_score:.2f} confidence. {weight_note}"
    if not eval_agreement:
        summary += f" (ML: {ml_label}, AI: {llm_label})"

    explanation = f"""
### Analysis Results:
- **ML Model:** {ml_label} (Confidence: {ml_score:.2f})
- **AI Classifier:** {llm_label} (Confidence: {llm_score:.2f})
- **DeepEval Reasoning Score:** {eval_score:.2f}/1.0

### AI Reasoning:
{state.get("llm_reasoning", "No reasoning provided.")}
"""

    result = {
        "final_label": final_label,
        "final_score": final_score,
        "summary": summary,
        "explanation": explanation,
        "eval_agreement": eval_agreement,
        "eval_confidence_delta": eval_confidence_delta,
        "ml_weight": ml_weight,
        "llm_weight": llm_weight,
    }
    print(">>> [NODE] Finished Aggregator Node.")
    return result
