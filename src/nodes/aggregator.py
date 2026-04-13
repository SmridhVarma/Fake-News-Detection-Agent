"""
aggregator_node — Combines ML + LLM scores into a final verdict.

Produces the final classification label, score, and human-readable summary.
"""

from src.state import AgentState

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
    
    # Convert both signals into P(REAL) on [0, 1] before averaging.
    # ML score is already probability of REAL from predict_proba(class=1),
    # so do NOT invert it when ML label is FAKE.
    p_real_ml = ml_score

    # LLM score is confidence for its predicted label, so map to P(REAL).
    p_real_llm = llm_score if llm_label == "REAL" else 1.0 - llm_score
    
    # 50/50 split
    p_real_final = (p_real_ml + p_real_llm) / 2.0
    
    final_label = "REAL" if p_real_final >= 0.5 else "FAKE"
    final_score = p_real_final if final_label == "REAL" else 1.0 - p_real_final
    
    summary = f"The article is classified as {final_label} with {final_score:.2f} confidence. "
    if eval_agreement:
        summary += "Both the ML model and the AI agreed on this verdict."
    else:
        summary += f"There was disagreement (ML predicted {ml_label}, AI predicted {llm_label}); the final verdict weighed both inputs."
        
    explanation = f"""
### Analysis Results:
- **ML Model:** {ml_label} (Confidence: {ml_score:.2f})
- **AI Classifier:** {llm_label} (Confidence: {llm_score:.2f})
- **DeepEval Reasoning Score:** {state.get("eval_score", 0.0):.2f}/1.0

### AI Reasoning:
{state.get("llm_reasoning", "No reasoning provided.")}
"""

    result = {
        "final_label": final_label,
        "final_score": final_score,
        "summary": summary,
        "explanation": explanation,
        "eval_agreement": eval_agreement,
        "eval_confidence_delta": eval_confidence_delta
    }
    print(">>> [NODE] Finished Aggregator Node.")
    return result
