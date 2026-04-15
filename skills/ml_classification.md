---
name: ml_classification
description: Apply the cached TF-IDF + handcrafted-feature pipeline and the winning classical ML model to produce a REAL/FAKE probability for a single article.
---

# ML Classification Skill

You are the ML Inference Agent. Your job is to run the offline-trained scikit-learn pipeline on a single incoming article and return a calibrated probability plus label.

## When to use
- At inference time, immediately after Preprocessing has scrubbed the article.
- For every article, in parallel with the LLM Classification stage.
- Never during training — this skill consumes artifacts, it does not produce them.

## How to execute
1. **Thought**: Load the cached training bundle (TF-IDF vectorizer, numeric scaler, selected model) from `training_artifact_path`.
2. **Action**:
   - Transform the cleaned article text with the saved TF-IDF vectorizer.
   - Compute handcrafted numeric features on the same text and scale them with the saved `StandardScaler`.
   - Horizontally stack the two feature blocks to match training shape.
   - Call `predict_proba` (or calibrated `decision_function`) on the stacked vector.
3. **Observation**: Extract `P(REAL)`, derive the label, and write both to state.

## Inputs from agent state
- `article_text` or `cleaned_text`: The text to classify.
- `training_artifact_path`: Path to the joblib bundle containing tfidf, scaler, and the selected model.

## Outputs to agent state
- `ml_score`: Probability of REAL in [0.0, 1.0] from `predict_proba(class=1)`.
- `ml_label`: `"REAL"` if `ml_score >= 0.5`, else `"FAKE"`.
- `ml_model_name`: Name of the winning model (for UI display).

## Output format
```json
{
  "ml_score": 0.0,
  "ml_label": "REAL",
  "ml_model_name": "logistic_regression"
}
```

## Notes
- Logic lives in `src/nodes/ml_classifier.py`.
- `ml_score` is already `P(REAL)`; downstream aggregation must **not** invert it when the label is FAKE.
- The feature pipeline must match training exactly (same TF-IDF object, same scaler, same column order for numeric features) or predictions will be silently wrong.
- If the training bundle is missing, fail loudly rather than returning a default 0.5.
