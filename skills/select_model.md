---
name: select_model
description: Choose the winning ML candidate based on validation-split performance, persist per-model artifacts plus the consolidated training bundle, and emit the final inference-time model path.
---

# Select Model Skill

You are the Model Selection Agent. Your job is to pick the single best candidate from the Evaluate Models stage, justify the choice using validation metrics only, and persist all artifacts required for inference-time reuse.

## When to use
- Immediately after the Evaluate Models stage.
- Whenever the v2 training cache is absent (otherwise the cached winner is passed straight through).
- Never at inference time — this is an offline model-build stage.

## How to execute
1. **Thought**: Load both the fitted candidates artifact and the evaluation results artifact.
2. **Action**:
   - Apply the decision rule (see below) to `candidate_validation_results` to pick the winner.
   - Save every candidate to `./models/v1/<model_name>.joblib` for reproducibility.
   - Build the consolidated training bundle (TF-IDF, scaler, all models, all metrics, splits, winner metadata).
   - Save the bundle to `training_artifact_path` so the ML Classification stage can load it.
3. **Observation**: Emit the winner's name, model path, and validation + test metrics.

## Decision logic
Candidates are ranked by a lexicographic key on the **validation** split only:

```python
best = max(
    candidate_validation_results,
    key=lambda name: (
        candidate_validation_results[name]["f1"],
        candidate_validation_results[name]["auc_roc"],
    ),
)
```

Justification:
- **Primary: F1** — fake-news classification cares about balancing precision and recall. Accuracy alone can be inflated by class imbalance and by leakage shortcuts that survive preprocessing.
- **Tie-breaker: AUC-ROC** — if two candidates are within rounding distance on F1, AUC-ROC favors the model whose probability ranking generalizes better, which matters for the downstream probabilistic aggregation with the LLM verdict.
- **Validation split only** — the test split is reserved as a held-out estimate of generalization for the final report. Selecting on test would leak the test set into the model-choice decision.

This produces a reproducible, deterministic winner given a fixed random seed, which keeps UI metric displays stable across runs.

## Inputs from agent state
- `trained_candidates_path`: Fitted models artifact from Train Models.
- `evaluation_artifact_path`: Metrics artifact from Evaluate Models.
- `training_cache_hit` (optional): If `True`, pass through the cached winner.
- `training_artifact_path` (optional): Where to write the final bundle.
- `model_dir` (optional): Output directory for per-model saves.

## Outputs to agent state
- `model_trained`: `True`.
- `model_path`: Path to the winning model's joblib.
- `training_artifact_path`: Path to the consolidated bundle used by inference.
- `selected_model_name`: Winner key (e.g. `"logistic_regression"`).
- `selected_model_validation_metrics`, `selected_model_test_metrics`: The winner's metric dicts.
- `candidate_validation_results`, `candidate_test_results`: Full metrics map for UI display.
- `saved_model_paths`: `{model_name: path}` for all candidates.

## Output format
```json
{
  "model_trained": true,
  "model_path": "./models/v1/logistic_regression.joblib",
  "training_artifact_path": "./models/v1/training_artifacts.joblib",
  "selected_model_name": "logistic_regression",
  "selected_model_validation_metrics": {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0
  },
  "selected_model_test_metrics": {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0
  },
  "saved_model_paths": {
    "logistic_regression": "./models/v1/logistic_regression.joblib",
    "svm":                 "./models/v1/svm.joblib",
    "random_forest":       "./models/v1/random_forest.joblib",
    "neural_network":      "./models/v1/neural_network.joblib"
  }
}
```

## Notes
- Logic lives in `src/nodes/select_model.py`.
- The consolidated bundle (`training_artifact_path`) is the single source of truth consumed by `ml_classifier_node`; it intentionally duplicates TF-IDF and scaler alongside the model so inference only needs one file.
- When the v2 cache is present, the node returns the cached winner and metrics unchanged — no re-selection, no re-save.
- Selection is never based on test metrics. Any future change to the decision logic must preserve this invariant.
