---
name: evaluate_models
description: Compute classification metrics (accuracy, precision, recall, F1, AUC-ROC) for every trained ML candidate on the validation and test splits, and record them for the Select Model stage and UI visualization.
mode: organisational
---

# Evaluate Models Skill

You are the Model Evaluation Agent. Your job is to measure every fitted candidate on both the validation and test splits, produce a structured metrics report, and persist it for downstream selection and user-facing display.

## When to use
- Immediately after the Train Models stage, once the fitted candidates artifact is on disk.
- Whenever the v2 training cache is absent — this stage is skipped otherwise.
- Never at inference time on a single article — this is an offline model-build stage only.

## How to execute
1. **Thought**: Confirm that `trained_candidates_path` is present in state and loadable.
2. **Action**:
   - Load the trained candidates artifact (fitted models + split matrices).
   - For each candidate, run `predict` on the validation matrix and, where available, `predict_proba(class=1)`; otherwise fall back to `decision_function` for the AUC score.
   - Compute the metric tuple on the validation split.
   - Repeat on the test split for diagnostic (not selection) purposes.
   - Persist `candidate_validation_results` and `candidate_test_results` to `./models/v1/evaluation_results.joblib`.
3. **Observation**: Emit both metric dictionaries back to state.

## Metrics computed
For each candidate, on **both** the validation and test splits:
- **accuracy** — `sklearn.metrics.accuracy_score`
- **precision** — `precision_score(zero_division=0)`
- **recall** — `recall_score(zero_division=0)`
- **f1** — `f1_score(zero_division=0)`
- **auc_roc** — `roc_auc_score` using `predict_proba[:, 1]` when available, else `decision_function`.

All values are rounded to 4 decimal places.

## Visualizations produced
Two plots are generated automatically and saved to `{model_dir}/plots/`:

- **`roc_curves.png`** — ROC curve overlay for all four candidates on the test split. Each model is colour-coded; the diagonal random-classifier baseline is included. AUC values are shown in the legend.
- **`confusion_matrices.png`** — Side-by-side confusion matrix heatmap (Blues palette) for every candidate on the test split, with FAKE/REAL axis labels.

Both images are also loaded and displayed in the Jupyter notebook's evaluation section.

## Inputs from agent state
- `trained_candidates_path`: Path written by the Train Models stage.
- `training_cache_hit` (optional): If `True`, skip and pass through.
- `model_dir` (optional): Output directory for the evaluation artifact.

## Outputs to agent state
- `evaluation_artifact_path`: Path to the evaluation results joblib.
- `candidate_validation_results`: `{model_name: metrics_dict}` on the validation split.
- `candidate_test_results`: `{model_name: metrics_dict}` on the test split.
- `roc_curve_path`: File path to the saved ROC curve overlay PNG.
- `confusion_matrix_path`: File path to the saved confusion matrix heatmap PNG.
- `training_cache_hit`: Propagated flag.

## Output format
```json
{
  "evaluation_artifact_path": "./models/v1/evaluation_results.joblib",
  "candidate_validation_results": {
    "logistic_regression": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0},
    "svm":                 {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0},
    "random_forest":       {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0},
    "neural_network":      {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0}
  },
  "candidate_test_results": { "...": "same shape as validation_results" },
  "roc_curve_path": "./models/v1/plots/roc_curves.png",
  "confusion_matrix_path": "./models/v1/plots/confusion_matrices.png",
  "training_cache_hit": false
}
```

## Notes
- Logic lives in `src/nodes/evaluate_models.py`.
- Test metrics are computed for **diagnostics only** — they must not feed into model selection, or the test split stops being a held-out estimate of generalization.
- `LinearSVC` wrapped in `CalibratedClassifierCV` exposes `predict_proba`, so AUC uses the calibrated probabilities rather than the raw `decision_function`.
- When the cache is hit, metrics are read from the cached training bundle in the Select Model stage; this stage emits no new metrics.
