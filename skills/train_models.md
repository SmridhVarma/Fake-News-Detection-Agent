---
name: train_models
description: Fit four traditional ML candidates (Logistic Regression, calibrated LinearSVC, Random Forest, MLP) on TF-IDF + handcrafted features and persist them for downstream evaluation and selection.
---

# Train Models Skill

You are the Model Training Agent. Your job is to fit the full set of classical ML candidates on the preprocessed training split and persist the fitted estimators to disk so the Evaluate Models and Select Model stages can consume them.

## When to use
- Once per model-build phase, after Preprocessing has produced the train/val/test splits.
- Whenever the v2 training cache (`./models/v2/training_artifacts.joblib`) is absent.
- Never at inference time — this stage is auto-skipped when cached artifacts exist.

## How to execute
1. **Thought**: Confirm that preprocessing artifacts exist and contain train/val/test splits plus the numeric feature column list.
2. **Action**:
   - Load `preprocessing_artifact_path` from state.
   - Build TF-IDF (`ngram_range=(1,2)`, `max_features=20000`, `min_df=3`, `max_df=0.95`) on `text_ml`.
   - Scale numeric handcrafted features with `StandardScaler`.
   - Horizontally stack TF-IDF and scaled numeric features into a single sparse matrix per split.
   - Fit each candidate on the training matrix.
   - Save all fitted estimators + feature transformers + splits to `./models/v1/trained_candidates.joblib`.
3. **Observation**: Emit the intermediate artifact path and candidate names.

## Candidate models and configurations
- **logistic_regression** — `LogisticRegression(max_iter=1000, random_state=random_state)`.
- **svm** — `CalibratedClassifierCV(LinearSVC(random_state=random_state, dual="auto"), cv=3)`. Calibration produces proper `predict_proba` scores for downstream averaging.
- **random_forest** — `RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)`.
- **neural_network** — `MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam", alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=20, early_stopping=True, random_state=random_state)`.

Feature pipeline (shared across all candidates):
- TF-IDF: `max_features=20000`, `ngram_range=(1, 2)`, `min_df=3`, `max_df=0.95`.
- Numeric scaler: `StandardScaler()` on `[sub_variance, mean_subjectivity, lexical_density, caps_ratio, has_dateline]`.
- Final feature matrix: `hstack([tfidf_matrix, csr_matrix(scaled_numeric)])`.

## Inputs from agent state
- `preprocessing_artifact_path`: Joblib produced by the Preprocessing stage.
- `model_dir` (optional): Output directory for intermediate artifacts (default `./models/v1`).

## Outputs to agent state
- `trained_candidates_path`: Path to the joblib holding the fitted models, fitted transformers, and split matrices.
- `candidate_model_names`: List of candidate model keys for logging/UI.
- `training_cache_hit`: `True` if the node short-circuited due to the v2 cache.
- `model_dir`: Resolved output directory.

## Output format
```json
{
  "trained_candidates_path": "./models/v1/trained_candidates.joblib",
  "candidate_model_names": ["logistic_regression", "svm", "random_forest", "neural_network"],
  "training_cache_hit": false,
  "model_dir": "./models/v1"
}
```

## Notes
- Logic lives in `src/nodes/train_models.py`.
- The TF-IDF vectorizer and `StandardScaler` are persisted inside the intermediate artifact so the Evaluate and Select stages (and inference) can reproduce the exact same feature pipeline.
- When `./models/v2/training_artifacts.joblib` exists, this node returns immediately with `training_cache_hit=True` — all downstream training stages honor this flag and skip their own work.
- Model selection itself happens in the Select Model stage, not here — this skill is purely a fit step.
