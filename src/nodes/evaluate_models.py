"""
evaluate_models_node — Compute validation and test metrics for every
trained candidate and produce an evaluation artifact consumed by
select_model_node.

Metrics: accuracy, precision, recall, F1, AUC-ROC on both the validation
and test splits. Skips if the v2 training cache is already present.
"""

import os

from src.state import AgentState
from src.utils.training_artifacts import load_artifacts, save_artifacts
from src.nodes.training import compute_metrics

V2_TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"


def evaluate_model(model, X_val, y_val, X_test, y_test) -> tuple[dict, dict]:
    y_val_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_val_score = model.predict_proba(X_val)[:, 1]
    else:
        y_val_score = model.decision_function(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_score)

    y_test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_test_score = model.predict_proba(X_test)[:, 1]
    else:
        y_test_score = model.decision_function(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_score)

    return val_metrics, test_metrics


def evaluate_models_node(state: AgentState) -> dict:
    print("\n>>> [NODE] Starting Evaluate Models Node...")

    if state.get("training_cache_hit") or os.path.exists(V2_TRAINING_ARTIFACT_PATH):
        print(">>> [LOG] Skipping evaluation — v2 training cache is present.")
        print(">>> [NODE] Finished Evaluate Models Node.")
        return {"evaluation_artifact_path": None, "training_cache_hit": True}

    trained_candidates_path = state.get("trained_candidates_path")
    if not trained_candidates_path:
        raise ValueError("trained_candidates_path missing. Run train_models_node first.")

    bundle = load_artifacts(trained_candidates_path)
    if bundle is None:
        raise ValueError(f"Could not load trained candidates at {trained_candidates_path}.")

    trained_models = bundle["trained_models"]
    X_val = bundle["X_val_final"]
    X_test = bundle["X_test_final"]
    y_val = bundle["y_val"]
    y_test = bundle["y_test"]

    candidate_validation_results = {}
    candidate_test_results = {}
    for name, model in trained_models.items():
        val_metrics, test_metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
        candidate_validation_results[name] = val_metrics
        candidate_test_results[name] = test_metrics

    model_dir = state.get("model_dir", "./models/v1")
    eval_bundle = {
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
    }
    eval_path = save_artifacts(eval_bundle, path=f"{model_dir}/evaluation_results.joblib")

    print(">>> [NODE] Finished Evaluate Models Node.")
    return {
        "evaluation_artifact_path": eval_path,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "training_cache_hit": False,
    }
