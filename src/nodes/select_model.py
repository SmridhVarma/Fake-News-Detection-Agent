"""
select_model_node — Pick the winning candidate based on validation
metrics, persist per-model artifacts plus the consolidated training
bundle, and emit the final paths consumed by ml_classifier_node.

Decision rule: rank candidates by (F1 desc, AUC-ROC desc) on the
validation split. Test metrics are never used for selection.
"""

import os

from src.state import AgentState
from src.utils.training_artifacts import load_artifacts, save_artifacts, save_model
from src.nodes.training import select_best_model

V2_TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"


def select_model_node(state: AgentState) -> dict:
    print("\n>>> [NODE] Starting Select Model Node...")

    if state.get("training_cache_hit") or os.path.exists(V2_TRAINING_ARTIFACT_PATH):
        cached = load_artifacts(V2_TRAINING_ARTIFACT_PATH)
        if cached is not None:
            print(f">>> [LOG] Using cached v2 training artifacts at {V2_TRAINING_ARTIFACT_PATH}.")
            print(">>> [NODE] Finished Select Model Node.")
            return {
                "model_trained": True,
                "model_path": cached.get("selected_model_path"),
                "training_artifact_path": V2_TRAINING_ARTIFACT_PATH,
                "candidate_validation_results": cached.get("candidate_validation_results"),
                "candidate_test_results": cached.get("candidate_test_results"),
                "candidate_results": cached.get("candidate_results"),
                "selected_model_name": cached.get("selected_model_name"),
                "selected_model_validation_metrics": cached.get("selected_model_validation_metrics"),
                "selected_model_test_metrics": cached.get("selected_model_test_metrics"),
                "selected_model_metrics": cached.get("selected_model_metrics"),
                "saved_model_paths": cached.get("saved_model_paths"),
            }

    trained_candidates_path = state.get("trained_candidates_path")
    eval_artifact_path = state.get("evaluation_artifact_path")
    if not trained_candidates_path or not eval_artifact_path:
        raise ValueError("Missing train/evaluate artifacts. Run train_models_node and evaluate_models_node first.")

    trained_bundle = load_artifacts(trained_candidates_path)
    eval_bundle = load_artifacts(eval_artifact_path)
    if trained_bundle is None or eval_bundle is None:
        raise ValueError("Could not load intermediate artifacts.")

    trained_models = trained_bundle["trained_models"]
    candidate_validation_results = eval_bundle["candidate_validation_results"]
    candidate_test_results = eval_bundle["candidate_test_results"]

    model_dir = state.get("model_dir", "./models/v1")
    os.makedirs(model_dir, exist_ok=True)

    saved_model_paths = {}
    for name, model in trained_models.items():
        saved_model_paths[name] = save_model(model, path=f"{model_dir}/{name}.joblib")

    best_model_name = select_best_model(candidate_validation_results)
    best_validation_metrics = candidate_validation_results[best_model_name]
    best_test_metrics = candidate_test_results[best_model_name]
    best_model_path = saved_model_paths[best_model_name]

    training_bundle = {
        "train_df": trained_bundle["train_df"],
        "val_df": trained_bundle["val_df"],
        "test_df": trained_bundle["test_df"],
        "numeric_feature_cols": trained_bundle["numeric_feature_cols"],
        "tfidf_vectorizer": trained_bundle["tfidf_vectorizer"],
        "numeric_scaler": trained_bundle["numeric_scaler"],
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "selected_model_path": best_model_path,
        "saved_model_paths": saved_model_paths,
        "preprocessing_summary": trained_bundle.get("preprocessing_summary", {}),
    }

    training_artifact_path = state.get(
        "training_artifact_path",
        f"{model_dir}/training_artifacts.joblib",
    )
    final_path = save_artifacts(training_bundle, path=training_artifact_path)

    print(">>> [NODE] Finished Select Model Node.")
    return {
        "model_trained": True,
        "model_path": best_model_path,
        "training_artifact_path": final_path,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "saved_model_paths": saved_model_paths,
    }
