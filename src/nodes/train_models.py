"""
train_models_node — Fit candidate classical ML models.

Builds TF-IDF + handcrafted feature matrices from the preprocessing
artifacts and fits Logistic Regression, calibrated LinearSVC, Random
Forest, and MLP candidates. Persists an intermediate artifact consumed
by the downstream evaluate_models and select_model nodes.

Skips fitting entirely if ./models/v2/training_artifacts.joblib exists
(inference bypass for cached models).
"""

import os

from src.state import AgentState
from src.utils.training_artifacts import load_artifacts, save_artifacts
from src.nodes.training import build_traditional_feature_matrices

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


V2_TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"


def build_candidates(random_state: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "svm": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=random_state, dual="auto"),
            cv=3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=20,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def train_models_node(state: AgentState) -> dict:
    print("\n>>> [NODE] Starting Train Models Node...")

    if os.path.exists(V2_TRAINING_ARTIFACT_PATH):
        print(f">>> [LOG] v2 training artifacts found at {V2_TRAINING_ARTIFACT_PATH}. Skipping training.")
        print(">>> [NODE] Finished Train Models Node.")
        return {"trained_candidates_path": None, "training_cache_hit": True}

    preprocess_artifact_path = state.get(
        "preprocessing_artifact_path",
        "./models/v1/preprocessing_artifacts.joblib",
    )
    model_dir = state.get("model_dir", "./models/v1")
    os.makedirs(model_dir, exist_ok=True)

    artifacts = load_artifacts(preprocess_artifact_path)
    if artifacts is None:
        raise ValueError("Preprocessing artifacts not found. Run preprocess_data_node first.")

    train_df = artifacts["train_df"]
    val_df = artifacts["val_df"]
    test_df = artifacts["test_df"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]
    random_state = artifacts.get("random_state", 42)

    bundle = build_traditional_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_feature_cols=numeric_feature_cols,
    )

    candidates = build_candidates(random_state=random_state)
    trained_models = {}
    for name, model in candidates.items():
        model.fit(bundle["X_train_final"], bundle["y_train"])
        trained_models[name] = model

    intermediate = {
        "trained_models": trained_models,
        "X_train_final": bundle["X_train_final"],
        "X_val_final": bundle["X_val_final"],
        "X_test_final": bundle["X_test_final"],
        "y_train": bundle["y_train"],
        "y_val": bundle["y_val"],
        "y_test": bundle["y_test"],
        "tfidf_vectorizer": bundle["tfidf_vectorizer"],
        "numeric_scaler": bundle["numeric_scaler"],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "random_state": random_state,
        "preprocessing_summary": artifacts.get("preprocessing_summary", {}),
    }

    trained_path = save_artifacts(intermediate, path=f"{model_dir}/trained_candidates.joblib")

    print(">>> [NODE] Finished Train Models Node.")
    return {
        "trained_candidates_path": trained_path,
        "training_cache_hit": False,
        "candidate_model_names": list(trained_models.keys()),
        "model_dir": model_dir,
    }
