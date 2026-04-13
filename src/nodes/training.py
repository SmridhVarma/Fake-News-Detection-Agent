import os
import shutil
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.state import AgentState
from src.utils.training_artifacts import (
    load_artifacts,
    save_artifacts,
    save_model,
)


# =========================
# Utility helpers
# =========================

def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
    }


def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def select_best_model(results: dict) -> str:
    return sorted(
        results.keys(),
        key=lambda name: (results[name]["f1"], results[name]["auc_roc"]),
        reverse=True
    )[0]


# =========================
# Traditional ML preparation
# =========================

def build_traditional_feature_matrices(train_df, val_df, test_df, numeric_feature_cols):
    X_text_train = train_df["text_ml"]
    X_text_val = val_df["text_ml"]
    X_text_test = test_df["text_ml"]

    X_num_train = train_df[numeric_feature_cols].copy()
    X_num_val = val_df[numeric_feature_cols].copy()
    X_num_test = test_df[numeric_feature_cols].copy()

    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )

    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_val_tfidf = tfidf.transform(X_text_val)
    X_test_tfidf = tfidf.transform(X_text_test)

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train_final = hstack([X_train_tfidf, csr_matrix(X_num_train_scaled)])
    X_val_final = hstack([X_val_tfidf, csr_matrix(X_num_val_scaled)])
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_num_test_scaled)])

    return {
        "X_train_final": X_train_final,
        "X_val_final": X_val_final,
        "X_test_final": X_test_final,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
    }


def train_traditional_models(X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, random_state=42):
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "svm": CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=random_state,
                dual="auto"
            ),
            cv=3
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

    candidate_validation_results = {}
    candidate_test_results = {}
    trained_models = {}

    for model_name, model in candidates.items():
        model.fit(X_train_final, y_train)

        # Validation set evaluation
        y_val_pred = model.predict(X_val_final)
        if hasattr(model, "predict_proba"):
            y_val_score = model.predict_proba(X_val_final)[:, 1]
        else:
            y_val_score = model.decision_function(X_val_final)
        candidate_validation_results[model_name] = compute_metrics(y_val, y_val_pred, y_val_score)

        # Test set evaluation
        y_test_pred = model.predict(X_test_final)
        if hasattr(model, "predict_proba"):
            y_test_score = model.predict_proba(X_test_final)[:, 1]
        else:
            y_test_score = model.decision_function(X_test_final)
        candidate_test_results[model_name] = compute_metrics(y_test, y_test_pred, y_test_score)

        trained_models[model_name] = model

    return candidate_validation_results, candidate_test_results, trained_models


# =========================
# Main training node
# =========================

def training_node(state: AgentState) -> dict:
    """
    Train and compare traditional ML models using TF-IDF + Handcrafted features.
    Integrates upstream validation split support with local diagnostic logs and v2 bypass.
    """
    print("\n>>> [NODE] Starting Training Node...")
    
    # Check for v2 artifacts to skip retraining
    v2_training_artifact_path = "./models/v2/training_artifacts.joblib"
    if os.path.exists(v2_training_artifact_path):
        print(f">>> [LOG] v2 training artifacts found at {v2_training_artifact_path}. Skipping model fitting.")
        v2_artifacts = load_artifacts(v2_training_artifact_path)
        if v2_artifacts:
            print(">>> [NODE] Finished Training Node.")
            return {
                "model_trained": True,
                "model_path": v2_artifacts.get("selected_model_path"),
                "training_artifact_path": v2_training_artifact_path,
                "candidate_validation_results": v2_artifacts.get("candidate_validation_results"),
                "candidate_test_results": v2_artifacts.get("candidate_test_results"),
                "candidate_results": v2_artifacts.get("candidate_results"),
                "selected_model_name": v2_artifacts.get("selected_model_name"),
                "selected_model_validation_metrics": v2_artifacts.get("selected_model_validation_metrics"),
                "selected_model_test_metrics": v2_artifacts.get("selected_model_test_metrics"),
                "selected_model_metrics": v2_artifacts.get("selected_model_metrics"),
                "saved_model_paths": v2_artifacts.get("saved_model_paths"),
            }

    preprocess_artifact_path = state.get(
        "preprocessing_artifact_path",
        "./models/v1/preprocessing_artifacts.joblib"
    )
    training_artifact_path = state.get(
        "training_artifact_path",
        "./models/v1/training_artifacts.joblib"
    )
    model_dir = state.get(
        "model_dir",
        "./models/v1"
    )

    os.makedirs(model_dir, exist_ok=True)

    artifacts = load_artifacts(preprocess_artifact_path)

    if artifacts is None:
        raise ValueError("Preprocessing artifacts not found. Run preprocess_data_node first.")

    train_df = artifacts["train_df"]
    val_df = artifacts["val_df"]
    test_df = artifacts["test_df"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]
    random_state = artifacts.get("random_state", 42)

    # =========================
    # Traditional models
    # =========================
    traditional_bundle = build_traditional_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_feature_cols=numeric_feature_cols,
    )

    X_train_final = traditional_bundle["X_train_final"]
    X_val_final = traditional_bundle["X_val_final"]
    X_test_final = traditional_bundle["X_test_final"]
    y_train = traditional_bundle["y_train"]
    y_val = traditional_bundle["y_val"]
    y_test = traditional_bundle["y_test"]
    tfidf = traditional_bundle["tfidf_vectorizer"]
    scaler = traditional_bundle["numeric_scaler"]

    candidate_validation_results, candidate_test_results, traditional_models = train_traditional_models(
        X_train_final=X_train_final,
        X_val_final=X_val_final,
        X_test_final=X_test_final,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        random_state=random_state,
    )

    saved_model_paths = {}

    for model_name, model in traditional_models.items():
        model_path = save_model(model, path=f"{model_dir}/{model_name}.joblib")
        saved_model_paths[model_name] = model_path

    # =========================
    # Select winner based on validation performance
    # =========================
    best_model_name = select_best_model(candidate_validation_results)
    best_validation_metrics = candidate_validation_results[best_model_name]
    best_test_metrics = candidate_test_results[best_model_name]
    best_model_path = saved_model_paths[best_model_name]

    training_bundle = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "selected_model_path": best_model_path,
        "saved_model_paths": saved_model_paths,
        "preprocessing_summary": artifacts.get("preprocessing_summary", {}),
    }

    final_artifact_path = save_artifacts(
        training_bundle,
        path=training_artifact_path
    )

    result = {
        "model_trained": True,
        "model_path": best_model_path,
        "training_artifact_path": final_artifact_path,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "saved_model_paths": saved_model_paths,
    }
    print(">>> [NODE] Finished Training Node.")
    return result