"""
ml_classifier_node — Phase 1: ML model inference.

Loads the selected trained model and returns a confidence score + label.
Supports:
- classical sklearn models trained on TF-IDF + handcrafted features
"""

import os
import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.state import AgentState
from src.utils.training_artifacts import load_artifacts, load_model
from src.utils.preprocessing import (
    clean_text,
)
from src.utils.ingestion_tools import calculate_article_scores


def _label_from_score(score: float) -> str:
    return "REAL" if score >= 0.5 else "FAKE"


def _run_classical_model(state: AgentState, artifacts: dict, selected_model_name: str) -> dict:
    selected_model_path = artifacts["saved_model_paths"][selected_model_name]
    
    # Fix paths for v2 models if they reference './models/v2/'
    if not os.path.exists(selected_model_path):
        if "./models/" in selected_model_path:
            alt_path = selected_model_path.replace("./models/", "./")
            if os.path.exists(alt_path):
                selected_model_path = alt_path

    model = load_model(selected_model_path)
    if model is None:
        raise ValueError(f"Classical model not found at: {selected_model_path}")

    tfidf = artifacts["tfidf_vectorizer"]
    scaler = artifacts["numeric_scaler"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]

    raw_text = state.get("article_text", "") or state.get("raw_input", "")
    article_text_ml = state.get("article_text_ml", "")

    if not article_text_ml:
        article_text_ml = clean_text(raw_text)

    # 1. Text vector
    X_text = tfidf.transform([article_text_ml])

    # 2. Handcrafted numeric features from original/raw text
    feature_dict = calculate_article_scores(raw_text)

    numeric_values = []
    for col in numeric_feature_cols:
        value = feature_dict.get(col, 0)
        if isinstance(value, bool):
            value = int(value)
        numeric_values.append(value)

    X_num = np.array([numeric_values], dtype=float)
    X_num_scaled = scaler.transform(X_num)

    # 3. Combine TF-IDF + numeric
    X_final = hstack([
        X_text,
        csr_matrix(X_num_scaled),
    ])

    # 4. Predict
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X_final)[:, 1][0])
    elif hasattr(model, "decision_function"):
        # fallback: squash decision score into [0,1]
        decision = float(model.decision_function(X_final)[0])
        score = 1 / (1 + np.exp(-decision))
    else:
        pred = int(model.predict(X_final)[0])
        score = float(pred)

    label = _label_from_score(score)

    return {
        "ml_score": round(score, 4),
        "ml_label": label,
    }


def ml_classifier_node(state: AgentState) -> dict:
    """Run the selected trained ML model on the article and return score + label."""
    print("\n>>> [NODE] Starting ML Classifier Node...")
    artifact_path = state.get("training_artifact_path", "./models/training_artifacts.joblib")

    # Fallback to v2 if models/training_artifacts.joblib doesn't exist
    if not os.path.exists(artifact_path) or "v2" in artifact_path:
         v2_path = "./models/v2/training_artifacts.joblib"
         if os.path.exists(v2_path):
             artifact_path = v2_path

    artifacts = load_artifacts(artifact_path)
    if artifacts is None:
        raise ValueError("Training artifacts not found. Run training_node first.")

    selected_model_name = artifacts.get("selected_model_name")
    if not selected_model_name:
        raise ValueError("No selected model found in training artifacts.")

    result = _run_classical_model(state, artifacts, selected_model_name)
    print(">>> [NODE] Finished ML Classifier Node.")
    return result