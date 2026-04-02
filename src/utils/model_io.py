"""
model_io.py — Save / load trained ML models.
"""

import os
import joblib


def save_model(model, path: str = None) -> str:
    """Save a trained model to disk."""
    path = path or os.getenv("MODEL_SAVE_PATH", "./models/fake_news_clf.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str = None):
    """Load a trained model from disk. Returns None if not found."""
    path = path or os.getenv("MODEL_SAVE_PATH", "./models/fake_news_clf.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)
