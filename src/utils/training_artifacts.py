import os
import joblib


DEFAULT_PREPROCESS_ARTIFACT_PATH = "./models/preprocessing_artifacts.joblib"
DEFAULT_TRAINING_ARTIFACT_PATH = "./models/training_artifacts.joblib"
DEFAULT_MODEL_PATH = "./models/fake_news_clf.joblib"


def save_artifacts(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifacts(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)