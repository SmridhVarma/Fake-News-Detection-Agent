import os
import json
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

# Transformer imports
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
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

def build_traditional_feature_matrices(train_df, test_df, numeric_feature_cols):
    X_text_train = train_df["text_ml"]
    X_text_test = test_df["text_ml"]

    X_num_train = train_df[numeric_feature_cols].copy()
    X_num_test = test_df[numeric_feature_cols].copy()

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )

    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_test_tfidf = tfidf.transform(X_text_test)

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train_final = hstack([
        X_train_tfidf,
        csr_matrix(X_num_train_scaled),
    ])

    X_test_final = hstack([
        X_test_tfidf,
        csr_matrix(X_num_test_scaled),
    ])

    return {
        "X_train_final": X_train_final,
        "X_test_final": X_test_final,
        "y_train": y_train,
        "y_test": y_test,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
    }


def train_traditional_models(X_train_final, X_test_final, y_train, y_test, random_state=42):
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),

        # Linear SVM is the better SVM variant for sparse TF-IDF text.
        # CalibratedClassifierCV gives probability estimates for ROC-AUC.
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

    results = {}
    trained_models = {}

    for model_name, model in candidates.items():
        model.fit(X_train_final, y_train)

        y_pred = model.predict(X_test_final)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_final)[:, 1]
        else:
            # Fallback if ever needed
            y_score = model.decision_function(X_test_final)

        metrics = compute_metrics(y_test, y_pred, y_score)
        results[model_name] = metrics
        trained_models[model_name] = model

    return results, trained_models


# =========================
# Transformer training
# =========================

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_transformer_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./models/transformer_fake_news",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    ensure_clean_dir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = train_df["text_llm"].fillna("").tolist()
    test_texts = test_df["text_llm"].fillna("").tolist()

    y_train = train_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=512,
    )

    train_dataset = NewsDataset(train_encodings, y_train)
    test_dataset = NewsDataset(test_encodings, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
    )

    def transformer_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        preds = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc_roc": roc_auc_score(labels, probs),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=transformer_metrics,
    )

    trainer.train()

    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)

    metrics = compute_metrics(np.array(y_test), preds, probs)

    # Save tokenizer + model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "metrics": metrics,
        "model_dir": output_dir,
        "transformer_model_name": model_name,
    }


# =========================
# Main training node
# =========================

def training_node(state: AgentState) -> dict:
    """
    Train and compare:
    - Logistic Regression
    - SVM (linear SVM with calibration)
    - Random Forest
    - Neural Network (MLP)
    - Transformer (DistilBERT by default)

    Traditional models use:
    TF-IDF(text_ml) + handcrafted numeric features

    Transformer uses:
    text_llm only
    """

    preprocess_artifact_path = state.get(
    "preprocessing_artifact_path",
    "./models/preprocessing_artifacts.joblib"
    )
    artifacts = load_artifacts(preprocess_artifact_path)

    if artifacts is None:
        raise ValueError("Preprocessing artifacts not found. Run preprocess_data_node first.")

    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]
    random_state = artifacts.get("random_state", 42)

    include_transformer = state.get("include_transformer", True)
    transformer_model_name = state.get("transformer_model_name", "distilbert-base-uncased")
    transformer_epochs = state.get("transformer_epochs", 2)
    transformer_batch_size = state.get("transformer_batch_size", 8)
    transformer_learning_rate = state.get("transformer_learning_rate", 2e-5)

    # =========================
    # Traditional models
    # =========================
    traditional_bundle = build_traditional_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        numeric_feature_cols=numeric_feature_cols,
    )

    X_train_final = traditional_bundle["X_train_final"]
    X_test_final = traditional_bundle["X_test_final"]
    y_train = traditional_bundle["y_train"]
    y_test = traditional_bundle["y_test"]
    tfidf = traditional_bundle["tfidf_vectorizer"]
    scaler = traditional_bundle["numeric_scaler"]

    traditional_results, traditional_models = train_traditional_models(
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )

    all_results = dict(traditional_results)
    saved_model_paths = {}

    # Save all traditional models
    for model_name, model in traditional_models.items():
        model_path = save_model(model, path=f"./models/{model_name}.joblib")
        saved_model_paths[model_name] = model_path

    # =========================
    # Transformer
    # =========================
    transformer_info = None

    if include_transformer:
        transformer_info = train_transformer_model(
            train_df=train_df,
            test_df=test_df,
            model_name=transformer_model_name,
            output_dir="./models/transformer_fake_news",
            epochs=transformer_epochs,
            batch_size=transformer_batch_size,
            learning_rate=transformer_learning_rate,
        )

        all_results["transformer"] = transformer_info["metrics"]
        saved_model_paths["transformer"] = transformer_info["model_dir"]

    # =========================
    # Select winner
    # =========================
    best_model_name = select_best_model(all_results)
    best_metrics = all_results[best_model_name]
    best_model_path = saved_model_paths[best_model_name]

    # Save shared artifacts needed later
    training_bundle = {
        "train_df": train_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
        "candidate_results": all_results,
        "selected_model_name": best_model_name,
        "selected_model_metrics": best_metrics,
        "selected_model_path": best_model_path,
        "saved_model_paths": saved_model_paths,
        "transformer_model_name": transformer_model_name if include_transformer else None,
    }

    final_artifact_path = save_artifacts(
    training_bundle,
    path="./models/training_artifacts.joblib"
    )

    return {
        "model_trained": True,
        "model_path": best_model_path,
        "training_artifact_path": final_artifact_path,
        "candidate_results": all_results,
        "selected_model_name": best_model_name,
        "selected_model_metrics": best_metrics,
        "saved_model_paths": saved_model_paths,
    }