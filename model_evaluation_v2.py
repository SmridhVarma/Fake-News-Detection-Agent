import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils.training_artifacts import load_artifacts, load_model

# Optional transformer support
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"
OUTPUT_DIR = "./evaluation_outputs/v2_detailed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
    }


def get_classical_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))
    else:
        preds = model.predict(X)
        return preds.astype(float)


def evaluate_classical_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = get_classical_scores(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
    return y_pred, y_score, metrics


def evaluate_transformer_model(model_dir, texts, y_true):
    if not TRANSFORMER_AVAILABLE:
        raise ImportError("transformers/torch not available for transformer evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    probs = []
    preds = []

    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            p = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        probs.append(float(p[1]))
        preds.append(int(np.argmax(p)))

    y_score = np.array(probs)
    y_pred = np.array(preds)
    metrics = compute_metrics(np.array(y_true), y_pred, y_score)
    return y_pred, y_score, metrics


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(roc_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{model_name} (AUC={d['auc']:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison Across Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_bars(metrics_df, save_path):
    plot_df = metrics_df[metrics_df["model"] != "majority_baseline"].copy()

    metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(plot_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metric_cols):
        ax.bar(x + i * width, plot_df[metric], width, label=metric)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(plot_df["model"], rotation=20)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def select_best_model(metrics_df):
    """
    Deterministic decision logic:
    1. Highest F1
    2. Tie-breaker: highest AUC-ROC
    3. Second tie-breaker: highest accuracy
    """
    candidate_df = metrics_df[metrics_df["model"] != "majority_baseline"].copy()
    candidate_df = candidate_df.sort_values(
        by=["f1", "auc_roc", "accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return candidate_df.iloc[0]["model"], candidate_df


# =========================================================
# LOAD ARTIFACTS
# =========================================================
print("\n=== LOADING TRAINING ARTIFACTS ===")
artifacts = load_artifacts(TRAINING_ARTIFACT_PATH)

if artifacts is None:
    raise ValueError("Training artifacts not found. Run preprocessing + training first.")

required_keys = [
    "train_df",
    "test_df",
    "numeric_feature_cols",
    "tfidf_vectorizer",
    "numeric_scaler",
    "saved_model_paths",
]
missing = [k for k in required_keys if k not in artifacts]
if missing:
    raise ValueError(f"Training artifact file is missing required keys: {missing}")

train_df = artifacts["train_df"]
test_df = artifacts["test_df"]
numeric_feature_cols = artifacts["numeric_feature_cols"]
tfidf = artifacts["tfidf_vectorizer"]
scaler = artifacts["numeric_scaler"]
saved_model_paths = artifacts["saved_model_paths"]

print("Loaded training artifacts successfully.")


# =========================================================
# REBUILD TEST MATRICES FOR CLASSICAL MODELS
# =========================================================
print("\n=== REBUILDING TEST MATRICES ===")

X_text_test = test_df["text_ml"]
X_num_test = test_df[numeric_feature_cols].copy()
y_test = test_df["label"].astype(int)

X_test_tfidf = tfidf.transform(X_text_test)
X_num_test_scaled = scaler.transform(X_num_test)

X_test_full = hstack([
    X_test_tfidf,
    csr_matrix(X_num_test_scaled),
])

print("Test matrix rebuilt.")


# =========================================================
# BASELINE
# =========================================================
print("\n=== BASELINE ===")
majority_class = Counter(train_df["label"]).most_common(1)[0][0]
majority_accuracy = (y_test == majority_class).mean()

metrics_rows = [{
    "model": "majority_baseline",
    "accuracy": round(float(majority_accuracy), 4),
    "precision": None,
    "recall": None,
    "f1": None,
    "auc_roc": None,
}]

print(f"Majority baseline accuracy: {majority_accuracy:.4f}")


# =========================================================
# EVALUATE ALL TRAINED MODELS
# =========================================================
print("\n=== EVALUATING ALL TRAINED MODELS ===")

roc_data = {}
all_predictions = {}
all_scores = {}

for model_name, model_path in saved_model_paths.items():
    print(f"\nEvaluating: {model_name}")

    if model_name == "transformer":
        if not TRANSFORMER_AVAILABLE:
            print("Skipping transformer: transformers/torch not available.")
            continue

        try:
            y_pred, y_score, metrics = evaluate_transformer_model(
                model_dir=model_path,
                texts=test_df["text_llm"].fillna("").tolist(),
                y_true=y_test.tolist(),
            )
        except Exception as e:
            print(f"Skipping transformer due to error: {e}")
            continue
    else:
        model = load_model(model_path)
        if model is None:
            print(f"Skipping {model_name}: could not load model.")
            continue

        y_pred, y_score, metrics = evaluate_classical_model(
            model=model,
            X_test=X_test_full,
            y_test=y_test,
        )

    all_predictions[model_name] = y_pred
    all_scores[model_name] = y_score

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_test, y_score),
    }

    row = {"model": model_name}
    row.update(metrics)
    metrics_rows.append(row)

    print(metrics)


# =========================================================
# METRICS TABLE
# =========================================================
print("\n=== METRICS SUMMARY TABLE ===")
metrics_df = pd.DataFrame(metrics_rows)

selected_model_name, sorted_candidates_df = select_best_model(metrics_df)
metrics_df_final = pd.concat(
    [
        sorted_candidates_df,
        metrics_df[metrics_df["model"] == "majority_baseline"]
    ],
    ignore_index=True
)

print(metrics_df_final)

metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
metrics_df_final.to_csv(metrics_csv_path, index=False)
print(f"\nSaved metrics summary table to: {metrics_csv_path}")


# =========================================================
# PLOTS
# =========================================================
print("\n=== GENERATING PLOTS ===")

# Bar chart across metrics
metric_bar_path = os.path.join(OUTPUT_DIR, "metric_comparison_bar_chart.png")
plot_metric_bars(metrics_df_final, metric_bar_path)
print(f"Saved metric comparison bar chart to: {metric_bar_path}")

# ROC curves
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
plot_roc_curves(roc_data, roc_path)
print(f"Saved ROC comparison plot to: {roc_path}")

# Confusion matrix for each model
for model_name, y_pred in all_predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(
        cm=cm,
        labels=["FAKE (0)", "REAL (1)"],
        title=f"Confusion Matrix — {model_name}",
        save_path=cm_path,
    )
    print(f"Saved confusion matrix for {model_name} to: {cm_path}")


# =========================================================
# SELECTION LOGIC + RATIONALE
# =========================================================
print("\n=== MODEL SELECTION LOGIC ===")
print("Selection rule:")
print("1. Choose the model with the highest F1-score on held-out test data.")
print("2. If tied, choose the model with the higher AUC-ROC.")
print("3. If still tied, choose the model with the higher accuracy.")

winner_row = sorted_candidates_df.iloc[0].to_dict()

print(f"\nSelected model: {selected_model_name}")
print("Winning metrics:")
print(winner_row)

if len(sorted_candidates_df) > 1:
    runner_up_row = sorted_candidates_df.iloc[1].to_dict()
    print("\nRunner-up model:")
    print(runner_up_row)

    print("\nSelection rationale:")
    print(
        f"- {selected_model_name} was selected because it had the strongest "
        f"F1-score on held-out test data."
    )
    print(
        f"- Winner F1: {winner_row['f1']:.4f} | "
        f"Runner-up F1: {runner_up_row['f1']:.4f}"
    )
    print(
        f"- Winner AUC-ROC: {winner_row['auc_roc']:.4f} | "
        f"Runner-up AUC-ROC: {runner_up_row['auc_roc']:.4f}"
    )
    print(
        f"- This rule prioritizes balanced performance between precision and recall, "
        f"which is important in fake-news detection because both false positives and "
        f"false negatives matter."
    )


# =========================================================
# CLASSIFICATION REPORTS
# =========================================================
print("\n=== CLASSIFICATION REPORTS ===")
for model_name, y_pred in all_predictions.items():
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, digits=4))


# =========================================================
# FINAL INTERPRETATION HOOKS
# =========================================================
print("\n=== INTERPRETATION NOTES ===")
print("\nAll evaluation outputs saved under:")
print(OUTPUT_DIR)