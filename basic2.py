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


def plot_metric_bars(metrics_df, save_path, title):
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
    ax.set_title(title)
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
    "val_df",
    "test_df",
    "numeric_feature_cols",
    "tfidf_vectorizer",
    "numeric_scaler",
    "saved_model_paths",
    "candidate_validation_results",
    "candidate_test_results",
]
missing = [k for k in required_keys if k not in artifacts]
if missing:
    raise ValueError(f"Training artifact file is missing required keys: {missing}")

train_df = artifacts["train_df"]
val_df = artifacts["val_df"]
test_df = artifacts["test_df"]
numeric_feature_cols = artifacts["numeric_feature_cols"]
tfidf = artifacts["tfidf_vectorizer"]
scaler = artifacts["numeric_scaler"]
saved_model_paths = artifacts["saved_model_paths"]
candidate_validation_results = artifacts["candidate_validation_results"]
candidate_test_results = artifacts["candidate_test_results"]

print("Loaded training artifacts successfully.")

if "preprocessing_summary" in artifacts:
    print("\nPreprocessing summary:")
    print(artifacts["preprocessing_summary"])


# =========================================================
# REBUILD TEST MATRICES
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

print(f"Majority baseline accuracy: {majority_accuracy:.4f}")

baseline_row = {
    "model": "majority_baseline",
    "accuracy": round(float(majority_accuracy), 4),
    "precision": None,
    "recall": None,
    "f1": None,
    "auc_roc": None,
}


# =========================================================
# VALIDATION METRICS TABLE
# =========================================================
print("\n=== VALIDATION METRICS SUMMARY ===")
validation_metrics_df = pd.DataFrame([
    {"model": model_name, **metrics}
    for model_name, metrics in candidate_validation_results.items()
])

selected_model_name, sorted_validation_df = select_best_model(validation_metrics_df)
validation_metrics_final = pd.concat(
    [sorted_validation_df, pd.DataFrame([baseline_row])],
    ignore_index=True
)

print(validation_metrics_final)

validation_csv_path = os.path.join(OUTPUT_DIR, "validation_metrics_summary.csv")
validation_metrics_final.to_csv(validation_csv_path, index=False)
print(f"\nSaved validation metrics summary to: {validation_csv_path}")


# =========================================================
# TEST METRICS TABLE
# =========================================================
print("\n=== TEST METRICS SUMMARY ===")
test_metrics_df = pd.DataFrame([
    {"model": model_name, **metrics}
    for model_name, metrics in candidate_test_results.items()
])

sorted_test_df = test_metrics_df.sort_values(
    by=["f1", "auc_roc", "accuracy"],
    ascending=[False, False, False]
).reset_index(drop=True)

test_metrics_final = pd.concat(
    [sorted_test_df, pd.DataFrame([baseline_row])],
    ignore_index=True
)

print(test_metrics_final)

test_csv_path = os.path.join(OUTPUT_DIR, "test_metrics_summary.csv")
test_metrics_final.to_csv(test_csv_path, index=False)
print(f"\nSaved test metrics summary to: {test_csv_path}")


# =========================================================
# PLOTS
# =========================================================
print("\n=== GENERATING PLOTS ===")

validation_bar_path = os.path.join(OUTPUT_DIR, "validation_metric_comparison_bar_chart.png")
plot_metric_bars(
    validation_metrics_final,
    validation_bar_path,
    "Validation Metric Comparison Across Models"
)
print(f"Saved validation metric comparison bar chart to: {validation_bar_path}")

test_bar_path = os.path.join(OUTPUT_DIR, "test_metric_comparison_bar_chart.png")
plot_metric_bars(
    test_metrics_final,
    test_bar_path,
    "Test Metric Comparison Across Models"
)
print(f"Saved test metric comparison bar chart to: {test_bar_path}")


# =========================================================
# EVALUATE ALL TRAINED MODELS ON TEST
# =========================================================
print("\n=== EVALUATING ALL TRAINED MODELS ON TEST SET ===")

roc_data = {}
all_predictions = {}

for model_name, model_path in saved_model_paths.items():
    print(f"\nEvaluating: {model_name}")

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

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_test, y_score),
    }

    print(metrics)

roc_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
plot_roc_curves(roc_data, roc_path)
print(f"\nSaved ROC comparison plot to: {roc_path}")

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
print("1. Choose the model with the highest F1-score on validation data.")
print("2. If tied, choose the model with the higher AUC-ROC on validation data.")
print("3. If still tied, choose the model with the higher accuracy on validation data.")

winner_val_row = sorted_validation_df.iloc[0].to_dict()

print(f"\nSelected model based on validation: {selected_model_name}")
print("Winning validation metrics:")
print(winner_val_row)

selected_test_metrics = candidate_test_results[selected_model_name]
print("\nSelected model test metrics:")
print(selected_test_metrics)

if len(sorted_validation_df) > 1:
    runner_up_val_row = sorted_validation_df.iloc[1].to_dict()
    print("\nRunner-up model (validation):")
    print(runner_up_val_row)

    print("\nSelection rationale:")
    print(
        f"- {selected_model_name} was selected because it had the strongest "
        f"F1-score on validation data."
    )
    print(
        f"- Winner validation F1: {winner_val_row['f1']:.4f} | "
        f"Runner-up validation F1: {runner_up_val_row['f1']:.4f}"
    )
    print(
        f"- Winner validation AUC-ROC: {winner_val_row['auc_roc']:.4f} | "
        f"Runner-up validation AUC-ROC: {runner_up_val_row['auc_roc']:.4f}"
    )
    print(
        f"- Final reported test performance for the selected model was: "
        f"{selected_test_metrics}"
    )


# =========================================================
# CLASSIFICATION REPORTS
# =========================================================
print("\n=== CLASSIFICATION REPORTS (TEST SET) ===")
for model_name, y_pred in all_predictions.items():
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, digits=4))


# =========================================================
# FINAL INTERPRETATION HOOKS
# =========================================================
print("\n=== INTERPRETATION NOTES ===")
print("""
Use this structure in the report:
- Model choice was made on validation performance, not test performance.
- Final reported performance comes from the held-out test set.
- Validation metrics justify selection.
- Test metrics quantify final generalization.
- Confusion matrices and ROC curves support the evaluation visually.
""")

print("\nAll evaluation outputs saved under:")
print(OUTPUT_DIR)