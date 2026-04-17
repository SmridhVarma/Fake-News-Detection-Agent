import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
    }


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))
    preds = model.predict(X)
    return preds.astype(float)


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


def plot_roc_curves(roc_data, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{model_name} (AUC={d['auc']:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_bars(metrics_df, save_path, title):
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(metrics_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metric_cols):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df["model"], rotation=20)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(training_artifact_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    artifacts = load_artifacts(training_artifact_path)
    if artifacts is None:
        raise ValueError(f"Training artifacts not found: {training_artifact_path}")

    required_keys = [
        "test_df",
        "numeric_feature_cols",
        "tfidf_vectorizer",
        "numeric_scaler",
        "saved_model_paths",
    ]
    missing = [k for k in required_keys if k not in artifacts]
    if missing:
        raise ValueError(f"Training artifact missing required keys: {missing}")

    test_df = artifacts["test_df"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]
    tfidf = artifacts["tfidf_vectorizer"]
    scaler = artifacts["numeric_scaler"]
    saved_model_paths = artifacts["saved_model_paths"]

    X_text_test = test_df["text_ml"]
    X_num_test = test_df[numeric_feature_cols].copy()
    y_test = test_df["label"].astype(int)

    X_test_tfidf = tfidf.transform(X_text_test)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_test_full = hstack([X_test_tfidf, csr_matrix(X_num_test_scaled)])

    metrics_rows = []
    roc_data = []

    roc_dict = {}
    summary_lines = []
    summary_lines.append(f"Artifact path: {training_artifact_path}")
    summary_lines.append(f"Selected model in artifact: {artifacts.get('selected_model_name')}")
    summary_lines.append("")

    for model_name, model_path in saved_model_paths.items():
        model = load_model(model_path)
        if model is None:
            summary_lines.append(f"{model_name}: FAILED TO LOAD")
            continue

        y_pred = model.predict(X_test_full)
        y_score = get_scores(model, X_test_full)

        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics_rows.append({"model": model_name, **metrics})

        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(
            cm,
            ["FAKE (0)", "REAL (1)"],
            f"Confusion Matrix — {model_name}",
            cm_path,
        )

        report = classification_report(y_test, y_pred, digits=4)
        report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_dict[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc_score(y_test, y_score),
        }

        summary_lines.append(f"{model_name}: {metrics}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["f1", "auc_roc", "accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    metrics_csv = os.path.join(output_dir, "all_family_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    metric_bar_path = os.path.join(output_dir, "all_family_metric_bar_chart.png")
    plot_metric_bars(metrics_df, metric_bar_path, "Metrics by Model Family")

    roc_path = os.path.join(output_dir, "all_family_roc_curves.png")
    plot_roc_curves(roc_dict, roc_path, "ROC Curves by Model Family")

    summary_txt = os.path.join(output_dir, "family_evaluation_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Saved:")
    print(metrics_csv)
    print(metric_bar_path)
    print(roc_path)
    print(summary_txt)
    print("\nMetrics table:")
    print(metrics_df)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python family_evaluation.py <training_artifact_path> <output_dir>"
        )

    main(sys.argv[1], sys.argv[2])