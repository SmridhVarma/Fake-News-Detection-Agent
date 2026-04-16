"""
evaluate_models_node — Compute validation and test metrics for every
trained candidate and produce an evaluation artifact consumed by
select_model_node.

Metrics: accuracy, precision, recall, F1, AUC-ROC on both the validation
and test splits. Also generates and saves a confusion matrix heatmap and
ROC curve overlay for all candidates. Skips if the v2 training cache is
already present.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

from src.state import AgentState
from src.utils.training_artifacts import load_artifacts, save_artifacts
from src.nodes.training import compute_metrics

V2_TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"


def evaluate_model(model, X_val, y_val, X_test, y_test) -> tuple[dict, dict]:
    y_val_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_val_score = model.predict_proba(X_val)[:, 1]
    else:
        y_val_score = model.decision_function(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_score)

    y_test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_test_score = model.predict_proba(X_test)[:, 1]
    else:
        y_test_score = model.decision_function(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_score)

    return val_metrics, test_metrics


MODEL_COLORS = {
    "logistic_regression": "#2563eb",
    "svm": "#16a34a",
    "random_forest": "#d97706",
    "neural_network": "#dc2626",
}


def generate_evaluation_plots(trained_models, X_test, y_test, output_dir: str) -> dict:
    """Generate confusion matrix heatmap (best model) and ROC curve overlay (all models)."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # ── 1. ROC Curve overlay for all candidates ──────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.50)")

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, None)
        label = f"{name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})"
        ax_roc.plot(fpr, tpr, lw=2, color=color, label=label)

    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curves — All Candidate Models (Test Split)", fontsize=13, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig_roc)
    plot_paths["roc_curve_path"] = roc_path
    print(f">>> [LOG] ROC curve saved to {roc_path}")

    # ── 2. Confusion Matrix grid for all candidates ───────────────────────────
    n_models = len(trained_models)
    fig_cm, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig_cm.suptitle("Confusion Matrices — All Candidate Models (Test Split)", fontsize=13, fontweight="bold", y=1.02)
    fig_cm.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cm)
    plot_paths["confusion_matrix_path"] = cm_path
    print(f">>> [LOG] Confusion matrices saved to {cm_path}")

    return plot_paths


def evaluate_models_node(state: AgentState) -> dict:
    print("\n>>> [NODE] Starting Evaluate Models Node...")

    if state.get("training_cache_hit") or os.path.exists(V2_TRAINING_ARTIFACT_PATH):
        print(">>> [LOG] Skipping evaluation — v2 training cache is present.")
        print(">>> [NODE] Finished Evaluate Models Node.")
        return {"evaluation_artifact_path": None, "training_cache_hit": True}

    trained_candidates_path = state.get("trained_candidates_path")
    if not trained_candidates_path:
        raise ValueError("trained_candidates_path missing. Run train_models_node first.")

    bundle = load_artifacts(trained_candidates_path)
    if bundle is None:
        raise ValueError(f"Could not load trained candidates at {trained_candidates_path}.")

    trained_models = bundle["trained_models"]
    X_val = bundle["X_val_final"]
    X_test = bundle["X_test_final"]
    y_val = bundle["y_val"]
    y_test = bundle["y_test"]

    candidate_validation_results = {}
    candidate_test_results = {}
    for name, model in trained_models.items():
        val_metrics, test_metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
        candidate_validation_results[name] = val_metrics
        candidate_test_results[name] = test_metrics

    model_dir = state.get("model_dir", "./models/v1")

    # Generate and save evaluation visualizations
    plots_dir = os.path.join(model_dir, "plots")
    plot_paths = generate_evaluation_plots(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        output_dir=plots_dir,
    )

    eval_bundle = {
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        **plot_paths,
    }
    eval_path = save_artifacts(eval_bundle, path=f"{model_dir}/evaluation_results.joblib")

    print(">>> [NODE] Finished Evaluate Models Node.")
    return {
        "evaluation_artifact_path": eval_path,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "training_cache_hit": False,
        **plot_paths,
    }
