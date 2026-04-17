import os
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.training_artifacts import load_artifacts


V1_PATH = "./models/v1/training_artifacts.joblib"
V2_PATH = "./models/v2/training_artifacts.joblib"
OUTPUT_DIR = "./evaluation_outputs/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def artifact_to_df(artifacts, version_name, metric_key):
    rows = []
    for model_name, metrics in artifacts[metric_key].items():
        row = {"version": version_name, "model": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


print("\n=== LOADING V1/V2 ARTIFACTS ===")
v1 = load_artifacts(V1_PATH)
v2 = load_artifacts(V2_PATH)

if v1 is None:
    raise ValueError("Missing v1 training artifacts.")
if v2 is None:
    raise ValueError("Missing v2 training artifacts.")

v1_val_df = artifact_to_df(v1, "v1", "candidate_validation_results")
v2_val_df = artifact_to_df(v2, "v2", "candidate_validation_results")
validation_combined_df = pd.concat([v1_val_df, v2_val_df], ignore_index=True)

v1_test_df = artifact_to_df(v1, "v1", "candidate_test_results")
v2_test_df = artifact_to_df(v2, "v2", "candidate_test_results")
test_combined_df = pd.concat([v1_test_df, v2_test_df], ignore_index=True)

print("\n=== VALIDATION METRICS ===")
print(validation_combined_df)

print("\n=== TEST METRICS ===")
print(test_combined_df)

validation_csv_path = os.path.join(OUTPUT_DIR, "v1_v2_validation_metrics_comparison.csv")
test_csv_path = os.path.join(OUTPUT_DIR, "v1_v2_test_metrics_comparison.csv")

validation_combined_df.to_csv(validation_csv_path, index=False)
test_combined_df.to_csv(test_csv_path, index=False)

print(f"\nSaved validation comparison table to: {validation_csv_path}")
print(f"Saved test comparison table to: {test_csv_path}")

print("\n=== SELECTED MODELS ===")
print("v1 selected model:", v1.get("selected_model_name"))
print("v1 selected validation metrics:", v1.get("selected_model_validation_metrics"))
print("v1 selected test metrics:", v1.get("selected_model_test_metrics"))
print()
print("v2 selected model:", v2.get("selected_model_name"))
print("v2 selected validation metrics:", v2.get("selected_model_validation_metrics"))
print("v2 selected test metrics:", v2.get("selected_model_test_metrics"))

print("\n=== PREPROCESSING SUMMARY ===")
print("v1 preprocessing summary:")
print(v1.get("preprocessing_summary", {}))
print()
print("v2 preprocessing summary:")
print(v2.get("preprocessing_summary", {}))

plot_df = test_combined_df.pivot(index="model", columns="version", values="f1")
plot_df.plot(kind="bar", figsize=(10, 6))
plt.title("Test F1 Score Comparison: v1 vs v2")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "v1_v2_test_f1_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved F1 comparison chart to: {plot_path}")