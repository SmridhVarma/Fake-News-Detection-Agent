import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.utils.training_artifacts import load_artifacts


TRAINING_ARTIFACT_PATH = "./models/v2/training_artifacts.joblib"
OUTPUT_DIR = "./evaluation_outputs/v2_sense_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    decision = model.decision_function(X)
    return 1 / (1 + np.exp(-decision))


def remove_first_sentence(text):
    if not isinstance(text, str):
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return ""


print("\n=== LOAD TRAINING ARTIFACTS (V2) ===")
artifacts = load_artifacts(TRAINING_ARTIFACT_PATH)

if artifacts is None:
    raise ValueError("No v2 training artifacts found.")

required_keys = ["train_df", "val_df", "test_df", "tfidf_vectorizer"]
missing = [k for k in required_keys if k not in artifacts]
if missing:
    raise ValueError(f"Training artifact file is missing required keys: {missing}")

train_df = artifacts["train_df"].copy()
val_df = artifacts["val_df"].copy()
test_df = artifacts["test_df"].copy()
tfidf = artifacts["tfidf_vectorizer"]

print("Loaded training artifacts successfully.")
print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))

summary_lines = []
summary_lines.append("=== SENSE CHECK V2 ===")
summary_lines.append(f"Train rows: {len(train_df)}")
summary_lines.append(f"Validation rows: {len(val_df)}")
summary_lines.append(f"Test rows: {len(test_df)}")
summary_lines.append(f"Preprocessing summary: {artifacts.get('preprocessing_summary', {})}")

# =========================================================
# STEP 1: HIGH-SIMILARITY OVERLAP CHECK
# =========================================================
print("\n=== STEP 1: HIGH-SIMILARITY TRAIN-TEST CHECK ===")

X_train = tfidf.transform(train_df["text_ml"])
X_test = tfidf.transform(test_df["text_ml"])

sim_matrix = cosine_similarity(X_test, X_train)
max_sim = sim_matrix.max(axis=1)
argmax_sim = sim_matrix.argmax(axis=1)

threshold_counts = {
    "gt_0_95": int((max_sim > 0.95).sum()),
    "gt_0_98": int((max_sim > 0.98).sum()),
    "gt_0_99": int((max_sim > 0.99).sum()),
}

print("Similarity threshold counts:", threshold_counts)
summary_lines.append(f"Similarity threshold counts: {threshold_counts}")

overlap_df = pd.DataFrame({
    "test_index": test_df.index,
    "matched_train_index": [train_df.index[i] for i in argmax_sim],
    "max_cosine_similarity": max_sim,
    "test_label": test_df["label"].values,
    "matched_train_label": [train_df.iloc[i]["label"] for i in argmax_sim],
    "test_text_preview": test_df["raw_text"].fillna(test_df["text_ml"]).astype(str).str[:250].values,
    "matched_train_text_preview": [str(train_df.iloc[i].get("raw_text", train_df.iloc[i]["text_ml"]))[:250] for i in argmax_sim],
}).sort_values("max_cosine_similarity", ascending=False)

overlap_csv_path = os.path.join(OUTPUT_DIR, "v2_high_similarity_matches.csv")
overlap_df.to_csv(overlap_csv_path, index=False)
print(f"Saved high-similarity pairs to: {overlap_csv_path}")

print("\nTop 10 highest-similarity examples:")
print(overlap_df.head(10)[["test_index", "matched_train_index", "max_cosine_similarity", "test_label", "matched_train_label"]])

# =========================================================
# STEP 2: TOP PREDICTIVE WORDS
# =========================================================
print("\n=== STEP 2: TOP PREDICTIVE WORDS ===")

vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, max_df=0.95)
X_train_bow = vectorizer.fit_transform(train_df["text_ml"])
y_train = train_df["label"].astype(int)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_bow, y_train)

feature_names = np.array(vectorizer.get_feature_names_out())
coef = lr.coef_[0]

top_real_idx = np.argsort(coef)[-30:][::-1]
top_fake_idx = np.argsort(coef)[:30]

top_real_df = pd.DataFrame({
    "term": feature_names[top_real_idx],
    "coefficient": coef[top_real_idx],
    "class": "real"
})

top_fake_df = pd.DataFrame({
    "term": feature_names[top_fake_idx],
    "coefficient": coef[top_fake_idx],
    "class": "fake"
})

top_words_df = pd.concat([top_real_df, top_fake_df], ignore_index=True)

top_words_csv_path = os.path.join(OUTPUT_DIR, "v2_top_predictive_words.csv")
top_words_df.to_csv(top_words_csv_path, index=False)

print("\nTop REAL indicators:")
print(top_real_df["term"].tolist())

print("\nTop FAKE indicators:")
print(top_fake_df["term"].tolist())

summary_lines.append("Top REAL indicators:")
summary_lines.append(", ".join(top_real_df["term"].tolist()))
summary_lines.append("Top FAKE indicators:")
summary_lines.append(", ".join(top_fake_df["term"].tolist()))

# =========================================================
# STEP 3: FIRST-SENTENCE REMOVAL ABLATION
# =========================================================
print("\n=== STEP 3: FIRST-SENTENCE REMOVAL ABLATION ===")

train_ablate = train_df.copy()
test_ablate = test_df.copy()

source_train_col = "raw_text" if "raw_text" in train_ablate.columns else "text_ml"
source_test_col = "raw_text" if "raw_text" in test_ablate.columns else "text_ml"

train_ablate["ablate_raw"] = train_ablate[source_train_col].apply(remove_first_sentence)
test_ablate["ablate_raw"] = test_ablate[source_test_col].apply(remove_first_sentence)

train_ablate["ablate_text_ml"] = train_ablate["ablate_raw"].fillna("").astype(str).str.strip()
test_ablate["ablate_text_ml"] = test_ablate["ablate_raw"].fillna("").astype(str).str.strip()

train_ablate = train_ablate[train_ablate["ablate_text_ml"] != ""].copy()
test_ablate = test_ablate[test_ablate["ablate_text_ml"] != ""].copy()

y_train_ablate = train_ablate["label"].astype(int)
y_test_ablate = test_ablate["label"].astype(int)

tfidf_ablate = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
)

X_train_ablate = tfidf_ablate.fit_transform(train_ablate["ablate_text_ml"])
X_test_ablate = tfidf_ablate.transform(test_ablate["ablate_text_ml"])

lr_ablate = LogisticRegression(max_iter=1000, random_state=42)
lr_ablate.fit(X_train_ablate, y_train_ablate)

y_pred_lr_ablate = lr_ablate.predict(X_test_ablate)
y_score_lr_ablate = get_scores(lr_ablate, X_test_ablate)
lr_ablate_metrics = compute_metrics(y_test_ablate, y_pred_lr_ablate, y_score_lr_ablate)

svm_ablate = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42, dual="auto"),
    cv=3
)
svm_ablate.fit(X_train_ablate, y_train_ablate)

y_pred_svm_ablate = svm_ablate.predict(X_test_ablate)
y_score_svm_ablate = svm_ablate.predict_proba(X_test_ablate)[:, 1]
svm_ablate_metrics = compute_metrics(y_test_ablate, y_pred_svm_ablate, y_score_svm_ablate)

ablation_df = pd.DataFrame([
    {"model": "logistic_regression_no_first_sentence", **lr_ablate_metrics},
    {"model": "svm_no_first_sentence", **svm_ablate_metrics},
])

ablation_csv_path = os.path.join(OUTPUT_DIR, "v2_first_sentence_ablation.csv")
ablation_df.to_csv(ablation_csv_path, index=False)

print("\nAblation results:")
print(ablation_df)

summary_lines.append("Ablation results:")
summary_lines.append(ablation_df.to_string(index=False))

# =========================================================
# SAVE SUMMARY TXT
# =========================================================
summary_txt_path = os.path.join(OUTPUT_DIR, "v2_sense_check_summary.txt")
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\nSaved summary to: {summary_txt_path}")
print(f"Saved top words to: {top_words_csv_path}")
print(f"Saved ablation results to: {ablation_csv_path}")