import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.training_artifacts import load_artifacts


# =========================================================
# CONFIG
# =========================================================
TRAINING_ARTIFACT_PATH = "./models/v1/training_artifacts.joblib"
OUTPUT_DIR = "./evaluation_outputs/v1_sense_check"


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


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))
    else:
        preds = model.predict(X)
        return preds.astype(float)


def remove_first_sentence(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)
    if len(parts) == 1:
        return ""
    return parts[1].strip()


# =========================================================
# LOAD ARTIFACTS
# =========================================================
print("\n=== LOADING TRAINING ARTIFACTS ===")
artifacts = load_artifacts(TRAINING_ARTIFACT_PATH)

if artifacts is None:
    raise ValueError("Training artifacts not found. Run preprocessing + training first.")

required_keys = ["train_df", "test_df", "tfidf_vectorizer"]
missing = [k for k in required_keys if k not in artifacts]
if missing:
    raise ValueError(f"Training artifact file is missing required keys: {missing}")

train_df = artifacts["train_df"].copy()
test_df = artifacts["test_df"].copy()
tfidf = artifacts["tfidf_vectorizer"]

print("Loaded training artifacts successfully.")
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))


# =========================================================
# STEP 1: NEAR-DUPLICATE CHECK
# =========================================================
print("\n=== STEP 1: NEAR-DUPLICATE SIMILARITY CHECK ===")

X_train = tfidf.transform(train_df["text_ml"])
X_test = tfidf.transform(test_df["text_ml"])

max_sims = []
nearest_train_idx = []

batch_size = 200
for start in range(0, X_test.shape[0], batch_size):
    end = min(start + batch_size, X_test.shape[0])
    sims = cosine_similarity(X_test[start:end], X_train)
    max_sims.extend(sims.max(axis=1))
    nearest_train_idx.extend(sims.argmax(axis=1))

max_sims = np.array(max_sims)
nearest_train_idx = np.array(nearest_train_idx)

sim_summary = pd.Series(max_sims).describe()
print(sim_summary)

print("\nCount with max similarity > 0.95:", int((max_sims > 0.95).sum()))
print("Count with max similarity > 0.98:", int((max_sims > 0.98).sum()))
print("Count with max similarity > 0.99:", int((max_sims > 0.99).sum()))

near_dup_df = pd.DataFrame({
    "test_index": test_df.index,
    "max_similarity_to_train": max_sims,
    "nearest_train_index": nearest_train_idx,
}).sort_values("max_similarity_to_train", ascending=False)

print("\nTop 10 most similar test rows to training set:")
print(near_dup_df.head(10))

# Optional: show a few examples
print("\n=== SAMPLE NEAR-DUPLICATE PAIRS ===")
for _, row in near_dup_df.head(3).iterrows():
    test_pos = int(row["test_index"])
    train_pos = int(row["nearest_train_index"])

    test_text = test_df.loc[test_pos, "raw_text"] if "raw_text" in test_df.columns else test_df.loc[test_pos, "text_ml"]
    train_text = train_df.iloc[train_pos]["raw_text"] if "raw_text" in train_df.columns else train_df.iloc[train_pos]["text_ml"]

    print("\n" + "=" * 100)
    print(f"Similarity: {row['max_similarity_to_train']:.4f}")
    print("\nTEST SNIPPET:")
    print(str(test_text)[:600])
    print("\nMATCHED TRAIN SNIPPET:")
    print(str(train_text)[:600])


# =========================================================
# STEP 2: TOP PREDICTIVE WORDS
# =========================================================
print("\n=== STEP 2: TOP PREDICTIVE WORDS (LOGISTIC REGRESSION) ===")

y_train = train_df["label"].astype(int)
y_test = test_df["label"].astype(int)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

feature_names = np.array(tfidf.get_feature_names_out())
coefs = lr.coef_[0]

top_real_idx = np.argsort(coefs)[-30:][::-1]
top_fake_idx = np.argsort(coefs)[:30]

top_real_df = pd.DataFrame({
    "feature": feature_names[top_real_idx],
    "coefficient": coefs[top_real_idx]
})

top_fake_df = pd.DataFrame({
    "feature": feature_names[top_fake_idx],
    "coefficient": coefs[top_fake_idx]
})

print("\nTop 30 features pushing toward REAL (label=1):")
print(top_real_df)

print("\nTop 30 features pushing toward FAKE (label=0):")
print(top_fake_df)

print("\nWhat to look for:")
print("- source names")
print("- dateline/location patterns")
print("- publisher artifacts")
print("- repeated partisan/topic buzzwords")


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

print("\nAblation results:")
print(ablation_df)

# =========================================================
# STEP 4: OPTIONAL REFERENCE AGAINST ORIGINAL TRAINED RESULTS
# =========================================================
print("\n=== STEP 4: REFERENCE AGAINST ORIGINAL TRAINED RESULTS ===")
if "candidate_results" in artifacts:
    original_rows = []
    for model_name, metrics in artifacts["candidate_results"].items():
        row = {"model": model_name}
        row.update(metrics)
        original_rows.append(row)

    original_df = pd.DataFrame(original_rows)
    print(original_df)
else:
    print("No candidate_results found in artifacts.")


# =========================================================
# STEP 5: SUMMARY INTERPRETATION
# =========================================================
print("\n=== STEP 5: HOW TO INTERPRET THIS ===")
print("""
1. Near-duplicate similarity:
   - If many test examples have max train similarity above 0.95 or 0.98,
     train/test overlap is likely inflating performance.

2. Top predictive words:
   - If the strongest features are source or publisher-like cues, the model
     may be exploiting dataset artifacts rather than deeper misinformation reasoning.

3. First-sentence removal ablation:
   - If performance drops sharply, the opening sentence carries too much signal,
     which may indicate dateline/source-style shortcuts.
   - If performance stays very high, the dataset itself may simply be highly separable.

4. Put together:
   - Mild class imbalance + low majority baseline + high text-only performance
     usually means the dataset is easy and/or artifact-rich rather than the model
     being magically near-perfect in a real-world sense.
""")