import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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


print("\n=== LOAD SAVED TRAINING ARTIFACTS (V1) ===")
full_artifacts = load_artifacts("./models/v1/training_artifacts.joblib")

if full_artifacts is None:
    raise ValueError("No v1 training artifacts found. Run v1 training first.")

required_keys = [
    "train_df",
    "test_df",
    "numeric_feature_cols",
    "selected_model_name",
    "saved_model_paths",
    "tfidf_vectorizer",
    "numeric_scaler",
]

missing = [k for k in required_keys if k not in full_artifacts]
if missing:
    raise ValueError(
        f"V1 training artifact file is missing keys: {missing}. "
        f"Run v1 training again first."
    )

train_df = full_artifacts["train_df"]
test_df = full_artifacts["test_df"]
numeric_feature_cols = full_artifacts["numeric_feature_cols"]

print("Loaded v1 training artifacts successfully.")


print("\nValidation metrics for selected model:")
print(full_artifacts.get("selected_model_validation_metrics", {}))

print("\nTest metrics for selected model:")
print(full_artifacts.get("selected_model_test_metrics", full_artifacts.get("selected_model_metrics", {})))



print("\n=== STEP 1: CLASS BALANCE CHECK (V1) ===")
print("\nTrain class counts:")
print(train_df["label"].value_counts())

print("\nTrain class proportions:")
print(train_df["label"].value_counts(normalize=True))

print("\nTest class counts:")
print(test_df["label"].value_counts())

print("\nTest class proportions:")
print(test_df["label"].value_counts(normalize=True))


print("\n=== STEP 2: MAJORITY-CLASS BASELINE (V1) ===")
majority_class = Counter(train_df["label"]).most_common(1)[0][0]
majority_accuracy = (test_df["label"] == majority_class).mean()

print("Majority class in train set:", majority_class)
print("Majority-class baseline accuracy:", round(majority_accuracy, 4))


print("\n=== STEP 3: SAVED MODEL CONFUSION MATRIX + REPORT (V1) ===")
selected_model_name = full_artifacts["selected_model_name"]
selected_model_path = full_artifacts["saved_model_paths"][selected_model_name]
model = load_model(selected_model_path)

tfidf = full_artifacts["tfidf_vectorizer"]
scaler = full_artifacts["numeric_scaler"]

X_text_test = test_df["text_ml"]
X_num_test = test_df[numeric_feature_cols].copy()
y_test = test_df["label"].astype(int)

X_test_tfidf = tfidf.transform(X_text_test)
X_num_test_scaled = scaler.transform(X_num_test)

X_test_full = hstack([
    X_test_tfidf,
    csr_matrix(X_num_test_scaled),
])

y_pred = model.predict(X_test_full)

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test_full)[:, 1]
else:
    decision = model.decision_function(X_test_full)
    y_score = 1 / (1 + np.exp(-decision))

print("Selected model:", selected_model_name)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nSaved-model metrics:")
print(compute_metrics(y_test, y_pred, y_score))


print("\n=== STEP 4: TEXT-ONLY VS TEXT+FEATURES COMPARISON (V1) ===")

X_text_train = train_df["text_ml"]
X_text_test = test_df["text_ml"]

X_num_train = train_df[numeric_feature_cols].copy()
X_num_test = test_df[numeric_feature_cols].copy()

y_train = train_df["label"].astype(int)
y_test = test_df["label"].astype(int)

# Text-only
tfidf_text_only = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
)

X_train_text_only = tfidf_text_only.fit_transform(X_text_train)
X_test_text_only = tfidf_text_only.transform(X_text_test)

svm_text_only = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42, dual="auto"),
    cv=3
)
svm_text_only.fit(X_train_text_only, y_train)

y_pred_text_only = svm_text_only.predict(X_test_text_only)
y_score_text_only = svm_text_only.predict_proba(X_test_text_only)[:, 1]

text_only_metrics = compute_metrics(y_test, y_pred_text_only, y_score_text_only)

# Text + features
tfidf_full = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
)

X_train_tfidf = tfidf_full.fit_transform(X_text_train)
X_test_tfidf = tfidf_full.transform(X_text_test)

scaler_full = StandardScaler()
X_num_train_scaled = scaler_full.fit_transform(X_num_train)
X_num_test_scaled = scaler_full.transform(X_num_test)

X_train_full = hstack([
    X_train_tfidf,
    csr_matrix(X_num_train_scaled),
])

X_test_full_compare = hstack([
    X_test_tfidf,
    csr_matrix(X_num_test_scaled),
])

svm_full = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42, dual="auto"),
    cv=3
)
svm_full.fit(X_train_full, y_train)

y_pred_full = svm_full.predict(X_test_full_compare)
y_score_full = svm_full.predict_proba(X_test_full_compare)[:, 1]

full_metrics = compute_metrics(y_test, y_pred_full, y_score_full)

print("\nText-only SVM metrics:")
print(text_only_metrics)

print("\nText + handcrafted features SVM metrics:")
print(full_metrics)