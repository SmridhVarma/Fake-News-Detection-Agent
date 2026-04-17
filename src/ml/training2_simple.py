import os
import numpy as np

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
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


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
    }


def select_best_model(results: dict) -> str:
    return sorted(
        results.keys(),
        key=lambda name: (results[name]["f1"], results[name]["auc_roc"]),
        reverse=True
    )[0]


def build_traditional_feature_matrices(train_df, val_df, test_df, numeric_feature_cols):
    X_text_train = train_df["text_ml"]
    X_text_val = val_df["text_ml"]
    X_text_test = test_df["text_ml"]

    X_num_train = train_df[numeric_feature_cols].copy()
    X_num_val = val_df[numeric_feature_cols].copy()
    X_num_test = test_df[numeric_feature_cols].copy()

    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )

    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_val_tfidf = tfidf.transform(X_text_val)
    X_test_tfidf = tfidf.transform(X_text_test)

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train_final = hstack([X_train_tfidf, csr_matrix(X_num_train_scaled)])
    X_val_final = hstack([X_val_tfidf, csr_matrix(X_num_val_scaled)])
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_num_test_scaled)])

    return {
        "X_train_final": X_train_final,
        "X_val_final": X_val_final,
        "X_test_final": X_test_final,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
    }


def get_model_score_array(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))
    else:
        preds = model.predict(X)
        return preds.astype(float)


def tune_logistic_regression(X_train, y_train, random_state, cv_folds, n_jobs):
    base_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    param_grid = {
        "C": [0.5, 1.0, 2.0],
        "solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, round(grid.best_score_, 4)


def tune_svm(X_train, y_train, random_state, cv_folds, n_jobs):
    # Tune LinearSVC first, then calibrate the best estimator
    base_model = LinearSVC(
        random_state=random_state,
        dual="auto",
        max_iter=5000,
    )

    param_grid = {
        "C": [0.5, 1.0, 2.0],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_linear_svc = grid.best_estimator_

    calibrated_model = CalibratedClassifierCV(
        estimator=best_linear_svc,
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    return calibrated_model, grid.best_params_, round(grid.best_score_, 4)


def tune_random_forest(X_train, y_train, random_state, cv_folds, n_jobs):
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, round(grid.best_score_, 4)


def tune_neural_network(X_train, y_train, random_state, cv_folds, n_jobs):
    base_model = MLPClassifier(
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=20,
        early_stopping=True,
        random_state=random_state,
    )

    param_grid = {
        "hidden_layer_sizes": [(128,), (64, 128)],
        "alpha": [1e-4, 1e-3],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, round(grid.best_score_, 4)


def train_traditional_models(
    X_train_final,
    X_val_final,
    X_test_final,
    y_train,
    y_val,
    y_test,
    random_state=42,
    enable_tuning=True,
    cv_folds=3,
    grid_n_jobs=-1,
):
    candidate_validation_results = {}
    candidate_test_results = {}
    trained_models = {}
    best_params_by_model = {}
    cv_best_scores = {}

    if enable_tuning:
        tuned_candidates = {
            "logistic_regression": tune_logistic_regression(
                X_train_final, y_train, random_state, cv_folds, grid_n_jobs
            ),
            "svm": tune_svm(
                X_train_final, y_train, random_state, cv_folds, grid_n_jobs
            ),
            "random_forest": tune_random_forest(
                X_train_final, y_train, random_state, cv_folds, grid_n_jobs
            ),
            "neural_network": tune_neural_network(
                X_train_final, y_train, random_state, cv_folds, grid_n_jobs
            ),
        }

        models = {name: tpl[0] for name, tpl in tuned_candidates.items()}
        best_params_by_model = {name: tpl[1] for name, tpl in tuned_candidates.items()}
        cv_best_scores = {name: tpl[2] for name, tpl in tuned_candidates.items()}
    else:
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
            "svm": CalibratedClassifierCV(
                estimator=LinearSVC(
                    random_state=random_state,
                    dual="auto",
                    max_iter=5000,
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

        for model_name, model in models.items():
            model.fit(X_train_final, y_train)
            best_params_by_model[model_name] = "default"
            cv_best_scores[model_name] = None

    for model_name, model in models.items():
        if enable_tuning:
            # already fitted inside tuning functions / GridSearch
            pass
        else:
            # already fitted just above
            pass

        y_val_pred = model.predict(X_val_final)
        y_val_score = get_model_score_array(model, X_val_final)
        candidate_validation_results[model_name] = compute_metrics(y_val, y_val_pred, y_val_score)

        y_test_pred = model.predict(X_test_final)
        y_test_score = get_model_score_array(model, X_test_final)
        candidate_test_results[model_name] = compute_metrics(y_test, y_test_pred, y_test_score)

        trained_models[model_name] = model

    return (
        candidate_validation_results,
        candidate_test_results,
        trained_models,
        best_params_by_model,
        cv_best_scores,
    )


def training_node(state: AgentState) -> dict:
    preprocess_artifact_path = state.get(
        "preprocessing_artifact_path",
        "./models/v1/preprocessing_artifacts.joblib"
    )
    training_artifact_path = state.get(
        "training_artifact_path",
        "./models/v1/training_artifacts.joblib"
    )
    model_dir = state.get(
        "model_dir",
        "./models/v1"
    )

    enable_tuning = state.get("enable_tuning", True)
    cv_folds = state.get("cv_folds", 3)
    grid_n_jobs = state.get("grid_n_jobs", -1)

    os.makedirs(model_dir, exist_ok=True)

    artifacts = load_artifacts(preprocess_artifact_path)

    if artifacts is None:
        raise ValueError("Preprocessing artifacts not found. Run preprocess_data_node first.")

    train_df = artifacts["train_df"]
    val_df = artifacts["val_df"]
    test_df = artifacts["test_df"]
    numeric_feature_cols = artifacts["numeric_feature_cols"]
    random_state = artifacts.get("random_state", 42)

    traditional_bundle = build_traditional_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_feature_cols=numeric_feature_cols,
    )

    X_train_final = traditional_bundle["X_train_final"]
    X_val_final = traditional_bundle["X_val_final"]
    X_test_final = traditional_bundle["X_test_final"]
    y_train = traditional_bundle["y_train"]
    y_val = traditional_bundle["y_val"]
    y_test = traditional_bundle["y_test"]
    tfidf = traditional_bundle["tfidf_vectorizer"]
    scaler = traditional_bundle["numeric_scaler"]

    (
        candidate_validation_results,
        candidate_test_results,
        traditional_models,
        best_params_by_model,
        cv_best_scores,
    ) = train_traditional_models(
        X_train_final=X_train_final,
        X_val_final=X_val_final,
        X_test_final=X_test_final,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        random_state=random_state,
        enable_tuning=enable_tuning,
        cv_folds=cv_folds,
        grid_n_jobs=grid_n_jobs,
    )

    saved_model_paths = {}

    for model_name, model in traditional_models.items():
        model_path = save_model(model, path=f"{model_dir}/{model_name}.joblib")
        saved_model_paths[model_name] = model_path

    best_model_name = select_best_model(candidate_validation_results)
    best_validation_metrics = candidate_validation_results[best_model_name]
    best_test_metrics = candidate_test_results[best_model_name]
    best_model_path = saved_model_paths[best_model_name]

    training_bundle = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "tfidf_vectorizer": tfidf,
        "numeric_scaler": scaler,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "selected_model_path": best_model_path,
        "saved_model_paths": saved_model_paths,
        "best_params_by_model": best_params_by_model,
        "cv_best_scores": cv_best_scores,
        "tuning_enabled": enable_tuning,
        "cv_folds": cv_folds,
        "preprocessing_summary": artifacts.get("preprocessing_summary", {}),
    }

    final_artifact_path = save_artifacts(
        training_bundle,
        path=training_artifact_path
    )

    return {
        "model_trained": True,
        "model_path": best_model_path,
        "training_artifact_path": final_artifact_path,
        "candidate_validation_results": candidate_validation_results,
        "candidate_test_results": candidate_test_results,
        "candidate_results": candidate_test_results,
        "selected_model_name": best_model_name,
        "selected_model_validation_metrics": best_validation_metrics,
        "selected_model_test_metrics": best_test_metrics,
        "selected_model_metrics": best_test_metrics,
        "saved_model_paths": saved_model_paths,
        "best_params_by_model": best_params_by_model,
        "cv_best_scores": cv_best_scores,
        "tuning_enabled": enable_tuning,
        "cv_folds": cv_folds,
    }