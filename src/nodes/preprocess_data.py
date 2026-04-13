import pandas as pd
import os
from sklearn.model_selection import train_test_split

from src.state import AgentState
from src.utils.preprocessing import (
    clean_text_for_transformers,
    clean_text_for_traditional_ml,
)
from src.utils.ingestion_tools import calculate_article_scores
from src.utils.training_artifacts import save_artifacts


def build_full_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("") if "title" in df.columns else ""
    text = df["text"].fillna("") if "text" in df.columns else ""
    return (title + " " + text).str.strip()


def preprocess_data_node(state: AgentState) -> dict:
    """
    V1 preprocessing with train/validation/test split and dual cleaning support.
    """
    print("\n>>> [NODE] Starting Preprocess Data Node...")
    
    # Check for v2 artifacts to skip retraining
    v2_artifact_path = "./models/v2/preprocessing_artifacts.joblib"
    if os.path.exists(v2_artifact_path):
        print(f">>> [LOG] v2 preprocessing artifacts found at {v2_artifact_path}. Skipping processing.")
        print(">>> [NODE] Finished Preprocess Data Node.")
        return {
            "preprocessing_artifact_path": v2_artifact_path,
        }

    fake_path = state.get("fake_csv_path", "./data/Fake.csv")
    true_path = state.get("true_csv_path", "./data/True.csv")

    train_size = state.get("train_size", 0.70)
    val_size = state.get("val_size", 0.10)
    test_size = state.get("test_size", 0.20)

    random_state = state.get("random_state", 42)

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = 0
    df_true["label"] = 1

    df_fake["raw_text"] = build_full_text(df_fake)
    df_true["raw_text"] = build_full_text(df_true)

    df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)

    df["raw_text"] = df["raw_text"].fillna("").astype(str).str.strip()
    df = df[df["raw_text"] != ""].copy()
    df = df.drop_duplicates(subset=["raw_text"]).reset_index(drop=True)

    # 6. Create cleaned text columns (Integrated Upstream Dual Cleaning)
    df["text_llm"] = df["raw_text"].apply(clean_text_for_transformers)
    df["text_ml"] = df["raw_text"].apply(clean_text_for_traditional_ml)

    feature_dicts = df["raw_text"].apply(calculate_article_scores)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    df = pd.concat([df, feature_df], axis=1)

    df["has_dateline"] = df["has_dateline"].astype(int)

    numeric_feature_cols = [
        "sub_variance",
        "mean_subjectivity",
        "lexical_density",
        "caps_ratio",
        "has_dateline",
    ]

    # Integrated Upstream 3-way Split
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=df["label"],
    )

    val_fraction_of_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction_of_temp),
        random_state=random_state,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    artifacts = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "random_state": random_state,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }

    artifact_path = save_artifacts(
        artifacts,
        path="./models/v1/preprocessing_artifacts.joblib"
    )

    result = {
        "preprocessed_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "numeric_feature_cols": numeric_feature_cols,
        "preprocessing_artifact_path": artifact_path,
    }
    print(">>> [NODE] Finished Preprocess Data Node.")
    return result