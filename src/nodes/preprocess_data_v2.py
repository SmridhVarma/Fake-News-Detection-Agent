import re
import unicodedata
import pandas as pd
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


def canonicalize_for_dedup(text: str) -> str:
    """
    Stronger canonical key for duplicate detection.
    Helps catch rows that differ only by punctuation, quotes, spacing, etc.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = clean_text_for_transformers(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data_node(state: AgentState) -> dict:
    """
    V2 preprocessing with:
    - stronger canonical deduplication
    - train / validation / test split
    - saved preprocessing summary for before-vs-after comparison
    """

    fake_path = state.get("fake_csv_path", "./data/Fake.csv")
    true_path = state.get("true_csv_path", "./data/True.csv")

    # =========================================================
    # PLACEHOLDERS: EDIT THESE %
    # Must sum to 1.0
    # =========================================================
    train_size = state.get("train_size", 0.70)
    val_size = state.get("val_size", 0.10)
    test_size = state.get("test_size", 0.20)

    random_state = state.get("random_state", 42)

    if round(train_size + val_size + test_size, 10) != 1.0:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    # 1. Load raw datasets
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    raw_fake_rows = len(df_fake)
    raw_true_rows = len(df_true)

    # 2. Assign labels
    df_fake["label"] = 0
    df_true["label"] = 1

    # 3. Combine title + text
    df_fake["raw_text"] = build_full_text(df_fake)
    df_true["raw_text"] = build_full_text(df_true)

    # 4. Merge
    df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
    raw_total_rows = len(df)

    # 5. Remove empty rows
    df["raw_text"] = df["raw_text"].fillna("").astype(str).str.strip()
    before_empty_filter = len(df)
    df = df[df["raw_text"] != ""].copy()
    after_empty_filter = len(df)
    empty_rows_removed = before_empty_filter - after_empty_filter

    # 6. Stronger dedup key BEFORE split
    df["dedup_key"] = df["raw_text"].apply(canonicalize_for_dedup)

    before_empty_key_filter = len(df)
    df = df[df["dedup_key"] != ""].copy()
    after_empty_key_filter = len(df)
    empty_key_rows_removed = before_empty_key_filter - after_empty_key_filter

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["dedup_key"]).reset_index(drop=True)
    after_dedup = len(df)
    duplicate_rows_removed = before_dedup - after_dedup

    # 7. Create cleaned text columns
    df["text_llm"] = df["raw_text"].apply(clean_text_for_transformers)
    df["text_ml"] = df["raw_text"].apply(clean_text_for_traditional_ml)

    # 8. Handcrafted features
    feature_dicts = df["raw_text"].apply(calculate_article_scores)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    df = pd.concat([df, feature_df], axis=1)

    # 9. Clean types
    df["has_dateline"] = df["has_dateline"].astype(int)

    numeric_feature_cols = [
        "sub_variance",
        "mean_subjectivity",
        "lexical_density",
        "caps_ratio",
        "has_dateline",
    ]

    # 10. Train / val / test split
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

    # 11. Save preprocessing artifacts
    artifacts = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_feature_cols": numeric_feature_cols,
        "random_state": random_state,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "preprocessing_summary": {
            "raw_fake_rows": raw_fake_rows,
            "raw_true_rows": raw_true_rows,
            "raw_total_rows": raw_total_rows,
            "empty_rows_removed": empty_rows_removed,
            "empty_key_rows_removed": empty_key_rows_removed,
            "duplicate_rows_removed": duplicate_rows_removed,
            "final_rows_after_preprocessing": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_class_proportions": train_df["label"].value_counts(normalize=True).to_dict(),
            "val_class_proportions": val_df["label"].value_counts(normalize=True).to_dict(),
            "test_class_proportions": test_df["label"].value_counts(normalize=True).to_dict(),
        },
    }

    artifact_path = save_artifacts(
        artifacts,
        path="./models/v2/preprocessing_artifacts.joblib"
    )

    return {
        "preprocessed_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "numeric_feature_cols": numeric_feature_cols,
        "preprocessing_artifact_path": artifact_path,
        "empty_rows_removed": empty_rows_removed,
        "empty_key_rows_removed": empty_key_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
    }