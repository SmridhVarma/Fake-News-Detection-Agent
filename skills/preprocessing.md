---
name: preprocessing
description: Load the labelled Fake/True CSV dataset, clean and deduplicate articles, engineer handcrafted stylistic features, and produce stratified train/validation/test splits persisted as a joblib artifact for the Train Models stage.
mode: organisational
---

# Preprocessing Skill

You are the Data Preparation Agent. Your job is to transform raw CSV news data into clean, feature-enriched, stratified splits that the Train Models stage can consume directly. All decisions here — split ratios, feature columns, cleaning strategy — must be documented and reproducible.

## When to use
- Once per model-build phase, before Train Models runs.
- Whenever the v2 preprocessing cache (`./models/v2/preprocessing_artifacts.joblib`) is absent.
- Never at single-article inference time — this stage operates on the full labelled dataset.

## How to execute
1. **Thought**: Check for v2 cached preprocessing artifacts; skip if present.
2. **Action**:
   - Load `Fake.csv` (label=0) and `True.csv` (label=1) from `fake_csv_path` / `true_csv_path`.
   - Concatenate, fill nulls, strip whitespace, and deduplicate on `raw_text`.
   - Produce two cleaned text columns: `text_ml` (lowercased, punctuation-stripped for TF-IDF) and `text_llm` (light normalisation, preserves casing for transformer/LLM steps).
   - Compute handcrafted stylistic features per row via `calculate_article_scores`: `sub_variance`, `mean_subjectivity`, `lexical_density`, `caps_ratio`, `has_dateline`.
   - Stratified 3-way split: 70% train / 10% validation / 20% test (configurable via `train_size`, `val_size`, `test_size` in state).
   - Persist the artifact bundle to `./models/v1/preprocessing_artifacts.joblib`.
3. **Observation**: Emit row counts and the artifact path back to state.

## Inputs from agent state
- `fake_csv_path`: Path to the fake-news CSV (default `./data/Fake.csv`).
- `true_csv_path`: Path to the real-news CSV (default `./data/True.csv`).
- `train_size` (optional): Fraction for training split (default 0.70).
- `val_size` (optional): Fraction for validation split (default 0.10).
- `test_size` (optional): Fraction for test split (default 0.20).
- `random_state` (optional): Seed for reproducibility (default 42).

## Outputs to agent state
- `preprocessed_rows`: Total rows after deduplication and null removal.
- `train_rows`: Number of rows in the training split.
- `val_rows`: Number of rows in the validation split.
- `test_rows`: Number of rows in the test split.
- `numeric_feature_cols`: List of handcrafted feature column names used downstream (`["sub_variance", "mean_subjectivity", "lexical_density", "caps_ratio", "has_dateline"]`).
- `preprocessing_artifact_path`: Path to the persisted joblib bundle consumed by Train Models.

## Output format
```json
{
  "preprocessed_rows": 44898,
  "train_rows": 31428,
  "val_rows": 4490,
  "test_rows": 8980,
  "numeric_feature_cols": ["sub_variance", "mean_subjectivity", "lexical_density", "caps_ratio", "has_dateline"],
  "preprocessing_artifact_path": "./models/v1/preprocessing_artifacts.joblib"
}
```

## Notes
- Logic lives in `src/nodes/preprocess_data.py`.
- Dual cleaning (`text_ml` vs `text_llm`) is intentional: ML models need aggressive normalisation to reduce vocabulary size, while LLM and stylistic-feature stages benefit from preserved casing and punctuation.
- `has_dateline` is cast to `int` before saving so it can be scaled alongside the other numeric features without type errors.
- Stratified splitting ensures both the fake and real classes maintain their natural proportions in every split — critical for fair model evaluation on this near-balanced dataset.
- The v2 cache path (`./models/v2/`) is checked first; if present, this node returns immediately so inference runs bypass a full dataset reload.
