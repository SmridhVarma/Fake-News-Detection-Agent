from src.nodes.preprocess_data import preprocess_data_node
from src.ml.training2 import training_node

print("Running preprocessing (v1)...")
preprocess_output = preprocess_data_node({
    "fake_csv_path": "./data/Fake.csv",
    "true_csv_path": "./data/True.csv",

    "train_size": 0.70,
    "val_size": 0.20,
    "test_size": 0.10,

    "random_state": 42,
})
print("Preprocessing done.")
print(preprocess_output)

print("\nRunning training (v1) with basic hyperparameter tuning...")
train_output = training_node({
    "preprocessing_artifact_path": preprocess_output["preprocessing_artifact_path"],
    "training_artifact_path": "./models/v1/training_artifacts.joblib",
    "model_dir": "./models/v1",

    # tuning controls
    "enable_tuning": True,
    "cv_folds": 3,
    "grid_n_jobs": -1,
})
print("Training done.")
print(train_output)