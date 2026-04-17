from src.ml.preprocess_data_v2 import preprocess_data_node
from src.nodes.training import training_node

print("Running preprocessing (v2)...")
preprocess_output = preprocess_data_node({
    "fake_csv_path": "./data/Fake.csv",
    "true_csv_path": "./data/True.csv",

    "train_size": 0.70,
    "val_size": 0.10,
    "test_size": 0.20,

    "random_state": 42,
})
print("Preprocessing done.")
print(preprocess_output)

print("\nRunning training (v2)...")
train_output = training_node({
    "preprocessing_artifact_path": preprocess_output["preprocessing_artifact_path"],
    "training_artifact_path": "./models/v2/training_artifacts.joblib",
    "model_dir": "./models/v2",
})
print("Training done.")
print(train_output)