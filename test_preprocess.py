from src.nodes.preprocess_data import preprocess_data_node

print("Running preprocessing...")

preprocess_output = preprocess_data_node({
    "fake_csv_path": "./data/Fake.csv",
    "true_csv_path": "./data/True.csv",
    "test_size": 0.2,
    "random_state": 42,
})

print("\nPreprocessing finished.")
print(preprocess_output)