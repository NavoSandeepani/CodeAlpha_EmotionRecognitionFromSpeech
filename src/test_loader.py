from data_loader import load_ravdess_dataset

# Load dataset (relative path – very important)
X, y = load_ravdess_dataset("data/RAVDESS")

print("Total samples:", len(X))
print("X shape:", X.shape)

# Print first few labels to confirm emotions
print("First 10 labels:", y[:100])
