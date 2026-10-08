import pandas as pd

print("Loading datasets...")

X_train = pd.read_csv("data/X_train_augmented.csv")
X_test = pd.read_csv("data/X_test.csv")

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Create a hash for every row
train_hashes = pd.util.hash_pandas_object(X_train, index=False)
test_hashes = pd.util.hash_pandas_object(X_test, index=False)

# Find exact duplicate rows between train and test
overlap = len(set(train_hashes) & set(test_hashes))

print("\n==============================")
print("EXACT TRAIN-TEST DUPLICATE CHECK")
print("==============================")
print("Exact duplicate rows:", overlap)

if overlap == 0:
    print("\n✅ No exact train-test duplicates found.")
    print("The high accuracy is NOT explained by exact duplicate leakage.")
else:
    print("\n⚠️ Exact duplicates found!")
    print("Further investigation is required.")