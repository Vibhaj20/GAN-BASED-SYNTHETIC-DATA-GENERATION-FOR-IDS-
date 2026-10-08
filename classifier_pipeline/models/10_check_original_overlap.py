import pandas as pd

print("Loading original datasets...")

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

print("Original train shape:", X_train.shape)
print("Test shape          :", X_test.shape)

# Create row hashes
train_hashes = pd.util.hash_pandas_object(X_train, index=False)
test_hashes = pd.util.hash_pandas_object(X_test, index=False)

# Find exact feature-row overlap
common_hashes = set(train_hashes) & set(test_hashes)

print("\n==============================")
print("ORIGINAL TRAIN-TEST OVERLAP")
print("==============================")

print("Unique overlapping feature patterns:", len(common_hashes))

# Number of test rows that have a matching original training row
test_overlap_count = test_hashes.isin(common_hashes).sum()

print("Test rows overlapping with original train:", test_overlap_count)

if len(common_hashes) == 0:
    print("\n✅ No exact overlap in the original train/test data.")
    print("The overlap appeared after augmentation.")
else:
    print("\n⚠️ Exact overlap already exists in the original train/test data.")
    print("Therefore, the overlap was present BEFORE GAN augmentation.")