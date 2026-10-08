import pandas as pd

print("=" * 70)
print("COMMON CLEAN TEST SET")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD BOTH TRAINING DATASETS
# ------------------------------------------------------------

print("\nLoading original training data...")
X_train_original = pd.read_csv("data/X_train.csv")

print("Loading GAN-augmented training data...")
X_train_augmented = pd.read_csv(
    "data/X_train_augmented.csv"
)

print("\nOriginal train:", X_train_original.shape)
print("Augmented train:", X_train_augmented.shape)

# ------------------------------------------------------------
# 2. LOAD TEST DATA
# ------------------------------------------------------------

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test_multiclass.csv").iloc[:, 0]

print("Test:", X_test.shape)

# ------------------------------------------------------------
# 3. CREATE HASHES
# ------------------------------------------------------------

print("\nCreating row hashes...")

original_hashes = set(
    pd.util.hash_pandas_object(
        X_train_original,
        index=False
    )
)

augmented_hashes = set(
    pd.util.hash_pandas_object(
        X_train_augmented,
        index=False
    )
)

# Combine all training hashes
all_training_hashes = (
    original_hashes |
    augmented_hashes
)

# ------------------------------------------------------------
# 4. FIND OVERLAPPING TEST ROWS
# ------------------------------------------------------------

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

overlap_mask = test_hashes.isin(
    all_training_hashes
)

clean_mask = ~overlap_mask

X_test_common_clean = X_test.loc[
    clean_mask
].reset_index(drop=True)

y_test_common_clean = y_test.loc[
    clean_mask
].reset_index(drop=True)

# ------------------------------------------------------------
# 5. RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMMON CLEAN TEST RESULTS")
print("=" * 70)

print("\nTotal test samples:",
      len(X_test))

print(
    "Samples overlapping with ORIGINAL training:",
    test_hashes.isin(original_hashes).sum()
)

print(
    "Samples overlapping with GAN training:",
    test_hashes.isin(augmented_hashes).sum()
)

print(
    "Samples overlapping with EITHER:",
    overlap_mask.sum()
)

print(
    "COMMON CLEAN TEST SAMPLES:",
    len(X_test_common_clean)
)

print("\nCommon clean class distribution:")
print(
    y_test_common_clean
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 6. SAVE COMMON CLEAN TEST SET
# ------------------------------------------------------------

X_test_common_clean.to_csv(
    "data/X_test_common_clean.csv",
    index=False
)

y_test_common_clean.to_csv(
    "data/y_test_common_clean.csv",
    index=False
)

print("\nSaved:")
print("data/X_test_common_clean.csv")
print("data/y_test_common_clean.csv")

print("\n" + "=" * 70)
print("COMMON CLEAN TEST SET COMPLETE")
print("=" * 70)