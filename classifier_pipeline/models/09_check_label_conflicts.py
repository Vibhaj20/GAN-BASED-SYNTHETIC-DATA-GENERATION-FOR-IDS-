import pandas as pd

print("Loading datasets...")

X_train = pd.read_csv("data/X_train_augmented.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train_multiclass_augmented.csv")
y_test = pd.read_csv("data/y_test_multiclass.csv")

train_label_col = y_train.columns[0]
test_label_col = y_test.columns[0]

# Create hashes
train_hash = pd.util.hash_pandas_object(X_train, index=False)
test_hash = pd.util.hash_pandas_object(X_test, index=False)

# Training hash + labels
train_data = pd.DataFrame({
    "hash": train_hash,
    "train_label": y_train[train_label_col].values
})

# Test hash + labels
test_data = pd.DataFrame({
    "hash": test_hash,
    "test_label": y_test[test_label_col].values
})

# Only hashes appearing in both train and test
common_hashes = set(train_hash) & set(test_hash)

train_common = train_data[
    train_data["hash"].isin(common_hashes)
]

test_common = test_data[
    test_data["hash"].isin(common_hashes)
]

# Find all labels associated with each training hash
train_label_sets = (
    train_common
    .groupby("hash")["train_label"]
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)

# Find all labels associated with each test hash
test_label_sets = (
    test_common
    .groupby("hash")["test_label"]
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)

# Combine
analysis = train_label_sets.merge(
    test_label_sets,
    on="hash",
    how="inner"
)

# Check whether the sets of labels differ
analysis["label_conflict"] = analysis.apply(
    lambda row: set(row["train_label"]) != set(row["test_label"]),
    axis=1
)

conflicts = analysis[analysis["label_conflict"]]

print("\n==============================")
print("ACTUAL LABEL CONFLICT ANALYSIS")
print("==============================")

print("Unique shared feature patterns:", len(analysis))
print("Shared patterns with label conflicts:", len(conflicts))

print("\nExamples of conflicting patterns:")

if len(conflicts) > 0:
    print(conflicts.head(20).to_string(index=False))
else:
    print("No conflicting feature patterns found.")

print("\n==============================")
print("TRAIN LABEL SETS")
print("==============================")

if len(conflicts) > 0:
    print(
        conflicts["train_label"]
        .value_counts()
        .head(20)
    )

print("\nDone.")