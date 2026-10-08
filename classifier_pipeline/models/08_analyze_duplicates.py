import pandas as pd

print("Loading datasets...")

X_train = pd.read_csv("data/X_train_augmented.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train_multiclass_augmented.csv")
y_test = pd.read_csv("data/y_test_multiclass.csv")

# Get label column names
train_label_col = y_train.columns[0]
test_label_col = y_test.columns[0]

# Create hashes for feature rows
train_hash = pd.util.hash_pandas_object(X_train, index=False)
test_hash = pd.util.hash_pandas_object(X_test, index=False)

# Map training row hash -> training label
train_hash_label = pd.DataFrame({
    "hash": train_hash,
    "label": y_train[train_label_col].values
})

# Add hash to test data
test_info = pd.DataFrame({
    "hash": test_hash,
    "test_label": y_test[test_label_col].values
})

# Keep only test rows that occur in training
duplicates = test_info[test_info["hash"].isin(set(train_hash))].copy()

# Get one training label for each duplicated hash
train_labels = train_hash_label.drop_duplicates("hash")

duplicates = duplicates.merge(
    train_labels,
    on="hash",
    how="left"
)

print("\n==============================")
print("DUPLICATE LABEL ANALYSIS")
print("==============================")

print("Total duplicate test rows:", len(duplicates))

same_label = (duplicates["test_label"] == duplicates["label"]).sum()
different_label = (duplicates["test_label"] != duplicates["label"]).sum()

print("Duplicates with SAME label:", same_label)
print("Duplicates with DIFFERENT label:", different_label)

print("\nTest-label distribution among duplicates:")
print(duplicates["test_label"].value_counts().sort_index())

print("\nTraining-label distribution among duplicates:")
print(duplicates["label"].value_counts().sort_index())

if different_label == 0:
    print("\n✅ All exact duplicates have the same class label.")
else:
    print("\n⚠️ Some exact duplicates have DIFFERENT class labels.")