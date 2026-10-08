import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("BASELINE DECISION TREE - CLEAN TEST")
print("=" * 60)

# Load original training data
X_train = pd.read_csv("data/X_train.csv")

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test_multiclass.csv").iloc[:, 0]

# Find exact train-test overlaps
train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

train_hash_set = set(train_hashes)

overlap_mask = test_hashes.isin(train_hash_set)

# Keep only test rows with NO exact match in original training data
clean_mask = ~overlap_mask

X_test_clean = X_test.loc[clean_mask].reset_index(drop=True)
y_test_clean = y_test.loc[clean_mask].reset_index(drop=True)

print("\nTotal test samples:", len(X_test))
print("Overlapping test samples:", overlap_mask.sum())
print("Clean test samples:", len(X_test_clean))

# Load already-trained baseline model
model = joblib.load(
    "models/baseline_decision_tree_model.pkl"
)

# Predict clean test set
y_pred = model.predict(X_test_clean)

# Accuracy
accuracy = accuracy_score(
    y_test_clean,
    y_pred
)

print("\n" + "=" * 60)
print("CLEAN BASELINE PERFORMANCE")
print("=" * 60)

print(f"\nClean Test Accuracy: {accuracy * 100:.2f}%")

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\nClassification Report:")

print(
    classification_report(
        y_test_clean,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\n" + "=" * 60)
print("CLEAN BASELINE DT COMPLETE")
print("=" * 60)