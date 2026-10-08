import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from tensorflow.keras.models import load_model
from autogluon.tabular import TabularPredictor


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

print("=" * 70)
print("CLEAN TEST EVALUATION - EXACT OVERLAP REMOVED")
print("=" * 70)

print("\nLoading data...")

X_train = pd.read_csv("data/X_train_augmented.csv")
X_test = pd.read_csv("data/X_test.csv")

y_test = pd.read_csv(
    "data/y_test_multiclass.csv"
).iloc[:, 0]

print("Training data:", X_train.shape)
print("Test data    :", X_test.shape)


# ============================================================
# 2. FIND EXACT TRAIN-TEST OVERLAPS
# ============================================================

print("\nFinding exact train-test overlaps...")

train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

train_hash_set = set(train_hashes)

# True = test row exists somewhere in training data
overlap_mask = test_hashes.isin(train_hash_set)

clean_mask = ~overlap_mask

print("\nTotal test samples:", len(X_test))
print("Overlapping test samples:", overlap_mask.sum())
print("Clean test samples:", clean_mask.sum())


# ============================================================
# 3. CREATE CLEAN TEST SET
# ============================================================

X_test_clean = X_test.loc[
    clean_mask
].reset_index(drop=True)

y_test_clean = y_test.loc[
    clean_mask
].reset_index(drop=True)

print("\nClean test shape:", X_test_clean.shape)
print("Clean labels shape:", y_test_clean.shape)

print("\nClean test class distribution:")
print(
    y_test_clean
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAVED MODELS")
print("=" * 70)


# Decision Tree
print("\nLoading Decision Tree...")
dt_model = joblib.load(
    "models/decision_tree_model.pkl"
)


# XGBoost
print("Loading XGBoost...")
xgb_model = joblib.load(
    "models/xgboost_model.pkl"
)


# CNN
print("Loading CNN...")
cnn_model = load_model(
    "models/cnn_model.keras"
)


# AutoGluon
print("Loading AutoGluon...")
ag_predictor = TabularPredictor.load(
    "models/autogluon"
)

print("\nAll models loaded successfully!")


# ============================================================
# 5. CLASS NAMES
# ============================================================

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]


# ============================================================
# 6. DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE - CLEAN TEST")
print("=" * 70)

dt_pred = dt_model.predict(
    X_test_clean
)

dt_accuracy = accuracy_score(
    y_test_clean,
    dt_pred
)

print(
    f"\nClean Test Accuracy: "
    f"{dt_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test_clean,
        dt_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 7. XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST - CLEAN TEST")
print("=" * 70)

xgb_pred = xgb_model.predict(
    X_test_clean
)

xgb_accuracy = accuracy_score(
    y_test_clean,
    xgb_pred
)

print(
    f"\nClean Test Accuracy: "
    f"{xgb_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test_clean,
        xgb_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 8. CNN
# ============================================================

print("\n" + "=" * 70)
print("CNN - CLEAN TEST")
print("=" * 70)

X_test_cnn = X_test_clean.values.astype(
    "float32"
)

X_test_cnn = X_test_cnn.reshape(
    X_test_cnn.shape[0],
    X_test_cnn.shape[1],
    1
)

cnn_probabilities = cnn_model.predict(
    X_test_cnn,
    batch_size=256,
    verbose=1
)

cnn_pred = np.argmax(
    cnn_probabilities,
    axis=1
)

cnn_accuracy = accuracy_score(
    y_test_clean,
    cnn_pred
)

print(
    f"\nClean Test Accuracy: "
    f"{cnn_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test_clean,
        cnn_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 9. AUTOGLUON
# ============================================================

print("\n" + "=" * 70)
print("AUTOGLUON - CLEAN TEST")
print("=" * 70)

ag_pred = ag_predictor.predict(
    X_test_clean
)

ag_accuracy = accuracy_score(
    y_test_clean,
    ag_pred
)

print(
    f"\nClean Test Accuracy: "
    f"{ag_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test_clean,
        ag_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 10. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLEAN TEST MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "XGBoost",
        "CNN",
        "AutoGluon"
    ],
    "Clean_Test_Accuracy": [
        dt_accuracy,
        xgb_accuracy,
        cnn_accuracy,
        ag_accuracy
    ]
})

comparison["Clean_Test_Accuracy_%"] = (
    comparison["Clean_Test_Accuracy"] * 100
)

print("\n")
print(
    comparison[
        ["Model", "Clean_Test_Accuracy_%"]
    ].to_string(index=False)
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

comparison.to_csv(
    "results/clean_test_model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print(
    "results/clean_test_model_comparison.csv"
)

print("\n" + "=" * 70)
print("CLEAN TEST EVALUATION COMPLETE")
print("=" * 70)