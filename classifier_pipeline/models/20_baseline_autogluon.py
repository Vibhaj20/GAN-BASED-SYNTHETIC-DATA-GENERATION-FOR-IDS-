import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report


print("=" * 70)
print("BASELINE AUTOGLUON - BEFORE GAN")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL TRAINING DATA
# ============================================================

print("\nLoading original training data...")

X_train = pd.read_csv(
    "data/X_train.csv"
)

y_train = pd.read_csv(
    "data/y_train_multiclass.csv"
).iloc[:, 0]

train_data = X_train.copy()
train_data["attack_class"] = y_train

print("Training data shape:", train_data.shape)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = pd.read_csv(
    "data/X_test.csv"
)

y_test = pd.read_csv(
    "data/y_test_multiclass.csv"
).iloc[:, 0]

test_data = X_test.copy()
test_data["attack_class"] = y_test

print("Test data shape:", test_data.shape)


# ============================================================
# 3. TRAIN AUTOGLUON
# ============================================================

print("\nTraining AutoGluon...")

predictor = TabularPredictor(
    label="attack_class",
    path="models/autogluon_baseline",
    problem_type="multiclass",
    eval_metric="accuracy"
)

predictor.fit(
    train_data=train_data,
    presets="medium_quality",
    time_limit=600
)


# ============================================================
# 4. PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = predictor.predict(
    X_test
)


# ============================================================
# 5. EVALUATE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("BASELINE AUTOGLUON PERFORMANCE")
print("=" * 70)

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

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
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Baseline AutoGluon"],
    "Accuracy": [accuracy],
    "Accuracy_Percent": [accuracy * 100]
})

results.to_csv(
    "results/baseline_autogluon_results.csv",
    index=False
)

print("\nResults saved:")
print("results/baseline_autogluon_results.csv")

print("\nModel saved:")
print("models/autogluon_baseline")

print("\n" + "=" * 70)
print("BASELINE AUTOGLUON COMPLETE")
print("=" * 70)