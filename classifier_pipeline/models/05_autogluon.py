import pandas as pd
import os

from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. FILE PATHS
# ============================================================

TRAIN_X = "data/X_train_augmented.csv"
TRAIN_Y = "data/y_train_multiclass_augmented.csv"

TEST_X = "data/X_test.csv"
TEST_Y = "data/y_test_multiclass.csv"

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("AUTOGLUON - IDS CLASSIFIER")
print("=" * 60)

print("\nLoading training data...")

X_train = pd.read_csv(TRAIN_X)
y_train = pd.read_csv(TRAIN_Y).iloc[:, 0]

X_test = pd.read_csv(TEST_X)
y_test = pd.read_csv(TEST_Y).iloc[:, 0]

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# ============================================================
# 3. COMBINE FEATURES + LABEL
# ============================================================

label_column = "attack_class"

train_data = X_train.copy()
train_data[label_column] = y_train.values

test_data = X_test.copy()
test_labels = y_test.copy()

print("\nTraining data prepared.")
print("Training shape:", train_data.shape)

# ============================================================
# 4. CREATE AUTOGLUON PREDICTOR
# ============================================================

print("\n" + "=" * 60)
print("STARTING AUTOGLUON TRAINING...")
print("=" * 60)

os.makedirs("models/autogluon", exist_ok=True)

predictor = TabularPredictor(
    label=label_column,
    path="models/autogluon",
    problem_type="multiclass",
    eval_metric="accuracy"
)

# ============================================================
# 5. TRAIN AUTOGLUON
# ============================================================

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600
)

print("\nAutoGluon training completed!")

# ============================================================
# 6. LEADERBOARD
# ============================================================

print("\n" + "=" * 60)
print("AUTOGLUON LEADERBOARD")
print("=" * 60)

leaderboard = predictor.leaderboard(
    test_data.assign(**{label_column: test_labels}),
    silent=True
)

print(leaderboard)

leaderboard.to_csv(
    "results/autogluon_leaderboard.csv",
    index=False
)

# ============================================================
# 7. PREDICTION

print("\nGenerating predictions...")

predictions = predictor.predict(test_data)

print("Prediction completed!")

# 8. ACCURACY

accuracy = accuracy_score(
    test_labels,
    predictions
)

print("\n" + "=" * 60)
print("AUTOGLUON PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 9. CLASSIFICATION REPORT

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\nClassification Report:")

report = classification_report(
    test_labels,
    predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# 10. CONFUSION MATRIX

cm = confusion_matrix(
    test_labels,
    predictions,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)

# 11. SAVE CONFUSION MATRIX

os.makedirs("results", exist_ok=True)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    "results/autogluon_confusion_matrix.csv"
)

# 12. SAVE RESULTS

results = {
    "Model": ["AutoGluon"],
    "Accuracy": [accuracy]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/autogluon_results.csv",
    index=False
)

print("\nResults saved:")
print("results/autogluon_results.csv")
print("results/autogluon_confusion_matrix.csv")
print("results/autogluon_leaderboard.csv")

print("\n" + "=" * 60)
print("AUTOGLUON COMPLETE")
print("=" * 60)