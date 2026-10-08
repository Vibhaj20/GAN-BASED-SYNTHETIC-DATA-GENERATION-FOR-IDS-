import pandas as pd
import os
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. FILE PATHS

TRAIN_X = "data/X_train_augmented.csv"
TRAIN_Y = "data/y_train_multiclass_augmented.csv"

TEST_X = "data/X_test.csv"
TEST_Y = "data/y_test_multiclass.csv"

# 2. LOAD DATA

print("=" * 60)
print("XGBOOST - IDS CLASSIFIER")
print("=" * 60)

print("\nLoading training data...")

X_train = pd.read_csv(TRAIN_X)
y_train = pd.read_csv(TRAIN_Y)

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nLoading test data...")

X_test = pd.read_csv(TEST_X)
y_test = pd.read_csv(TEST_Y)

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# 3. EXTRACT LABEL COLUMN

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

# 4. CREATE XGBOOST MODEL

print("\n" + "=" * 60)
print("CREATING XGBOOST MODEL...")
print("=" * 60)

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# 5. TRAIN MODEL

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("\nXGBoost training completed!")

# 6. PREDICTION

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Prediction completed!")

# 7. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 8. CLASSIFICATION REPORT

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

# 9. CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)


# 10. SAVE CONFUSION MATRIX


cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

os.makedirs("results", exist_ok=True)

cm_df.to_csv(
    "results/xgboost_confusion_matrix.csv"
)


# 11. SAVE MODEL


model_path = "models/xgboost_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)


# 12. SAVE RESULTS


results = {
    "Model": ["XGBoost"],
    "Accuracy": [accuracy]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/xgboost_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/xgboost_results.csv")
print("results/xgboost_confusion_matrix.csv")

print("\n" + "=" * 60)
print("XGBOOST COMPLETE")
print("=" * 60)