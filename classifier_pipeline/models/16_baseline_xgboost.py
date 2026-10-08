import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


print("=" * 70)
print("BASELINE XGBOOST - BEFORE GAN")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL TRAINING DATA
# ============================================================

print("\nLoading original training data...")

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv(
    "data/y_train_multiclass.csv"
).iloc[:, 0]

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)


# ============================================================
# 2. LOAD SAME TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv(
    "data/y_test_multiclass.csv"
).iloc[:, 0]

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)


# ============================================================
# 3. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

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


# ============================================================
# 4. TRAIN
# ============================================================

print("\nTraining baseline XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 5. PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Prediction completed!")


# ============================================================
# 6. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("BASELINE XGBOOST PERFORMANCE")
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
# 7. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/baseline_xgboost_model.pkl"
)

print("\nModel saved:")
print("models/baseline_xgboost_model.pkl")


print("\n" + "=" * 70)
print("BASELINE XGBOOST COMPLETE")
print("=" * 70)