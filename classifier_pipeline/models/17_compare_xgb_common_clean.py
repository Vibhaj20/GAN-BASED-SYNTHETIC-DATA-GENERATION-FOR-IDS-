import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

print("=" * 70)
print("XGBOOST - COMMON CLEAN TEST COMPARISON")
print("=" * 70)

# ============================================================
# 1. LOAD COMMON CLEAN TEST SET
# ============================================================

X_test = pd.read_csv(
    "data/X_test_common_clean.csv"
)

y_test = pd.read_csv(
    "data/y_test_common_clean.csv"
).iloc[:, 0]

print("\nCommon clean test shape:", X_test.shape)
print("Common clean labels:", y_test.shape)


# ============================================================
# 2. LOAD ORIGINAL XGBOOST
# ============================================================

print("\nLoading original/baseline XGBoost...")

baseline_model = joblib.load(
    "models/baseline_xgboost_model.pkl"
)


# ============================================================
# 3. LOAD GAN XGBOOST
# ============================================================

print("Loading GAN-augmented XGBoost...")

gan_model = joblib.load(
    "models/xgboost_model.pkl"
)


# ============================================================
# 4. PREDICTIONS
# ============================================================

print("\nGenerating baseline predictions...")

baseline_pred = baseline_model.predict(
    X_test
)

print("Generating GAN predictions...")

gan_pred = gan_model.predict(
    X_test
)


# ============================================================
# 5. ACCURACY
# ============================================================

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

gan_accuracy = accuracy_score(
    y_test,
    gan_pred
)

print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

print(
    f"\nOriginal XGBoost Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"GAN XGBoost Accuracy:      "
    f"{gan_accuracy * 100:.2f}%"
)

print(
    f"Difference:                "
    f"{(gan_accuracy - baseline_accuracy) * 100:.2f} percentage points"
)


# ============================================================
# 6. ORIGINAL XGBOOST REPORT
# ============================================================

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\n" + "=" * 70)
print("ORIGINAL XGBOOST - COMMON CLEAN TEST")
print("=" * 70)

print(
    classification_report(
        y_test,
        baseline_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 7. GAN XGBOOST REPORT
# ============================================================

print("\n" + "=" * 70)
print("GAN XGBOOST - COMMON CLEAN TEST")
print("=" * 70)

print(
    classification_report(
        y_test,
        gan_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 8. SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Original XGBoost",
        "GAN-Augmented XGBoost"
    ],
    "Accuracy": [
        baseline_accuracy,
        gan_accuracy
    ],
    "Accuracy_Percent": [
        baseline_accuracy * 100,
        gan_accuracy * 100
    ]
})

comparison.to_csv(
    "results/xgb_common_clean_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("results/xgb_common_clean_comparison.csv")

print("\n" + "=" * 70)
print("XGBOOST COMPARISON COMPLETE")
print("=" * 70)