import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 70)
print("AUTOGLUON: BEFORE GAN vs AFTER GAN")
print("COMMON-CLEAN TEST SET")
print("=" * 70)


# ============================================================
# 1. LOAD COMMON-CLEAN TEST DATA
# ============================================================

X_test = pd.read_csv(
    "data/X_test_common_clean.csv"
)

y_test = pd.read_csv(
    "data/y_test_common_clean.csv"
).iloc[:, 0]

print("\nCommon-clean test shape:", X_test.shape)
print("Common-clean labels:", y_test.shape)


# ============================================================
# 2. LOAD BASELINE AUTOGLUON
# ============================================================

print("\nLoading baseline AutoGluon...")

baseline_predictor = TabularPredictor.load(
    "models/autogluon_baseline"
)


# ============================================================
# 3. LOAD GAN AUTOGLUON
# ============================================================

print("Loading GAN AutoGluon...")

gan_predictor = TabularPredictor.load(
    "models/autogluon"
)


# ============================================================
# 4. PREDICTIONS
# ============================================================

print("\nPredicting with baseline AutoGluon...")

baseline_pred = baseline_predictor.predict(
    X_test
)

print("\nPredicting with GAN AutoGluon...")

gan_pred = gan_predictor.predict(
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

difference = gan_accuracy - baseline_accuracy


print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

print(
    f"\nBaseline AutoGluon (Before GAN): "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"GAN AutoGluon (After GAN):       "
    f"{gan_accuracy * 100:.2f}%"
)

print(
    f"Difference:                       "
    f"{difference * 100:+.2f} percentage points"
)


# ============================================================
# 6. CLASSIFICATION REPORT - BASELINE
# ============================================================

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\n" + "=" * 70)
print("BASELINE AUTOGLUON - CLASSIFICATION REPORT")
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
# 7. CLASSIFICATION REPORT - GAN
# ============================================================

print("\n" + "=" * 70)
print("GAN AUTOGLUON - CLASSIFICATION REPORT")
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
# 8. CONFUSION MATRICES
# ============================================================

baseline_cm = confusion_matrix(
    y_test,
    baseline_pred,
    labels=[0, 1, 2, 3, 4]
)

gan_cm = confusion_matrix(
    y_test,
    gan_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\n" + "=" * 70)
print("BASELINE AUTOGLUON - CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        baseline_cm,
        index=class_names,
        columns=class_names
    )
)

print("\n" + "=" * 70)
print("GAN AUTOGLUON - CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        gan_cm,
        index=class_names,
        columns=class_names
    )
)


# ============================================================
# 9. SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Baseline AutoGluon",
        "GAN AutoGluon"
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
    "results/autogluon_before_after_common_clean.csv",
    index=False
)


# ============================================================
# 10. SAVE CONFUSION MATRICES
# ============================================================

pd.DataFrame(
    baseline_cm,
    index=class_names,
    columns=class_names
).to_csv(
    "results/baseline_autogluon_common_clean_confusion_matrix.csv"
)

pd.DataFrame(
    gan_cm,
    index=class_names,
    columns=class_names
).to_csv(
    "results/gan_autogluon_common_clean_confusion_matrix.csv"
)


print("\nResults saved:")
print("results/autogluon_before_after_common_clean.csv")
print("results/baseline_autogluon_common_clean_confusion_matrix.csv")
print("results/gan_autogluon_common_clean_confusion_matrix.csv")

print("\n" + "=" * 70)
print("AUTOGLUON COMPARISON COMPLETE")
print("=" * 70)