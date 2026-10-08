import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 70)
print("CNN: BEFORE GAN vs AFTER GAN")
print("COMMON-CLEAN TEST SET")
print("=" * 70)


# ============================================================
# 1. LOAD COMMON-CLEAN TEST DATA
# ============================================================

X_test = pd.read_csv(
    "data/X_test_common_clean.csv"
).values.astype("float32")

y_test = pd.read_csv(
    "data/y_test_common_clean.csv"
).iloc[:, 0].values.astype("int32")

print("\nCommon-clean test shape:", X_test.shape)
print("Common-clean labels:", y_test.shape)


# ============================================================
# 2. RESHAPE FOR CNN
# ============================================================

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


# ============================================================
# 3. LOAD BASELINE CNN
# ============================================================

print("\nLoading baseline CNN...")

baseline_model = load_model(
    "models/baseline_cnn_model.keras"
)


# ============================================================
# 4. LOAD GAN CNN
# ============================================================

print("Loading GAN CNN...")

gan_model = load_model(
    "models/cnn_model.keras"
)


# ============================================================
# 5. BASELINE CNN PREDICTION
# ============================================================

print("\nPredicting with baseline CNN...")

baseline_probabilities = baseline_model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

baseline_pred = np.argmax(
    baseline_probabilities,
    axis=1
)


# ============================================================
# 6. GAN CNN PREDICTION
# ============================================================

print("\nPredicting with GAN CNN...")

gan_probabilities = gan_model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

gan_pred = np.argmax(
    gan_probabilities,
    axis=1
)


# ============================================================
# 7. ACCURACY
# ============================================================

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

gan_accuracy = accuracy_score(
    y_test,
    gan_pred
)

difference = (
    gan_accuracy - baseline_accuracy
)


# ============================================================
# 8. PRINT ACCURACY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

print(
    f"\nBaseline CNN (Before GAN): "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"GAN CNN (After GAN):       "
    f"{gan_accuracy * 100:.2f}%"
)

print(
    f"Difference:                 "
    f"{difference * 100:+.2f} percentage points"
)


# ============================================================
# 9. CLASSIFICATION REPORT - BASELINE
# ============================================================

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\n" + "=" * 70)
print("BASELINE CNN - CLASSIFICATION REPORT")
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
# 10. CLASSIFICATION REPORT - GAN
# ============================================================

print("\n" + "=" * 70)
print("GAN CNN - CLASSIFICATION REPORT")
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
# 11. CONFUSION MATRICES
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
print("BASELINE CNN - CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        baseline_cm,
        index=class_names,
        columns=class_names
    )
)

print("\n" + "=" * 70)
print("GAN CNN - CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        gan_cm,
        index=class_names,
        columns=class_names
    )
)


# ============================================================
# 12. SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Baseline CNN",
        "GAN CNN"
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
    "results/cnn_before_after_common_clean.csv",
    index=False
)


# ============================================================
# 13. SAVE CONFUSION MATRICES
# ============================================================

pd.DataFrame(
    baseline_cm,
    index=class_names,
    columns=class_names
).to_csv(
    "results/baseline_cnn_common_clean_confusion_matrix.csv"
)

pd.DataFrame(
    gan_cm,
    index=class_names,
    columns=class_names
).to_csv(
    "results/gan_cnn_common_clean_confusion_matrix.csv"
)


print("\nResults saved to:")
print("results/cnn_before_after_common_clean.csv")
print("results/baseline_cnn_common_clean_confusion_matrix.csv")
print("results/gan_cnn_common_clean_confusion_matrix.csv")

print("\n" + "=" * 70)
print("CNN COMPARISON COMPLETE")
print("=" * 70)