import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("BASELINE DECISION TREE - BEFORE GAN")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL TRAINING DATA
# ------------------------------------------------------------

print("\nLoading original training data...")

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train_multiclass.csv").iloc[:, 0]

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

# ------------------------------------------------------------
# 2. LOAD SAME TEST DATA
# ------------------------------------------------------------

print("\nLoading test data...")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test_multiclass.csv").iloc[:, 0]

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# ------------------------------------------------------------
# 3. CREATE DECISION TREE
# ------------------------------------------------------------

print("\nCreating Decision Tree...")

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=None
)

# ------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------

print("\nTraining baseline Decision Tree...")

model.fit(X_train, y_train)

print("Training completed!")

# ------------------------------------------------------------
# 5. PREDICT
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 6. PERFORMANCE
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("BASELINE DECISION TREE PERFORMANCE")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

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

# ------------------------------------------------------------
# 7. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------------------
# 8. SAVE BASELINE MODEL
# ------------------------------------------------------------

joblib.dump(
    model,
    "models/baseline_decision_tree_model.pkl"
)

print("\nModel saved:")
print("models/baseline_decision_tree_model.pkl")

print("\n" + "=" * 60)
print("BASELINE DECISION TREE COMPLETE")
print("=" * 60)