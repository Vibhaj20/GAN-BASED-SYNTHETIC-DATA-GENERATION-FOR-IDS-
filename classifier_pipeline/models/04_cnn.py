import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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
print("CNN - IDS CLASSIFIER")
print("=" * 60)

print("\nLoading training data...")

X_train = pd.read_csv(TRAIN_X).values.astype("float32")
y_train = pd.read_csv(TRAIN_Y).iloc[:, 0].values.astype("int32")

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nLoading test data...")

X_test = pd.read_csv(TEST_X).values.astype("float32")
y_test = pd.read_csv(TEST_Y).iloc[:, 0].values.astype("int32")

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# 3. RESHAPE DATA FOR 1D CNN

# 101 features -> 101 time/feature positions with 1 channel

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("\nCNN training shape:", X_train.shape)
print("CNN test shape:", X_test.shape)

# 4. BUILD CNN MODEL

print("\n" + "=" * 60)
print("BUILDING CNN MODEL...")
print("=" * 60)

model = Sequential([
    
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(101, 1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(5, activation="softmax")
])

# 5. COMPILE MODEL

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN architecture:")
model.summary()

# 6. EARLY STOPPING

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 7. TRAIN CNN

print("\n" + "=" * 60)
print("TRAINING CNN...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)

print("\nCNN training completed!")

# 8. PREDICTION

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

print("Prediction completed!")

# 9. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("CNN MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 10. CLASSIFICATION REPORT

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# 11. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)

# 12. SAVE CONFUSION MATRIX

os.makedirs("results", exist_ok=True)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    "results/cnn_confusion_matrix.csv"
)

# 13. SAVE MODEL

model.save("models/cnn_model.keras")

print("\nCNN model saved to:")
print("models/cnn_model.keras")

# 14. SAVE RESULTS

results = {
    "Model": ["CNN"],
    "Accuracy": [accuracy]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/cnn_results.csv",
    index=False
)

# 15. SAVE TRAINING HISTORY

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "results/cnn_training_history.csv",
    index=False
)

print("\nResults saved:")
print("results/cnn_results.csv")
print("results/cnn_confusion_matrix.csv")
print("results/cnn_training_history.csv")

print("\n" + "=" * 60)
print("CNN COMPLETE")
print("=" * 60)