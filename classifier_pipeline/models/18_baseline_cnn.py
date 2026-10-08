import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


print("=" * 70)
print("BASELINE CNN - BEFORE GAN")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL TRAINING DATA
# ============================================================

print("\nLoading original training data...")

X_train = pd.read_csv(
    "data/X_train.csv"
).values.astype("float32")

y_train = pd.read_csv(
    "data/y_train_multiclass.csv"
).iloc[:, 0].values.astype("int32")

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = pd.read_csv(
    "data/X_test.csv"
).values.astype("float32")

y_test = pd.read_csv(
    "data/y_test_multiclass.csv"
).iloc[:, 0].values.astype("int32")

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)


# ============================================================
# 3. RESHAPE FOR CNN
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nCNN training shape:", X_train.shape)
print("CNN test shape:", X_test.shape)


# ============================================================
# 4. BUILD SAME CNN ARCHITECTURE
# ============================================================

print("\nBuilding CNN...")

model = Sequential([

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(101, 1)
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        5,
        activation="softmax"
    )
])


# ============================================================
# 5. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 6. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining baseline CNN...")

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


# ============================================================
# 8. PREDICTION
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 9. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("BASELINE CNN PERFORMANCE")
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
# 10. SAVE MODEL
# ============================================================

model.save(
    "models/baseline_cnn_model.keras"
)

print("\nModel saved:")
print("models/baseline_cnn_model.keras")


print("\n" + "=" * 70)
print("BASELINE CNN COMPLETE")
print("=" * 70)