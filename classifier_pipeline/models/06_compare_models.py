import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# 1. MODEL NAMES AND CONFUSION MATRIX FILES
# ============================================================

models = {
    "Decision Tree": "results/decision_tree_confusion_matrix.csv",
    "XGBoost": "results/xgboost_confusion_matrix.csv",
    "CNN": "results/cnn_confusion_matrix.csv",
    "AutoGluon": "results/autogluon_confusion_matrix.csv"
}

class_names = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
    "Normal"
]

# ============================================================
# 2. STORE FINAL RESULTS
# ============================================================

all_results = []

os.makedirs("results", exist_ok=True)

# ============================================================
# 3. PROCESS EACH MODEL
# ============================================================

for model_name, file_path in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # Read confusion matrix
    cm_df = pd.read_csv(file_path, index_col=0)

    cm = cm_df.values

    print("\nConfusion Matrix:")
    print(cm_df)

    # --------------------------------------------------------
    # Calculate TP, FN, FP, TN
    # --------------------------------------------------------

    total = cm.sum()

    true_positives = cm.diagonal()

    false_negatives = cm.sum(axis=1) - true_positives

    false_positives = cm.sum(axis=0) - true_positives

    true_negatives = (
        total
        - true_positives
        - false_negatives
        - false_positives
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = true_positives / (
        true_positives + false_positives
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = true_positives / (
        true_positives + false_negatives
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = 2 * (
        precision * recall
    ) / (
        precision + recall
    )

    # Handle possible NaN
    precision = pd.Series(precision).fillna(0).values
    recall = pd.Series(recall).fillna(0).values
    f1 = pd.Series(f1).fillna(0).values

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = true_positives.sum() / total

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # --------------------------------------------------------
    # Weighted metrics
    # --------------------------------------------------------

    class_support = cm.sum(axis=1)

    weighted_precision = (
        precision * class_support
    ).sum() / total

    weighted_recall = (
        recall * class_support
    ).sum() / total

    weighted_f1 = (
        f1 * class_support
    ).sum() / total

    # --------------------------------------------------------
    # Print per-class results
    # --------------------------------------------------------

    print("\nPer-Class Metrics:")

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name}: "
            f"Precision={precision[i]:.4f}, "
            f"Recall={recall[i]:.4f}, "
            f"F1={f1[i]:.4f}"
        )

    # --------------------------------------------------------
    # Print overall results
    # --------------------------------------------------------

    print("\nOverall Metrics:")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted Precision:{weighted_precision:.4f}")
    print(f"Weighted Recall:   {weighted_recall:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results.append({
        "Model": model_name,

        "Accuracy": accuracy,

        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,

        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1,

        "R2L Precision": precision[2],
        "R2L Recall": recall[2],
        "R2L F1": f1[2],

        "U2R Precision": precision[3],
        "U2R Recall": recall[3],
        "U2R F1": f1[3]
    })


# ============================================================
# 4. CREATE FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(all_results)

# Sort by accuracy
comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

# ============================================================
# 5. SAVE FINAL COMPARISON
# ============================================================

output_file = "results/final_model_comparison.csv"

comparison.to_csv(
    output_file,
    index=False
)

# ============================================================
# 6. DISPLAY FINAL TABLE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFinal comparison saved to:")
print(output_file)

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)