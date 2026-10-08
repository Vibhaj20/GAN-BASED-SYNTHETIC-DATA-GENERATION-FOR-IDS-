import pandas as pd
import os

DATA_DIR = "data"

files = [
    "X_train_augmented.csv",
    "y_train_multiclass_augmented.csv",
    "X_test.csv",
    "y_test_multiclass.csv",
    "y_test_binary.csv"
]

print("=" * 60)
print("CHECKING IDS CLASSIFIER DATA")
print("=" * 60)

for file in files:
    path = os.path.join(DATA_DIR, file)

    if os.path.exists(path):
        df = pd.read_csv(path)

        print(f"\n{file}")
        print("Shape:", df.shape)
        print("Columns:", len(df.columns))

        if "y_" in file:
            print("Unique labels:", df.iloc[:, 0].value_counts().sort_index().to_dict())

    else:
        print(f"\nMISSING: {file}")