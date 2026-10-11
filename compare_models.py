from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "training" / "sentinel2_samples.csv"

FEATURES = [
    "B02_mean",
    "B03_mean",
    "B04_mean",
    "B08_mean",
]

TARGET = "label"


def main():
    print("=" * 70)
    print("PERBANDINGAN MODEL KLASIFIKASI SAWAH / NON-SAWAH")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=42))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        results.append({
            "Model": name,
            "Accuracy_Mean": scores.mean(),
            "Accuracy_Std": scores.std(),
            "Fold_1": scores[0],
            "Fold_2": scores[1],
            "Fold_3": scores[2],
            "Fold_4": scores[3],
            "Fold_5": scores[4],
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "Accuracy_Mean",
        ascending=False
    )

    print("\nHasil 5-Fold Cross Validation:\n")

    display_df = results_df.copy()
    display_df["Accuracy_Mean"] = display_df["Accuracy_Mean"].round(4)
    display_df["Accuracy_Std"] = display_df["Accuracy_Std"].round(4)

    print(display_df.to_string(index=False))

    output_path = (
        BASE
        / "data"
        / "output"
        / "model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nHasil disimpan ke:")
    print(output_path)

    best = results_df.iloc[0]

    print("\nModel terbaik:")
    print(best["Model"])
    print(
        f"Mean Accuracy = "
        f"{best['Accuracy_Mean']:.4f}"
    )


if __name__ == "__main__":
    main()