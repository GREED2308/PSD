from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE = Path(__file__).resolve().parent

CSV_PATH = (
    BASE / "data" / "training"
    / "sentinel2_samples_manual.csv"
)

MODEL_PATH = (
    BASE / "data" / "output"
    / "random_forest_sawah.joblib"
)

RESULT_PATH = (
    BASE / "data" / "output"
    / "random_forest_evaluation.csv"
)

FEATURES = [
    "B02_mean",
    "B03_mean",
    "B04_mean",
    "B08_mean",
]

TARGET = "label"


def main():
    print("=" * 70)
    print("RANDOM FOREST — KLASIFIKASI SAWAH / NON-SAWAH")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    print("\nData:", df.shape)
    print("\nDistribusi label:")
    print(df[TARGET].value_counts())

    X = df[FEATURES].astype(np.float32)
    y = df[TARGET].astype(str)

    # Hold-out 70:30
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("EVALUASI HOLD-OUT 70:30")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["non_sawah", "sawah"]
        )
    )

    # 5-fold cross validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n5-Fold Cross Validation:")
    print("Scores :", cv_scores)
    print(f"Mean   : {cv_scores.mean():.4f}")
    print(f"Std    : {cv_scores.std():.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    # Latih ulang model final dengan seluruh data manual 100 sample
    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    final_model.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "classes": ["non_sawah", "sawah"],
        },
        MODEL_PATH,
    )

    eval_df = pd.DataFrame({
        "metric": [
            "holdout_accuracy",
            "cv_mean_accuracy",
            "cv_std_accuracy",
        ],
        "value": [
            accuracy,
            cv_scores.mean(),
            cv_scores.std(),
        ]
    })

    eval_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\nModel final disimpan:")
    print(MODEL_PATH)

    print("\nEvaluasi disimpan:")
    print(RESULT_PATH)


if __name__ == "__main__":
    main()
