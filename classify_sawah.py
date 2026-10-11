from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

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
    print("=" * 60)
    print("KLASIFIKASI SAWAH / NON-SAWAH")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    print(f"Data       : {df.shape}")
    print("\nDistribusi label:")
    print(df[TARGET].value_counts())

    # Pastikan fitur yang diperlukan tersedia
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di CSV: {missing}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Train-test split dengan stratifikasi
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("\nPembagian data:")
    print(f"Training : {len(X_train)}")
    print(f"Testing  : {len(X_test)}")

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    # Simpan model dan prediksi test
    output = X_test.copy()
    output["actual"] = y_test.values
    output["predicted"] = y_pred

    output_path = BASE / "data" / "output" / "classification_test.csv"
    output.to_csv(output_path, index=False)

    print(f"\nHasil test disimpan:")
    print(output_path)


if __name__ == "__main__":
    main()