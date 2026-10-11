from __future__ import annotations

"""Baseline comparison for the six-class Sentinel-2 land-cover project.

Runs two dataset variants (all reviewed samples and quality-screened samples),
using a stratified train/test split and 5-fold CV on the training split only.
Outputs metrics, per-class reports, confusion matrices, split assignments, and
serialized fitted estimators. This is a random-split baseline, not spatial CV.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = [
    "B01_mean", "B02_mean", "B03_mean", "B04_mean", "B05_mean", "B06_mean",
    "B07_mean", "B08_mean", "B8A_mean", "B09_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean",
]
EXPECTED_CLASSES = ["sawah", "bangunan", "mangrove", "lahan_hijau", "laut", "danau"]
DATASETS = {
    "all_reviewed": PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6" / "dataset_modeling_v6_all_reviewed.csv",
    "quality_screened": PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6" / "dataset_modeling_v6_quality_screened.csv",
}
OUTPUT_DIR = PROJECT_DIR / "outputs" / "modeling_v6_baseline"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def make_models(min_class_count: int) -> dict[str, Pipeline]:
    n_neighbors = min(5, max(1, min_class_count - 1))
    return {
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced", min_samples_leaf=2)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE,
                class_weight="balanced_subsample", min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "class_name", "status", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di {path.name}: {missing}")
    df = df.copy()
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df = df[df["status"].astype(str).str.upper().eq("OK")].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Invalid or missing input values are imputed within each training fold only.
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = df["class_name"]
    valid_target = y.isin(EXPECTED_CLASSES)
    df, X, y = df.loc[valid_target].copy(), X.loc[valid_target].copy(), y.loc[valid_target].copy()
    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise ValueError(f"sample_id duplikat ditemukan: {dupes}")
    if X.isna().all(axis=1).any():
        ids = df.loc[X.isna().all(axis=1), "sample_id"].tolist()[:10]
        raise ValueError(f"Sampel tanpa satu pun fitur numerik valid: {ids}")
    return df.reset_index(drop=True), y.reset_index(drop=True), X.reset_index(drop=True)


def run_variant(variant: str, path: Path) -> list[dict]:
    df, y, X = load_dataset(path)
    counts = y.value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"Varian {variant} tidak memiliki keenam kelas: {counts.to_dict()}")
    min_count = int(counts.min())
    if min_count < 6:
        raise ValueError(f"Jumlah minimum per kelas terlalu kecil untuk split + CV 5-fold: {counts.to_dict()}")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    out_dir = OUTPUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df = df[["sample_id", "class_name"]].copy()
    split_df["split"] = "train"
    split_df.loc[test_idx, "split"] = "test"
    split_df.to_csv(out_dir / "split_train_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"class_name": EXPECTED_CLASSES,
                  "all_count": [int(counts[c]) for c in EXPECTED_CLASSES],
                  "train_count": [int(y_train.value_counts().get(c, 0)) for c in EXPECTED_CLASSES],
                  "test_count": [int(y_test.value_counts().get(c, 0)) for c in EXPECTED_CLASSES]}).to_csv(
        out_dir / "distribusi_kelas.csv", index=False, encoding="utf-8-sig"
    )

    cv_splits = min(N_SPLITS, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}
    results = []
    for model_name, estimator in make_models(min_count).items():
        print(f"[{variant}] {model_name}: CV {cv_splits}-fold + evaluasi test ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_result = cross_validate(
                clone(estimator), X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=None, error_score="raise"
            )
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_test)
        metrics = {
            "dataset_variant": variant,
            "model": model_name,
            "n_samples": int(len(df)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "features": ";".join(FEATURES),
            "cv_folds": cv_splits,
            "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_result["test_accuracy"], ddof=1)) if cv_splits > 1 else 0.0,
            "cv_balanced_accuracy_mean": float(np.mean(cv_result["test_balanced_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv_result["test_f1_macro"])),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "test_precision_macro": float(precision_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_test, pred, labels=EXPECTED_CLASSES, average="weighted", zero_division=0)),
            "test_cohen_kappa": float(cohen_kappa_score(y_test, pred, labels=EXPECTED_CLASSES)),
        }
        results.append(metrics)
        report = classification_report(
            y_test, pred, labels=EXPECTED_CLASSES, target_names=EXPECTED_CLASSES,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            out_dir / f"classification_report_{model_name}.csv", encoding="utf-8-sig"
        )
        cm = confusion_matrix(y_test, pred, labels=EXPECTED_CLASSES)
        pd.DataFrame(cm, index=EXPECTED_CLASSES, columns=EXPECTED_CLASSES).to_csv(
            out_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig"
        )
        pd.DataFrame({"sample_id": test_df["sample_id"].to_numpy(),
                      "y_true": y_test.to_numpy(), "y_pred": pred}).to_csv(
            out_dir / f"prediksi_test_{model_name}.csv", index=False, encoding="utf-8-sig"
        )
        joblib.dump(fitted, out_dir / f"model_{model_name}.joblib")

    pd.DataFrame(results).sort_values("test_f1_macro", ascending=False).to_csv(
        out_dir / "perbandingan_model.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "dataset_variant": variant,
        "source_file": str(path),
        "n_samples_used": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "features": FEATURES,
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds_on_training_only": cv_splits,
        "validation_note": "Random stratified split; not spatially blocked. Nearby/geographically correlated samples may make estimates optimistic.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    best = max(results, key=lambda r: r["test_f1_macro"])
    print(f"[{variant}] selesai: {len(df)} sampel. F1-macro test tertinggi (eksperimen ini): {best['model']} = {best['test_f1_macro']:.3f}", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Bandingkan baseline model klasifikasi tutupan lahan enam kelas v6.")
    parser.add_argument("--variant", choices=["all_reviewed", "quality_screened", "both"], default="both",
                        help="Varian dataset yang diuji (default: both).")
    args = parser.parse_args()
    print("=== Baseline modeling Sentinel-2 v6 ===")
    print(f"Fitur: {len(FEATURES)} (12 band *_mean + 4 indeks *_mean)")
    print("Model: Decision Tree, Random Forest, SVM RBF, KNN, Logistic Regression")
    print("Split: stratified 80/20; CV 5-fold hanya pada data training")
    print("Catatan: evaluasi acak ini belum menguji generalisasi spasial.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = ["all_reviewed", "quality_screened"] if args.variant == "both" else [args.variant]
    all_results = []
    try:
        for variant in variants:
            all_results.extend(run_variant(variant, DATASETS[variant]))
        combined = pd.DataFrame(all_results).sort_values(["dataset_variant", "test_f1_macro"], ascending=[True, False])
        combined.to_csv(OUTPUT_DIR / "perbandingan_semua_varian.csv", index=False, encoding="utf-8-sig")
        print("\n=== Ringkasan F1-macro test ===")
        print(combined[["dataset_variant", "model", "cv_f1_macro_mean", "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nSemua laporan: {OUTPUT_DIR}")
        print("Penting: jangan memilih model hanya dari skor test tunggal; pertimbangkan CV, per kelas, bias spasial, dan kualitas label.")
        return 0
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
