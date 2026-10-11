from __future__ import annotations

"""Spatial-block holdout evaluation for Sentinel-2 six-class land-cover v6.

Uses GeoJSON polygon centroids to assign spatial blocks, chooses one holdout
split without looking at model scores, and evaluates KNN and SVM-RBF on both
all-reviewed and quality-screened CSVs. This is a spatial holdout experiment,
not a guarantee of province-wide generalization.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6"
GEOJSON_PATH = PROJECT_DIR / "outputs" / "dataset_v6_review" / "dataset_6kelas_v6_kandidat_representatif_303.geojson"
DATASETS = {
    "all_reviewed": BASE_DIR / "dataset_modeling_v6_all_reviewed.csv",
    "quality_screened": BASE_DIR / "dataset_modeling_v6_quality_screened.csv",
}
OUTPUT_DIR = PROJECT_DIR / "outputs" / "modeling_v6_spatial_validation"
EXPECTED_CLASSES = ["sawah", "bangunan", "mangrove", "lahan_hijau", "laut", "danau"]
FEATURES = [
    "B01_mean", "B02_mean", "B03_mean", "B04_mean", "B05_mean", "B06_mean",
    "B07_mean", "B08_mean", "B8A_mean", "B09_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean",
]
RANDOM_STATE = 42


def load_centroids(block_size_m: float) -> pd.DataFrame:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON tidak ditemukan: {GEOJSON_PATH}")
    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    # Azimuthal-equidistant projection centered on East Java; used only to make
    # approximate metric spatial blocks over the study area.
    local_crs = "+proj=aeqd +lat_0=-7.5 +lon_0=112.5 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    rows = []
    seen = set()
    for feature in fc.get("features", []):
        props = feature.get("properties") or {}
        sid = str(props.get("sample_id", "")).strip()
        geom_obj = feature.get("geometry")
        if not sid or not geom_obj:
            continue
        if sid in seen:
            raise ValueError(f"sample_id duplikat pada GeoJSON: {sid}")
        seen.add(sid)
        geom = shape(geom_obj)
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Geometri kosong/tidak valid untuk {sid}; perbaiki sebelum evaluasi.")
        # Centroid in the local metric projection, not centroid in longitude/latitude.
        geom_local = shapely_transform(transformer.transform, geom)
        centroid_local = geom_local.centroid
        centroid_wgs84 = geom.centroid
        gx = math.floor(centroid_local.x / block_size_m)
        gy = math.floor(centroid_local.y / block_size_m)
        rows.append({
            "sample_id": sid,
            "longitude_centroid": float(centroid_wgs84.x),
            "latitude_centroid": float(centroid_wgs84.y),
            "block_x": int(gx),
            "block_y": int(gy),
            "spatial_block_id": f"B{gx}_{gy}",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("GeoJSON tidak berisi geometri sampel valid.")
    return out


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset tidak ditemukan: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "class_name", "status", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di {path.name}: {missing}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df = df[df["status"].astype(str).str.upper().eq("OK")].copy()
    df = df[df["class_name"].isin(EXPECTED_CLASSES)].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"sample_id duplikat pada {path.name}")
    if df[FEATURES].isna().all(axis=1).any():
        bad = df.loc[df[FEATURES].isna().all(axis=1), "sample_id"].tolist()[:10]
        raise ValueError(f"Sampel tanpa nilai fitur valid pada {path.name}: {bad}")
    return df.reset_index(drop=True)


def make_model(name: str, min_train_class_count: int):
    if name == "KNN":
        k = min(5, max(1, min_train_class_count - 1))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ])
    if name == "SVM_RBF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        ])
    raise ValueError(name)


def score_split_candidate(merged: pd.DataFrame, variants: dict[str, pd.DataFrame], train_idx, test_idx):
    # The same spatial blocks must be used in both data variants so the
    # comparisons share the same geographic holdout as far as sample filtering allows.
    test_fraction_errors = []
    class_distribution_errors = []
    for variant, vdf in variants.items():
        v_indices = vdf.index.to_numpy()
        test_ids = set(merged.iloc[test_idx]["sample_id"])
        train_ids = set(merged.iloc[train_idx]["sample_id"])
        v_test = vdf[vdf["sample_id"].isin(test_ids)]
        v_train = vdf[vdf["sample_id"].isin(train_ids)]
        if v_train.empty or v_test.empty:
            return None
        if set(v_train["class_name"]) != set(EXPECTED_CLASSES) or set(v_test["class_name"]) != set(EXPECTED_CLASSES):
            return None
        test_counts = v_test["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
        train_counts = v_train["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
        if int(test_counts.min()) < 2 or int(train_counts.min()) < 4:
            return None
        frac = len(v_test) / len(vdf)
        if not 0.15 <= frac <= 0.35:
            return None
        test_fraction_errors.append(abs(frac - 0.20))
        overall_props = vdf["class_name"].value_counts(normalize=True).reindex(EXPECTED_CLASSES, fill_value=0)
        test_props = v_test["class_name"].value_counts(normalize=True).reindex(EXPECTED_CLASSES, fill_value=0)
        class_distribution_errors.append(float(np.abs(overall_props - test_props).mean()))
    # Select a split for size and class representation only, never based on model scores.
    return float(np.mean(test_fraction_errors) + 2.0 * np.mean(class_distribution_errors))


def choose_spatial_split(merged: pd.DataFrame, variants: dict[str, pd.DataFrame], n_trials: int = 10000):
    groups = merged["spatial_block_id"].to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError(
            f"Hanya {len(np.unique(groups))} blok spasial. Tidak cukup untuk mencari holdout yang representatif; "
            "coba --block-size-km lebih kecil (misalnya 10 atau 15)."
        )
    splitter = GroupShuffleSplit(n_splits=n_trials, test_size=0.20, random_state=RANDOM_STATE)
    best = None
    best_score = float("inf")
    for train_idx, test_idx in splitter.split(merged, merged["class_name"], groups):
        score = score_split_candidate(merged, variants, train_idx, test_idx)
        if score is not None and score < best_score:
            best_score = score
            best = (train_idx.copy(), test_idx.copy())
    if best is None:
        # Relax the per-class minimum from 2 to 1 but retain all six classes in both
        # subsets and keep the same spatial split for both variants.
        for train_idx, test_idx in GroupShuffleSplit(n_splits=n_trials, test_size=0.20, random_state=RANDOM_STATE + 1).split(merged, merged["class_name"], groups):
            valid = True
            errors = []
            for _, vdf in variants.items():
                ids_test = set(merged.iloc[test_idx]["sample_id"])
                ids_train = set(merged.iloc[train_idx]["sample_id"])
                v_test = vdf[vdf["sample_id"].isin(ids_test)]
                v_train = vdf[vdf["sample_id"].isin(ids_train)]
                if set(v_train["class_name"]) != set(EXPECTED_CLASSES) or set(v_test["class_name"]) != set(EXPECTED_CLASSES):
                    valid = False
                    break
                if v_test["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0).min() < 1:
                    valid = False
                    break
                frac = len(v_test) / len(vdf)
                if not 0.10 <= frac <= 0.40:
                    valid = False
                    break
                overall = vdf["class_name"].value_counts(normalize=True).reindex(EXPECTED_CLASSES, fill_value=0)
                testp = v_test["class_name"].value_counts(normalize=True).reindex(EXPECTED_CLASSES, fill_value=0)
                errors.append(abs(frac - 0.20) + 2 * float(np.abs(overall - testp).mean()))
            if valid:
                score = float(np.mean(errors))
                if score < best_score:
                    best_score = score
                    best = (train_idx.copy(), test_idx.copy())
        if best is None:
            raise ValueError(
                "Tidak ditemukan split blok spasial yang mempertahankan keenam kelas pada train dan test "
                "untuk kedua varian. Jangan gunakan split acak sebagai pengganti diam-diam; coba ukuran blok lain "
                "atau gunakan leave-location-out berdasarkan wilayah yang dipilih secara eksplisit."
            )
        print("PERINGATAN: split dipilih dengan minimal 1 sampel per kelas pada test; interpretasikan metrik per kelas secara hati-hati.")
    return best[0], best[1], best_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi holdout spasial untuk model KNN dan SVM RBF.")
    parser.add_argument("--block-size-km", type=float, default=20.0,
                        help="Ukuran blok spasial (km), default 20. Coba 10/15/25 jika split representatif tidak ditemukan.")
    parser.add_argument("--trials", type=int, default=10000,
                        help="Jumlah split kandidat yang dicoba (default 10000).")
    args = parser.parse_args()
    if args.block_size_km <= 0:
        raise ValueError("block-size-km harus > 0")

    block_size_m = args.block_size_km * 1000.0
    print("=== Evaluasi holdout spasial Sentinel-2 v6 ===")
    print(f"GeoJSON : {GEOJSON_PATH}")
    print(f"Ukuran blok : {args.block_size_km:g} km (grid metrik proyeksi lokal)")
    print("Metode split: semua sampel dalam blok spasial yang sama ditempatkan bersama.")
    print("Model: KNN dan SVM RBF; split dicari berdasarkan cakupan kelas/ukuran, bukan skor model.")

    centroids = load_centroids(block_size_m)
    datasets = {name: load_csv(path) for name, path in DATASETS.items()}
    full = datasets["all_reviewed"].merge(centroids, on="sample_id", how="left", validate="one_to_one")
    if full["spatial_block_id"].isna().any():
        missing = full.loc[full["spatial_block_id"].isna(), "sample_id"].tolist()[:10]
        raise ValueError(f"Sampel tidak cocok dengan GeoJSON: {missing}")
    for variant, df in datasets.items():
        missing_ids = sorted(set(df["sample_id"]) - set(centroids["sample_id"]))
        if missing_ids:
            raise ValueError(f"ID dari {variant} tidak ada di GeoJSON: {missing_ids[:10]}")

    train_idx, test_idx, split_score = choose_spatial_split(full, datasets, n_trials=args.trials)
    train_ids = set(full.iloc[train_idx]["sample_id"])
    test_ids = set(full.iloc[test_idx]["sample_id"])
    group_train = set(full.iloc[train_idx]["spatial_block_id"])
    group_test = set(full.iloc[test_idx]["spatial_block_id"])
    if group_train & group_test:
        raise RuntimeError("Kebocoran blok ditemukan: blok yang sama masuk train dan test.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = full[["sample_id", "class_name", "spatial_block_id", "longitude_centroid", "latitude_centroid"]].copy()
    split["split"] = np.where(split["sample_id"].isin(test_ids), "test", "train")
    split.to_csv(OUTPUT_DIR / "spatial_split_shared.csv", index=False, encoding="utf-8-sig")

    metrics_all = []
    for variant, vdf in datasets.items():
        v = vdf.merge(centroids, on="sample_id", how="left", validate="one_to_one")
        train = v[v["sample_id"].isin(train_ids)].copy()
        test = v[v["sample_id"].isin(test_ids)].copy()
        train_counts = train["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
        test_counts = test["class_name"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
        variant_dir = OUTPUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "class_name": EXPECTED_CLASSES,
            "train_count": [int(train_counts[c]) for c in EXPECTED_CLASSES],
            "test_count": [int(test_counts[c]) for c in EXPECTED_CLASSES],
        }).to_csv(variant_dir / "distribusi_kelas_spasial.csv", index=False, encoding="utf-8-sig")
        X_train = train[FEATURES].to_numpy(dtype=float)
        y_train = train["class_name"].to_numpy()
        X_test = test[FEATURES].to_numpy(dtype=float)
        y_test = test["class_name"].to_numpy()
        min_train_class = int(train_counts.min())

        for model_name in ["KNN", "SVM_RBF"]:
            print(f"[{variant}] {model_name}: {len(train)} train / {len(test)} spatial test ...", flush=True)
            model = make_model(model_name, min_train_class)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            row = {
                "dataset_variant": variant,
                "model": model_name,
                "n_samples": len(v),
                "n_train": len(train),
                "n_test": len(test),
                "n_train_spatial_blocks": len(group_train),
                "n_test_spatial_blocks": len(group_test),
                "block_size_km": args.block_size_km,
                "test_accuracy": accuracy_score(y_test, pred),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "test_precision_macro": precision_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0),
                "test_recall_macro": recall_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0),
                "test_f1_macro": f1_score(y_test, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0),
                "test_f1_weighted": f1_score(y_test, pred, labels=EXPECTED_CLASSES, average="weighted", zero_division=0),
                "test_cohen_kappa": cohen_kappa_score(y_test, pred, labels=EXPECTED_CLASSES),
            }
            metrics_all.append(row)
            report = classification_report(y_test, pred, labels=EXPECTED_CLASSES, target_names=EXPECTED_CLASSES,
                                           output_dict=True, zero_division=0)
            pd.DataFrame(report).transpose().to_csv(variant_dir / f"classification_report_{model_name}.csv", encoding="utf-8-sig")
            cm = confusion_matrix(y_test, pred, labels=EXPECTED_CLASSES)
            pd.DataFrame(cm, index=EXPECTED_CLASSES, columns=EXPECTED_CLASSES).to_csv(
                variant_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig")
            pred_df = test[["sample_id", "class_name", "spatial_block_id", "longitude_centroid", "latitude_centroid"]].copy()
            pred_df["y_pred"] = pred
            pred_df["correct"] = pred_df["class_name"].to_numpy() == pred
            pred_df.to_csv(variant_dir / f"prediksi_spatial_test_{model_name}.csv", index=False, encoding="utf-8-sig")
            # Save model trained only on the spatial training blocks.
            import joblib
            joblib.dump(model, variant_dir / f"model_spatial_{model_name}.joblib")

    metrics_df = pd.DataFrame(metrics_all)
    metrics_df.to_csv(OUTPUT_DIR / "perbandingan_spatial_models.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "geojson": str(GEOJSON_PATH),
        "block_size_km": args.block_size_km,
        "block_projection": "+proj=aeqd +lat_0=-7.5 +lon_0=112.5 +datum=WGS84 +units=m +no_defs",
        "split_score_class_balance_only": split_score,
        "random_state": RANDOM_STATE,
        "train_sample_count_all_reviewed": len(train_ids),
        "test_sample_count_all_reviewed": len(test_ids),
        "train_block_count": len(group_train),
        "test_block_count": len(group_test),
        "classes": EXPECTED_CLASSES,
        "features": FEATURES,
        "interpretation": "Single spatial-block holdout. No spatial block is shared by train and test. Metrics are more spatially cautious than random split but depend on block size and selected held-out areas.",
    }
    (OUTPUT_DIR / "spatial_validation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Ringkasan evaluasi spasial ===")
    print(metrics_df[["dataset_variant", "model", "n_train", "n_test", "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nBlok train/test: {len(group_train)} / {len(group_test)}; overlap={len(group_train & group_test)}")
    print(f"Output: {OUTPUT_DIR}")
    print("Catatan: ini satu spatial holdout; jangan menyamakan hasilnya langsung dengan random split tanpa menjelaskan perbedaan desain evaluasi.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
