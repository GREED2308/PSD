from __future__ import annotations

"""Prepare verified v6 Sentinel-2 feature tables for model experiments.

This script does not modify the GeoJSON master or the original feature CSV.
The --labels-reviewed flag records the user's confirmation that labels and
polygon boundaries were checked visually. That flag is an audit declaration,
not independent proof that a label is correct.

Outputs:
- dataset_modeling_v6_all_reviewed.csv: all rows with OK extraction and complete features
- dataset_modeling_v6_quality_screened.csv: conservative quality-screened subset
- dataset_modeling_v6_quality_review_excluded.csv: rows excluded from the screened subset
- dataset_modeling_v6_summary_by_class.csv: counts by class for both variants
- dataset_modeling_v6_summary.txt: concise report
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_DIR
    / "outputs"
    / "sentinel2_temporal_composite_v6"
    / "fitur_sentinel2a_l2a_v6_temporal_2026-09-01_2026-10-10.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6"

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI"]
FEATURE_COLUMNS = [f"{name}_mean" for name in BANDS + INDICES]
EXPECTED_CLASSES = {
    "sawah": 1,
    "bangunan": 2,
    "mangrove": 3,
    "lahan_hijau": 4,
    "laut": 5,
    "danau": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Siapkan dataset model v6 tanpa mengubah file master."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path CSV fitur komposit temporal.")
    parser.add_argument(
        "--labels-reviewed",
        action="store_true",
        help="Tegaskan bahwa label dan batas poligon telah diperiksa visual oleh pengguna.",
    )
    parser.add_argument("--min-clear-pixels", type=int, default=10, help="Ambang minimum piksel spasial valid untuk subset tersaring (default 10).")
    parser.add_argument("--min-valid-fraction", type=float, default=0.50, help="Ambang minimum fraksi piksel valid (default 0.50).")
    parser.add_argument("--min-mean-observations", type=float, default=2.0, help="Ambang minimum rata-rata observasi bersih per piksel (default 2).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.labels_reviewed:
        print("Berhenti: sebelum menandai label sebagai telah diperiksa, jalankan ulang dengan --labels-reviewed.")
        print("Gunakan flag itu hanya jika label DAN batas poligon sudah diperiksa secara visual.")
        return 2
    if not args.input.exists():
        print(f"ERROR: CSV input tidak ditemukan: {args.input}")
        return 1
    if args.min_clear_pixels < 1:
        print("ERROR: --min-clear-pixels harus minimal 1.")
        return 1
    if not 0 <= args.min_valid_fraction <= 1:
        print("ERROR: --min-valid-fraction harus berada di antara 0 dan 1.")
        return 1
    if args.min_mean_observations < 0:
        print("ERROR: --min-mean-observations tidak boleh negatif.")
        return 1

    try:
        df = pd.read_csv(args.input, dtype={"sample_id": str}, low_memory=False)
    except Exception as exc:
        print(f"ERROR membaca CSV: {type(exc).__name__}: {exc}")
        return 1

    required = ["sample_id", "class_name", "status", "clear_pixel_count", "valid_pixel_fraction_of_polygon", "mean_clear_observations_per_valid_pixel", *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: kolom wajib tidak ditemukan:")
        for col in missing:
            print(f" - {col}")
        print("Periksa apakah CSV berasal dari extract_features_temporal_composite_v6.py.")
        return 1
    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(keep=False), "sample_id"].astype(str).tolist()[:10]
        print(f"ERROR: sample_id duplikat terdeteksi, contoh: {dupes}")
        return 1

    numeric_cols = ["clear_pixel_count", "valid_pixel_fraction_of_polygon", "mean_clear_observations_per_valid_pixel", *FEATURE_COLUMNS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # This is a derivative audit annotation; source label_status and flags remain untouched.
    df["manual_visual_review_declaration"] = "USER_CONFIRMED_LABEL_AND_GEOMETRY_VISUALLY_CHECKED"
    df["feature_table_source"] = str(args.input)
    df["quality_screen_min_clear_pixels"] = args.min_clear_pixels
    df["quality_screen_min_valid_fraction"] = args.min_valid_fraction
    df["quality_screen_min_mean_observations"] = args.min_mean_observations

    # Validate class names / IDs when IDs are present.
    if "class_id" in df.columns:
        df["class_id_numeric"] = pd.to_numeric(df["class_id"], errors="coerce")
        mismatch = []
        for _, row in df.iterrows():
            expected = EXPECTED_CLASSES.get(str(row["class_name"]))
            actual = row["class_id_numeric"]
            if expected is not None and pd.notna(actual) and int(actual) != expected:
                mismatch.append((row["sample_id"], row["class_name"], int(actual), expected))
        if mismatch:
            print("ERROR: class_id dan class_name tidak konsisten; contoh:")
            for item in mismatch[:10]:
                print(f" - sample_id={item[0]} class_name={item[1]} class_id={item[2]} expected={item[3]}")
            return 1
        df.drop(columns=["class_id_numeric"], inplace=True)

    status_ok = df["status"].astype(str).eq("OK")
    features_complete = np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    clear_ok = df["clear_pixel_count"].ge(args.min_clear_pixels)
    fraction_ok = df["valid_pixel_fraction_of_polygon"].ge(args.min_valid_fraction)
    temporal_ok = df["mean_clear_observations_per_valid_pixel"].ge(args.min_mean_observations)

    reasons = []
    for i in range(len(df)):
        row_reasons = []
        if not status_ok.iloc[i]:
            row_reasons.append("EXTRACTION_STATUS_NOT_OK")
        if not features_complete[i]:
            row_reasons.append("MISSING_OR_NONFINITE_FEATURES")
        if not clear_ok.iloc[i]:
            row_reasons.append(f"CLEAR_PIXELS_LT{args.min_clear_pixels}")
        if not fraction_ok.iloc[i]:
            row_reasons.append(f"VALID_FRACTION_LT{args.min_valid_fraction:g}")
        if not temporal_ok.iloc[i]:
            row_reasons.append(f"TEMPORAL_SUPPORT_LT{args.min_mean_observations:g}")
        reasons.append(";".join(row_reasons))

    df["quality_screen_flags"] = reasons
    df["eligible_in_quality_screened_variant"] = [
        bool(status_ok.iloc[i] and features_complete[i] and clear_ok.iloc[i] and fraction_ok.iloc[i] and temporal_ok.iloc[i])
        for i in range(len(df))
    ]

    # Full reviewed baseline still requires successful extraction and complete features.
    all_mask = status_ok & features_complete
    all_reviewed = df.loc[all_mask].copy()
    screened = df.loc[df["eligible_in_quality_screened_variant"]].copy()
    excluded = df.loc[~df["eligible_in_quality_screened_variant"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / "dataset_modeling_v6_all_reviewed.csv"
    screened_path = OUTPUT_DIR / "dataset_modeling_v6_quality_screened.csv"
    excluded_path = OUTPUT_DIR / "dataset_modeling_v6_quality_review_excluded.csv"
    summary_class_path = OUTPUT_DIR / "dataset_modeling_v6_summary_by_class.csv"
    summary_path = OUTPUT_DIR / "dataset_modeling_v6_summary.txt"

    all_reviewed.to_csv(all_path, index=False, encoding="utf-8-sig")
    screened.to_csv(screened_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    classes = sorted(set(df["class_name"].dropna().astype(str)), key=lambda x: (EXPECTED_CLASSES.get(x, 999), x))
    summary_rows = []
    for cls in classes:
        full_count = int((all_reviewed["class_name"].astype(str) == cls).sum())
        filtered_count = int((screened["class_name"].astype(str) == cls).sum())
        summary_rows.append({
            "class_name": cls,
            "all_reviewed_complete_features": full_count,
            "quality_screened_count": filtered_count,
            "excluded_from_quality_screen": full_count - filtered_count,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_class_path, index=False, encoding="utf-8-sig")

    lines = [
        "=== Persiapan dataset modeling v6 ===",
        f"Input: {args.input}",
        f"Rows input: {len(df)}",
        f"Unique sample_id: {df['sample_id'].nunique()}",
        f"User declaration: labels and polygon boundaries visually checked",
        f"Successful extraction + complete features: {len(all_reviewed)}",
        f"Quality-screened subset: {len(screened)}",
        f"Flagged out of quality-screened subset: {len(excluded)}",
        f"Filter: status=OK, all features finite, clear_pixel_count >= {args.min_clear_pixels}, valid_pixel_fraction_of_polygon >= {args.min_valid_fraction}, mean_clear_observations_per_valid_pixel >= {args.min_mean_observations}",
        "Note: thresholds are operational review criteria, not universal scientific standards.",
        "Note: the all-reviewed variant retains low-support samples to enable a baseline/sensitivity comparison.",
        "",
        "Counts by class:",
        summary_df.to_string(index=False),
        "",
        "Files:",
        str(all_path),
        str(screened_path),
        str(excluded_path),
        str(summary_class_path),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Persiapan dataset modeling v6 selesai ===")
    print(f"Input: {args.input}")
    print(f"Baris input: {len(df)} | sample_id unik: {df['sample_id'].nunique()}")
    print(f"Berhasil diekstrak + fitur lengkap: {len(all_reviewed)}")
    print(f"Varian semua sampel terverifikasi: {len(all_reviewed)}")
    print(f"Varian tersaring kualitas: {len(screened)}")
    print(f"Ditahan dari varian tersaring: {len(excluded)}")
    print("\nRingkasan per kelas:")
    print(summary_df.to_string(index=False))
    print("\nFile dibuat:")
    for path in [all_path, screened_path, excluded_path, summary_class_path, summary_path]:
        print(f" - {path}")
    print("\nCatatan: file sumber dan GeoJSON master tidak diubah.")
    print("Catatan: label diverifikasi berdasarkan pernyataan pengguna, bukan verifikasi otomatis oleh skrip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
