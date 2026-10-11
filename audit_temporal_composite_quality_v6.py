from __future__ import annotations

"""Audit quality of Sentinel-2 temporal-composite sample features.

Run from the project root, e.g.:
    python .\audit_temporal_composite_quality_v6.py

Thresholds are preliminary review flags only. No sample is automatically deleted.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6" /
    "fitur_sentinel2a_l2a_v6_temporal_2026-09-01_2026-10-10.csv"
)
OUTPUT_DIR = INPUT_PATH.parent

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI"]
FEATURE_COLS = [f"{b}_mean" for b in BANDS] + [f"{i}_mean" for i in INDICES]


def num(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input tidak ditemukan: {INPUT_PATH}")
        print("Pastikan skrip berada di root proyek PSD dan CSV komposit sudah ada.")
        return 1

    df = pd.read_csv(INPUT_PATH, dtype={"sample_id": str})
    if df.empty:
        print("CSV input kosong.")
        return 1

    for col in ["polygon_pixel_count", "valid_pixel_count", "clear_pixel_count",
                "valid_pixel_fraction_of_polygon", "clear_pixel_fraction_of_polygon",
                "mean_clear_observations_per_valid_pixel", "max_clear_observations_per_valid_pixel"]:
        df[col + "_num"] = num(df, col)

    # Missing band/index mean values are counted as a basic numeric integrity check.
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["missing_feature_mean_count"] = df[FEATURE_COLS].isna().sum(axis=1)

    flags: list[str] = []
    for _, r in df.iterrows():
        row_flags: list[str] = []
        if str(r.get("status", "")) != "OK":
            row_flags.append("STATUS_NOT_OK")
        clear_n = r.get("clear_pixel_count_num", np.nan)
        if pd.isna(clear_n) or clear_n < 1:
            row_flags.append("NO_VALID_SPATIAL_PIXELS")
        elif clear_n < 10:
            row_flags.append("VERY_FEW_SPATIAL_PIXELS_LT10")
        elif clear_n < 25:
            row_flags.append("FEW_SPATIAL_PIXELS_LT25")

        frac = r.get("clear_pixel_fraction_of_polygon_num", np.nan)
        if pd.isna(frac):
            row_flags.append("CLEAR_FRACTION_MISSING")
        elif frac < 0.50:
            row_flags.append("LOW_CLEAR_FRACTION_LT50PCT")

        obs = r.get("mean_clear_observations_per_valid_pixel_num", np.nan)
        if pd.isna(obs):
            row_flags.append("TEMPORAL_OBSERVATION_COUNT_MISSING")
        elif obs < 2:
            row_flags.append("LOW_TEMPORAL_SUPPORT_MEAN_LT2")

        if int(r.get("missing_feature_mean_count", 0)) > 0:
            row_flags.append("FEATURE_MEANS_MISSING")

        # Spectral indices should be inside [-1, 1] when finite. This only flags
        # gross numerical anomalies, not whether a value makes thematic sense.
        for idx in INDICES:
            v = r.get(f"{idx}_mean", np.nan)
            if pd.notna(v) and (v < -1.0001 or v > 1.0001):
                row_flags.append(f"{idx}_OUT_OF_RANGE")
        flags.append(";".join(dict.fromkeys(row_flags)) if row_flags else "REVIEWABLE_NOT_AUTO_APPROVED")

    df["quality_review_flags"] = flags
    df["needs_manual_review"] = df["quality_review_flags"] != "REVIEWABLE_NOT_AUTO_APPROVED"

    # Remove helper numeric columns for clean export but use them for summaries.
    per_sample_path = OUTPUT_DIR / "audit_temporal_v6_per_sampel.csv"
    per_review_path = OUTPUT_DIR / "audit_temporal_v6_perlu_review.csv"
    per_class_path = OUTPUT_DIR / "audit_temporal_v6_per_kelas.csv"

    df.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
    df[df["needs_manual_review"]].to_csv(per_review_path, index=False, encoding="utf-8-sig")

    summaries = []
    for class_name, g in df.groupby("class_name", dropna=False):
        clear = num(g, "clear_pixel_count")
        frac = num(g, "clear_pixel_fraction_of_polygon")
        obs = num(g, "mean_clear_observations_per_valid_pixel")
        status = g.get("status", pd.Series("", index=g.index)).astype(str)
        summaries.append({
            "class_name": class_name,
            "sample_count": int(len(g)),
            "status_OK_count": int((status == "OK").sum()),
            "status_non_OK_count": int((status != "OK").sum()),
            "clear_pixel_min": float(clear.min()) if clear.notna().any() else np.nan,
            "clear_pixel_median": float(clear.median()) if clear.notna().any() else np.nan,
            "clear_pixel_max": float(clear.max()) if clear.notna().any() else np.nan,
            "samples_clear_pixels_lt10": int((clear < 10).sum()),
            "samples_clear_pixels_lt25": int((clear < 25).sum()),
            "samples_clear_fraction_lt50pct": int((frac < 0.5).sum()),
            "median_mean_clear_observations_per_pixel": float(obs.median()) if obs.notna().any() else np.nan,
            "samples_mean_clear_observations_lt2": int((obs < 2).sum()),
            "samples_missing_feature_means": int((g[FEATURE_COLS].isna().any(axis=1)).sum()),
            "samples_flagged_for_review": int(g["needs_manual_review"].sum()),
        })
    summary_df = pd.DataFrame(summaries).sort_values("class_name")
    summary_df.to_csv(per_class_path, index=False, encoding="utf-8-sig")

    print("=== Audit kualitas komposit temporal v6 ===")
    print(f"Input: {INPUT_PATH}")
    print(f"Jumlah baris: {len(df)}")
    print(f"sample_id unik: {df['sample_id'].nunique() if 'sample_id' in df.columns else 'kolom sample_id tidak ada'}")
    print("\nStatus ekstraksi:")
    print(df["status"].value_counts(dropna=False).to_string() if "status" in df.columns else "Kolom status tidak ditemukan")
    print("\nFlag kualitas awal (hanya prioritas review):")
    print(df["quality_review_flags"].value_counts(dropna=False).to_string())
    print("\nRingkasan per kelas:")
    print(summary_df.to_string(index=False))
    print("\nFile dibuat:")
    print(f" - {per_sample_path}")
    print(f" - {per_class_path}")
    print(f" - {per_review_path}")
    print("Ambang awal: <10 piksel, <25 piksel, fraksi <0.50, dan rata-rata observasi <2.")
    print("Semua ambang hanya menandai untuk review; sampel tidak dihapus otomatis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
