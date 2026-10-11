from __future__ import annotations

"""Audit luas geometri sampel dan kaitkan dengan kualitas komposit Sentinel-2.

Tidak mengubah atau menghapus dataset sumber. Ambang hanya penanda prioritas review.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

PROJECT = Path(__file__).resolve().parent
GEOJSON_PATH = PROJECT / "outputs" / "dataset_v6_review" / "dataset_6kelas_v6_kandidat_representatif_303.geojson"
FEATURES_PATH = PROJECT / "outputs" / "sentinel2_temporal_composite_v6" / "fitur_sentinel2a_l2a_v6_temporal_2026-09-01_2026-10-10.csv"
OUTPUT_DIR = PROJECT / "outputs" / "sentinel2_temporal_composite_v6"


def utm_epsg(geom) -> int:
    lon, lat = float(geom.centroid.x), float(geom.centroid.y)
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32700 + zone) if lat < 0 else (32600 + zone)


def main() -> int:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON tidak ditemukan: {GEOJSON_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"CSV fitur tidak ditemukan: {FEATURES_PATH}")

    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    geom_rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        sid = str(props.get("sample_id", "")).strip()
        geom_obj = feat.get("geometry")
        if not sid or not geom_obj:
            continue
        geom = shape(geom_obj)
        if geom.is_empty or not geom.is_valid:
            geom_rows.append({
                "sample_id": sid,
                "geometry_valid": bool(geom.is_valid and not geom.is_empty),
                "geometry_area_m2": np.nan,
                "geometry_equivalent_10m_pixels": np.nan,
                "geometry_review_flag": "INVALID_OR_EMPTY_GEOMETRY",
            })
            continue
        epsg = utm_epsg(geom)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        projected = shapely_transform(transformer.transform, geom)
        area = float(projected.area)
        eq_pixels = area / 100.0
        flags = []
        if eq_pixels < 10:
            flags.append("GEOMETRY_LT10_EQUIV_10M_PIXELS")
        elif eq_pixels < 25:
            flags.append("GEOMETRY_LT25_EQUIV_10M_PIXELS")
        if area <= 0:
            flags.append("ZERO_AREA")
        geom_rows.append({
            "sample_id": sid,
            "geometry_valid": True,
            "geometry_utm_epsg": epsg,
            "geometry_area_m2": area,
            "geometry_equivalent_10m_pixels": eq_pixels,
            "geometry_review_flag": ";".join(flags) if flags else "GEOMETRY_SIZE_NOT_FLAGGED",
        })

    geom_df = pd.DataFrame(geom_rows)
    features = pd.read_csv(FEATURES_PATH, dtype={"sample_id": str})
    if "sample_id" not in features.columns:
        raise ValueError("Kolom sample_id tidak ditemukan pada CSV fitur.")
    merged = features.merge(geom_df, on="sample_id", how="left", validate="one_to_one")
    merged["clear_pixels_per_1000m2"] = np.where(
        pd.to_numeric(merged.get("geometry_area_m2"), errors="coerce") > 0,
        pd.to_numeric(merged.get("clear_pixel_count"), errors="coerce") /
        pd.to_numeric(merged.get("geometry_area_m2"), errors="coerce") * 1000.0,
        np.nan,
    )

    def reasons(row) -> str:
        flags = []
        geom_flag = str(row.get("geometry_review_flag", ""))
        if geom_flag.startswith("INVALID") or "GEOMETRY_LT" in geom_flag or "ZERO_AREA" in geom_flag:
            flags.append(geom_flag)
        clear = pd.to_numeric(row.get("clear_pixel_count"), errors="coerce")
        frac = pd.to_numeric(row.get("valid_pixel_fraction_of_polygon"), errors="coerce")
        obs = pd.to_numeric(row.get("mean_clear_observations_per_valid_pixel"), errors="coerce")
        if pd.notna(clear) and clear < 10:
            flags.append("CLEAR_PIXELS_LT10")
        elif pd.notna(clear) and clear < 25:
            flags.append("CLEAR_PIXELS_LT25")
        if pd.notna(frac) and frac < 0.5:
            flags.append("VALID_PIXEL_FRACTION_LT50PCT")
        if pd.notna(obs) and obs < 2:
            flags.append("TEMPORAL_SUPPORT_MEAN_LT2")
        return ";".join(dict.fromkeys(flags)) if flags else "REVIEWABLE_NOT_AUTO_APPROVED"

    merged["geometry_and_raster_review_flags"] = merged.apply(reasons, axis=1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sample = OUTPUT_DIR / "audit_geometry_temporal_v6_per_sampel.csv"
    needs_review = OUTPUT_DIR / "audit_geometry_temporal_v6_perlu_review.csv"
    per_class = OUTPUT_DIR / "audit_geometry_temporal_v6_per_kelas.csv"

    merged.sort_values(["class_name", "geometry_area_m2"], na_position="first").to_csv(
        per_sample, index=False, encoding="utf-8-sig"
    )
    flagged = merged[merged["geometry_and_raster_review_flags"] != "REVIEWABLE_NOT_AUTO_APPROVED"].copy()
    flagged.sort_values(["class_name", "geometry_area_m2"], na_position="first").to_csv(
        needs_review, index=False, encoding="utf-8-sig"
    )

    merged["_clear"] = pd.to_numeric(merged.get("clear_pixel_count"), errors="coerce")
    merged["_area"] = pd.to_numeric(merged.get("geometry_area_m2"), errors="coerce")
    merged["_fraction"] = pd.to_numeric(merged.get("valid_pixel_fraction_of_polygon"), errors="coerce")
    merged["_obs"] = pd.to_numeric(merged.get("mean_clear_observations_per_valid_pixel"), errors="coerce")
    summary = merged.groupby("class_name", dropna=False).agg(
        sample_count=("sample_id", "count"),
        geometry_area_m2_min=("_area", "min"),
        geometry_area_m2_median=("_area", "median"),
        geometry_area_m2_max=("_area", "max"),
        equivalent_10m_pixels_median=("geometry_equivalent_10m_pixels", "median"),
        clear_pixel_count_median=("_clear", "median"),
        samples_geometry_lt10_equiv_pixels=("geometry_review_flag", lambda s: s.astype(str).str.contains("GEOMETRY_LT10_EQUIV").sum()),
        samples_clear_pixels_lt10=("_clear", lambda s: (s < 10).sum()),
        samples_valid_fraction_lt50pct=("_fraction", lambda s: (s < 0.5).sum()),
        samples_temporal_support_lt2=("_obs", lambda s: (s < 2).sum()),
        samples_flagged=("geometry_and_raster_review_flags", lambda s: (s != "REVIEWABLE_NOT_AUTO_APPROVED").sum()),
    ).reset_index()
    summary.to_csv(per_class, index=False, encoding="utf-8-sig")

    print("=== Audit geometri + kualitas komposit temporal v6 ===")
    print(f"GeoJSON: {GEOJSON_PATH}")
    print(f"CSV fitur: {FEATURES_PATH}")
    print(f"Sampel fitur: {len(features)} | geometri: {len(geom_df)} | ID unik fitur: {features.sample_id.nunique()}")
    print("Luas geometri dihitung dalam UTM lokal per sampel; ekuivalen piksel = luas/100 m² (perkiraan, bukan hitungan raster pasti).")
    print("Ringkasan per kelas:")
    print(summary.to_string(index=False))
    print("\nFile dibuat:")
    print(f" - {per_sample}")
    print(f" - {per_class}")
    print(f" - {needs_review}")
    print("Semua flag hanya prioritas review; tidak ada fitur/sampel yang dihapus otomatis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
