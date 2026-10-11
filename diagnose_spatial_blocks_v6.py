from __future__ import annotations

"""Diagnose spatial-block class coverage for Sentinel-2 land-cover v6.

This diagnostic does not train models or select splits based on scores. It reports
whether classes are geographically concentrated and writes block/class tables
so that an appropriate spatial validation design can be chosen explicitly.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

PROJECT_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = PROJECT_DIR / "outputs" / "dataset_v6_review" / "dataset_6kelas_v6_kandidat_representatif_303.geojson"
BASE_DIR = PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6"
DATASETS = {
    "all_reviewed": BASE_DIR / "dataset_modeling_v6_all_reviewed.csv",
    "quality_screened": BASE_DIR / "dataset_modeling_v6_quality_screened.csv",
}
OUTPUT_DIR = PROJECT_DIR / "outputs" / "modeling_v6_spatial_diagnostic"
CLASSES = ["sawah", "bangunan", "mangrove", "lahan_hijau", "laut", "danau"]
LOCAL_CRS = "+proj=aeqd +lat_0=-7.5 +lon_0=112.5 +datum=WGS84 +units=m +no_defs"


def load_centroids(block_size_m: float) -> pd.DataFrame:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON tidak ditemukan: {GEOJSON_PATH}")
    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    transformer = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True)
    rows = []
    seen = set()
    for feature in fc.get("features", []):
        props = feature.get("properties") or {}
        sid = str(props.get("sample_id", "")).strip()
        geom_obj = feature.get("geometry")
        if not sid or geom_obj is None:
            continue
        if sid in seen:
            raise ValueError(f"sample_id duplikat pada GeoJSON: {sid}")
        seen.add(sid)
        geom = shape(geom_obj)
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Geometri kosong/tidak valid: {sid}")
        geom_local = shapely_transform(transformer.transform, geom)
        c_local = geom_local.centroid
        c_wgs = geom.centroid
        bx = math.floor(c_local.x / block_size_m)
        by = math.floor(c_local.y / block_size_m)
        rows.append({
            "sample_id": sid,
            "longitude_centroid": float(c_wgs.x),
            "latitude_centroid": float(c_wgs.y),
            "block_x": int(bx),
            "block_y": int(by),
            "spatial_block_id": f"B{bx}_{by}",
        })
    if not rows:
        raise ValueError("GeoJSON tidak berisi geometri sampel valid.")
    return pd.DataFrame(rows)


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    needed = {"sample_id", "class_name", "status"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang di {path.name}: {sorted(missing)}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df = df[df["status"].astype(str).str.upper().eq("OK")]
    df = df[df["class_name"].isin(CLASSES)]
    if df["sample_id"].duplicated().any():
        raise ValueError(f"sample_id duplikat pada {path.name}")
    return df.reset_index(drop=True)


def diagnose_size(block_size_km: float, centroids: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    out_dir = OUTPUT_DIR / f"block_{block_size_km:g}km"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Ukuran blok {block_size_km:g} km ===")

    merged_all = datasets["all_reviewed"].merge(centroids, on="sample_id", how="left", validate="one_to_one")
    if merged_all["spatial_block_id"].isna().any():
        missing = merged_all.loc[merged_all["spatial_block_id"].isna(), "sample_id"].tolist()
        raise ValueError(f"Sampel all_reviewed tidak cocok dengan GeoJSON, contoh: {missing[:10]}")

    # Pivot from all samples; quality-screened membership is marked separately.
    q_ids = set(datasets["quality_screened"]["sample_id"])
    merged_all["in_quality_screened"] = merged_all["sample_id"].isin(q_ids)
    class_block_all = (
        merged_all.groupby(["spatial_block_id", "class_name"]).size()
        .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    )
    class_block_all.to_csv(out_dir / "block_class_counts_all_reviewed.csv", encoding="utf-8-sig")

    block_rows = []
    all_class_counts = class_block_all.copy()
    qdf = merged_all[merged_all["in_quality_screened"]]
    class_block_q = (
        qdf.groupby(["spatial_block_id", "class_name"]).size()
        .unstack(fill_value=0).reindex(index=class_block_all.index, columns=CLASSES, fill_value=0)
    )
    class_block_q.to_csv(out_dir / "block_class_counts_quality_screened.csv", encoding="utf-8-sig")

    for block_id in class_block_all.index:
        r = {"spatial_block_id": block_id,
             "all_reviewed_count": int(all_class_counts.loc[block_id].sum()),
             "quality_screened_count": int(class_block_q.loc[block_id].sum())}
        for c in CLASSES:
            r[f"all_{c}"] = int(all_class_counts.loc[block_id, c])
            r[f"quality_{c}"] = int(class_block_q.loc[block_id, c])
        block_rows.append(r)
    pd.DataFrame(block_rows).sort_values("all_reviewed_count", ascending=False).to_csv(
        out_dir / "block_sample_counts.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for variant, vdf in datasets.items():
        merged = vdf.merge(centroids, on="sample_id", how="left", validate="one_to_one")
        if merged["spatial_block_id"].isna().any():
            missing = merged.loc[merged["spatial_block_id"].isna(), "sample_id"].tolist()
            raise ValueError(f"Sampel {variant} tidak cocok dengan GeoJSON, contoh: {missing[:10]}")
        grouped = merged.groupby(["spatial_block_id", "class_name"]).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=CLASSES, fill_value=0)
        n_blocks = int(merged["spatial_block_id"].nunique())
        print(f"Varian {variant}: {len(merged)} sampel, {n_blocks} blok")
        for c in CLASSES:
            counts = grouped[c]
            present = counts[counts > 0]
            n_samp = int((merged["class_name"] == c).sum())
            n_blocks_class = int(len(present))
            top = int(present.max()) if not present.empty else 0
            top_share = top / n_samp if n_samp else 0.0
            blocks_list = ";".join(f"{idx}:{int(val)}" for idx, val in present.sort_values(ascending=False).items())
            row = {
                "block_size_km": block_size_km,
                "dataset_variant": variant,
                "class_name": c,
                "sample_count": n_samp,
                "blocks_containing_class": n_blocks_class,
                "top_block_sample_count": top,
                "top_block_share_of_class": top_share,
                "necessary_condition_class_in_both_sides": n_blocks_class >= 2,
                "block_counts_desc": blocks_list,
            }
            summary_rows.append(row)
            print(f"  {c:13s}: n={n_samp:3d}, blok={n_blocks_class:2d}, "
                  f"maks/blok={top:2d}, proporsi di blok terbesar={top_share:.1%}, "
                  f"bisa ada di train+test secara perlu={n_blocks_class >= 2}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "class_spatial_concentration.csv", index=False, encoding="utf-8-sig")
    print(f"File diagnostik: {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Audit distribusi kelas di blok spasial; tidak melatih model.")
    p.add_argument("--block-sizes-km", type=float, nargs="+", default=[5, 10, 20],
                   help="Ukuran blok yang dibandingkan (default: 5 10 20)")
    args = p.parse_args()
    if any(x <= 0 for x in args.block_sizes_km):
        raise ValueError("Semua ukuran blok harus > 0")

    print("=== Diagnostik distribusi kelas pada blok spasial ===")
    print(f"GeoJSON: {GEOJSON_PATH}")
    datasets = {k: read_dataset(v) for k, v in DATASETS.items()}
    print("Jumlah sampel:", ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))
    for size_km in args.block_sizes_km:
        centroids = load_centroids(size_km * 1000.0)
        diagnose_size(size_km, centroids, datasets)
    print(f"\nSelesai. Ringkasan tersimpan di: {OUTPUT_DIR}")
    print("Catatan: kelas yang hanya muncul dalam satu blok tidak mungkin tampil di train dan test sekaligus tanpa membagi blok tersebut.")
    print("Ini audit distribusi, bukan pemilihan pembagian berdasarkan skor model.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
