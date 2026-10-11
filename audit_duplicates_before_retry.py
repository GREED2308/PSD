from __future__ import annotations
import csv, json, sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict

try:
    import pandas as pd
    from shapely.geometry import shape
    from pyproj import Transformer
    from shapely.ops import transform as geom_transform
except ImportError as exc:
    print(f"Dependensi belum tersedia: {exc}")
    print("Jalankan: python -m pip install pandas shapely pyproj")
    raise SystemExit(1)

PROJECT_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
FEATURES_PATH = PROJECT_DIR / "outputs" / "sentinel2_features_csv" / "fitur_sentinel2a_l2a_6kelas_retried.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_features_csv"
EXACT_OUT = OUT_DIR / "audit_duplikasi_geometri_persis.csv"
NEAR_OUT = OUT_DIR / "audit_overlap_hampir_duplikat_bangunan.csv"

if not GEOJSON_PATH.exists():
    raise FileNotFoundError(f"GeoJSON tidak ditemukan: {GEOJSON_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"CSV fitur retry tidak ditemukan: {FEATURES_PATH}")

with GEOJSON_PATH.open("r", encoding="utf-8") as f:
    gj = json.load(f)
if gj.get("type") != "FeatureCollection":
    raise ValueError("Input bukan GeoJSON FeatureCollection")

records = []
for feat in gj.get("features", []):
    props = feat.get("properties") or {}
    geom_data = feat.get("geometry")
    if not geom_data:
        continue
    geom = shape(geom_data)
    if geom.is_empty:
        continue
    records.append({
        "sample_id": str(props.get("sample_id", "")),
        "class_id": str(props.get("class_id", "")),
        "class_name": str(props.get("class_name", "")),
        "label_status_geojson": str(props.get("label_status", "")),
        "usable_for_training_geojson": props.get("usable_for_training_now", props.get("usable_for_training", "")),
        "source_file": str(props.get("source_file", "")),
        "geom": geom,
    })

features_df = pd.read_csv(FEATURES_PATH, dtype={"sample_id": str})
if "sample_id" not in features_df.columns:
    raise ValueError("CSV fitur tidak memiliki kolom sample_id")
status_cols = [c for c in ["sample_id", "status", "clear_pixel_count", "acquisition_date_requested", "retry_used_alternative_date"] if c in features_df.columns]
status_map = features_df[status_cols].drop_duplicates("sample_id").set_index("sample_id").to_dict(orient="index")

# Group exact topological matches using normalized WKB when available; fall back to pairwise equals.
groups = []
remaining = set(range(len(records)))
for i in range(len(records)):
    if i not in remaining:
        continue
    group = [i]
    remaining.remove(i)
    for j in list(remaining):
        if records[i]["geom"].equals(records[j]["geom"]):
            group.append(j)
            remaining.remove(j)
    if len(group) > 1:
        groups.append(group)

exact_rows = []
for group in groups:
    for i, j in combinations(group, 2):
        a, b = records[i], records[j]
        sa = status_map.get(a["sample_id"], {})
        sb = status_map.get(b["sample_id"], {})
        same_class = a["class_id"] == b["class_id"]
        exact_rows.append({
            "sample_id_a": a["sample_id"], "class_a": a["class_name"],
            "sample_id_b": b["sample_id"], "class_b": b["class_name"],
            "same_class": same_class,
            "status_a": sa.get("status", ""), "clear_pixels_a": sa.get("clear_pixel_count", ""),
            "date_a": sa.get("acquisition_date_requested", ""),
            "status_b": sb.get("status", ""), "clear_pixels_b": sb.get("clear_pixel_count", ""),
            "date_b": sb.get("acquisition_date_requested", ""),
            "label_status_a": a["label_status_geojson"], "label_status_b": b["label_status_geojson"],
            "source_file_a": a["source_file"], "source_file_b": b["source_file"],
            "recommended_action": "KEEP_ONE_REPRESENTATIVE_AND_EXCLUDE_DUPLICATE_FROM_SPLIT" if same_class else "LABEL_CONFLICT_MANUAL_REVIEW_REQUIRED",
        })

OUT_DIR.mkdir(parents=True, exist_ok=True)
exact_fields = list(exact_rows[0].keys()) if exact_rows else ["sample_id_a", "class_a", "sample_id_b", "class_b", "recommended_action"]
with EXACT_OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=exact_fields)
    writer.writeheader()
    writer.writerows(exact_rows)

# Find near-duplicate building polygons; calculate area overlap in equal-area CRS (EPSG:6933).
transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
def projected(g):
    return geom_transform(transformer.transform, g)

buildings = [r for r in records if r["class_id"] == "2" and r["geom"].is_valid]
proj_geoms = {r["sample_id"]: projected(r["geom"]) for r in buildings}
near_rows = []
for idx, (a, b) in enumerate(combinations(buildings, 2), 1):
    ga, gb = a["geom"], b["geom"]
    if ga.equals(gb):
        continue
    pa, pb = proj_geoms[a["sample_id"]], proj_geoms[b["sample_id"]]
    if not pa.intersects(pb):
        continue
    min_area = min(pa.area, pb.area)
    union_area = pa.union(pb).area
    if min_area <= 0 or union_area <= 0:
        continue
    overlap_area = pa.intersection(pb).area
    frac_smaller = overlap_area / min_area
    iou = overlap_area / union_area
    if frac_smaller >= 0.95:
        sa = status_map.get(a["sample_id"], {})
        sb = status_map.get(b["sample_id"], {})
        near_rows.append({
            "sample_id_a": a["sample_id"], "sample_id_b": b["sample_id"],
            "class_name": "bangunan",
            "overlap_pct_of_smaller_polygon": round(frac_smaller * 100, 4),
            "intersection_over_union_pct": round(iou * 100, 4),
            "status_a": sa.get("status", ""), "clear_pixels_a": sa.get("clear_pixel_count", ""),
            "status_b": sb.get("status", ""), "clear_pixels_b": sb.get("clear_pixel_count", ""),
            "recommended_action": "VISUAL_REVIEW_POSSIBLE_NEAR_DUPLICATE; DO_NOT_SPLIT_ACROSS_TRAIN_TEST",
        })
near_fields = list(near_rows[0].keys()) if near_rows else ["sample_id_a", "sample_id_b", "class_name", "overlap_pct_of_smaller_polygon", "intersection_over_union_pct", "recommended_action"]
with NEAR_OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=near_fields)
    writer.writeheader()
    writer.writerows(near_rows)

print("=== Audit duplikasi geometri sebelum retry lanjutan ===")
print(f"Dataset: {GEOJSON_PATH}")
print(f"Fitur geometri terbaca: {len(records)}")
print(f"Grup exact duplicate: {len(groups)}")
print(f"Pasangan exact duplicate: {len(exact_rows)}")
print(f"Pasangan bangunan overlap >=95% area poligon lebih kecil (tidak identik): {len(near_rows)}")
print(f"CSV exact duplicate: {EXACT_OUT}")
print(f"CSV near duplicate: {NEAR_OUT}")
print("Tidak ada geometri/label yang otomatis dihapus atau diubah.")
