from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as transform_geom

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
PAIR_CSV = PROJECT_DIR / "outputs" / "sentinel2_search" / "cakupan_produk_per_sampel.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_search"

# Equal-area projection for area-fraction calculations.
# Input GeoJSON / STAC footprints are expected in WGS84 longitude/latitude.
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def read_sample_geometries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON sampel tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError("Dataset sampel bukan FeatureCollection GeoJSON.")

    result = []
    for idx, feat in enumerate(data.get("features", []), start=1):
        props = feat.get("properties") or {}
        sid = props.get("sample_id")
        if not sid:
            sid = feat.get("id") or f"FEATURE_{idx:04d}"
        geom_json = feat.get("geometry")
        if not geom_json:
            result.append({
                "sample_id": str(sid),
                "class_id": props.get("class_id"),
                "class_name": props.get("class_name"),
                "source_file": props.get("source_file"),
                "geom_wgs84": None,
                "geom_area": None,
                "geometry_issue": "geometri kosong",
            })
            continue
        try:
            geom = shape(geom_json)
            issue = ""
            if geom.is_empty:
                issue = "geometri kosong"
                geom_area = None
            else:
                if not geom.is_valid:
                    issue = "geometri tidak valid; hitung area memakai buffer(0) hanya untuk audit"
                    geom = geom.buffer(0)
                projected = transform_geom(TRANSFORMER.transform, geom)
                geom_area = projected.area if not projected.is_empty else None
            result.append({
                "sample_id": str(sid),
                "class_id": props.get("class_id"),
                "class_name": props.get("class_name"),
                "source_file": props.get("source_file"),
                "geom_wgs84": geom,
                "geom_area": geom_area,
                "geometry_issue": issue,
            })
        except Exception as e:
            result.append({
                "sample_id": str(sid),
                "class_id": props.get("class_id"),
                "class_name": props.get("class_name"),
                "source_file": props.get("source_file"),
                "geom_wgs84": None,
                "geom_area": None,
                "geometry_issue": f"gagal membaca geometri: {e}",
            })
    return result


def get_stac_item(session: requests.Session, url: str) -> dict[str, Any]:
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=40)
            resp.raise_for_status()
            item = resp.json()
            if not isinstance(item, dict) or not item.get("geometry"):
                raise ValueError("Respons STAC tidak berisi geometri item.")
            return item
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"Gagal membaca item STAC {url}: {last_error}")


def safe_geometry(geom_json: dict[str, Any]):
    geom = shape(geom_json)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def main() -> int:
    if not PAIR_CSV.exists():
        raise FileNotFoundError(f"CSV hasil audit sebelumnya tidak ditemukan: {PAIR_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = read_sample_geometries(SAMPLE_PATH)
    pair_df = pd.read_csv(PAIR_CSV)
    required = {"item_id", "item_self_link", "acquisition_date", "mgrs_tile", "cloud_cover_pct"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang dari CSV pasangan: {sorted(missing)}")

    products_df = (
        pair_df.dropna(subset=["item_id", "item_self_link"])
        .drop_duplicates("item_id")[
            ["item_id", "item_self_link", "acquisition_date", "mgrs_tile", "cloud_cover_pct", "platform"]
        ]
        .reset_index(drop=True)
    )
    if products_df.empty:
        raise ValueError("Tidak ada item STAC dengan URL yang dapat dibaca.")

    print("=== Audit fraksi cakupan footprint produk terhadap poligon sampel ===")
    print(f"Folder proyek : {PROJECT_DIR}")
    print(f"Jumlah sampel : {len(samples)}")
    print(f"Jumlah produk unik dari audit sebelumnya: {len(products_df)}")
    print("Perhitungan luas menggunakan EPSG:6933 (equal-area).")
    print("Catatan: ini mengukur footprint geometris, bukan validitas piksel/awan.")

    session = requests.Session()
    session.headers.update({"User-Agent": "PSD-EastJava-Sentinel2-Coverage-Audit/1.0"})
    product_geoms: dict[str, Any] = {}
    fetch_errors = []

    for n, row in products_df.iterrows():
        item_id = str(row["item_id"])
        try:
            item = get_stac_item(session, str(row["item_self_link"]))
            product_geoms[item_id] = safe_geometry(item["geometry"])
            print(f"[{n + 1}/{len(products_df)}] OK {row['mgrs_tile']} {row['acquisition_date']}")
        except Exception as e:
            fetch_errors.append({"item_id": item_id, "url": row["item_self_link"], "error": str(e)})
            print(f"[{n + 1}/{len(products_df)}] ERROR {item_id}: {e}")

    details: list[dict[str, Any]] = []
    for item in samples:
        sample_geom = item["geom_wgs84"]
        sample_area = item["geom_area"]
        if sample_geom is None or not sample_area or sample_area <= 0:
            continue
        sample_projected = transform_geom(TRANSFORMER.transform, sample_geom)

        for _, prod in products_df.iterrows():
            item_id = str(prod["item_id"])
            if item_id not in product_geoms:
                continue
            product_geom = product_geoms[item_id]
            if not sample_geom.intersects(product_geom):
                continue

            inter = sample_geom.intersection(product_geom)
            if inter.is_empty:
                continue
            inter_projected = transform_geom(TRANSFORMER.transform, inter)
            inter_area = max(0.0, float(inter_projected.area))
            fraction = min(1.0, inter_area / sample_area) if sample_area > 0 else 0.0
            if fraction <= 0:
                continue

            details.append({
                "sample_id": item["sample_id"],
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "source_file": item["source_file"],
                "item_id": item_id,
                "acquisition_date": prod["acquisition_date"],
                "mgrs_tile": prod["mgrs_tile"],
                "cloud_cover_pct": prod["cloud_cover_pct"],
                "intersection_area_m2_equal_area": round(inter_area, 3),
                "sample_area_m2_equal_area": round(float(sample_area), 3),
                "sample_area_covered_pct": round(fraction * 100, 5),
                "coverage_level": (
                    "FULL_OR_NEAR_FULL_>=99.5%"
                    if fraction >= 0.995 else "PARTIAL_>0_TO_<99.5%"
                ),
            })

    detail_df = pd.DataFrame(details)
    detail_path = OUT_DIR / "audit_footprint_coverage_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    sample_rows = []
    if detail_df.empty:
        max_by_sample = {}
    else:
        max_by_sample = (
            detail_df.sort_values(
                ["sample_id", "sample_area_covered_pct", "cloud_cover_pct"],
                ascending=[True, False, True],
            )
            .drop_duplicates("sample_id")
            .set_index("sample_id")
            .to_dict("index")
        )

    for s in samples:
        best = max_by_sample.get(s["sample_id"])
        max_pct = float(best["sample_area_covered_pct"]) if best else 0.0
        sample_rows.append({
            "sample_id": s["sample_id"],
            "class_id": s["class_id"],
            "class_name": s["class_name"],
            "source_file": s["source_file"],
            "max_footprint_coverage_pct": round(max_pct, 5),
            "best_item_id_by_coverage": best["item_id"] if best else "",
            "best_date": best["acquisition_date"] if best else "",
            "best_tile": best["mgrs_tile"] if best else "",
            "best_scene_cloud_cover_pct": best["cloud_cover_pct"] if best else None,
            "status_by_footprint": (
                "FULL_OR_NEAR_FULL_>=99.5%"
                if max_pct >= 99.5 else
                "PARTIAL_ONLY"
                if max_pct > 0 else
                "NO_FOOTPRINT_INTERSECTION"
            ),
        })

    sample_df = pd.DataFrame(sample_rows)
    sample_path = OUT_DIR / "audit_footprint_coverage_per_sampel.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")

    # Class summary: count of samples with at least one product meeting each coverage threshold.
    if "class_id" in sample_df.columns:
        class_summary = (
            sample_df.groupby(["class_id", "class_name"], dropna=False)
            .agg(
                jumlah_sampel=("sample_id", "nunique"),
                cakupan_penuh_min_satu_produk=("status_by_footprint", lambda x: int((x == "FULL_OR_NEAR_FULL_>=99.5%").sum())),
                hanya_cakupan_parsial=("status_by_footprint", lambda x: int((x == "PARTIAL_ONLY").sum())),
                tanpa_irisan=("status_by_footprint", lambda x: int((x == "NO_FOOTPRINT_INTERSECTION").sum())),
                min_cakupan_persen=("max_footprint_coverage_pct", "min"),
                median_cakupan_persen=("max_footprint_coverage_pct", "median"),
            )
            .reset_index()
        )
    else:
        class_summary = pd.DataFrame()
    class_path = OUT_DIR / "audit_footprint_coverage_per_kelas.csv"
    class_summary.to_csv(class_path, index=False, encoding="utf-8-sig")

    errors_path = OUT_DIR / "audit_footprint_item_fetch_errors.csv"
    pd.DataFrame(fetch_errors, columns=["item_id", "url", "error"]).to_csv(errors_path, index=False, encoding="utf-8-sig")

    print("\n=== Ringkasan ===")
    print(f"Produk dengan footprint berhasil dibaca: {len(product_geoms)}/{len(products_df)}")
    print(f"Pasangan sample-produk dengan fraksi irisan > 0: {len(detail_df)}")
    if not sample_df.empty:
        print("Sampel dengan minimal satu produk >=99.5% cakupan:",
              int((sample_df.status_by_footprint == "FULL_OR_NEAR_FULL_>=99.5%").sum()), "/", len(sample_df))
        print("Sampel yang hanya punya irisan parsial:",
              int((sample_df.status_by_footprint == "PARTIAL_ONLY").sum()))
        print("Sampel tanpa irisan footprint:", int((sample_df.status_by_footprint == "NO_FOOTPRINT_INTERSECTION").sum()))
        print("\nRingkasan per kelas:")
        print(class_summary.to_string(index=False))
    print("\nFile keluaran:")
    print(f"  {detail_path}")
    print(f"  {sample_path}")
    print(f"  {class_path}")
    if fetch_errors:
        print(f"  {errors_path} (periksa kegagalan metadata item)")
    print("\nPENTING: cakupan footprint >=99.5% tetap tidak membuktikan piksel sampel bebas awan atau punya data valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
