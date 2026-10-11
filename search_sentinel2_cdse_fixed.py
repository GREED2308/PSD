"""Search Copernicus Data Space STAC for Sentinel-2A L2A scenes intersecting East Java and samples.

Run from the project folder with:
    python scripts/search_sentinel2_cdse.py

This searches product metadata only. It does NOT download large satellite products.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union
from pystac_client import Client

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
DATE_RANGE = "2026-09-01T00:00:00Z/2026-10-10T23:59:59Z"
MAX_CLOUD_COVER = 30
MAX_ITEMS = 5000
PAGE_LIMIT = 100

PROJECT_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH_CANDIDATES = [
    PROJECT_DIR / "data" / "geojson" / "jawa_timur_2D.geojson",
    PROJECT_DIR / "data" / "geojson" / "jawa_timur.geojson",
]
SAMPLES_NAME = "dataset_6kelas.geojson"
SAMPLE_PATH_CANDIDATES = [
    PROJECT_DIR / "data" / "geojson" / SAMPLES_NAME,
    PROJECT_DIR / SAMPLES_NAME,
    PROJECT_DIR / "data" / SAMPLES_NAME,
    PROJECT_DIR / "data" / "geojson" / "dataset_6kelas_gabungan_v5_353_kandidat.geojson",
]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_search"


def _drop_extra_coordinate_dimensions(value: Any) -> Any:
    """Convert coordinate positions with Z/M dimensions to X/Y, ignoring nonnumeric extras."""
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return value[:2]
        return [_drop_extra_coordinate_dimensions(part) for part in value]
    return value


def read_geojson(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GeoJSON tidak valid: {path}\nDetail: {exc}") from exc
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": data}]
    usable = [f for f in features if f.get("geometry")]
    # Work in memory only: support raw BIG GeoJSON with extra Z/M coordinates (e.g. [x, y, z, null]).
    for feature in usable:
        geom = feature.get("geometry") or {}
        if "coordinates" in geom:
            geom["coordinates"] = _drop_extra_coordinate_dimensions(geom["coordinates"])
    if not usable:
        raise ValueError(f"Tidak ada geometri pada file: {path}")
    return {"raw": data, "features": usable}


def load_shapes(path: Path):
    geo = read_geojson(path)
    geoms = []
    for feature in geo["features"]:
        geom = shape(feature["geometry"])
        if not geom.is_empty:
            geoms.append(geom)
    if not geoms:
        raise ValueError(f"Tidak ada geometri yang dapat dibaca: {path}")
    return geo, geoms


def find_boundary_file() -> Path:
    for candidate in BOUNDARY_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(p) for p in BOUNDARY_PATH_CANDIDATES)
    raise FileNotFoundError(
        "File batas Jawa Timur tidak ditemukan. Letakkan jawa_timur_2D.geojson "
        f"atau jawa_timur.geojson pada salah satu lokasi berikut:\n{checked}"
    )


def find_sample_file() -> Path:
    for candidate in SAMPLE_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(p) for p in SAMPLE_PATH_CANDIDATES)
    raise FileNotFoundError(
        "GeoJSON dataset enam kelas tidak ditemukan. Letakkan file "
        f"{SAMPLES_NAME} di salah satu lokasi berikut, atau ubah SAMPLE_PATH_CANDIDATES:\n{checked}"
    )


def is_sentinel_2a(item_id: str, properties: dict[str, Any]) -> bool | None:
    platform = str(properties.get("platform", "")).strip().lower()
    if platform:
        return platform in {"sentinel-2a", "s2a"}
    # Fallback for catalogue items whose identifier carries the platform name.
    text = (item_id + " " + str(properties.get("title", ""))).upper()
    if "S2A_" in text or text.startswith("S2A"):
        return True
    if "S2B_" in text or text.startswith("S2B"):
        return False
    return None


def main() -> int:
    print("=== Pencarian katalog Copernicus Sentinel-2A L2A ===")
    print(f"Folder proyek : {PROJECT_DIR}")
    boundary_path = find_boundary_file()
    print(f"Batas provinsi: {boundary_path}")
    print(f"Periode       : {DATE_RANGE}")
    print(f"Tutupan awan  : <= {MAX_CLOUD_COVER}%")

    boundary_geo, boundary_shapes = load_shapes(boundary_path)
    province_geom = unary_union(boundary_shapes)

    sample_path = find_sample_file()
    sample_geo, sample_shapes = load_shapes(sample_path)
    sample_union = unary_union(sample_shapes)

    all_bounds = [province_geom.bounds, sample_union.bounds]
    bbox = [
        min(bounds[0] for bounds in all_bounds),
        min(bounds[1] for bounds in all_bounds),
        max(bounds[2] for bounds in all_bounds),
        max(bounds[3] for bounds in all_bounds),
    ]
    print(f"Dataset sampel: {sample_path}")
    print(f"Jumlah fitur sampel terbaca: {len(sample_shapes)}")
    print("BBox pencarian [min_lon, min_lat, max_lon, max_lat]:")
    print("  " + ", ".join(f"{v:.6f}" for v in bbox))
    print("Menghubungi katalog STAC CDSE... (pencarian metadata, tidak mengunduh raster)")

    try:
        catalog = Client.open(STAC_URL)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lte": MAX_CLOUD_COVER}},
            limit=PAGE_LIMIT,
            max_items=MAX_ITEMS,
        )
        items = list(search.items())
    except Exception as exc:
        print("GAGAL mengakses STAC API.")
        print(f"Detail: {exc}")
        print("Periksa koneksi internet atau coba ulang beberapa saat lagi.")
        return 1

    print(f"Kandidat dari katalog sebelum filter platform/geometri: {len(items)}")
    rows = []
    skipped_platform = 0
    unknown_platform = 0
    skipped_geometry = 0
    for item in items:
        props = item.properties or {}
        platform_match = is_sentinel_2a(item.id, props)
        # Preserve only known S2A scenes. Unknown platform values are reported separately.
        if platform_match is not True:
            if platform_match is False:
                skipped_platform += 1
            else:
                unknown_platform += 1
            continue

        try:
            scene_geom = shape(item.geometry) if item.geometry else None
        except Exception:
            scene_geom = None
        if scene_geom is None or scene_geom.is_empty:
            intersects_province = None
            intersects_samples = None
        else:
            intersects_province = bool(scene_geom.intersects(province_geom))
            intersects_samples = bool(scene_geom.intersects(sample_union))
            if not (intersects_province or intersects_samples):
                skipped_geometry += 1
                continue

        cloud = props.get("eo:cloud_cover")
        dt = props.get("datetime") or props.get("start_datetime") or ""
        tile = props.get("s2:mgrs_tile") or props.get("mgrs:tile") or ""
        rows.append({
            "item_id": item.id,
            "datetime_utc": dt,
            "platform": props.get("platform", "unknown"),
            "platform_match_s2a": platform_match,
            "cloud_cover_pct": cloud,
            "mgrs_tile": tile,
            "intersects_east_java_boundary": intersects_province,
            "intersects_any_training_candidate": intersects_samples,
            "bbox_min_lon": item.bbox[0] if item.bbox and len(item.bbox) >= 4 else None,
            "bbox_min_lat": item.bbox[1] if item.bbox and len(item.bbox) >= 4 else None,
            "bbox_max_lon": item.bbox[2] if item.bbox and len(item.bbox) >= 4 else None,
            "bbox_max_lat": item.bbox[3] if item.bbox and len(item.bbox) >= 4 else None,
            "item_self_link": next((lnk.href for lnk in item.links if lnk.rel == "self"), ""),
        })

    if not rows:
        print("Tidak ada item yang lolos filter lokasi/platform. Coba periode lebih lebar atau awan <= 50%.")
        return 0

    df = pd.DataFrame(rows)
    # Sort best scenes first: known S2A, less cloud, then earlier/consistent date.
    df["_platform_sort"] = df["platform_match_s2a"].map({True: 0, None: 1, False: 2}).fillna(1)
    df["_cloud_sort"] = pd.to_numeric(df["cloud_cover_pct"], errors="coerce").fillna(101)
    df = df.sort_values(["_platform_sort", "_cloud_sort", "datetime_utc"], ascending=[True, True, True])
    df = df.drop(columns=["_platform_sort", "_cloud_sort"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "kandidat_sentinel2a_l2a.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n=== SELESAI ===")
    print(f"Hasil relevan disimpan: {csv_path}")
    print(f"Jumlah hasil tersimpan: {len(df)}")
    print(f"Item non-S2A yang dilewati: {skipped_platform}")
    print(f"Item dengan platform tidak teridentifikasi yang dilewati: {unknown_platform}")
    print(f"Item yang tidak beririsan dengan batas/sampel: {skipped_geometry}")
    print("\n10 kandidat teratas:")
    print(df[["datetime_utc", "platform", "cloud_cover_pct", "mgrs_tile", "intersects_east_java_boundary", "intersects_any_training_candidate", "item_id"]].head(10).to_string(index=False))
    print("\nCatatan: hasil ini hanya metadata kandidat; belum ada produk/raster yang diunduh.")
    print("Periksa CSV sebelum memilih produk, karena cloud cover scene tidak menjamin bebas awan pada setiap sampel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
