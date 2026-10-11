from __future__ import annotations

"""Extract polygon-level Sentinel-2 L2A features directly to CSV.

- Reads credentials locally from .env.
- Uses full-day time windows (not a one-minute timestamp window).
- Requests small, sample-specific 10 m UTM subsets from Sentinel Hub Process API.
- Reads GeoTIFF responses in memory only; no TIFF files are written to disk.
- Computes zonal reflectance statistics after masking to the sample polygon,
  dataMask, and SCL classes 4/5/6.
- Checkpoints CSV after every sample and can resume successful rows.

Important:
- Features remain candidate labels; this script does not verify labels.
- All L2A bands available in Sentinel Hub except B10 are requested. B10 is not
  exposed as a surface-reflectance band in the Sentinel-2 L2A collection.
- Different native band resolutions (10/20/60 m) are sampled on a 10 m output
  grid using nearest-neighbour resampling. Record this in the methodology.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import rasterio
from dotenv import load_dotenv
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
SAMPLES_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
SEARCH_DIR = PROJECT_DIR / "outputs" / "sentinel2_search"
ASSIGNMENT_PATH = SEARCH_DIR / "penugasan_sampel_produk_sentinel2a.csv"
COVERAGE_DETAIL_PATH = SEARCH_DIR / "audit_footprint_coverage_detail.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_features_csv"
OUTPUT_PATH = OUTPUT_DIR / "fitur_sentinel2a_l2a_6kelas.csv"
FAILURES_PATH = OUTPUT_DIR / "fitur_sentinel2a_l2a_6kelas_gagal.csv"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

SPECTRAL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12",
]
OUTPUT_BANDS = SPECTRAL_BANDS + ["SCL", "dataMask"]
CLEAR_SCL_CLASSES = {4, 5, 6}  # vegetation, not-vegetated, water
CLOUD_SHADOW_SCL_CLASSES = {3, 8, 9, 10, 11}
TARGET_RESOLUTION_M = 10.0
BBOX_PADDING_M = 20.0

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B11", "B12", "SCL", "dataMask"],
    output: { bands: 14, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05,
          sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09,
          sample.B11, sample.B12, sample.SCL, sample.dataMask];
}
"""


def get_token(session: requests.Session) -> str:
    load_dotenv(ENV_PATH, override=False)
    client_id = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            f"Kredensial COPERNICUS_CLIENT_ID/SECRET tidak terbaca dari {ENV_PATH}."
        )
    resp = session.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=45,
    )
    if not resp.ok:
        try:
            body = resp.json()
            detail = body.get("error_description") or body.get("error") or str(body)
        except ValueError:
            detail = resp.text[:400]
        raise RuntimeError(f"OAuth gagal (HTTP {resp.status_code}): {detail}")
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth merespons tanpa access_token.")
    return token


def load_samples() -> list[dict[str, Any]]:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"GeoJSON sampel tidak ditemukan: {SAMPLES_PATH}")
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    features = fc.get("features", [])
    records = []
    seen = set()
    for feature in features:
        props = feature.get("properties") or {}
        sample_id = str(props.get("sample_id", "")).strip()
        geom_obj = feature.get("geometry")
        if not sample_id or not geom_obj:
            continue
        if sample_id in seen:
            raise ValueError(f"sample_id duplikat ditemukan: {sample_id}")
        seen.add(sample_id)
        geom = shape(geom_obj)
        if geom.is_empty:
            continue
        if not geom.is_valid:
            raise ValueError(
                f"Geometri {sample_id} tidak valid. Perbaiki pada tahap audit, "
                "jangan diperbaiki diam-diam saat ekstraksi."
            )
        records.append({"sample_id": sample_id, "properties": props, "geometry": geom})
    if not records:
        raise RuntimeError("Tidak ada fitur sampel valid yang bisa dibaca.")
    return records


def build_assignment_map() -> dict[str, dict[str, Any]]:
    """Prefer provided assignment; fall back to full-cover footprint audit."""
    if ASSIGNMENT_PATH.exists():
        df = pd.read_csv(ASSIGNMENT_PATH, dtype={"sample_id": str})
        date_col = "selected_acquisition_date"
        item_col = "selected_item_id"
        tile_col = "selected_mgrs_tile"
        cloud_col = "scene_cloud_cover_pct"
        coverage_col = "sample_footprint_coverage_pct"
        source_name = str(ASSIGNMENT_PATH)
    elif COVERAGE_DETAIL_PATH.exists():
        df = pd.read_csv(COVERAGE_DETAIL_PATH, dtype={"sample_id": str})
        df = df[pd.to_numeric(df["sample_area_covered_pct"], errors="coerce") >= 99.5].copy()
        # Preserve the earlier selection strategy: prefer 18 Sep; when no full-cover
        # product exists on that date, prefer 1 Oct; otherwise use lowest scene cloud.
        acq_dates = df["acquisition_date"].astype(str)
        df["_date_priority"] = acq_dates.map({"2026-09-18": 0, "2026-10-01": 1}).fillna(2)
        df = df.sort_values(
            ["sample_id", "_date_priority", "cloud_cover_pct"],
            ascending=[True, True, True],
        ).drop_duplicates("sample_id", keep="first")
        df["sample_footprint_coverage_pct"] = df["sample_area_covered_pct"]
        df["selected_item_id"] = df["item_id"]
        df["selected_acquisition_date"] = df["acquisition_date"]
        df["selected_mgrs_tile"] = df["mgrs_tile"]
        df["scene_cloud_cover_pct"] = df["cloud_cover_pct"]
        date_col = "selected_acquisition_date"
        item_col = "selected_item_id"
        tile_col = "selected_mgrs_tile"
        cloud_col = "scene_cloud_cover_pct"
        coverage_col = "sample_footprint_coverage_pct"
        source_name = str(COVERAGE_DETAIL_PATH)
    else:
        raise FileNotFoundError(
            "File penugasan produk tidak ditemukan. Unduh penugasan_sampel_produk_sentinel2a.csv "
            f"ke {SEARCH_DIR}, atau pastikan {COVERAGE_DETAIL_PATH} tersedia."
        )

    assignments: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row.get("sample_id", "")).strip()
        if not sid or sid in assignments:
            continue
        acq_date = str(row.get(date_col, ""))[:10]
        try:
            date.fromisoformat(acq_date)
        except ValueError:
            continue
        assignments[sid] = {
            "assigned_item_id": str(row.get(item_col, "")),
            "acquisition_date": acq_date,
            "mgrs_tile": str(row.get(tile_col, "")),
            "scene_cloud_cover_pct": as_float(row.get(cloud_col)),
            "sample_footprint_coverage_pct": as_float(row.get(coverage_col)),
            "assignment_source": source_name,
        }
    return assignments


def as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_utm_epsg(geom_wgs84) -> int:
    # Select a UTM zone based on geometry center; East Java spans zones 49S/50S.
    lon = float(geom_wgs84.centroid.x)
    lat = float(geom_wgs84.centroid.y)
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32700 + zone) if lat < 0 else (32600 + zone)


def project_geometry(geom_wgs84, epsg: int):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return shapely_transform(transformer.transform, geom_wgs84)


def make_request_bbox(geom_utm):
    minx, miny, maxx, maxy = geom_utm.bounds
    # Expand bbox around the polygon so edge pixels remain available for masking.
    minx -= BBOX_PADDING_M
    miny -= BBOX_PADDING_M
    maxx += BBOX_PADDING_M
    maxy += BBOX_PADDING_M
    return [minx, miny, maxx, maxy]


def process_api_request(
    session: requests.Session, token: str, bbox: list[float], epsg: int, acq_date: str
) -> tuple[requests.Response | None, str, str]:
    d = date.fromisoformat(acq_date)
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": crs_uri},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start.isoformat().replace("+00:00", "Z"),
                            "to": end.isoformat().replace("+00:00", "Z"),
                        },
                        "maxCloudCoverage": 100,
                        "mosaickingOrder": "leastCC",
                    },
                    "processing": {
                        "upsampling": "NEAREST",
                        "downsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "resx": TARGET_RESOLUTION_M,
            "resy": TARGET_RESOLUTION_M,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}}
            ],
        },
        "evalscript": EVALSCRIPT,
    }

    active_token = token
    for attempt in range(1, 4):
        try:
            response = session.post(
                PROCESS_URL,
                json=payload,
                headers={"Authorization": f"Bearer {active_token}", "Accept": "image/tiff"},
                timeout=180,
            )
        except requests.RequestException as exc:
            if attempt == 3:
                return None, f"KONEKSI: {type(exc).__name__}: {exc}", active_token
            time.sleep(2 * attempt)
            continue

        if response.status_code == 401 and attempt < 3:
            active_token = get_token(session)
            time.sleep(1)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 * attempt)
            continue
        if not response.ok:
            try:
                j = response.json()
                detail = j.get("error", {}).get("message") or j.get("message") or str(j)
            except ValueError:
                detail = response.text[:300]
            return response, f"HTTP {response.status_code}: {detail}", active_token
        return response, "OK", active_token
    return None, "GAGAL_SETELAH_RETRY", active_token


def base_record(sample: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    p = sample["properties"]
    out = {
        "sample_id": sample["sample_id"],
        "class_id": p.get("class_id"),
        "class_name": p.get("class_name"),
        "label_status": p.get("label_status", p.get("validation_status", "")),
        "usable_for_training_now": p.get("usable_for_training_now", p.get("usable_for_training", False)),
        "source_file": p.get("source_file", ""),
        "original_label": p.get("original_label", ""),
        "assigned_item_id": assignment.get("assigned_item_id", ""),
        "acquisition_date_requested": assignment.get("acquisition_date", ""),
        "assigned_mgrs_tile": assignment.get("mgrs_tile", ""),
        "scene_cloud_cover_metadata_pct": assignment.get("scene_cloud_cover_pct"),
        "sample_footprint_coverage_pct": assignment.get("sample_footprint_coverage_pct"),
        "target_resolution_m": TARGET_RESOLUTION_M,
        "resampling_method": "NEAREST; output grid 10 m; native bands 20/60 m resampled",
        "cloud_mask_scl_clear_classes": "4|5|6",
        "status": "PENDING",
        "error": "",
    }
    for band in SPECTRAL_BANDS:
        out[f"{band}_mean"] = np.nan
        out[f"{band}_median"] = np.nan
        out[f"{band}_std"] = np.nan
    for index in ["NDVI", "NDWI", "MNDWI", "NDBI"]:
        out[f"{index}_mean"] = np.nan
        out[f"{index}_median"] = np.nan
    out.update({
        "utm_epsg": None,
        "raster_width_px": None,
        "raster_height_px": None,
        "polygon_pixel_count": 0,
        "dataMask_valid_pixel_count": 0,
        "clear_pixel_count": 0,
        "valid_pixel_fraction_of_polygon": np.nan,
        "clear_pixel_fraction_of_polygon": np.nan,
        "scl_cloud_shadow_pixel_count": 0,
        "scl_class_2_dark_pixel_count": 0,
        "scl_class_4_vegetation_pixel_count": 0,
        "scl_class_5_bare_nonvegetated_pixel_count": 0,
        "scl_class_6_water_pixel_count": 0,
        "mask_mode": "pixel_center",
    })
    return out


def stats(values: np.ndarray, prefix: str, record: dict[str, Any]) -> None:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    record[f"{prefix}_mean"] = float(vals.mean())
    record[f"{prefix}_median"] = float(np.median(vals))
    if prefix in SPECTRAL_BANDS:
        record[f"{prefix}_std"] = float(vals.std(ddof=0))


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > 1e-8)
    out[good] = (a[good] - b[good]) / denom[good]
    return out


def extract_one(
    sample: dict[str, Any], assignment: dict[str, Any],
    session: requests.Session, token: str,
) -> tuple[dict[str, Any], str]:
    record = base_record(sample, assignment)
    sid = sample["sample_id"]
    acq_date = assignment.get("acquisition_date", "")
    if not acq_date:
        record["status"] = "NO_ASSIGNMENT"
        record["error"] = "Tanggal akuisisi tidak ditemukan pada file penugasan."
        return record, token

    epsg = get_utm_epsg(sample["geometry"])
    geom_utm = project_geometry(sample["geometry"], epsg)
    bbox = make_request_bbox(geom_utm)
    response, message, token = process_api_request(session, token, bbox, epsg, acq_date)
    if response is None or not response.ok:
        record["status"] = "API_ERROR"
        record["error"] = message
        return record, token

    try:
        with MemoryFile(response.content) as mem:
            with mem.open() as ds:
                if ds.count != len(OUTPUT_BANDS):
                    record["status"] = "UNEXPECTED_BANDS"
                    record["error"] = f"Diharapkan {len(OUTPUT_BANDS)} band, diterima {ds.count}."
                    return record, token
                arr = ds.read().astype(np.float32)
                record["utm_epsg"] = epsg
                record["raster_width_px"] = ds.width
                record["raster_height_px"] = ds.height

                geom_ds = project_geometry(sample["geometry"], ds.crs.to_epsg() or epsg)
                inside = geometry_mask(
                    [mapping(geom_ds)],
                    out_shape=(ds.height, ds.width),
                    transform=ds.transform,
                    invert=True,
                    all_touched=False,
                )
                mask_mode = "pixel_center"
                if not inside.any():
                    inside = geometry_mask(
                        [mapping(geom_ds)],
                        out_shape=(ds.height, ds.width),
                        transform=ds.transform,
                        invert=True,
                        all_touched=True,
                    )
                    mask_mode = "all_touched_fallback"
                record["mask_mode"] = mask_mode
                polygon_count = int(inside.sum())
                record["polygon_pixel_count"] = polygon_count
                if polygon_count == 0:
                    record["status"] = "NO_POLYGON_PIXELS"
                    record["error"] = "Tidak ada piksel raster yang menyentuh poligon sampel."
                    return record, token

                band_arrays = {band: arr[i] for i, band in enumerate(SPECTRAL_BANDS)}
                scl = np.rint(arr[len(SPECTRAL_BANDS)]).astype(np.int16)
                data_mask = arr[len(SPECTRAL_BANDS) + 1]
                dm_valid = inside & np.isfinite(data_mask) & (data_mask > 0)
                record["dataMask_valid_pixel_count"] = int(dm_valid.sum())
                scl_cloud_shadow = dm_valid & np.isin(scl, list(CLOUD_SHADOW_SCL_CLASSES))
                scl_dark = dm_valid & (scl == 2)
                scl_veg = dm_valid & (scl == 4)
                scl_bare = dm_valid & (scl == 5)
                scl_water = dm_valid & (scl == 6)
                record["scl_cloud_shadow_pixel_count"] = int(scl_cloud_shadow.sum())
                record["scl_class_2_dark_pixel_count"] = int(scl_dark.sum())
                record["scl_class_4_vegetation_pixel_count"] = int(scl_veg.sum())
                record["scl_class_5_bare_nonvegetated_pixel_count"] = int(scl_bare.sum())
                record["scl_class_6_water_pixel_count"] = int(scl_water.sum())

                clear = dm_valid & np.isin(scl, list(CLEAR_SCL_CLASSES))
                for band_arr in band_arrays.values():
                    clear &= np.isfinite(band_arr)
                clear_count = int(clear.sum())
                record["clear_pixel_count"] = clear_count
                record["valid_pixel_fraction_of_polygon"] = float(dm_valid.sum() / polygon_count)
                record["clear_pixel_fraction_of_polygon"] = float(clear_count / polygon_count)
                if clear_count == 0:
                    record["status"] = "NO_CLEAR_PIXELS"
                    record["error"] = (
                        "Ada footprint/piksel data, tetapi tidak ada piksel poligon yang lolos "
                        "mask SCL kelas 4/5/6. Jangan gunakan baris ini untuk training."
                    )
                    return record, token

                for band in SPECTRAL_BANDS:
                    stats(band_arrays[band][clear], band, record)

                ndvi = normalized_difference(band_arrays["B08"], band_arrays["B04"])
                ndwi = normalized_difference(band_arrays["B03"], band_arrays["B08"])
                mndwi = normalized_difference(band_arrays["B03"], band_arrays["B11"])
                ndbi = normalized_difference(band_arrays["B11"], band_arrays["B08"])
                for name, value in [("NDVI", ndvi), ("NDWI", ndwi), ("MNDWI", mndwi), ("NDBI", ndbi)]:
                    stats(value[clear], name, record)

                record["status"] = "OK"
                record["error"] = ""
    except Exception as exc:
        record["status"] = "PROCESSING_ERROR"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record, token


def save_checkpoint(rows: dict[str, dict[str, Any]], sample_order: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = [rows[sid] for sid in sample_order if sid in rows]
    df = pd.DataFrame(ordered)
    temp = OUTPUT_PATH.with_suffix(".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(OUTPUT_PATH)
    failed = df[df.get("status", pd.Series(dtype=str)).astype(str) != "OK"] if not df.empty else df
    failed.to_csv(FAILURES_PATH, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ekstrak fitur Sentinel-2 L2A poligon ke CSV; tanpa menyimpan TIFF.")
    parser.add_argument("--limit", type=int, default=0, help="Uji hanya N sampel pertama, misalnya --limit 5. Default: semua sampel.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Jeda antar permintaan API dalam detik (default 0.15).")
    args = parser.parse_args()

    print("=== Ekstraksi fitur Sentinel-2A L2A langsung ke CSV ===")
    print(f"Proyek       : {PROJECT_DIR}")
    print(f"File .env    : {ENV_PATH}")
    print(f"Dataset      : {SAMPLES_PATH}")
    print(f"Output CSV   : {OUTPUT_PATH}")
    print("Metode       : Process API per poligon; TIFF hanya dibaca di memori, tidak disimpan.")
    print("Band         : " + ", ".join(SPECTRAL_BANDS) + ", SCL, dataMask")
    print("Resolusi     : target 10 m; band native 20/60 m di-resample nearest-neighbour.")
    print("Mask         : dataMask > 0 dan SCL kelas 4/5/6.")
    print("Waktu        : satu hari penuh berdasarkan tanggal penugasan, bukan interval satu menit.")

    try:
        samples = load_samples()
        assignment_map = build_assignment_map()
        missing = [s["sample_id"] for s in samples if s["sample_id"] not in assignment_map]
        print(f"Jumlah sampel: {len(samples)}")
        print(f"Penugasan   : {len(assignment_map)}")
        if missing:
            print(f"PERINGATAN: {len(missing)} sampel tidak memiliki penugasan tanggal; contohnya: {missing[:8]}")

        if args.limit > 0:
            samples = samples[:args.limit]
            print(f"Mode uji    : hanya {len(samples)} sampel pertama")
        else:
            print("Mode        : seluruh sampel; CSV disimpan setiap selesai satu sampel")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows: dict[str, dict[str, Any]] = {}
        if OUTPUT_PATH.exists():
            try:
                prev = pd.read_csv(OUTPUT_PATH, dtype={"sample_id": str})
                for _, row in prev.iterrows():
                    sid = str(row.get("sample_id", ""))
                    if sid:
                        rows[sid] = row.where(pd.notna(row), None).to_dict()
                print(f"Checkpoint ditemukan: {len(rows)} baris sebelumnya akan dipertahankan.")
            except Exception as exc:
                print(f"PERINGATAN: checkpoint tidak dapat dibaca; mulai file baru. Detail: {exc}")
                rows = {}

        session = requests.Session()
        token = get_token(session)
        print("OAuth        : berhasil (kredensial/token tidak ditampilkan)")

        order = [s["sample_id"] for s in samples]
        total = len(samples)
        for idx, sample in enumerate(samples, start=1):
            sid = sample["sample_id"]
            existing = rows.get(sid)
            if existing and str(existing.get("status")) == "OK" and as_float(existing.get("clear_pixel_count")):
                print(f"[{idx}/{total}] SKIP {sid} — hasil valid sudah ada di checkpoint")
                continue
            assignment = assignment_map.get(sid, {})
            row, token = extract_one(sample, assignment, session, token)
            rows[sid] = row
            save_checkpoint(rows, order)
            clear_count = row.get("clear_pixel_count", 0)
            print(f"[{idx}/{total}] {sid} | {row.get('class_name')} | {row.get('status')} | clear pixels={clear_count}")
            if row.get("error"):
                print(f"    Detail: {str(row['error'])[:220]}")
            time.sleep(max(0.0, args.sleep))

        final_rows = [rows[sid] for sid in order if sid in rows]
        df = pd.DataFrame(final_rows)
        print("\n=== SELESAI ===")
        print(f"CSV fitur        : {OUTPUT_PATH}")
        print(f"CSV non-OK       : {FAILURES_PATH}")
        print(f"Baris ditulis    : {len(df)}")
        if not df.empty and "status" in df.columns:
            print("Status:")
            print(df["status"].value_counts(dropna=False).to_string())
        print("Catatan: baris OK tetap perlu audit jumlah piksel dan validitas label; jangan langsung training tanpa pemeriksaan.")
        return 0
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
