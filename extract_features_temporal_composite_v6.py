from __future__ import annotations

"""Extract polygon-level Sentinel-2 L2A features as a temporal median composite.

This is a separate v6 script: it does not overwrite the original extraction output.
It requests a common date window for every sample and composes per-pixel median
reflectance from observations passing dataMask and SCL classes 4/5/6.

Candidate labels are not validated by this script. Review geometry, label quality,
clear-pixel coverage, and acquisition metadata before model training.
Native 10/20/60 m bands are requested on a 10 m output grid using nearest-neighbour
resampling for compatibility with the existing project workflow.
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
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_SAMPLES_PATH = (
    PROJECT_DIR / "outputs" / "dataset_v6_review" /
    "dataset_6kelas_v6_kandidat_representatif_303.geojson"
)
SAMPLES_PATH = DEFAULT_SAMPLES_PATH
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_temporal_composite_v6"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

SPECTRAL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12",
]
# Composite TIFF: 12 per-pixel temporal-median reflectance bands + number of
# clear observations used for each pixel + a mask indicating at least one valid observation.
OUTPUT_BANDS = SPECTRAL_BANDS + ["valid_observation_count", "valid_mask"]
CLEAR_SCL_CLASSES = {4, 5, 6}
TARGET_RESOLUTION_M = 10.0
BBOX_PADDING_M = 20.0

EVALSCRIPT = r"""//VERSION=3
const BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                    "B08", "B8A", "B09", "B11", "B12"];

function setup() {
  return {
    input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B11", "B12", "SCL", "dataMask"],
    mosaicking: "ORBIT",
    output: { bands: 14, sampleType: "FLOAT32" }
  };
}

function median(values) {
  if (values.length === 0) return 0;
  values.sort(function(a, b) { return a - b; });
  var mid = Math.floor(values.length / 2);
  if (values.length % 2 === 1) return values[mid];
  return (values[mid - 1] + values[mid]) / 2;
}

function evaluatePixel(samples) {
  var valid = [];
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (!(s.dataMask > 0) || [4, 5, 6].indexOf(Math.round(s.SCL)) === -1) continue;
    var bandsOK = true;
    for (var b = 0; b < BAND_NAMES.length; b++) {
      if (!isFinite(s[BAND_NAMES[b]])) { bandsOK = false; break; }
    }
    if (bandsOK) valid.push(s);
  }

  if (valid.length === 0) {
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  }

  var result = [];
  for (var j = 0; j < BAND_NAMES.length; j++) {
    var vals = [];
    for (var k = 0; k < valid.length; k++) vals.push(valid[k][BAND_NAMES[j]]);
    result.push(median(vals));
  }
  result.push(valid.length); // Number of clear observations used at this pixel.
  result.push(1);             // At least one clear observation exists.
  return result;
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
    session: requests.Session, token: str, bbox: list[float], epsg: int,
    start_date: date, end_date_inclusive: date,
) -> tuple[requests.Response | None, str, str]:
    """Request a per-pixel median over the same inclusive date window for every sample."""
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_exclusive = datetime.combine(
        end_date_inclusive + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": crs_uri},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": start.isoformat().replace("+00:00", "Z"),
                        "to": end_exclusive.isoformat().replace("+00:00", "Z"),
                    },
                    # Keep this at 100: local SCL masking in the evalscript decides
                    # whether an individual pixel can contribute to the composite.
                    "maxCloudCoverage": 100,
                    "mosaickingOrder": "leastCC",
                },
                "processing": {
                    "upsampling": "NEAREST",
                    "downsampling": "NEAREST",
                },
            }],
        },
        "output": {
            "resx": TARGET_RESOLUTION_M,
            "resy": TARGET_RESOLUTION_M,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
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
                timeout=240,
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
            time.sleep(3 * attempt)
            continue
        if not response.ok:
            try:
                body = response.json()
                detail = body.get("error", {}).get("message") or body.get("message") or str(body)
            except ValueError:
                detail = response.text[:400]
            return response, f"HTTP {response.status_code}: {detail}", active_token
        return response, "OK", active_token
    return None, "GAGAL_SETELAH_RETRY", active_token


def base_record(
    sample: dict[str, Any], start_date: date, end_date_inclusive: date,
) -> dict[str, Any]:
    p = sample["properties"]
    out = {
        "sample_id": sample["sample_id"],
        "class_id": p.get("class_id"),
        "class_name": p.get("class_name"),
        "label_status": p.get("label_status", p.get("validation_status", "")),
        "usable_for_training_now": p.get("usable_for_training_now", p.get("usable_for_training", False)),
        "source_file": p.get("source_file", ""),
        "original_label": p.get("original_label", ""),
        "composite_start_date": start_date.isoformat(),
        "composite_end_date_inclusive": end_date_inclusive.isoformat(),
        "composite_end_datetime_exclusive_utc": (
            datetime.combine(end_date_inclusive + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            .isoformat().replace("+00:00", "Z")
        ),
        "temporal_composite_method": "per-pixel median of SCL 4/5/6 observations within common date window",
        "target_resolution_m": TARGET_RESOLUTION_M,
        "resampling_method": "NEAREST; output grid 10 m; native bands 20/60 m resampled",
        "cloud_mask_scl_clear_classes": "4|5|6",
        "index_method": "indices computed from temporal-median reflectance composite",
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
        "valid_pixel_count": 0,
        "clear_pixel_count": 0,
        "valid_pixel_fraction_of_polygon": np.nan,
        "clear_pixel_fraction_of_polygon": np.nan,
        "mean_clear_observations_per_valid_pixel": np.nan,
        "max_clear_observations_per_valid_pixel": 0,
        "clear_observation_pixel_count_total": 0,
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
    sample: dict[str, Any], start_date: date, end_date_inclusive: date,
    session: requests.Session, token: str,
) -> tuple[dict[str, Any], str]:
    record = base_record(sample, start_date, end_date_inclusive)
    epsg = get_utm_epsg(sample["geometry"])
    geom_utm = project_geometry(sample["geometry"], epsg)
    bbox = make_request_bbox(geom_utm)
    response, message, token = process_api_request(
        session, token, bbox, epsg, start_date, end_date_inclusive
    )
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
                record["utm_epsg"] = ds.crs.to_epsg() or epsg
                record["raster_width_px"] = ds.width
                record["raster_height_px"] = ds.height

                geom_ds = project_geometry(sample["geometry"], ds.crs.to_epsg() or epsg)
                inside = geometry_mask(
                    [mapping(geom_ds)], out_shape=(ds.height, ds.width),
                    transform=ds.transform, invert=True, all_touched=False,
                )
                mask_mode = "pixel_center"
                if not inside.any():
                    inside = geometry_mask(
                        [mapping(geom_ds)], out_shape=(ds.height, ds.width),
                        transform=ds.transform, invert=True, all_touched=True,
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
                obs_count = arr[len(SPECTRAL_BANDS)]
                valid_mask = arr[len(SPECTRAL_BANDS) + 1]
                valid_any = inside & np.isfinite(valid_mask) & (valid_mask > 0)
                record["valid_pixel_count"] = int(valid_any.sum())
                clear = (
                    inside & np.isfinite(obs_count) & (obs_count > 0) &
                    np.isfinite(valid_mask) & (valid_mask > 0)
                )
                for band_arr in band_arrays.values():
                    clear &= np.isfinite(band_arr)
                clear_count = int(clear.sum())
                record["clear_pixel_count"] = clear_count
                record["valid_pixel_fraction_of_polygon"] = float(valid_any.sum() / polygon_count)
                record["clear_pixel_fraction_of_polygon"] = float(clear_count / polygon_count)
                if clear_count:
                    obs_values = obs_count[clear]
                    record["mean_clear_observations_per_valid_pixel"] = float(obs_values.mean())
                    record["max_clear_observations_per_valid_pixel"] = int(obs_values.max())
                    record["clear_observation_pixel_count_total"] = int(obs_values.sum())
                if clear_count == 0:
                    record["status"] = "NO_CLEAR_PIXELS_IN_COMPOSITE"
                    record["error"] = (
                        "Tidak ada piksel poligon dengan observasi SCL 4/5/6 yang valid "
                        "dalam jendela komposit. Jangan gunakan baris ini untuk training."
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


def save_checkpoint(
    rows: dict[str, dict[str, Any]], sample_order: list[str],
    output_path: Path, failures_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[sid] for sid in sample_order if sid in rows]
    df = pd.DataFrame(ordered)
    temp = output_path.with_suffix(".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(output_path)
    failed = df[df.get("status", pd.Series(dtype=str)).astype(str) != "OK"] if not df.empty else df
    failed.to_csv(failures_path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ekstrak fitur Sentinel-2 L2A sebagai komposit median temporal dengan jendela tanggal seragam."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH,
                        help="GeoJSON sampel; default subset representatif v6.")
    parser.add_argument("--start-date", default="2026-09-01", help="Tanggal awal inklusif YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-10-10", help="Tanggal akhir inklusif YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=0, help="Uji hanya N sampel pertama. Default: semua.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Jeda antar permintaan API dalam detik.")
    parser.add_argument("--fresh", action="store_true", help="Mulai checkpoint baru untuk jendela tanggal ini.")
    args = parser.parse_args()

    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    except ValueError:
        parser.error("Tanggal harus berformat YYYY-MM-DD.")
    if end_date < start_date:
        parser.error("--end-date harus sama dengan atau setelah --start-date.")
    if not args.samples.exists():
        print(f"Gagal: GeoJSON sampel tidak ditemukan: {args.samples}")
        print("Pastikan file subset representatif v6 tersedia, atau gunakan --samples <path>. ")
        return 1

    date_tag = f"{start_date.isoformat()}_{end_date.isoformat()}"
    output_path = OUTPUT_DIR / f"fitur_sentinel2a_l2a_v6_temporal_{date_tag}.csv"
    failures_path = OUTPUT_DIR / f"fitur_sentinel2a_l2a_v6_temporal_{date_tag}_gagal.csv"

    print("=== Ekstraksi fitur Sentinel-2A L2A: komposit median temporal ===")
    print(f"Proyek       : {PROJECT_DIR}")
    print(f"File .env    : {ENV_PATH}")
    print(f"Dataset      : {args.samples}")
    print(f"Rentang      : {start_date.isoformat()} s.d. {end_date.isoformat()} (inklusif)")
    print(f"Output CSV   : {output_path}")
    print("Metode       : median reflektansi per piksel dari observasi SCL 4/5/6 dalam rentang yang sama.")
    print("Band         : " + ", ".join(SPECTRAL_BANDS) + ", jumlah observasi valid, mask valid")
    print("Resolusi     : grid output 10 m; band native 20/60 m di-resample nearest-neighbour.")
    print("Catatan      : tanggal observasi bersih dapat berbeda per lokasi; jumlah observasi dicatat per piksel.")

    try:
        # load_samples reads the project-wide SAMPLES_PATH constant.
        # Override it here with the user-selected/default v6 GeoJSON path.
        global SAMPLES_PATH
        SAMPLES_PATH = args.samples
        samples = load_samples()
        print(f"Jumlah sampel: {len(samples)}")
        if args.limit > 0:
            samples = samples[:args.limit]
            print(f"Mode uji     : {len(samples)} sampel pertama")
        else:
            print("Mode         : seluruh sampel; checkpoint disimpan setiap sampel")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows: dict[str, dict[str, Any]] = {}
        if args.fresh and output_path.exists():
            output_path.unlink()
            if failures_path.exists():
                failures_path.unlink()
            print("Checkpoint   : lama untuk jendela ini dihapus sesuai --fresh")
        if output_path.exists():
            try:
                prev = pd.read_csv(output_path, dtype={"sample_id": str})
                for _, row in prev.iterrows():
                    sid = str(row.get("sample_id", "")).strip()
                    if sid:
                        rows[sid] = row.where(pd.notna(row), None).to_dict()
                print(f"Checkpoint   : {len(rows)} baris sebelumnya ditemukan")
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
            same_window = existing and (
                str(existing.get("composite_start_date")) == start_date.isoformat() and
                str(existing.get("composite_end_date_inclusive")) == end_date.isoformat()
            )
            if same_window and str(existing.get("status")) == "OK" and (as_float(existing.get("clear_pixel_count")) or 0) > 0:
                print(f"[{idx}/{total}] SKIP {sid} — hasil OK untuk jendela tanggal ini sudah ada")
                continue
            row, token = extract_one(sample, start_date, end_date, session, token)
            rows[sid] = row
            save_checkpoint(rows, order, output_path, failures_path)
            print(f"[{idx}/{total}] {sid} | {row.get('class_name')} | {row.get('status')} | clear pixels={row.get('clear_pixel_count', 0)}")
            if row.get("error"):
                print(f"    Detail: {str(row['error'])[:220]}")
            time.sleep(max(0.0, args.sleep))

        final_rows = [rows[sid] for sid in order if sid in rows]
        df = pd.DataFrame(final_rows)
        print("\n=== SELESAI ===")
        print(f"CSV fitur      : {output_path}")
        print(f"CSV non-OK     : {failures_path}")
        print(f"Baris ditulis  : {len(df)}")
        if not df.empty and "status" in df.columns:
            print("Status:")
            print(df["status"].value_counts(dropna=False).to_string())
        print("Catatan: OK bukan berarti label benar; audit kualitas dan validitas label tetap wajib sebelum training.")
        return 0
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
