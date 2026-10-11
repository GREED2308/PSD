from __future__ import annotations

"""Predict a 10 m class raster over a representative East Java AOI.

This is intentionally a 40 x 40 km pilot AOI in UTM 49S, chosen because sample
centroids from all six project classes occur inside it. It is NOT a province-wide
classification. Sentinel-2 L2A temporal median is requested from CDSE Process API
for four 20 x 20 km tiles. Tiles are checkpointed and can be resumed.

Input model: existing quality_screened model pipeline (SVM_RBF by default).
Input bands: the same 12 reflectance bands used by the training workflow.
Indexes: computed per output pixel using the same formulas as training features.
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

import joblib
import numpy as np
import pandas as pd
import rasterio
import requests
from dotenv import load_dotenv
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.enums import Resampling
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

ROOT = Path(os.environ.get("PSD_PROJECT_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
ENV_PATH = ROOT / ".env"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

FEATURES = [
    "B01_mean", "B02_mean", "B03_mean", "B04_mean", "B05_mean", "B06_mean",
    "B07_mean", "B08_mean", "B8A_mean", "B09_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean",
]
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CLASS_TO_ID = {"sawah": 1, "bangunan": 2, "mangrove": 3, "lahan_hijau": 4, "laut": 5, "danau": 6}
CLASS_LABELS = {
    0: ("Tidak ada data / tidak lolos mask", "#00000000"),
    1: ("Sawah", "#D6A23A"),
    2: ("Bangunan", "#E76F51"),
    3: ("Mangrove", "#287271"),
    4: ("Lahan hijau", "#6A994E"),
    5: ("Perairan terbuka / laut", "#277DA1"),
    6: ("Danau / ranu", "#7B61A8"),
}

# Selected pilot AOI: one 40x40 km grid cell in EPSG:32749 containing sample
# centroids from all six classes. Four 20x20 km API tiles, output at 10 m.
CRS_EPSG = 32749
AOI_XMIN = 680_000.0
AOI_YMIN = 9_200_000.0
AOI_SIZE_M = 40_000.0
TILE_SIZE_M = 20_000.0
RESOLUTION_M = 10.0
TILE_PIXELS = int(TILE_SIZE_M / RESOLUTION_M)  # 2000
START_DATE = "2026-09-01"
END_DATE = "2026-10-10"

EVALSCRIPT = r'''//VERSION=3
const BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"];
function setup() {
  return {
    input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL", "dataMask"],
    mosaicking: "ORBIT",
    output: { bands: 14, sampleType: "UINT16", nodataValue: 0 }
  };
}
function median(values) {
  if (values.length === 0) return 0;
  values.sort(function(a, b) { return a - b; });
  var mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}
function evaluatePixel(samples) {
  var valid = [];
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (!(s.dataMask > 0) || [4, 5, 6].indexOf(Math.round(s.SCL)) === -1) continue;
    var ok = true;
    for (var b = 0; b < BANDS.length; b++) {
      if (!isFinite(s[BANDS[b]])) { ok = false; break; }
    }
    if (ok) valid.push(s);
  }
  if (valid.length === 0) return [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
  var out = [];
  for (var j = 0; j < BANDS.length; j++) {
    var vals = [];
    for (var k = 0; k < valid.length; k++) vals.push(valid[k][BANDS[j]]);
    var reflectance = median(vals);
    out.push(Math.round(Math.max(0, Math.min(6.5, reflectance)) * 10000));
  }
  out.push(valid.length); // clear observations used at this pixel
  out.push(1);             // at least one clear observation exists
  return out;
}
'''


def find_boundary_path() -> Path:
    candidates = [
        ROOT / "data" / "geojson" / "jawa_timur_2D.geojson",
        ROOT / "data" / "geojson" / "jawa_timur_AOI_ringan.geojson",
        ROOT / "jawa_timur_2D.geojson",
        ROOT / "data" / "geojson" / "jawa_timur.geojson",
    ]
    for path in candidates:
        if path.exists():
            try:
                _ = load_boundary(path)
                return path
            except Exception:
                continue
    raise FileNotFoundError(
        "Batas Jawa Timur tidak ditemukan/terbaca. Pastikan tersedia "
        "data\\geojson\\jawa_timur_2D.geojson (disarankan) atau jawa_timur_AOI_ringan.geojson."
    )


def load_boundary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    geoms = []
    for feat in fc.get("features", []):
        if feat.get("geometry"):
            geoms.append(shape(feat["geometry"]))
    if not geoms:
        raise ValueError(f"Tidak ada geometri pada {path}")
    geom = unary_union(geoms)
    if not geom.is_valid:
        raise ValueError(f"Geometri batas tidak valid: {path}")
    return geom


def get_token(session: requests.Session) -> str:
    load_dotenv(ENV_PATH, override=False)
    client_id = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(f"Kredensial COPERNICUS_CLIENT_ID/SECRET tidak terbaca dari {ENV_PATH}")
    resp = session.post(TOKEN_URL, data={
        "grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret,
    }, timeout=45)
    if not resp.ok:
        try:
            j = resp.json()
            detail = j.get("error_description") or j.get("error") or str(j)
        except ValueError:
            detail = resp.text[:300]
        raise RuntimeError(f"OAuth gagal HTTP {resp.status_code}: {detail}")
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth berhasil tetapi access_token tidak ada.")
    return token


def process_tile(session: requests.Session, token: str, x0: float, y0: float,
                 start: date, end_inclusive: date) -> tuple[bytes, str]:
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_inclusive + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    bbox = [x0, y0, x0 + TILE_SIZE_M, y0 + TILE_SIZE_M]
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{CRS_EPSG}"
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": crs_uri}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": start_dt.isoformat().replace("+00:00", "Z"),
                        "to": end_dt.isoformat().replace("+00:00", "Z"),
                    },
                    "maxCloudCoverage": 100,
                    "mosaickingOrder": "leastCC",
                },
                "processing": {"upsampling": "NEAREST", "downsampling": "NEAREST"},
            }],
        },
        "output": {
            "width": TILE_PIXELS,
            "height": TILE_PIXELS,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    active_token = token
    for attempt in range(1, 4):
        try:
            resp = session.post(PROCESS_URL, json=payload,
                                headers={"Authorization": f"Bearer {active_token}", "Accept": "image/tiff"}, timeout=300)
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"Koneksi gagal setelah retry: {type(exc).__name__}: {exc}") from exc
            time.sleep(3 * attempt)
            continue
        if resp.status_code == 401 and attempt < 3:
            active_token = get_token(session)
            time.sleep(1)
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(4 * attempt)
            continue
        if not resp.ok:
            try:
                body = resp.json()
                detail = body.get("error", {}).get("message") or body.get("message") or str(body)
            except ValueError:
                detail = resp.text[:400]
            raise RuntimeError(f"Process API HTTP {resp.status_code}: {detail}")
        return resp.content, active_token
    raise RuntimeError("Permintaan tile gagal setelah retry.")


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.zeros(a.shape, dtype=np.float32)
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > 1e-8)
    out[good] = (a[good] - b[good]) / denom[good]
    return out


def feature_chunk(refl: np.ndarray) -> np.ndarray:
    # refl: (12, rows, cols) floating surface reflectance.
    ndvi = normalized_difference(refl[7], refl[3])
    ndwi = normalized_difference(refl[2], refl[7])
    mndwi = normalized_difference(refl[2], refl[10])
    ndbi = normalized_difference(refl[10], refl[7])
    return np.stack([*refl, ndvi, ndwi, mndwi, ndbi], axis=-1).astype(np.float32, copy=False)


def infer_tile(dataset_bytes: bytes, model, boundary_utm, tile_x0: float, tile_y0: float,
               temp_tile_path: Path, class_to_id: dict[str, int]) -> tuple[int, int]:
    temp_tile_path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryFile(dataset_bytes) as mem:
        with mem.open() as src:
            if src.count != 14:
                raise ValueError(f"Tile harus punya 14 band (12 reflectansi + observasi + mask), ditemukan {src.count}.")
            if src.width != TILE_PIXELS or src.height != TILE_PIXELS:
                raise ValueError(f"Dimensi tile bukan {TILE_PIXELS}x{TILE_PIXELS}: {src.width}x{src.height}")
            raw = src.read(out_dtype="uint16")
            transform = src.transform
            tile_box = box(tile_x0, tile_y0, tile_x0 + TILE_SIZE_M, tile_y0 + TILE_SIZE_M)
            local_boundary = boundary_utm.intersection(tile_box)
            if local_boundary.is_empty:
                boundary_mask = np.zeros((src.height, src.width), dtype=bool)
            else:
                boundary_mask = geometry_mask([mapping(local_boundary)],
                                              out_shape=(src.height, src.width), transform=transform,
                                              invert=True, all_touched=False)
    valid_mask = (raw[13] > 0) & (raw[12] > 0) & boundary_mask
    prediction = np.zeros((TILE_PIXELS, TILE_PIXELS), dtype=np.uint8)
    counts = np.zeros(7, dtype=np.int64)
    # Process small row batches to keep memory bounded.
    rows_per_chunk = 128
    for r0 in range(0, TILE_PIXELS, rows_per_chunk):
        r1 = min(TILE_PIXELS, r0 + rows_per_chunk)
        good = valid_mask[r0:r1]
        if not good.any():
            continue
        reflectance = raw[:12, r0:r1].astype(np.float32) / 10000.0
        cube = feature_chunk(reflectance)  # (rows, cols, 16)
        X = cube[good]
        Xdf = pd.DataFrame(X, columns=FEATURES)
        classes = model.predict(Xdf)
        codes = np.fromiter((class_to_id.get(str(c).strip().lower(), 0) for c in classes),
                            count=len(classes), dtype=np.uint8)
        target = prediction[r0:r1]
        target[good] = codes
        values, freqs = np.unique(codes, return_counts=True)
        for val, freq in zip(values, freqs):
            if 0 <= int(val) <= 6:
                counts[int(val)] += int(freq)
    out_profile = {
        "driver": "GTiff", "width": TILE_PIXELS, "height": TILE_PIXELS, "count": 1,
        "dtype": "uint8", "crs": f"EPSG:{CRS_EPSG}", "transform": transform,
        "nodata": 0, "compress": "DEFLATE", "predictor": 2, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(temp_tile_path, "w", **out_profile) as dst:
        dst.write(prediction, 1)
        dst.update_tags(
            model_name="SVM_RBF or KNN trained from polygon-level sample features",
            temporal_window=f"{START_DATE} to {END_DATE} inclusive",
            description="10 m class IDs: 1 sawah, 2 bangunan, 3 mangrove, 4 lahan_hijau, 5 laut, 6 danau; 0 nodata",
            scope="representative 40x40 km AOI; not entire East Java",
        )
    return int(valid_mask.sum()), int(np.count_nonzero(prediction))


def write_aoi_geojson(path: Path, boundary_wgs84, boundary_utm):
    x0, y0 = AOI_XMIN, AOI_YMIN
    aoi_utm = box(x0, y0, x0 + AOI_SIZE_M, y0 + AOI_SIZE_M)
    clipped = boundary_utm.intersection(aoi_utm)
    to_wgs = Transformer.from_crs(f"EPSG:{CRS_EPSG}", "EPSG:4326", always_xy=True).transform
    clipped_wgs = shapely_transform(to_wgs, clipped)
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {
            "name": "AOI klasifikasi percontohan 40 x 40 km",
            "scope": "AOI percontohan di dalam Jawa Timur; bukan prediksi seluruh provinsi",
            "model": "SVM_RBF quality_screened",
            "resolution_m": 10,
            "period": f"{START_DATE} to {END_DATE}",
        }, "geometry": mapping(clipped_wgs),
    }]}
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def make_preview(raster_path: Path, png_path: Path, metadata_path: Path):
    with rasterio.open(raster_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        preview = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1), destination=preview,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=0, dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=0,
            resampling=Resampling.nearest,
        )
    rgba = np.zeros((preview.shape[0], preview.shape[1], 4), dtype=np.uint8)
    hex_to_rgb = {i: tuple(int(CLASS_LABELS[i][1][j:j+2], 16) for j in (1, 3, 5)) for i in range(1, 7)}
    for class_id, rgb in hex_to_rgb.items():
        m = preview == class_id
        rgba[m, :3] = rgb
        rgba[m, 3] = 190
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(png_path, optimize=True)
    meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    meta.update({
        "preview_png": str(png_path),
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "classes": {str(k): {"label": CLASS_LABELS[k][0], "color": CLASS_LABELS[k][1]} for k in range(1, 7)},
    })
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi raster 10 m untuk AOI percontohan Jawa Timur 40x40 km.")
    parser.add_argument("--model", choices=["SVM_RBF", "KNN"], default="SVM_RBF", help="Model tersaring kualitas untuk prediksi raster.")
    parser.add_argument("--variant", choices=["quality_screened"], default="quality_screened", help="Model dilatih pada dataset tersaring kualitas.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Uji maksimum N tile, misalnya 1; 0 memproses semua tile.")
    parser.add_argument("--plan-only", action="store_true", help="Validasi boundary dan tampilkan rencana tile tanpa memerlukan model atau mengirim permintaan API.")
    parser.add_argument("--fresh", action="store_true", help="Hapus cache tile model/tanggal ini dan proses ulang.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Jeda antar tile API (default 1 detik).")
    args = parser.parse_args()

    try:
        start = date.fromisoformat(START_DATE)
        end = date.fromisoformat(END_DATE)
        boundary_path = find_boundary_path()
        boundary_wgs84 = load_boundary(boundary_path)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{CRS_EPSG}", always_xy=True).transform
        boundary_utm = shapely_transform(to_utm, boundary_wgs84)
        model_path = ROOT / "outputs" / "modeling_v6_baseline" / args.variant / f"model_{args.model}.joblib"
        model = None
        if not args.plan_only:
            if not model_path.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
            model = joblib.load(model_path)
        out_dir = ROOT / "outputs" / "classification_raster_v6_aoi40km"
        tiles_dir = out_dir / f"tiles_{args.model}_{args.variant}_{START_DATE}_{END_DATE}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        raster_path = out_dir / f"classification_{args.model}_{args.variant}_10m_aoi40km.tif"
        metadata_path = out_dir / f"classification_{args.model}_{args.variant}_aoi40km_metadata.json"
        png_path = out_dir / f"classification_{args.model}_{args.variant}_aoi40km_preview.png"
        aoi_path = out_dir / "aoi_klasifikasi_percontohan_40x40km.geojson"

        print("=== Prediksi raster klasifikasi v6: AOI percontohan ===")
        print(f"Root proyek : {ROOT}")
        print(f"Boundary    : {boundary_path}")
        print(f"Model       : {model_path}")
        print(f"CRS/grid    : EPSG:{CRS_EPSG}; 10 m")
        print(f"Periode     : {START_DATE} s.d. {END_DATE} (inklusif)")
        print("AOI         : UTM [680000, 9200000, 720000, 9240000] (40 x 40 km)")
        print("Cakupan     : AOI percontohan di Jawa Timur, BUKAN seluruh provinsi.")
        print("Metode      : median temporal SCL 4/5/6, fitur pixel-level, model terlatih dari fitur poligon.")
        print("Tile API    : 4 tile 20 x 20 km; cache aktif untuk resume.")

        # Count sample centroids per tile to order the smoke test on the densest tile.
        samples_path = ROOT / "outputs" / "dataset_v6_review" / "dataset_6kelas_v6_kandidat_representatif_303.geojson"
        tile_counts: dict[tuple[int, int], int] = {}
        if samples_path.exists():
            with samples_path.open("r", encoding="utf-8") as f:
                sample_fc = json.load(f)
            for feature in sample_fc.get("features", []):
                if not feature.get("geometry"):
                    continue
                geom = shape(feature["geometry"])
                center = geom.centroid
                px, py = to_utm(center.x, center.y)
                tx = math.floor(px / TILE_SIZE_M) * TILE_SIZE_M
                ty = math.floor(py / TILE_SIZE_M) * TILE_SIZE_M
                if AOI_XMIN <= tx < AOI_XMIN + AOI_SIZE_M and AOI_YMIN <= ty < AOI_YMIN + AOI_SIZE_M:
                    tile_counts[(tx, ty)] = tile_counts.get((tx, ty), 0) + 1
        tile_specs = []
        for x0 in (AOI_XMIN, AOI_XMIN + TILE_SIZE_M):
            for y0 in (AOI_YMIN, AOI_YMIN + TILE_SIZE_M):
                geom = boundary_utm.intersection(box(x0, y0, x0 + TILE_SIZE_M, y0 + TILE_SIZE_M))
                if not geom.is_empty and geom.area > 1000:
                    tile_specs.append((x0, y0))
        tile_specs.sort(key=lambda t: tile_counts.get(t, 0), reverse=True)
        if not tile_specs:
            raise RuntimeError("AOI percontohan tidak memotong batas Jawa Timur. Periksa file batas/CRS.")
        if args.max_tiles > 0:
            todo_specs = tile_specs[:args.max_tiles]
        else:
            todo_specs = tile_specs
        print(f"Tile yang beririsan batas wilayah: {len(tile_specs)}; target run ini: {len(todo_specs)}")
        for tx, ty in tile_specs:
            area = boundary_utm.intersection(box(tx, ty, tx + TILE_SIZE_M, ty + TILE_SIZE_M)).area / 1e6
            print(f"  tile x={int(tx)} y={int(ty)} | centroid sampel={tile_counts.get((tx, ty), 0)} | luas batas di tile={area:.1f} km²")
        if args.plan_only:
            print("Plan-only: tidak ada permintaan API yang dikirim dan tidak ada raster dibuat.")
            return 0

        if args.fresh:
            for p in tiles_dir.glob("tile_*.tif"):
                p.unlink()
            print("Cache tile untuk model/tanggal ini dihapus sesuai --fresh.")

        session = requests.Session()
        token = get_token(session)
        print("OAuth        : berhasil; kredensial tidak ditampilkan")
        for i, (x0, y0) in enumerate(todo_specs, 1):
            tile_path = tiles_dir / f"tile_x{int(x0)}_y{int(y0)}.tif"
            if tile_path.exists():
                print(f"[{i}/{len(todo_specs)}] SKIP tile x={int(x0)} y={int(y0)} — cache tersedia")
                continue
            print(f"[{i}/{len(todo_specs)}] Meminta Sentinel-2 composite tile x={int(x0)} y={int(y0)} (sampel={tile_counts.get((x0,y0),0)}) ...", flush=True)
            content, token = process_tile(session, token, x0, y0, start, end)
            n_valid, n_predicted = infer_tile(content, model, boundary_utm, x0, y0, tile_path, CLASS_TO_ID)
            print(f"    OK | piksel valid dalam batas: {n_valid:,} | piksel diberi kelas: {n_predicted:,}", flush=True)
            pu_spent = None
            # Avoid printing response headers as some proxies vary; no credentials are printed.
            time.sleep(max(0.0, args.sleep))

        global_profile = {
            "driver": "GTiff", "width": int(AOI_SIZE_M / RESOLUTION_M), "height": int(AOI_SIZE_M / RESOLUTION_M),
            "count": 1, "dtype": "uint8", "crs": f"EPSG:{CRS_EPSG}",
            "transform": from_origin(AOI_XMIN, AOI_YMIN + AOI_SIZE_M, RESOLUTION_M, RESOLUTION_M),
            "nodata": 0, "compress": "DEFLATE", "predictor": 2, "tiled": True,
            "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(raster_path, "w", **global_profile) as dst:
            dst.write(np.zeros((int(AOI_SIZE_M / RESOLUTION_M), int(AOI_SIZE_M / RESOLUTION_M)), dtype=np.uint8), 1)
            dst.update_tags(
                model=args.model, training_variant=args.variant, composite_start=START_DATE, composite_end_inclusive=END_DATE,
                resolution_m=RESOLUTION_M, crs_epsg=CRS_EPSG,
                scope="representative AOI 40x40 km only; not a province-wide raster classification",
            )
            loaded = 0
            for x0, y0 in tile_specs:
                tile_path = tiles_dir / f"tile_x{int(x0)}_y{int(y0)}.tif"
                if not tile_path.exists():
                    continue
                with rasterio.open(tile_path) as tile_src:
                    col_off = int(round((x0 - AOI_XMIN) / RESOLUTION_M))
                    row_off = int(round((AOI_YMIN + AOI_SIZE_M - (y0 + TILE_SIZE_M)) / RESOLUTION_M))
                    dst.write(tile_src.read(1), 1, window=rasterio.windows.Window(col_off, row_off, TILE_PIXELS, TILE_PIXELS))
                    loaded += 1
        complete = loaded == len(tile_specs)
        write_aoi_geojson(aoi_path, boundary_wgs84, boundary_utm)
        # Area totals are model-predicted areas only for valid pixels in this AOI, not province-wide truth.
        area_rows = []
        with rasterio.open(raster_path) as classified:
            class_array = classified.read(1)
            total_valid_pixels = int(np.count_nonzero(class_array > 0))
            total_valid_area_ha = total_valid_pixels * (RESOLUTION_M * RESOLUTION_M) / 10000.0
            for class_id in range(1, 7):
                npx = int(np.count_nonzero(class_array == class_id))
                area_ha = npx * (RESOLUTION_M * RESOLUTION_M) / 10000.0
                area_rows.append({
                    "class_id": class_id,
                    "class_name": CLASS_LABELS[class_id][0],
                    "pixel_count": npx,
                    "predicted_area_ha": area_ha,
                    "share_of_valid_predicted_area_pct": (100.0 * area_ha / total_valid_area_ha) if total_valid_area_ha else 0.0,
                    "scope": "AOI percontohan saja; bukan luas kelas seluruh Jawa Timur",
                })
        area_summary_path = out_dir / f"summary_area_prediksi_{args.model}_{args.variant}_aoi40km.csv"
        pd.DataFrame(area_rows).to_csv(area_summary_path, index=False, encoding="utf-8-sig")
        metadata = {
            "project": "SIG Klasifikasi Tutupan Lahan Jawa Timur v6",
            "scope": "AOI percontohan 40 x 40 km di Jawa Timur; BUKAN klasifikasi raster seluruh provinsi",
            "model": args.model,
            "training_variant": args.variant,
            "model_path": str(model_path),
            "feature_names": FEATURES,
            "class_codes": {str(v): k for k, v in CLASS_TO_ID.items()},
            "class_labels": {str(k): v[0] for k, v in CLASS_LABELS.items()},
            "crs": f"EPSG:{CRS_EPSG}", "resolution_m": RESOLUTION_M,
            "aoi_bounds_utm": [AOI_XMIN, AOI_YMIN, AOI_XMIN + AOI_SIZE_M, AOI_YMIN + AOI_SIZE_M],
            "composite_start_date": START_DATE, "composite_end_date_inclusive": END_DATE,
            "temporal_method": "Per-pixel median of Sentinel-2 L2A observations passing dataMask and SCL classes 4/5/6",
            "spatial_method": "Prediction by 10 m pixel; output mask clipped to Jawa Timur administrative boundary",
            "processed_tiles": loaded, "total_intersecting_tiles": len(tile_specs), "complete_aoi": complete,
            "valid_predicted_pixels": total_valid_pixels, "valid_predicted_area_ha": total_valid_area_ha,
            "area_summary_csv": str(area_summary_path),
            "boundary_source": str(boundary_path), "aoi_geojson": str(aoi_path), "raster_path": str(raster_path),
            "warning": "Training features are polygon-level means; output is pixel-level. Treat map as exploratory, not validated province-wide land-cover inventory.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        make_preview(raster_path, png_path, metadata_path)
        website_assets = ROOT / "website" / "assets"
        website_assets.mkdir(parents=True, exist_ok=True)
        website_preview = website_assets / "classification_aoi40km.png"
        import shutil
        shutil.copy2(png_path, website_preview)
        summary = pd.DataFrame([
            {"class_id": k, "class_name": CLASS_LABELS[k][0], "color": CLASS_LABELS[k][1]}
            for k in range(1, 7)
        ])
        summary.to_csv(out_dir / "legend_classification_v6.csv", index=False, encoding="utf-8-sig")
        print("\n=== SELESAI ===")
        print(f"Raster klasifikasi : {raster_path}")
        print(f"Preview PNG        : {png_path}")
        print(f"Preview web statis  : {website_preview}")
        print(f"AOI GeoJSON        : {aoi_path}")
        print(f"Ringkasan luas AOI  : {area_summary_path}")
        print(f"Metadata            : {metadata_path}")
        print(f"Tile selesai        : {loaded}/{len(tile_specs)}")
        print(f"AOI lengkap         : {complete}")
        if not complete:
            print("PERINGATAN: hasil AOI masih parsial karena belum semua tile selesai. Jalankan ulang tanpa --max-tiles untuk melanjutkan cache.")
        print("Catatan: raster ini hanya AOI percontohan 40x40 km, bukan klasifikasi seluruh Jawa Timur.")
        # Partial output is an expected result when --max-tiles is explicitly used for the smoke test.
        return 0 if (complete or args.max_tiles > 0) else 2
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
