from __future__ import annotations

"""Diagnose Sentinel-2 SCL class counts for selected sample polygons.

Reads project .env, dataset_6kelas.geojson and the sample-product assignment CSV.
Makes small Process API requests and writes class-frequency CSV; does not save TIFFs
or print credentials/tokens. This mirrors the extractor's full-day least-CC query,
so it diagnoses the data/mask behavior used in the current workflow.
"""
import json, os, math, time, sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import rasterio
from dotenv import load_dotenv
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
SAMPLES_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
ASSIGNMENTS_PATH = PROJECT_DIR / "outputs" / "sentinel2_search" / "penugasan_sampel_produk_sentinel2a.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_features_csv"
OUTPUT_CSV = OUTPUT_DIR / "diagnostic_scl_class_counts.csv"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
SAMPLE_IDS = ["JTM0051", "JTM0074", "JTM0123", "JTM0201", "JTM0265", "JTM0151"]

EVALSCRIPT = """//VERSION=3
function setup() {
  return { input: ["B04", "SCL", "dataMask"], output: { bands: 3, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) { return [sample.B04, sample.SCL, sample.dataMask]; }
"""


def get_token(session):
    load_dotenv(ENV_PATH, override=False)
    cid = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
    secret = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise RuntimeError(f"Kredensial COPERNICUS_CLIENT_ID/SECRET tidak terbaca dari {ENV_PATH}")
    r = session.post(TOKEN_URL, data={"grant_type":"client_credentials", "client_id":cid, "client_secret":secret}, timeout=45)
    if not r.ok:
        raise RuntimeError(f"OAuth gagal (HTTP {r.status_code}); detail tidak ditampilkan untuk menghindari kebocoran kredensial.")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth respons tanpa access_token")
    return token


def get_epsg(geom):
    lon, lat = geom.centroid.x, geom.centroid.y
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32700 + zone) if lat < 0 else (32600 + zone)


def to_utm(geom, epsg):
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return shapely_transform(tr.transform, geom)


def main():
    for p in (SAMPLES_PATH, ASSIGNMENTS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"File diperlukan tidak ditemukan: {p}")
    fc = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    feats = {str(f.get("properties", {}).get("sample_id")): f for f in fc.get("features", [])}
    assignments = pd.read_csv(ASSIGNMENTS_PATH, dtype={"sample_id": str})
    assignments = assignments.set_index("sample_id", drop=False)
    missing = [sid for sid in SAMPLE_IDS if sid not in feats or sid not in assignments.index]
    if missing:
        raise RuntimeError(f"Sample/assignment tidak ditemukan: {missing}")

    sess = requests.Session()
    token = get_token(sess)
    print("=== Diagnostik histogram SCL per sampel ===")
    print("Sampel yang diuji:", ", ".join(SAMPLE_IDS))
    print("Mask tidak diterapkan pada histogram; ini untuk melihat semua kelas SCL.")
    print("Waktu permintaan: satu hari penuh menurut tanggal assignment; mosaickingOrder=leastCC.")
    print("OAuth berhasil; secret/token tidak ditampilkan.")
    rows=[]
    for idx, sid in enumerate(SAMPLE_IDS, 1):
        f = feats[sid]
        props = f.get("properties") or {}
        geom = shape(f["geometry"])
        a = assignments.loc[sid]
        acq = str(a["selected_acquisition_date"])[:10]
        item = str(a.get("selected_item_id", ""))
        tile = str(a.get("selected_mgrs_tile", ""))
        d = date.fromisoformat(acq)
        start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        epsg = get_epsg(geom)
        g_utm = to_utm(geom, epsg)
        minx,miny,maxx,maxy = g_utm.bounds
        bbox = [minx-20, miny-20, maxx+20, maxy+20]
        payload = {
            "input": {"bounds": {"bbox": bbox, "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                      "data": [{"type":"sentinel-2-l2a", "dataFilter": {"timeRange":{"from":start.isoformat().replace('+00:00','Z'),"to":end.isoformat().replace('+00:00','Z')}, "maxCloudCoverage":100, "mosaickingOrder":"leastCC"},
                                "processing":{"upsampling":"NEAREST","downsampling":"NEAREST"}}]},
            "output":{"resx":10,"resy":10,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
            "evalscript":EVALSCRIPT
        }
        response = sess.post(PROCESS_URL, json=payload, headers={"Authorization":f"Bearer {token}","Accept":"image/tiff"}, timeout=180)
        if not response.ok:
            rows.append({"sample_id":sid,"class_id":props.get("class_id"),"class_name":props.get("class_name"),"date_requested":acq,"assigned_tile":tile,"assigned_item_id":item,"status":f"HTTP_{response.status_code}","detail":response.text[:250]})
            print(f"[{idx}/{len(SAMPLE_IDS)}] {sid}: API error HTTP {response.status_code}")
            continue
        with MemoryFile(response.content) as mem:
            with mem.open() as ds:
                arr = ds.read().astype(np.float32)
                b04, sclf, dm = arr[0], arr[1], arr[2]
                scl = np.rint(sclf).astype(np.int16)
                geom_ds = to_utm(geom, ds.crs.to_epsg() or epsg)
                inside = geometry_mask([mapping(geom_ds)], out_shape=(ds.height,ds.width), transform=ds.transform, invert=True, all_touched=False)
                if not inside.any():
                    inside = geometry_mask([mapping(geom_ds)], out_shape=(ds.height,ds.width), transform=ds.transform, invert=True, all_touched=True)
                dm_valid = inside & np.isfinite(dm) & (dm > 0)
                row={"sample_id":sid,"class_id":props.get("class_id"),"class_name":props.get("class_name"),"date_requested":acq,"assigned_tile":tile,"assigned_item_id":item,"status":"OK","polygon_pixel_count":int(inside.sum()),"dataMask_valid_pixel_count":int(dm_valid.sum()),"B04_mean_all_valid":float(np.mean(b04[dm_valid])) if dm_valid.any() else np.nan,"scl_histogram_all_valid":";".join(f"{c}:{int(np.sum(dm_valid & (scl==c)))}" for c in range(12)),"dominant_scl_class":int(np.bincount(scl[dm_valid].clip(0,11), minlength=12).argmax()) if dm_valid.any() else None}
                for c in range(12):
                    row[f"scl_{c}_count"] = int(np.sum(dm_valid & (scl==c)))
                rows.append(row)
                hist = {c: int(np.sum(dm_valid & (scl==c))) for c in range(12) if np.sum(dm_valid & (scl==c)) > 0}
                print(f"[{idx}/{len(SAMPLE_IDS)}] {sid} {props.get('class_name')} date={acq} tile-assigned={tile} | dataMask={int(dm_valid.sum())} | SCL={hist}")
        time.sleep(0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\nOutput CSV:", OUTPUT_CSV)
    print("Band SCL: 0 no data, 1 saturated/defective, 2 dark area, 3 cloud shadow, 4 vegetation, 5 not vegetated, 6 water, 7 unclassified, 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus, 11 snow/ice.")

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
