from __future__ import annotations

"""Compare Sentinel Hub SCL outcomes across available candidate dates for selected samples.

This is a small diagnostic only. It does not change the feature CSV or label dataset,
not does it save TIFFs. Process API requests use a full UTC day window and can mosaic
available Sentinel-2 L2A data for that date; candidate item/tile columns are metadata
for interpreting the result, not a guarantee that Process API locks to that exact item.
"""
import os, json, math, time
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
ENV_PATH = PROJECT_DIR / '.env'
SAMPLES_PATH = PROJECT_DIR / 'data' / 'geojson' / 'dataset_6kelas.geojson'
COVERAGE_PATH = PROJECT_DIR / 'outputs' / 'sentinel2_search' / 'audit_footprint_coverage_detail.csv'
FEATURES_PATH = PROJECT_DIR / 'outputs' / 'sentinel2_features_csv' / 'fitur_sentinel2a_l2a_6kelas.csv'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'sentinel2_features_csv'
OUTPUT_PATH = OUTPUT_DIR / 'diagnose_alternative_dates_scl.csv'
TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
PROCESS_URL = 'https://sh.dataspace.copernicus.eu/process/v1'
SAMPLE_IDS = ['JTM0051', 'JTM0074', 'JTM0123', 'JTM0201', 'JTM0265', 'JTM0151']
CLEAR_CLASSES = {4, 5, 6}

EVALSCRIPT = '''//VERSION=3
function setup() {
  return { input: ["B04", "SCL", "dataMask"], output: { bands: 3, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) { return [sample.B04, sample.SCL, sample.dataMask]; }
'''

def get_token(session):
    load_dotenv(ENV_PATH, override=False)
    cid = os.getenv('COPERNICUS_CLIENT_ID', '').strip()
    secret = os.getenv('COPERNICUS_CLIENT_SECRET', '').strip()
    if not cid or not secret:
        raise RuntimeError(f'Kredensial OAuth tidak ditemukan pada {ENV_PATH}; isi file .env secara lokal.')
    r = session.post(TOKEN_URL, data={'grant_type':'client_credentials','client_id':cid,'client_secret':secret}, timeout=45)
    if not r.ok:
        raise RuntimeError(f'OAuth gagal (HTTP {r.status_code}); detail rahasia tidak ditampilkan.')
    token = r.json().get('access_token')
    if not token:
        raise RuntimeError('Respons OAuth tidak memuat access_token.')
    return token

def get_utm_epsg(geom):
    lon, lat = geom.centroid.x, geom.centroid.y
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32700 + zone) if lat < 0 else (32600 + zone)

def project_geometry(geom, epsg):
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    return shapely_transform(tr.transform, geom)

def process_day(session, token, geom, acq_date):
    epsg = get_utm_epsg(geom)
    geom_utm = project_geometry(geom, epsg)
    minx, miny, maxx, maxy = geom_utm.bounds
    pad = 20.0
    bbox = [minx-pad, miny-pad, maxx+pad, maxy+pad]
    day = date.fromisoformat(str(acq_date)[:10])
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    payload = {
        'input': {'bounds': {'bbox': bbox, 'properties': {'crs': f'http://www.opengis.net/def/crs/EPSG/0/{epsg}'}},
                  'data': [{'type':'sentinel-2-l2a','dataFilter': {'timeRange': {'from':start.isoformat().replace('+00:00','Z'), 'to':end.isoformat().replace('+00:00','Z')}, 'maxCloudCoverage':100, 'mosaickingOrder':'leastCC'}, 'processing': {'upsampling':'NEAREST','downsampling':'NEAREST'}}]},
        'output': {'resx':10,'resy':10,'responses':[{'identifier':'default','format':{'type':'image/tiff'}}]},
        'evalscript': EVALSCRIPT
    }
    r = session.post(PROCESS_URL, json=payload, headers={'Authorization':f'Bearer {token}','Accept':'image/tiff'}, timeout=180)
    if not r.ok:
        return {'status': f'HTTP_{r.status_code}', 'error': f'Process API HTTP {r.status_code}', 'dataMask_valid':0, 'clear_pixels':0, 'scl_histogram':''}
    with MemoryFile(r.content) as mem:
        with mem.open() as ds:
            arr = ds.read().astype(np.float32)
            b04, sclf, dm = arr[0], arr[1], arr[2]
            scl = np.rint(sclf).astype(np.int16)
            geom_ds = project_geometry(geom, ds.crs.to_epsg() or epsg)
            inside = geometry_mask([mapping(geom_ds)], out_shape=(ds.height, ds.width), transform=ds.transform, invert=True, all_touched=False)
            if not inside.any():
                inside = geometry_mask([mapping(geom_ds)], out_shape=(ds.height, ds.width), transform=ds.transform, invert=True, all_touched=True)
            valid = inside & np.isfinite(dm) & (dm > 0)
            hist = {int(c):int(np.sum(valid & (scl == c))) for c in range(12) if np.any(valid & (scl == c))}
            clear = valid & np.isfinite(b04) & np.isin(scl, list(CLEAR_CLASSES))
            return {'status':'OK', 'error':'', 'raster_width':ds.width, 'raster_height':ds.height,
                    'polygon_pixels':int(inside.sum()), 'dataMask_valid':int(valid.sum()), 'clear_pixels':int(clear.sum()),
                    'clear_fraction_of_valid':float(clear.sum()/valid.sum()) if valid.sum() else 0.0,
                    'scl_histogram':';'.join(f'{k}:{v}' for k,v in sorted(hist.items())),
                    'dominant_scl_class':max(hist, key=hist.get) if hist else None,
                    'B04_min':float(np.nanmin(b04[valid])) if valid.any() else np.nan,
                    'B04_max':float(np.nanmax(b04[valid])) if valid.any() else np.nan}

def main():
    for p in [SAMPLES_PATH, COVERAGE_PATH, FEATURES_PATH]:
        if not p.exists(): raise FileNotFoundError(f'File diperlukan tidak ditemukan: {p}')
    fc = json.loads(SAMPLES_PATH.read_text(encoding='utf-8'))
    features = {str(f.get('properties',{}).get('sample_id')): f for f in fc.get('features',[])}
    coverage = pd.read_csv(COVERAGE_PATH, dtype={'sample_id':str})
    coverage['sample_area_covered_pct'] = pd.to_numeric(coverage['sample_area_covered_pct'], errors='coerce')
    coverage['cloud_cover_pct'] = pd.to_numeric(coverage['cloud_cover_pct'], errors='coerce')
    coverage = coverage[coverage['sample_area_covered_pct'] >= 99.5].copy()
    prior = pd.read_csv(FEATURES_PATH, dtype={'sample_id':str})
    prior = prior.set_index('sample_id', drop=False)
    missing = [s for s in SAMPLE_IDS if s not in features or s not in prior.index]
    if missing: raise RuntimeError(f'Sampel tidak ada di input: {missing}')
    sess = requests.Session()
    token = get_token(sess)
    rows=[]
    total=0
    plan={}
    for sid in SAMPLE_IDS:
        rows_sid = coverage[coverage['sample_id'] == sid].copy()
        if rows_sid.empty:
            plan[sid] = []
            continue
        current_date = str(prior.loc[sid].get('acquisition_date_requested',''))[:10]
        rows_sid['acquisition_date'] = rows_sid['acquisition_date'].astype(str).str[:10]
        date_rows=[]
        for d, g in rows_sid.groupby('acquisition_date'):
            date_rows.append({'date':d,'cloud':float(g['cloud_cover_pct'].min()), 'item_ids':'|'.join(g['item_id'].astype(str).unique()), 'tiles':'|'.join(g['mgrs_tile'].astype(str).unique()), 'current': d == current_date})
        date_rows.sort(key=lambda x: (not x['current'], x['cloud']))
        plan[sid] = date_rows[:5]
        total += len(plan[sid])
    print('=== Perbandingan tanggal alternatif untuk sampel diagnostik ===')
    print('Sampel:', ', '.join(SAMPLE_IDS))
    print('Permintaan dibatasi maksimal 5 tanggal kandidat per sampel (berdasarkan footprint >=99.5%).')
    print('Catatan: Process API diminta untuk satu hari penuh; tile/item di CSV adalah metadata kandidat, bukan kunci item yang dipaksakan ke Process API.')
    print('Mengautentikasi OAuth...')
    print('OAuth berhasil; kredensial/token tidak ditampilkan.')
    idx=0
    for sid in SAMPLE_IDS:
        feat=features[sid]; props=feat.get('properties') or {}; geom=shape(feat['geometry'])
        if not plan[sid]:
            print(f'{sid}: tidak punya tanggal kandidat dengan cakupan penuh')
            continue
        for cand in plan[sid]:
            idx += 1
            result = process_day(sess, token, geom, cand['date'])
            row={'sample_id':sid,'class_id':props.get('class_id'),'class_name':props.get('class_name'),
                 'current_assigned_date':str(prior.loc[sid].get('acquisition_date_requested',''))[:10],
                 'candidate_date':cand['date'],'is_current_assigned_date':cand['current'],'scene_cloud_cover_metadata_pct':cand['cloud'],
                 'candidate_tiles':cand['tiles'],'candidate_item_ids':cand['item_ids'], **result}
            rows.append(row)
            print(f'[{idx}/{total}] {sid} {props.get("class_name")} date={cand["date"]} cloud_meta={cand["cloud"]:.2f}% | dataMask={result.get("dataMask_valid",0)} clear={result.get("clear_pixels",0)} SCL={result.get("scl_histogram","")} status={result.get("status")}')
            time.sleep(0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH,index=False,encoding='utf-8-sig')
    print('\nOutput CSV:', OUTPUT_PATH)
    if not df.empty:
        best=df.sort_values(['sample_id','clear_pixels','clear_fraction_of_valid'], ascending=[True,False,False]).drop_duplicates('sample_id')
        print('\nRingkasan: jumlah sampel yang setidaknya satu tanggal kandidatnya menghasilkan clear_pixels > 0:', int(best['clear_pixels'].gt(0).sum()), '/', len(SAMPLE_IDS))
        print('Tidak ada fitur yang ditimpa. Hasil ini hanya diagnostik pemilihan tanggal.')

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f'Gagal: {type(e).__name__}: {e}')
        raise SystemExit(1)
