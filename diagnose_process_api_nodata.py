from __future__ import annotations
import json, os, sys
from pathlib import Path
import requests
from dotenv import load_dotenv
import numpy as np
import rasterio
from rasterio.io import MemoryFile

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / '.env'
SAMPLES_PATH = PROJECT_DIR / 'data' / 'geojson' / 'dataset_6kelas.geojson'
OUT_DIR = PROJECT_DIR / 'outputs' / 'sentinel2_test' / 'diagnostic'
TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
PROCESS_URL = 'https://sh.dataspace.copernicus.eu/process/v1'
EVALSCRIPT = '''//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.SCL, sample.dataMask];
}
'''
CASES = [
    ('exact_1min', '2026-09-18T02:31:00Z', '2026-09-18T02:32:00Z'),
    ('full_day', '2026-09-18T00:00:00Z', '2026-09-19T00:00:00Z'),
    ('wider_period', '2026-09-15T00:00:00Z', '2026-10-02T00:00:00Z'),
]

def token():
    load_dotenv(ENV_PATH)
    cid = os.getenv('COPERNICUS_CLIENT_ID', '').strip()
    secret = os.getenv('COPERNICUS_CLIENT_SECRET', '').strip()
    if not cid or not secret:
        raise RuntimeError(f'Kredensial tidak terbaca dari {ENV_PATH}; periksa nama variabel di .env.')
    r = requests.post(TOKEN_URL, data={'grant_type':'client_credentials','client_id':cid,'client_secret':secret}, timeout=45)
    if not r.ok:
        try: msg = r.json().get('error_description') or r.json().get('error')
        except Exception: msg = r.text[:300]
        raise RuntimeError(f'OAuth gagal HTTP {r.status_code}: {msg}')
    return r.json()['access_token']

def get_bbox():
    with SAMPLES_PATH.open(encoding='utf-8') as f: fc = json.load(f)
    feat = next((x for x in fc['features'] if str(x.get('properties',{}).get('sample_id')) == 'JTM0001'), None)
    if not feat: raise RuntimeError('JTM0001 tidak ditemukan di dataset.')
    p = feat['properties']; lon=float(p['centroid_lon']); lat=float(p['centroid_lat']); h=0.002
    return [lon-h,lat-h,lon+h,lat+h], p

def main():
    print('=== Diagnostik NoData Sentinel Hub Process API ===')
    print(f'Proyek: {PROJECT_DIR}')
    print(f'Sampel: JTM0001 (bbox kecil di sekitar centroid)')
    bbox, props = get_bbox()
    print('BBox CRS84:', ', '.join(f'{v:.7f}' for v in bbox))
    tok = token(); print('OAuth: berhasil (secret/token tidak ditampilkan)')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, start, end in CASES:
        payload = {
          'input': {'bounds': {'bbox': bbox, 'properties': {'crs':'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}},
                    'data': [{'type':'sentinel-2-l2a','dataFilter': {'timeRange': {'from':start,'to':end}, 'maxCloudCoverage':100, 'mosaickingOrder':'leastCC'},
                              'processing': {'upsampling':'NEAREST','downsampling':'NEAREST'}}]},
          'output': {'width':48,'height':48,'responses':[{'identifier':'default','format':{'type':'image/tiff'}}]},
          'evalscript': EVALSCRIPT
        }
        print(f'\n[{name}] {start} -> {end}')
        try:
            r = requests.post(PROCESS_URL, json=payload, headers={'Authorization':f'Bearer {tok}','Accept':'image/tiff'}, timeout=180)
        except requests.RequestException as e:
            print('Request gagal:', str(e)); continue
        if not r.ok:
            try:
                x=r.json(); msg=x.get('error',{}).get('message') or x.get('message') or str(x)
            except Exception: msg=r.text[:500]
            print(f'HTTP {r.status_code}: {msg}'); continue
        try:
            with MemoryFile(r.content) as mem:
                with mem.open() as ds:
                    arr=ds.read()
                    if ds.count >= 6:
                        dm=arr[5]
                        valid=np.isfinite(dm)&(dm>0)
                        print(f'TIFF: {ds.width}x{ds.height}, bands={ds.count}, CRS={ds.crs}, bytes={len(r.content)}')
                        print(f'dataMask valid: {int(valid.sum())}/{valid.size}')
                        for i, label in enumerate(['B02','B03','B04','B08','SCL','dataMask']):
                            band=arr[i]
                            print(f'  {label}: min={np.nanmin(band):.6g}, max={np.nanmax(band):.6g}, unique<=8={np.unique(band)[:8].tolist()}')
                    else:
                        print(f'TIFF count band tidak sesuai: {ds.count}; bytes={len(r.content)}')
                    out=OUT_DIR/f'JTM0001_{name}.tif'
                    out.write_bytes(r.content)
                    print(f'Disimpan: {out}')
        except Exception as e:
            print('Respons HTTP sukses tetapi tidak dapat dibaca sebagai TIFF:', str(e), 'Content-Type=', r.headers.get('Content-Type'))
    print('\nSelesai. Ini tes diagnostik, bukan data untuk training.')
if __name__ == '__main__':
    try: main()
    except (OSError, ValueError, RuntimeError, requests.RequestException) as e:
        print('Gagal:', e); sys.exit(1)
