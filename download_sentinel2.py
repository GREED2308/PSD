import os
import json
from pathlib import Path

import geopandas as gpd
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE = Path(__file__).resolve().parent

GEOJSON_PATH = BASE / "data" / "geojson" / "menganti.geojson"
OUTPUT_DIR = BASE / "data" / "sentinel2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scene yang kita pilih dari Copernicus Browser
START_DATE = "2026-10-01T00:00:00Z"
END_DATE   = "2026-10-02T00:00:00Z"

MAX_CLOUD = 20

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"


# ============================================================
# 2. LOAD ENV
# ============================================================

load_dotenv(BASE / ".env")

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET "
        "tidak ditemukan di .env"
    )


# ============================================================
# 3. BACA GEOJSON MENGANTI
# ============================================================

print("Membaca GeoJSON...")

gdf = gpd.read_file(GEOJSON_PATH)

if gdf.empty:
    raise RuntimeError("GeoJSON kosong.")

if gdf.crs is None:
    raise RuntimeError("GeoJSON tidak memiliki CRS.")

print("CRS asli:", gdf.crs)

# UTM Zone 49S, satuan meter
gdf = gdf.to_crs(epsg=32749)

print("CRS output:", gdf.crs)
print("Jumlah geometry:", len(gdf))

geometry = gdf.geometry.union_all()

if geometry.is_empty:
    raise RuntimeError("Geometry AOI kosong.")

geojson_geometry = json.loads(
    gpd.GeoSeries(
        [geometry],
        crs="EPSG:32749"
    ).to_json()
)["features"][0]["geometry"]

minx, miny, maxx, maxy = geometry.bounds

print("BBOX UTM:")
print(f"  west  : {minx:.2f}")
print(f"  south : {miny:.2f}")
print(f"  east  : {maxx:.2f}")
print(f"  north : {maxy:.2f}")

print("Lebar AOI :", maxx - minx, "meter")
print("Tinggi AOI:", maxy - miny, "meter")


# ============================================================
# 4. AUTHENTICATION COPERNICUS
# ============================================================

print("\nMeminta OAuth token...")

client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)

token = oauth.fetch_token(
    token_url=TOKEN_URL,
    client_secret=CLIENT_SECRET,
    include_client_id=True,
)

print("Authentication berhasil.")


# ============================================================
# 5. EVALSCRIPT
# ============================================================

evalscript = """
//VERSION=3

function setup() {
    return {
        input: [
            {
                bands: [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "SCL",
                    "dataMask"
                ]
            }
        ],

        output: {
            bands: 6,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B08,
        sample.SCL,
        sample.dataMask
    ];
}
"""


# ============================================================
# 6. REQUEST SENTINEL-2 L2A
# ============================================================

payload = {
    "input": {
        "bounds": {
            "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/32749"
            },
            "bbox": [
                minx,
                miny,
                maxx,
                maxy
            ],
            "geometry": geojson_geometry
        },

        "data": [
            {
                "type": "sentinel-2-l2a",

                "dataFilter": {
                    "timeRange": {
                        "from": START_DATE,
                        "to": END_DATE
                    },
                    "maxCloudCoverage": MAX_CLOUD,
                    "mosaickingOrder": "leastCC"
                }
            }
        ]
    },

    "output": {
        "resx": 10,
        "resy": 10,

        "responses": [
            {
                "identifier": "default",
                "format": {
                    "type": "image/tiff"
                }
            }
        ]
    },

    "evalscript": evalscript
}


print("\nMengirim request Sentinel-2 L2A...")
print("Tanggal:", START_DATE, "sampai", END_DATE)
print("Max cloud:", MAX_CLOUD, "%")
print("Output: GeoTIFF 10 m")
print("Band: B02, B03, B04, B08, SCL, dataMask")


# ============================================================
# 7. REQUEST
# ============================================================

response = oauth.post(
    PROCESS_URL,
    json=payload,
    headers={
        "Accept": "image/tiff"
    },
    timeout=300
)

print("HTTP status:", response.status_code)

if response.status_code != 200:
    print("\nResponse error:")
    print(response.text)

response.raise_for_status()


# ============================================================
# 8. SIMPAN GEOTIFF
# ============================================================

output_file = OUTPUT_DIR / "menganti_sentinel2_2026-10-01.tif"

with open(output_file, "wb") as f:
    f.write(response.content)

print("\nDownload berhasil.")
print("File:", output_file)
print("Ukuran:", output_file.stat().st_size / (1024 ** 2), "MB")