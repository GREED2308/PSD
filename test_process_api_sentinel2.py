from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
SAMPLES_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_test"
OUTPUT_PATH = OUTPUT_DIR / "JTM0001_S2A_L2A_20260918_test.tif"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# Short, precise interval of the catalog acquisition:
# S2A_MSIL2A_20260918T023131...T49MGN
TIME_FROM = "2026-09-18T02:31:00Z"
TIME_TO = "2026-09-18T02:32:00Z"

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
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


def get_token() -> str:
    load_dotenv(ENV_PATH)
    client_id = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            f"Kredensial COPERNICUS_CLIENT_ID/SECRET tidak terbaca dari {ENV_PATH}."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=45,
    )
    if not response.ok:
        # Do not print request headers, credentials, or token.
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except (ValueError, AttributeError):
            detail = response.text[:500]
        raise RuntimeError(f"OAuth gagal (HTTP {response.status_code}): {detail}")

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth merespons tanpa access_token.")
    return token


def find_sample_bbox() -> tuple[list[float], dict]:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"GeoJSON sampel tidak ditemukan: {SAMPLES_PATH}")

    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        collection = json.load(f)

    sample = next(
        (feature for feature in collection.get("features", [])
         if str(feature.get("properties", {}).get("sample_id")) == "JTM0001"),
        None,
    )
    if sample is None:
        raise RuntimeError("Sampel JTM0001 tidak ditemukan di dataset_6kelas.geojson.")

    props = sample.get("properties", {})
    try:
        lon = float(props["centroid_lon"])
        lat = float(props["centroid_lat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Centroid JTM0001 tidak valid atau tidak tersedia.") from exc

    # Small 0.004-degree test box around the sample centroid (~440 m across).
    half = 0.002
    bbox = [lon - half, lat - half, lon + half, lat + half]
    return bbox, props


def main() -> int:
    print("=== Uji kecil Sentinel Hub Process API ===")
    print(f"Proyek       : {PROJECT_DIR}")
    print(f"File .env    : {ENV_PATH}")
    print(f"Sampel uji   : JTM0001")
    print(f"Rentang UTC  : {TIME_FROM} sampai {TIME_TO}")
    print("Band output  : B02, B03, B04, B08, SCL, dataMask")
    print("Resolusi uji : 48 x 48 piksel (permintaan kecil; bukan produk final)")

    try:
        bbox, props = find_sample_bbox()
        print(f"Kelas sampel : {props.get('class_name', 'tidak diketahui')}")
        print("BBox CRS84   : " + ", ".join(f"{v:.7f}" for v in bbox))
        token = get_token()
        print("OAuth        : berhasil (token tidak ditampilkan)")

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {"from": TIME_FROM, "to": TIME_TO},
                            "maxCloudCoverage": 30,
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
                "width": 48,
                "height": 48,
                "responses": [
                    {"identifier": "default", "format": {"type": "image/tiff"}}
                ],
            },
            "evalscript": EVALSCRIPT,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "image/tiff",
            },
            timeout=180,
        )
        if not response.ok:
            try:
                body = response.json()
                detail = body.get("error", {}).get("message") or body.get("message") or str(body)
            except (ValueError, AttributeError):
                detail = response.text[:1000]
            print(f"Process API gagal (HTTP {response.status_code}).")
            print(f"Detail: {detail}")
            print("Token dan secret tidak ditampilkan.")
            return 2

        content_type = response.headers.get("Content-Type", "(tidak diketahui)")
        content = response.content
        is_tiff = content[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
        if not is_tiff:
            print("Respons HTTP berhasil, tetapi isi respons tidak terlihat seperti TIFF.")
            print(f"Content-Type: {content_type}; bytes: {len(content)}")
            print(content[:500].decode("utf-8", errors="replace"))
            return 3

        OUTPUT_PATH.write_bytes(content)
        print("Process API   : BERHASIL")
        print(f"Content-Type  : {content_type}")
        print(f"Ukuran output : {len(content):,} bytes")
        print(f"GeoTIFF       : {OUTPUT_PATH}")

        try:
            import rasterio
            import numpy as np

            with rasterio.open(OUTPUT_PATH) as ds:
                print(f"Dimensi       : {ds.width} x {ds.height}")
                print(f"Jumlah band   : {ds.count}")
                print(f"CRS           : {ds.crs}")
                print(f"Bounds        : {ds.bounds}")
                if ds.count >= 6:
                    mask = ds.read(6)
                    valid = np.isfinite(mask) & (mask > 0)
                    print(f"Piksel dataMask valid: {int(valid.sum())}/{mask.size}")
                    if valid.sum() == 0:
                        print("PERINGATAN: TIFF terbentuk, tetapi tidak ada piksel dataMask valid pada area/waktu uji.")
        except ImportError:
            print("Catatan: rasterio belum terpasang; metadata internal TIFF belum dibaca.")

        print("Catatan: ini hanya uji koneksi dan data contoh, bukan ekstraksi final semua sampel.")
        return 0

    except requests.RequestException as exc:
        print(f"Kesalahan koneksi: {exc}")
        return 4
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Gagal: {exc}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
