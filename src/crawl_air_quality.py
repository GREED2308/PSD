from pathlib import Path
import os
import requests
import geopandas as gpd
from shapely.geometry import shape
from shapely import force_2d
from dotenv import load_dotenv

# =========================
# KONFIGURASI
# =========================

GEOJSON_PATH = Path("data/geojson/menganti.geojson")

OUTPUT_CO = Path("data/processed/menganti_CO_daily.csv")
OUTPUT_SO2 = Path("data/processed/menganti_SO2_daily.csv")

START_DATE = "2025-08-31T00:00:00Z"
END_DATE   = "2026-09-01T00:00:00Z"

BASE_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"

load_dotenv()

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

# =========================
# OAUTH
# =========================

def get_token():
    token_url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/protocol/openid-connect/token"
    )

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    response = requests.post(token_url, data=data)
    response.raise_for_status()

    return response.json()["access_token"]


# =========================
# GEOJSON
# =========================

def load_geometry():
    gdf = gpd.read_file(GEOJSON_PATH)

    # Pastikan CRS WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Hilangkan koordinat Z
    gdf["geometry"] = gdf.geometry.apply(force_2d)

    geometry = gdf.geometry.union_all()

    print("=== WILAYAH ===")
    print(f"Kode BPS   : {gdf.iloc[0].get('KDCBPS')}")
    print(f"Kecamatan  : {gdf.iloc[0].get('WADMKC')}")
    print(f"Kabupaten  : {gdf.iloc[0].get('WADMKK')}")
    print(f"CRS        : {gdf.crs}")
    print(f"Geometry   : {geometry.geom_type}")

    return geometry


# =========================
# EVALSCRIPT
# =========================

def build_evalscript(band_name):
    return f"""
//VERSION=3

function setup() {{
    return {{
        input: [{{
            bands: ["{band_name}", "dataMask"]
        }}],
        output: [
            {{
                id: "index",
                bands: 1,
                sampleType: "FLOAT32"
            }},
            {{
                id: "dataMask",
                bands: 1,
                sampleType: "UINT8"
            }}
        ]
    }};
}}

function evaluatePixel(sample) {{
    return {{
        index: [sample.{band_name}],
        dataMask: [sample.dataMask]
    }};
}}
"""


# =========================
# REQUEST STATISTICS
# =========================

def get_statistics(token, geometry, band_name):

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "geometry": geometry.__geo_interface__,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },
            "data": [
                {
                    "type": "sentinel-5p-l2",
                    "dataFilter": {},
                    "processing": {
                        "minQa": 50
                    }
                }
            ]
        },
        "aggregation": {
            "timeRange": {
                "from": START_DATE,
                "to": END_DATE
            },
            "aggregationInterval": {
                "of": "P1D"
            },
            "evalscript": build_evalscript(band_name),
            "resx": 0.01,
            "resy": 0.01
        }
    }

    url = BASE_URL

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("=== RESPONSE ERROR ===")
        print(response.text)

    response.raise_for_status()

    return response.json()


# =========================
# PARSE RESULT
# =========================

def parse_statistics(result, pollutant):

    rows = []

    for interval in result.get("data", []):

        interval_from = interval.get("interval", {}).get("from")

        if not interval_from:
            continue

        date = interval_from[:10]

        value = None

        try:
            value = (
                interval
                .get("outputs", {})
                .get("index", {})
                .get("bands", {})
                .get("B0", {})
                .get("stats", {})
                .get("mean")
            )
        except Exception:
            value = None

        rows.append({
            "date": date,
            pollutant: value
        })

    return rows


# =========================
# MAIN
# =========================

def main():

    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(
            "COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET belum ada di .env"
        )

    token = get_token()

    print("OAuth berhasil.")

    geometry = load_geometry()

    # =========================
    # CO
    # =========================

    print("\nMengambil CO...")

    co_result = get_statistics(
        token,
        geometry,
        "CO"
    )

    co_rows = parse_statistics(
        co_result,
        "CO"
    )

    import pandas as pd

    df_co = pd.DataFrame(co_rows)

    if not df_co.empty:
        df_co["date"] = pd.to_datetime(df_co["date"])
        df_co = df_co.sort_values("date").reset_index(drop=True)

        df_co = df_co[
            (df_co["date"] >= "2025-08-31") &
            (df_co["date"] <= "2026-08-31")
        ]

    OUTPUT_CO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_co.to_csv(
        OUTPUT_CO,
        index=False
    )

    print("\nSELESAI CRAWL CO")
    print(f"Tanggal awal  : {df_co['date'].min()}")
    print(f"Tanggal akhir : {df_co['date'].max()}")
    print(f"Jumlah baris  : {len(df_co)}")
    print(f"CO valid      : {df_co['CO'].notna().sum()}")
    print(f"CO missing    : {df_co['CO'].isna().sum()}")
    print(f"CSV           : {OUTPUT_CO}")

    # =========================
    # SO2
    # =========================

    print("\nMengambil SO2...")

    so2_result = get_statistics(
        token,
        geometry,
        "SO2"
    )

    so2_rows = parse_statistics(
        so2_result,
        "SO2"
    )

    df_so2 = pd.DataFrame(so2_rows)

    if not df_so2.empty:
        df_so2["date"] = pd.to_datetime(df_so2["date"])
        df_so2 = df_so2.sort_values("date").reset_index(drop=True)

        df_so2 = df_so2[
            (df_so2["date"] >= "2025-08-31") &
            (df_so2["date"] <= "2026-08-31")
        ]

    OUTPUT_SO2.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_so2.to_csv(
        OUTPUT_SO2,
        index=False
    )

    print("\nSELESAI CRAWL SO2")
    print(f"Tanggal awal  : {df_so2['date'].min()}")
    print(f"Tanggal akhir : {df_so2['date'].max()}")
    print(f"Jumlah baris  : {len(df_so2)}")
    print(f"SO2 valid     : {df_so2['SO2'].notna().sum()}")
    print(f"SO2 missing   : {df_so2['SO2'].isna().sum()}")
    print(f"CSV           : {OUTPUT_SO2}")


if __name__ == "__main__":
    main()