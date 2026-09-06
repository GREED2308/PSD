import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


# ============================================================
# CONFIG
# ============================================================

GEOJSON_PATH = Path("data/geojson/sidojangkung.geojson")
OUTPUT_PATH = Path("data/processed/sidojangkung_air_quality_daily.csv")

START_DATE = "2025-09-01T00:00:00Z"
END_DATE = "2026-08-31T00:00:00Z"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

STATS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"


# ============================================================
# POLLUTANTS
# ============================================================

POLLUTANTS = {
    "no2": {
        "band": "NO2",
        "qa": 75,
    },
    "co": {
        "band": "CO",
        "qa": 50,
    },
    "o3": {
        "band": "O3",
        "qa": 50,
    },
    "so2": {
        "band": "SO2",
        "qa": 50,
    },
}


# ============================================================
# EVALSCRIPT
# ============================================================

def build_evalscript(band: str) -> str:
    return f"""
//VERSION=3

function setup() {{
    return {{
        input: [{{
            bands: ["{band}", "dataMask"],
            units: ["MOL_M2", "DN"]
        }}],
        output: [
            {{
                id: "pollutant",
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

function evaluatePixel(samples) {{
    return {{
        pollutant: [samples.{band}],
        dataMask: [samples.dataMask]
    }};
}}
"""


# ============================================================
# OAUTH
# ============================================================

def create_session() -> OAuth2Session:
    load_dotenv()

    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "COPERNICUS_CLIENT_ID atau COPERNICUS_CLIENT_SECRET "
            "belum tersedia di .env"
        )

    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    oauth.fetch_token(
        token_url=TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )

    print("OAuth berhasil.")
    return oauth


# ============================================================
# LOAD GEOJSON
# ============================================================

def load_geometry():
    gdf = gpd.read_file(GEOJSON_PATH)

    if gdf.empty:
        raise RuntimeError("GeoJSON kosong.")

    if gdf.crs is None:
        raise RuntimeError("GeoJSON tidak memiliki CRS.")

    gdf = gdf.to_crs("EPSG:4326")

    row = gdf.iloc[0]

    print("Wilayah   :", row["WADMKD"])
    print("Kecamatan :", row["WADMKC"])
    print("Kabupaten :", row["WADMKK"])

    return gdf.geometry.iloc[0].__geo_interface__


# ============================================================
# REQUEST ONE POLLUTANT
# ============================================================

def request_pollutant(
    oauth: OAuth2Session,
    geometry: dict,
    name: str,
    band: str,
    qa: int,
) -> dict:

    print(f"\nMengambil {name.upper()}...")

    body = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/"
                        "OGC/1.3/CRS84"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-5p-l2",
                    "dataFilter": {},
                    "processing": {
                        "minQa": qa,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": START_DATE,
                "to": END_DATE,
            },
            "aggregationInterval": {
                "of": "P1D"
            },
            "lastIntervalBehavior": "SHORTEN",
            "evalscript": build_evalscript(band),
            "resx": 0.01,
            "resy": 0.01,
        },
        "calculations": {
            "pollutant": {
                "statistics": {
                    "default": {}
                }
            }
        },
    }

    response = oauth.post(
        STATS_URL,
        json=body,
        timeout=600,
    )

    print("HTTP status:", response.status_code)

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_result(result: dict, prefix: str) -> pd.DataFrame:
    rows = []

    for item in result.get("data", []):
        interval = item["interval"]

        stats = (
            item.get("outputs", {})
            .get("pollutant", {})
            .get("bands", {})
            .get("B0", {})
            .get("stats", {})
        )

        mean = stats.get("mean")
        minimum = stats.get("min")
        maximum = stats.get("max")
        stdev = stats.get("stDev")

        def finite_or_none(value):
            if value is None:
                return None

            try:
                value = float(value)
            except (TypeError, ValueError):
                return None

            return value if math.isfinite(value) else None

        mean = finite_or_none(mean)
        minimum = finite_or_none(minimum)
        maximum = finite_or_none(maximum)
        stdev = finite_or_none(stdev)

        rows.append({
            "date": interval["from"][:10],
            f"{prefix}_mean": mean,
            f"{prefix}_min": minimum,
            f"{prefix}_max": maximum,
            f"{prefix}_stdev": stdev,
            f"{prefix}_sample_count": stats.get("sampleCount", 0),
            f"{prefix}_nodata_count": stats.get("noDataCount", 0),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    oauth = create_session()
    geometry = load_geometry()

    frames = []

    for name, config in POLLUTANTS.items():
        result = request_pollutant(
            oauth=oauth,
            geometry=geometry,
            name=name,
            band=config["band"],
            qa=config["qa"],
        )

        df = parse_result(
            result=result,
            prefix=name,
        )

        print(
            f"{name.upper()}: "
            f"{len(df)} hari, "
            f"{df[f'{name}_mean'].notna().sum()} valid"
        )

        frames.append(df)

    # Gabungkan semua polutan berdasarkan tanggal
    final_df = frames[0]

    for df in frames[1:]:
        final_df = final_df.merge(
            df,
            on="date",
            how="outer",
        )

    final_df = final_df.sort_values("date")
    final_df = final_df.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== SELESAI ===")
    print("Jumlah hari:", len(final_df))
    print("Tanggal awal:", final_df["date"].min())
    print("Tanggal akhir:", final_df["date"].max())
    print("CSV:", OUTPUT_PATH)

    print("\nJumlah valid:")
    for pollutant in POLLUTANTS:
        print(
            pollutant.upper(),
            ":",
            final_df[f"{pollutant}_mean"].notna().sum(),
        )


if __name__ == "__main__":
    main()