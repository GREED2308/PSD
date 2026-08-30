import requests
from pathlib import Path

URL = (
    "https://gis.bnpb.go.id/server/rest/services/"
    "inarisk/batas_administrasi/MapServer/4/query"
)

params = {
    "where": (
        "WADMKD='Sidojangkung' "
        "AND WADMKC='Menganti' "
        "AND WADMKK='Gresik'"
    ),
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
}

response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()

data = response.json()

output = Path("data/geojson/sidojangkung.geojson")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(
    response.text,
    encoding="utf-8"
)

print(f"Disimpan ke: {output}")
print(f"Jumlah feature: {len(data.get('features', []))}")
