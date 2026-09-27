import requests
import geopandas as gpd
from io import BytesIO
from shapely.geometry import shape

# ============================================================
# KONFIGURASI
# ============================================================

KODE_KECAMATAN = "3525040"
NAMA_KECAMATAN = "Menganti"
NAMA_KABUPATEN = "Gresik"

OUTPUT_PATH = "data/geojson/menganti.geojson"

# Layer batas desa dari Geoportal Kementerian Pertanian
URL = (
    "https://geoportal.pertanian.go.id/arcgis/rest/services/"
    "Batas_Administrasi/MapServer/0/query"
)

PARAMS = {
    "where": "WADMKC = 'Menganti'",
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
}

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================

print("Mengambil data batas desa Kecamatan Menganti...")

response = requests.get(URL, params=PARAMS, timeout=120)
response.raise_for_status()

geojson_data = response.json()

print(
    f"Feature yang ditemukan: "
    f"{len(geojson_data.get('features', []))}"
)

# ============================================================
# 2. BACA KE GEOPANDAS
# ============================================================

gdf = gpd.GeoDataFrame.from_features(
    geojson_data["features"],
    crs="EPSG:4326"
)

if gdf.empty:
    raise ValueError(
        "Tidak ada polygon yang ditemukan untuk Kecamatan Menganti."
    )

print("\nKolom:")
print(gdf.columns.tolist())

# ============================================================
# 3. VALIDASI WILAYAH
# ============================================================

if "WADMKC" in gdf.columns:
    print("\nKecamatan yang ditemukan:")
    print(gdf["WADMKC"].unique())

# ============================================================
# 4. GABUNGKAN SEMUA DESA
# ============================================================

print("\nMenggabungkan polygon desa...")

geometry_menganti = gdf.geometry.union_all()

# ============================================================
# 5. BUAT GEOJSON FINAL
# ============================================================

hasil = gpd.GeoDataFrame(
    {
        "KDCBPS": [KODE_KECAMATAN],
        "WADMKC": [NAMA_KECAMATAN],
        "WADMKK": [NAMA_KABUPATEN],
        "geometry": [geometry_menganti],
    },
    crs="EPSG:4326",
)

# ============================================================
# 6. SIMPAN
# ============================================================

hasil.to_file(
    OUTPUT_PATH,
    driver="GeoJSON"
)

print("\n========================================")
print("BERHASIL")
print("========================================")
print(f"Output : {OUTPUT_PATH}")
print(f"CRS    : {hasil.crs}")
print(f"Jumlah feature : {len(hasil)}")
print(f"Kecamatan      : {hasil.iloc[0]['WADMKC']}")
print(f"Kabupaten      : {hasil.iloc[0]['WADMKK']}")
print(f"Kode BPS       : {hasil.iloc[0]['KDCBPS']}")
print("========================================")