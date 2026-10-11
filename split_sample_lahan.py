import geopandas as gpd
from pathlib import Path
import zipfile

INPUT = Path("data/training/sample_lahan_auto.shp")
OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baca polygon 100 sample
gdf = gpd.read_file(INPUT)

print("Total polygon:", len(gdf))
print("\nDistribusi label:")
print(gdf["label"].value_counts())

# Pisahkan
gdf_sawah = gdf[gdf["label"] == "sawah"].copy()
gdf_non_sawah = gdf[gdf["label"] == "non_sawah"].copy()

print("\nSawah     :", len(gdf_sawah))
print("Non-sawah :", len(gdf_non_sawah))

# ------------------------------------------------------------
# Simpan SHP sawah
# ------------------------------------------------------------

sawah_dir = OUTPUT_DIR / "sawah"
sawah_dir.mkdir(exist_ok=True)

sawah_shp = sawah_dir / "sawah.shp"

gdf_sawah.to_file(
    sawah_shp,
    driver="ESRI Shapefile"
)

# ------------------------------------------------------------
# Simpan SHP non-sawah
# ------------------------------------------------------------

non_dir = OUTPUT_DIR / "non_sawah"
non_dir.mkdir(exist_ok=True)

non_sawah_shp = non_dir / "non_sawah.shp"

gdf_non_sawah.to_file(
    non_sawah_shp,
    driver="ESRI Shapefile"
)

# ------------------------------------------------------------
# Fungsi ZIP semua komponen SHP
# ------------------------------------------------------------

def zip_shapefile(shp_path, zip_path):
    extensions = [
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg"
    ]

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for ext in extensions:
            file_path = shp_path.with_suffix(ext)

            if file_path.exists():
                z.write(
                    file_path,
                    arcname=file_path.name
                )


zip_shapefile(
    sawah_shp,
    OUTPUT_DIR / "sawah.zip"
)

zip_shapefile(
    non_sawah_shp,
    OUTPUT_DIR / "non_sawah.zip"
)

print("\n=== SELESAI ===")
print("Sawah SHP      :", sawah_shp)
print("Sawah ZIP      :", OUTPUT_DIR / "sawah.zip")
print("Non-sawah SHP  :", non_sawah_shp)
print("Non-sawah ZIP  :", OUTPUT_DIR / "non_sawah.zip")