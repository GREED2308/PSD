import numpy as np
import rasterio
from scipy import ndimage
from pyproj import Transformer

path = (
    r".\outputs\classification_raster_v6_aoi40km_offshore"
    r"\classification_SVM_RBF_quality_screened_10m_aoi40km_offshore.tif"
)

with rasterio.open(path) as src:
    raster = src.read(1)
    transform = src.transform
    crs = src.crs

mask = raster == 6
pixel_count = int(mask.sum())
area_km2 = pixel_count * 100 / 1_000_000

print("=== AUDIT KELAS 6: DANAU / RANU ===")
print(f"Jumlah piksel kelas 6 : {pixel_count:,}")
print(f"Luas prediksi         : {area_km2:.2f} km²")

# Kelompokkan piksel yang bersentuhan, termasuk secara diagonal.
structure = np.ones((3, 3), dtype=np.uint8)
labels, n_components = ndimage.label(mask, structure=structure)
sizes = np.bincount(labels.ravel())[1:]

print(f"Jumlah kelompok       : {n_components:,}")

if n_components:
    top_ids = (np.argsort(sizes)[::-1][:10] + 1).tolist()
    centers = ndimage.center_of_mass(mask, labels, top_ids)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    print("\n10 kelompok terbesar:")
    print("No | Piksel | Luas (km²) | Longitude | Latitude")

    for rank, (component_id, center) in enumerate(
        zip(top_ids, centers), start=1
    ):
        row, col = center
        x, y = transform * (col + 0.5, row + 0.5)
        lon, lat = to_wgs84.transform(x, y)
        count = int(sizes[component_id - 1])
        area = count * 100 / 1_000_000

        print(
            f"{rank:2} | {count:7,} | {area:10.3f} | "
            f"{lon:10.5f} | {lat:9.5f}"
        )
