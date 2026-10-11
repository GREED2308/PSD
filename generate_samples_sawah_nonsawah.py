import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from scipy.ndimage import uniform_filter


# ============================================================
# KONFIGURASI
# ============================================================

RASTER = Path("data/sentinel2/menganti_sentinel2_2026-10-01.tif")

OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHP_NAME = "sample_lahan_auto"

SAWAH_TARGET = 50
NON_SAWAH_TARGET = 50

# Ukuran satu polygon sample.
# 5 pixel x 5 pixel pada raster ~ 10 m/pixel => ~50 x 50 meter.
WINDOW_PIXELS = 5

# Minimum proporsi pixel yang harus memenuhi mask kelas
MIN_CLASS_FRACTION = 0.90

# Jarak minimum antartitik pusat sample.
# 20 pixel ~ 200 meter.
MIN_DISTANCE_PIXELS = 20

RANDOM_SEED = 42


# ============================================================
# 1. BACA SENTINEL-2A
# ============================================================

print("=" * 60)
print("GENERATE SAMPLE SAWAH + NON-SAWAH")
print("=" * 60)

if not RASTER.exists():
    raise FileNotFoundError(f"Raster tidak ditemukan: {RASTER}")

with rasterio.open(RASTER) as src:
    blue = src.read(1).astype("float32")       # B02
    green = src.read(2).astype("float32")      # B03
    red = src.read(3).astype("float32")        # B04
    nir = src.read(4).astype("float32")        # B08
    scl = src.read(5).astype("int16")          # SCL
    data_mask = src.read(6).astype("float32")  # dataMask

    transform = src.transform
    crs = src.crs
    height = src.height
    width = src.width
    nodata = src.nodata

print(f"Raster      : {RASTER}")
print(f"Ukuran      : {width} x {height}")
print(f"CRS         : {crs}")
print(f"Resolusi    : {transform.a:.3f} x {abs(transform.e):.3f} m")


# ============================================================
# 2. HITUNG NDVI
# ============================================================

denominator = nir + red

ndvi = np.divide(
    nir - red,
    denominator,
    out=np.full_like(nir, np.nan, dtype="float32"),
    where=denominator != 0,
)


# ============================================================
# 3. VALID PIXEL
# ============================================================

# dataMask = 1 -> data tersedia
#
# SCL:
# 0 = No Data
# 1 = Saturated/Defective
# 2 = Dark Area/Pixels
# 3 = Cloud Shadow
# 4 = Vegetation
# 5 = Not Vegetated
# 6 = Water
# 7 = Unclassified
# 8 = Cloud Medium Probability
# 9 = Cloud High Probability
# 10 = Thin Cirrus
# 11 = Snow/Ice

valid_base = (
    (data_mask > 0)
    & np.isfinite(ndvi)
    & ~np.isin(scl, [0, 1, 2, 3, 8, 9, 10, 11])
)

print("\nNilai SCL:", np.unique(scl))


# ============================================================
# 4. FUNGSI MEMBUAT KANDIDAT WINDOW
# ============================================================

def candidate_centers(mask, window_pixels=5, min_fraction=0.90):
    """
    Mengembalikan kandidat pusat polygon yang memiliki proporsi
    pixel kelas minimal min_fraction dalam window_pixels x window_pixels.
    """
    fraction = uniform_filter(
        mask.astype("float32"),
        size=window_pixels,
        mode="constant",
        cval=0.0,
    )

    radius = window_pixels // 2

    candidates = (fraction >= min_fraction)

    # Hindari tepi raster agar window lengkap.
    candidates[:radius, :] = False
    candidates[-radius:, :] = False
    candidates[:, :radius] = False
    candidates[:, -radius:] = False

    rows, cols = np.where(candidates)

    return np.column_stack((rows, cols))


# ============================================================
# 5. PILIH PUSAT SAMPLE DENGAN JARAK MINIMUM
# ============================================================

def select_spread_centers(
    candidates,
    target,
    min_distance_pixels=20,
    seed=42,
):
    """
    Memilih kandidat yang tersebar.
    Tidak menjamin 100% tersebar sempurna, tetapi menjaga jarak minimum.
    """
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=int)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    candidates = candidates[order]

    selected = []
    occupied = set()

    # Spatial hash sederhana.
    cell_size = max(1, min_distance_pixels)

    for row, col in candidates:
        cell_r = int(row // cell_size)
        cell_c = int(col // cell_size)

        too_close = False

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (cell_r + dr, cell_c + dc)

                if neighbor not in occupied:
                    continue

                # Cek jarak terhadap seluruh titik yang sudah terpilih
                for sr, sc in selected:
                    distance = np.hypot(row - sr, col - sc)

                    if distance < min_distance_pixels:
                        too_close = True
                        break

                if too_close:
                    break

            if too_close:
                break

        if not too_close:
            selected.append((int(row), int(col)))
            occupied.add((cell_r, cell_c))

            if len(selected) >= target:
                break

    return np.asarray(selected, dtype=int)


def choose_samples(mask_configs, target, label, seed):
    """
    Mencoba beberapa konfigurasi threshold sampai jumlah sample
    mencukupi.
    """
    for idx, (mask, description) in enumerate(mask_configs, start=1):
        candidates = candidate_centers(
            mask,
            window_pixels=WINDOW_PIXELS,
            min_fraction=MIN_CLASS_FRACTION if idx == 1 else max(0.80, MIN_CLASS_FRACTION - 0.05 * (idx - 1)),
        )

        print(
            f"{label}: konfigurasi {idx} | "
            f"kandidat={len(candidates)} | {description}"
        )

        selected = select_spread_centers(
            candidates,
            target=target,
            min_distance_pixels=MIN_DISTANCE_PIXELS,
            seed=seed + idx,
        )

        print(f"{label}: terpilih={len(selected)}")

        if len(selected) >= target:
            return selected[:target]

    raise RuntimeError(
        f"Tidak mendapatkan {target} kandidat {label}. "
        "Perlu melonggarkan threshold atau menggunakan citra tanggal lain."
    )


# ============================================================
# 6. MASK KELAS
# ============================================================

# SAWAH:
# Gunakan SCL vegetation + NDVI tinggi.
sawah_masks = [
    (
        valid_base
        & (scl == 4)
        & (ndvi >= 0.45)
        & (ndvi <= 0.90),
        "SCL=4, 0.45 <= NDVI <= 0.90",
    ),
    (
        valid_base
        & (scl == 4)
        & (ndvi >= 0.35)
        & (ndvi <= 0.90),
        "SCL=4, 0.35 <= NDVI <= 0.90",
    ),
    (
        valid_base
        & (scl == 4)
        & (ndvi >= 0.30),
        "SCL=4, NDVI >= 0.30",
    ),
]

# NON-SAWAH:
# Gunakan SCL=5 (not vegetated) dan NDVI rendah.
# Water sengaja tidak digunakan agar non-sawah tidak didominasi air.
non_sawah_masks = [
    (
        valid_base
        & (scl == 5)
        & (ndvi <= 0.20),
        "SCL=5, NDVI <= 0.20",
    ),
    (
        valid_base
        & (scl == 5)
        & (ndvi <= 0.30),
        "SCL=5, NDVI <= 0.30",
    ),
    (
        valid_base
        & (scl == 5)
        & (ndvi <= 0.35),
        "SCL=5, NDVI <= 0.35",
    ),
]


# ============================================================
# 7. PILIH 50 + 50 SAMPLE
# ============================================================

sawah_centers = choose_samples(
    sawah_masks,
    target=SAWAH_TARGET,
    label="SAWAH",
    seed=RANDOM_SEED,
)

non_sawah_centers = choose_samples(
    non_sawah_masks,
    target=NON_SAWAH_TARGET,
    label="NON-SAWAH",
    seed=RANDOM_SEED + 100,
)


# ============================================================
# 8. KONVERSI PUSAT MENJADI POLYGON
# ============================================================

def make_polygon_samples(centers, label, start_id):
    records = []
    radius = WINDOW_PIXELS // 2

    for i, (row, col) in enumerate(centers, start=start_id):
        row0 = int(row - radius)
        col0 = int(col - radius)

        window = Window(
            col_off=col0,
            row_off=row0,
            width=WINDOW_PIXELS,
            height=WINDOW_PIXELS,
        )

        left, bottom, right, top = window_bounds(
            window,
            transform,
        )

        from shapely.geometry import box

        polygon = box(
            left,
            bottom,
            right,
            top,
        )

        row_slice = slice(row0, row0 + WINDOW_PIXELS)
        col_slice = slice(col0, col0 + WINDOW_PIXELS)

        ndvi_window = ndvi[row_slice, col_slice]
        valid_window = valid_base[row_slice, col_slice]

        ndvi_mean = float(
            np.nanmean(ndvi_window)
        )

        valid_pct = float(
            np.mean(valid_window) * 100
        )

        records.append(
            {
                "sample_id": i,
                "label": label,
                "area_m2": float(polygon.area),
                "ndvi_mean": ndvi_mean,
                "valid_pct": valid_pct,
                "row": int(row),
                "col": int(col),
                "geometry": polygon,
            }
        )

    return records


sawah_records = make_polygon_samples(
    sawah_centers,
    "sawah",
    start_id=1,
)

non_sawah_records = make_polygon_samples(
    non_sawah_centers,
    "non_sawah",
    start_id=SAWAH_TARGET + 1,
)

records = sawah_records + non_sawah_records


# ============================================================
# 9. SIMPAN SHAPEFILE
# ============================================================

gdf = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=crs,
)

# Urutkan berdasarkan ID
gdf = gdf.sort_values("sample_id").reset_index(drop=True)

shp_path = OUTPUT_DIR / f"{SHP_NAME}.shp"

gdf.to_file(
    shp_path,
    driver="ESRI Shapefile",
    encoding="UTF-8",
)

print("\n=== HASIL SAMPLE ===")
print(gdf["label"].value_counts())
print("Total polygon:", len(gdf))
print("CRS:", gdf.crs)
print("File:", shp_path)


# ============================================================
# 10. SIMPAN CSV RINGKAS
# ============================================================

csv_path = OUTPUT_DIR / f"{SHP_NAME}.csv"

gdf.drop(columns="geometry").to_csv(
    csv_path,
    index=False,
)

print("CSV:", csv_path)


# ============================================================
# 11. ZIP SEMUA FILE SHAPEFILE
# ============================================================

zip_path = OUTPUT_DIR / f"{SHP_NAME}.zip"

extensions = [
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
]

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
) as zf:

    for ext in extensions:
        file_path = OUTPUT_DIR / f"{SHP_NAME}{ext}"

        if file_path.exists():
            zf.write(
                file_path,
                arcname=file_path.name,
            )

print("ZIP :", zip_path)

print("\n=== SELESAI ===")
print("50 sawah     :", (gdf["label"] == "sawah").sum())
print("50 non-sawah :", (gdf["label"] == "non_sawah").sum())
print("Total        :", len(gdf))

print(
    "\nCATATAN:"
    "\nSample dibuat otomatis berdasarkan NDVI + SCL."
    "\nSebelum dianggap ground truth final, buka SHP di QGIS"
    "\ndan periksa seluruh polygon secara visual."
)
