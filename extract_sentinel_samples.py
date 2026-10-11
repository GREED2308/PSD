import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path


# ============================================================
# 1. KONFIGURASI
# ============================================================

RASTER_PATH = Path(
    "data/sentinel2/menganti_sentinel2_2026-10-01.tif"
)

SAMPLE_PATH = Path(
    "data/training/sample_lahan_auto.shp"
)

OUTPUT_PATH = Path(
    "data/training/sentinel2_samples.csv"
)


# ============================================================
# 2. VALIDASI FILE
# ============================================================

if not RASTER_PATH.exists():
    raise FileNotFoundError(
        f"GeoTIFF tidak ditemukan:\n{RASTER_PATH}"
    )

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(
        f"Shapefile tidak ditemukan:\n{SAMPLE_PATH}"
    )


# ============================================================
# 3. BACA SHAPEFILE
# ============================================================

print("=" * 60)
print("EKSTRAKSI FITUR SENTINEL-2 DARI POLYGON")
print("=" * 60)

gdf = gpd.read_file(SAMPLE_PATH)

print(f"Jumlah polygon : {len(gdf)}")
print(f"CRS sample     : {gdf.crs}")

required_columns = ["sample_id", "label"]

for col in required_columns:
    if col not in gdf.columns:
        raise ValueError(
            f"Kolom '{col}' tidak ditemukan pada shapefile."
        )


# ============================================================
# 4. BUKA GEOTIFF
# ============================================================

with rasterio.open(RASTER_PATH) as src:

    print("\n=== INFORMASI RASTER ===")
    print("CRS       :", src.crs)
    print("Resolusi  :", src.res)
    print("Ukuran    :", src.width, "x", src.height)
    print("Jumlah band:", src.count)

    # --------------------------------------------------------
    # Pastikan CRS polygon sama dengan raster
    # --------------------------------------------------------

    if gdf.crs != src.crs:
        print("\nReproject sample ke CRS raster...")
        gdf = gdf.to_crs(src.crs)

    print("CRS sample setelah transform:", gdf.crs)

    # --------------------------------------------------------
    # Mapping band
    # --------------------------------------------------------
    #
    # Band 1 = B02
    # Band 2 = B03
    # Band 3 = B04
    # Band 4 = B08
    # Band 5 = SCL
    # Band 6 = dataMask
    # --------------------------------------------------------

    results = []

    print("\nMulai ekstraksi...\n")

    for idx, row in gdf.iterrows():

        sample_id = row["sample_id"]
        label = row["label"]
        geometry = row.geometry

        try:

            # ------------------------------------------------
            # Mask raster menggunakan polygon
            # ------------------------------------------------

            raster_data, _ = mask(
                src,
                [geometry],
                crop=True,
                filled=False
            )

            # ------------------------------------------------
            # Ambil masing-masing band
            # ------------------------------------------------

            b02 = raster_data[0].data.astype(np.float32)
            b03 = raster_data[1].data.astype(np.float32)
            b04 = raster_data[2].data.astype(np.float32)
            b08 = raster_data[3].data.astype(np.float32)
            scl = raster_data[4].data
            data_mask = raster_data[5].data

            # ------------------------------------------------
            # Valid pixel
            # ------------------------------------------------

            # Mask polygon
            mask_array = np.ma.getmaskarray(raster_data)

            # Pixel yang benar-benar berada di dalam polygon
            inside_polygon = ~mask_array[0]

            # Pixel valid di dalam polygon
            valid = (
                inside_polygon
                & np.isfinite(b02)
                & np.isfinite(b03)
                & np.isfinite(b04)
                & np.isfinite(b08)
                & (data_mask > 0)
            )

            # Total hanya pixel di dalam polygon
            total_pixels = int(inside_polygon.sum())

            # Total pixel valid di dalam polygon
            valid_pixels = int(valid.sum())

            if valid_pixels == 0:
                print(
                    f"[WARNING] Sample {sample_id}: "
                    "tidak memiliki pixel valid."
                )
                continue

            valid_pct = (
                valid_pixels / total_pixels
            ) * 100

            # ------------------------------------------------
            # Ambil nilai valid
            # ------------------------------------------------

            b02_valid = b02[valid]
            b03_valid = b03[valid]
            b04_valid = b04[valid]
            b08_valid = b08[valid]

            # SCL pada pixel valid
            scl_valid = scl[valid]

            # ------------------------------------------------
            # NDVI per pixel
            # ------------------------------------------------

            denominator = b08_valid + b04_valid

            ndvi = np.divide(
                b08_valid - b04_valid,
                denominator,
                out=np.full_like(
                    b08_valid,
                    np.nan,
                    dtype=np.float32
                ),
                where=denominator != 0
            )

            ndvi_valid = ndvi[np.isfinite(ndvi)]

            # ------------------------------------------------
            # SCL mode
            # ------------------------------------------------

            if len(scl_valid) > 0:
                scl_values, scl_counts = np.unique(
                    scl_valid,
                    return_counts=True
                )

                scl_mode = int(
                    scl_values[
                        np.argmax(scl_counts)
                    ]
                )
            else:
                scl_mode = np.nan

            # ------------------------------------------------
            # Hasil statistik
            # ------------------------------------------------

            result = {
                "sample_id": sample_id,
                "label": label,
                "area_m2": row.get("area_m2", np.nan),

                "valid_pct": valid_pct,
                "scl_mode": scl_mode,

                "B02_mean": np.mean(b02_valid),
                "B02_std": np.std(b02_valid),

                "B03_mean": np.mean(b03_valid),
                "B03_std": np.std(b03_valid),

                "B04_mean": np.mean(b04_valid),
                "B04_std": np.std(b04_valid),

                "B08_mean": np.mean(b08_valid),
                "B08_std": np.std(b08_valid),

                "NDVI_mean": (
                    np.mean(ndvi_valid)
                    if len(ndvi_valid) > 0
                    else np.nan
                ),

                "NDVI_std": (
                    np.std(ndvi_valid)
                    if len(ndvi_valid) > 0
                    else np.nan
                )
            }

            results.append(result)

            print(
                f"[{len(results):03d}/100] "
                f"Sample {sample_id:<3} | "
                f"{label:<10} | "
                f"Valid={valid_pct:6.2f}% | "
                f"NDVI={result['NDVI_mean']:.4f}"
            )

        except Exception as e:

            print(
                f"[ERROR] Sample {sample_id}: {e}"
            )


# ============================================================
# 5. DATAFRAME
# ============================================================

df_samples = pd.DataFrame(results)

if df_samples.empty:
    raise RuntimeError(
        "Tidak ada sample yang berhasil diekstrak."
    )


# ============================================================
# 6. VALIDASI
# ============================================================

print("\n" + "=" * 60)
print("VALIDASI DATASET")
print("=" * 60)

print("\nJumlah sample:")
print(df_samples["label"].value_counts())

print("\nShape:")
print(df_samples.shape)

print("\nMissing value:")
print(df_samples.isna().sum())

print("\nStatistik NDVI:")
print(
    df_samples["NDVI_mean"].describe()
)


# ============================================================
# 7. SIMPAN CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_samples.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCSV berhasil disimpan:")
print(OUTPUT_PATH)

print("\nPreview:")
print(
    df_samples.head()
)

print("\n=== SELESAI ===")