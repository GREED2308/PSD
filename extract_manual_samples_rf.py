from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


BASE = Path(__file__).resolve().parent

# === UBAH BAGIAN INI SESUAI FILE SAMPLE MANUAL KAMU ===
AOI_PATH = BASE / "data" / "geojson" / "menganti_aoi.geojson"
SAMPLE_PATH = BASE / "data" / "training" / "manual" / "samples.shp"
RASTER_PATH = BASE / "data" / "sentinel2" / "menganti_sentinel2_2026-10-01.tif"

OUTPUT_PATH = (
    BASE / "data" / "training" / "sentinel2_samples_manual.csv"
)

# Nama field label di shapefile.
# Isi atributnya harus: sawah / non_sawah
LABEL_FIELD = "label"

EXPECTED_PER_CLASS = 50

# Band Sentinel-2 pada GeoTIFF:
# 1=B02, 2=B03, 3=B04, 4=B08, 5=SCL, 6=dataMask
BAND_INFO = {
    "B02_mean": 1,
    "B03_mean": 2,
    "B04_mean": 3,
    "B08_mean": 4,
}


def main():
    print("=" * 70)
    print("EKSTRAKSI FITUR SAMPLE MANUAL")
    print("=" * 70)

    if not AOI_PATH.exists():
        raise FileNotFoundError(f"AOI tidak ditemukan: {AOI_PATH}")

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample manual tidak ditemukan: {SAMPLE_PATH}\n"
            "Buat polygon sample di QGIS dan isi field 'label' "
            "dengan sawah/non_sawah."
        )

    if not RASTER_PATH.exists():
        raise FileNotFoundError(f"Raster tidak ditemukan: {RASTER_PATH}")

    samples = gpd.read_file(SAMPLE_PATH)

    if LABEL_FIELD not in samples.columns:
        raise ValueError(
            f"Field '{LABEL_FIELD}' tidak ditemukan.\n"
            f"Field tersedia: {list(samples.columns)}"
        )

    samples[LABEL_FIELD] = (
        samples[LABEL_FIELD]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"sawah", "non_sawah"}
    invalid = sorted(set(samples[LABEL_FIELD]) - allowed)

    if invalid:
        raise ValueError(
            f"Label tidak valid: {invalid}. "
            f"Gunakan hanya 'sawah' atau 'non_sawah'."
        )

    print("\nJumlah sample:")
    print(samples[LABEL_FIELD].value_counts())

    for label in ["sawah", "non_sawah"]:
        n = int((samples[LABEL_FIELD] == label).sum())
        if n != EXPECTED_PER_CLASS:
            raise ValueError(
                f"Jumlah {label} = {n}, "
                f"harus tepat {EXPECTED_PER_CLASS}."
            )

    aoi = gpd.read_file(AOI_PATH)

    with rasterio.open(RASTER_PATH) as src:
        raster_crs = src.crs

        if samples.crs is None:
            raise ValueError("Sample tidak memiliki CRS.")

        if aoi.crs is None:
            raise ValueError("AOI tidak memiliki CRS.")

        samples = samples.to_crs(raster_crs)
        aoi = aoi.to_crs(raster_crs)

        # Cek sample berada di dalam/beririsan dengan AOI
        aoi_geom = aoi.geometry.union_all()
        outside = ~samples.geometry.intersects(aoi_geom)

        if outside.any():
            bad = samples.index[outside].tolist()
            raise ValueError(
                f"Ada sample di luar AOI: index {bad}"
            )

        rows = []

        for idx, row in samples.iterrows():
            geom = [row.geometry]

            values, _ = mask(
                src,
                geom,
                crop=True,
                filled=False
            )

            # values shape = (6, h, w)
            valid = ~values.mask[0]

            if not valid.any():
                raise ValueError(
                    f"Sample index {idx} tidak memiliki pixel raster valid."
                )

            result = {
                "sample_id": idx + 1,
                "label": row[LABEL_FIELD],
                "valid_pct": float(valid.mean() * 100.0),
            }

            for feature_name, band_number in BAND_INFO.items():
                band = values[band_number - 1]

                band_values = band.compressed().astype(np.float32)

                if len(band_values) == 0:
                    result[feature_name] = np.nan
                else:
                    result[feature_name] = float(
                        np.mean(band_values)
                    )

            rows.append(result)

    df = pd.DataFrame(rows)

    feature_cols = list(BAND_INFO.keys())

    if df[feature_cols].isna().any().any():
        bad = df[df[feature_cols].isna().any(axis=1)]
        raise ValueError(
            f"Ada sample dengan fitur kosong:\n{bad}"
        )

    # Statistik akhir
    print("\nJumlah sample final:")
    print(df["label"].value_counts())

    print("\nPreview:")
    print(df.head())

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nCSV berhasil dibuat:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
