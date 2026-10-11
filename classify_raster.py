from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


BASE = Path(__file__).resolve().parent

CSV_PATH = (
    BASE
    / "data"
    / "training"
    / "sentinel2_samples.csv"
)

RASTER_PATH = (
    BASE
    / "data"
    / "sentinel2"
    / "menganti_sentinel2_2026-10-01.tif"
)

OUTPUT_PATH = (
    BASE
    / "data"
    / "output"
    / "klasifikasi_sawah_knn.tif"
)

FEATURES = [
    "B02_mean",
    "B03_mean",
    "B04_mean",
    "B08_mean",
]

TARGET = "label"


def main():
    print("=" * 70)
    print("KLASIFIKASI CITRA SENTINEL-2A")
    print("MODEL: KNN")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Baca data training
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    X_train = df[FEATURES].astype(np.float32)
    y_train = df[TARGET].astype(str)

    print(f"\nTraining samples : {len(df)}")
    print("\nDistribusi label:")
    print(y_train.value_counts())

    # ---------------------------------------------------------
    # 2. Scaling
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # ---------------------------------------------------------
    # 3. Model KNN
    # ---------------------------------------------------------
    model = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    print("\nModel KNN berhasil dilatih.")

    # ---------------------------------------------------------
    # 4. Buka raster Sentinel-2
    # ---------------------------------------------------------
    with rasterio.open(RASTER_PATH) as src:

        print("\nInformasi raster:")
        print(f"Width  : {src.width}")
        print(f"Height : {src.height}")
        print(f"Bands  : {src.count}")
        print(f"CRS    : {src.crs}")
        print(f"Transform: {src.transform}")

        # Urutan raster:
        # 1 = B02
        # 2 = B03
        # 3 = B04
        # 4 = B08
        # 5 = SCL
        # 6 = dataMask

        b02 = src.read(1).astype(np.float32)
        b03 = src.read(2).astype(np.float32)
        b04 = src.read(3).astype(np.float32)
        b08 = src.read(4).astype(np.float32)

        data_mask = src.read(6)

        # -----------------------------------------------------
        # 5. Susun fitur setiap pixel
        # -----------------------------------------------------
        height = src.height
        width = src.width

        pixels = np.column_stack([
            b02.ravel(),
            b03.ravel(),
            b04.ravel(),
            b08.ravel(),
        ])

        valid = (
            (data_mask.ravel() > 0)
            & np.all(np.isfinite(pixels), axis=1)
        )

        print(f"\nTotal pixel : {pixels.shape[0]:,}")
        print(f"Pixel valid: {valid.sum():,}")

        # -----------------------------------------------------
        # 6. Scaling pixel
        # -----------------------------------------------------
        pixels_scaled = np.empty_like(pixels)

        pixels_scaled[valid] = scaler.transform(
            pixels[valid]
        )

        # -----------------------------------------------------
        # 7. Prediksi secara batch
        # -----------------------------------------------------
        prediction = np.full(
            pixels.shape[0],
            255,
            dtype=np.uint8,
        )

        # sawah = 1
        # non_sawah = 0

        label_to_code = {
            "non_sawah": 0,
            "sawah": 1,
        }

        valid_indices = np.where(valid)[0]

        batch_size = 50_000

        print("\nMulai klasifikasi...")

        for start in range(
            0,
            len(valid_indices),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(valid_indices),
            )

            idx = valid_indices[start:end]

            pred_labels = model.predict(
                pixels_scaled[idx]
            )

            prediction[idx] = np.array([
                label_to_code[label]
                for label in pred_labels
            ], dtype=np.uint8)

            print(
                f"Progress: {end:,}/{len(valid_indices):,}"
            )

        # -----------------------------------------------------
        # 8. Bentuk kembali menjadi raster
        # -----------------------------------------------------
        classified = prediction.reshape(
            height,
            width,
        )

        # -----------------------------------------------------
        # 9. Simpan GeoTIFF
        # -----------------------------------------------------
        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="lzw",
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile,
        ) as dst:

            dst.write(
                classified,
                1,
            )

            dst.set_band_description(
                1,
                "Klasifikasi Sawah/Non-Sawah",
            )

            dst.update_tags(
                1,
                CLASS_0="non_sawah",
                CLASS_1="sawah",
                NODATA="255",
                MODEL="KNN",
                N_NEIGHBORS="5",
            )

    # ---------------------------------------------------------
    # 10. Statistik hasil
    # ---------------------------------------------------------
    unique, counts = np.unique(
        classified,
        return_counts=True,
    )

    print("\n" + "=" * 70)
    print("HASIL KLASIFIKASI")
    print("=" * 70)

    for value, count in zip(
        unique,
        counts,
    ):
        if value == 0:
            name = "non_sawah"
        elif value == 1:
            name = "sawah"
        else:
            name = "nodata"

        print(
            f"{name:10s}: {count:,} pixel"
        )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()