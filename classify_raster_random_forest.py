from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio


BASE = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE / "data" / "output"
    / "random_forest_sawah.joblib"
)

RASTER_PATH = (
    BASE / "data" / "sentinel2"
    / "menganti_sentinel2_2026-10-01.tif"
)

OUTPUT_PATH = (
    BASE / "data" / "output"
    / "klasifikasi_sawah_random_forest.tif"
)

BATCH_SIZE = 100_000


def main():
    print("=" * 70)
    print("KLASIFIKASI SELURUH RASTER — RANDOM FOREST")
    print("=" * 70)

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]

    print("Features:", features)

    with rasterio.open(RASTER_PATH) as src:
        print("\nRaster:")
        print("Width :", src.width)
        print("Height:", src.height)
        print("Bands :", src.count)
        print("CRS   :", src.crs)
        print("Res   :", src.res)

        # 1=B02, 2=B03, 3=B04, 4=B08
        b02 = src.read(1).astype(np.float32)
        b03 = src.read(2).astype(np.float32)
        b04 = src.read(3).astype(np.float32)
        b08 = src.read(4).astype(np.float32)

        # Band 6 = dataMask
        data_mask = src.read(6)

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

        valid_indices = np.flatnonzero(valid)

        print("\nTotal pixel :", f"{len(pixels):,}")
        print("Valid pixel :", f"{len(valid_indices):,}")

        prediction = np.full(
            len(pixels),
            255,
            dtype=np.uint8
        )

        label_to_code = {
            "non_sawah": 0,
            "sawah": 1,
        }

        for start in range(
            0,
            len(valid_indices),
            BATCH_SIZE
        ):
            end = min(
                start + BATCH_SIZE,
                len(valid_indices)
            )

            indices = valid_indices[start:end]

            batch = pixels[indices]

            labels = model.predict(batch)

            prediction[indices] = np.array(
                [
                    label_to_code[label]
                    for label in labels
                ],
                dtype=np.uint8
            )

            print(
                f"Progress: {end:,}/{len(valid_indices):,}"
            )

        classified = prediction.reshape(
            src.height,
            src.width
        )

        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="lzw",
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile
        ) as dst:
            dst.write(classified, 1)
            dst.set_band_description(
                1,
                "Klasifikasi Sawah/Non-Sawah"
            )

            dst.update_tags(
                1,
                CLASS_0="non_sawah",
                CLASS_1="sawah",
                NODATA="255",
                MODEL="Random Forest",
                N_ESTIMATORS="300",
            )

        # Statistik
        valid_count = np.sum(
            np.isin(classified, [0, 1])
        )

        rows = []

        pixel_area_m2 = (
            abs(src.transform.a)
            * abs(src.transform.e)
        )

        for code, label in [
            (0, "non_sawah"),
            (1, "sawah")
        ]:
            n = int(
                np.sum(classified == code)
            )

            area_m2 = n * pixel_area_m2

            rows.append({
                "label": label,
                "pixel": n,
                "area_m2": area_m2,
                "area_km2": area_m2 / 1_000_000,
                "percent_valid": (
                    n / valid_count * 100
                ),
            })

        stats = pd.DataFrame(rows)

        print("\n" + "=" * 70)
        print("HASIL KLASIFIKASI")
        print("=" * 70)
        print(
            stats.to_string(index=False)
        )

    print("\nOutput raster:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
