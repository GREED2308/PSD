import rasterio

path = "data/sentinel2/menganti_sentinel2_2026-10-01.tif"

with rasterio.open(path) as src:
    print("=== INFORMASI GEOTIFF ===")
    print("Jumlah band :", src.count)
    print("Lebar       :", src.width)
    print("Tinggi      :", src.height)
    print("CRS         :", src.crs)
    print("Resolusi    :", src.res)
    print("Bounds      :", src.bounds)
    print("Dtype       :", src.dtypes)
    print("NoData      :", src.nodata)

    print("\nUrutan band:")
    for i, name in enumerate(
        ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
        start=1
    ):
        print(f"Band {i}: {name}")