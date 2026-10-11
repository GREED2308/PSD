import rasterio
import numpy as np
import matplotlib.pyplot as plt

path = "data/sentinel2/menganti_sentinel2_2026-10-01.tif"

with rasterio.open(path) as src:
    blue = src.read(1)   # B02
    green = src.read(2)  # B03
    red = src.read(3)    # B04
    nir = src.read(4)   # B08
    scl = src.read(5)   # SCL

def stretch(arr):
    p2, p98 = np.nanpercentile(arr, (2, 98))
    arr = (arr - p2) / (p98 - p2)
    return np.clip(arr, 0, 1)

# True Color
rgb = np.dstack([
    stretch(red),
    stretch(green),
    stretch(blue)
])

plt.figure(figsize=(12, 10))
plt.imshow(rgb)
plt.title("Sentinel-2A Menganti — True Color")
plt.axis("off")
plt.show()

# False Color
false_color = np.dstack([
    stretch(nir),
    stretch(red),
    stretch(green)
])

plt.figure(figsize=(12, 10))
plt.imshow(false_color)
plt.title("Sentinel-2A Menganti — False Color")
plt.axis("off")
plt.show()

# Nilai SCL
print("Nilai SCL unik:")
print(np.unique(scl))