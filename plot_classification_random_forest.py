from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


BASE = Path(__file__).resolve().parent

RASTER_PATH = (
    BASE / "data" / "output"
    / "klasifikasi_sawah_random_forest.tif"
)

with rasterio.open(RASTER_PATH) as src:
    classified = src.read(1)
    transform = src.transform

plot_data = classified.astype(float)
plot_data[plot_data == 255] = np.nan

cmap = ListedColormap([
    "#d9d9d9",  # non-sawah
    "#18c95b",  # sawah
])

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.imshow(
    plot_data,
    cmap=cmap,
    vmin=0,
    vmax=1
)

ax.set_title(
    "Klasifikasi Lahan Sawah dan Non-Sawah\n"
    "Sentinel-2A | Random Forest"
)

ax.axis("off")

legend = [
    Patch(
        facecolor="#d9d9d9",
        edgecolor="black",
        label="Non-Sawah"
    ),
    Patch(
        facecolor="#18c95b",
        edgecolor="black",
        label="Sawah"
    ),
]

ax.legend(
    handles=legend,
    loc="lower right"
)

plt.tight_layout()
plt.show()
