# Kode Random Forest PSD

## Urutan

1. Buat sample polygon manual di QGIS.
   - File: `data/training/manual/samples.shp`
   - Field: `label`
   - Isi label hanya `sawah` atau `non_sawah`
   - Tepat 50 sawah dan 50 non_sawah.

2. Pastikan AOI:
   `data/geojson/menganti_aoi.geojson`

3. Ekstrak fitur:
   ```powershell
   python .\extract_manual_samples_rf.py
   ```

4. Latih dan evaluasi Random Forest:
   ```powershell
   python .\train_random_forest.py
   ```

5. Klasifikasikan seluruh Sentinel-2A:
   ```powershell
   python .\classify_raster_random_forest.py
   ```

6. Tampilkan hasil:
   ```powershell
   python .\plot_classification_random_forest.py
   ```

Fitur yang digunakan:
- B02_mean
- B03_mean
- B04_mean
- B08_mean

Model:
- Random Forest
- 300 trees
- random_state=42
- class_weight="balanced"

Label ditentukan dari sample manual, bukan dari NDVI/SCL.
