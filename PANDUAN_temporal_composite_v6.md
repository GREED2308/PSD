# Panduan komposit temporal Sentinel-2 v6

## Tujuan
Skrip `extract_features_temporal_composite_v6.py` merupakan skrip baru yang terpisah dari ekstraksi harian lama. Setiap sampel diminta menggunakan jendela waktu yang sama, secara default **1 September–10 Oktober 2026 (inklusif)**. Pada setiap piksel, reflektansi dihitung sebagai median dari observasi Sentinel-2 L2A yang lolos `dataMask > 0` dan SCL kelas 4, 5, atau 6.

Komposit mengurangi ketergantungan pada satu tanggal, tetapi **tidak berarti semua piksel di seluruh Jawa Timur diamati pada tanggal aktual yang sama**. Tanggal valid dan jumlah observasi dapat berbeda berdasarkan lokasi; skrip menyimpan statistik jumlah observasi valid per piksel untuk membantu audit.

## Input dan output
- Input default: `outputs/dataset_v6_review/dataset_6kelas_v6_kandidat_representatif_303.geojson`
- Kredensial tetap dibaca dari `.env` pada direktori proyek; jangan membagikan file `.env`.
- Output baru disimpan di `outputs/sentinel2_temporal_composite_v6/` dengan nama yang memuat jendela tanggal. File CSV ekstraksi lama tidak ditimpa.
- Skrip mempertahankan band reflektansi B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12. B10 tidak diminta sebagai band reflektansi permukaan L2A.
- Band 20 m dan 60 m di-resample ke grid output 10 m menggunakan nearest-neighbour, sesuai workflow awal proyek. Ini harus dijelaskan dalam metodologi; grid 10 m tidak mengubah resolusi asli band 20/60 m.
- Indeks NDVI, NDWI, MNDWI, dan NDBI dihitung dari komposit reflektansi median.

## Uji yang disarankan
Jalankan dari direktori root proyek yang berisi `.env`:

```powershell
python .\extract_features_temporal_composite_v6.py --limit 3
```

Pastikan status API sukses, jumlah band benar, nilai fitur bukan seluruh nol/NaN, dan file dibuat di output baru. Bila hasil uji masuk akal, lanjutkan seluruh 303 sampel dengan perintah:

```powershell
python .\extract_features_temporal_composite_v6.py
```

Untuk tanggal lain, gunakan `--start-date YYYY-MM-DD --end-date YYYY-MM-DD`. `--end-date` inklusif. Untuk menghapus checkpoint pada jendela tanggal yang sama dan mulai ulang, gunakan `--fresh` dengan hati-hati.

## Sebelum training
1. Periksa status ekstraksi dan jumlah piksel bersih.
2. Tandai sampel dengan piksel valid terlalu sedikit atau fraksi valid rendah untuk inspeksi.
3. Validasi label dan geometri, terutama sampel danau, bangunan kecil, serta sampel dekat pesisir.
4. Cegah kebocoran data saat train/test split: sampel yang bertetangga, geometri duplikat, atau lokasi sumber yang sama tidak boleh tersebar antara train dan test tanpa aturan pemisahan yang jelas.
5. Baru setelah itu bandingkan algoritma dengan split dan metrik yang konsisten.

Skrip belum diuji terhadap kredensial, file GeoJSON, dan kuota API di komputer pengguna; lakukan uji 3 sampel dahulu sebelum ekstraksi penuh.
