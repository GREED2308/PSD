# PSD — Proyek Sains Data

## Analisis Parameter Polutan Atmosfer Desa Sidojangkung

Proyek ini merupakan bagian dari mata kuliah **Proyek Sains Data (PSD)** yang berfokus pada pengumpulan, eksplorasi, dan visualisasi data parameter polutan atmosfer di wilayah penelitian.

Wilayah penelitian yang digunakan adalah **Desa Sidojangkung, Kecamatan Menganti, Kabupaten Gresik, Jawa Timur**.

Batas wilayah penelitian ditentukan menggunakan data geospasial dalam format **GeoJSON** sehingga pengambilan data atmosfer dapat dibatasi pada wilayah yang menjadi objek penelitian.

---

## Tujuan Proyek

Proyek ini bertujuan untuk:

1. Mengumpulkan data parameter polutan atmosfer di wilayah Desa Sidojangkung.
2. Mengeksplorasi karakteristik dan kelengkapan data.
3. Mengidentifikasi missing value dan nilai yang tidak biasa.
4. Menganalisis perubahan parameter polutan berdasarkan waktu.
5. Menampilkan hasil analisis dalam bentuk visualisasi *time series*.

---

## Data Penelitian

Data parameter atmosfer diperoleh dari:

**Copernicus Sentinel-5P / TROPOMI Level-2**

Parameter yang dianalisis meliputi:

| Parameter | Nama |
|---|---|
| NO₂ | Nitrogen Dioxide |
| CO | Carbon Monoxide |
| O₃ | Ozone |
| SO₂ | Sulfur Dioxide |

Periode pengamatan:

**1 September 2025 – 30 Agustus 2026**

Data diolah menjadi interval harian untuk mendukung analisis *time series*.

> **Catatan:** Data Sentinel-5P/TROPOMI merupakan pengamatan atmosfer berbasis satelit. Nilai yang digunakan dalam proyek ini tidak sama dengan pengukuran langsung konsentrasi udara permukaan dari stasiun pemantauan kualitas udara.

---

## Wilayah Penelitian

**Desa Sidojangkung**  
Kecamatan Menganti  
Kabupaten Gresik  
Jawa Timur, Indonesia

Batas administratif wilayah penelitian disimpan dalam:

{download}Download administratif wilayah <../data/geojson/sidojangkung.geojson>