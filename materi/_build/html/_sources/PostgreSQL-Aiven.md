# Laporan Tugas Lanjutan PSD — PostgreSQL Aiven, KNIME, dan Statistik Data

## 1. Pendahuluan

Tugas ini merupakan lanjutan dari tahapan proyek Proyek Sains Data (PSD) yang telah dikerjakan sebelumnya. Pada tahap sebelumnya, data parameter polutan atmosfer Sentinel-5P telah dikumpulkan untuk wilayah Sidojangkung, kemudian diolah menjadi data time series harian dalam format CSV.

# 2. Dasar Data dan Wilayah Penelitian

## 2.1 Wilayah penelitian

Wilayah yang digunakan adalah:

- Desa: Sidojangkung
- Kecamatan: Menganti
- Kabupaten: Gresik
- Sistem koordinat: EPSG:4326

Batas wilayah diperoleh dalam format GeoJSON. `{download}Download administratif wilayah <../data/geojson/sidojangkung.geojson>` digunakan sebagai salah satu alat untuk melihat/mengelola data batas wilayah.

Geometry wilayah berupa polygon dengan batas geografis yang digunakan untuk membatasi area pengambilan data.

## 2.2 Data yang digunakan

Data berasal dari parameter atmosfer Sentinel-5P yang diproses menjadi statistik harian untuk empat parameter utama:

- NO₂
- CO
- O₃
- SO₂

File hasil pengolahan: {download}Data<../data/processed/sidojangkung_air_quality_daily.csv>

CSV memiliki 25 kolom, yaitu:

```text
date

NO2:
no2_mean
no2_min
no2_max
no2_stdev
no2_sample_count
no2_nodata_count

CO:
co_mean
co_min
co_max
co_stdev
co_sample_count
co_nodata_count

O3:
o3_mean
o3_min
o3_max
o3_stdev
o3_sample_count
o3_nodata_count

SO2:
so2_mean
so2_min
so2_max
so2_stdev
so2_sample_count
so2_nodata_count
```

Pada tahap statistik menggunakan KNIME, analisis difokuskan pada empat feature utama:

```text
no2_mean
co_mean
o3_mean
so2_mean
```

Kolom lainnya merupakan atribut pendukung atau hasil agregasi harian.

---

# 3. Pemeriksaan CSV Sebelum Dipindahkan

Sebelum memasukkan data ke Aiven, file diperiksa terlebih dahulu.

Hasil pemeriksaan menunjukkan bahwa:

- file CSV tersedia;
- header memiliki 25 kolom;
- tanggal tersimpan dalam format `YYYY-MM-DD`;
- nilai numerik tersimpan dalam format desimal/scientific notation;
- beberapa field dapat kosong;
- field kosong harus dipertahankan sebagai nilai `NULL` pada PostgreSQL.

Contoh baris yang memiliki nilai kosong:

```text
2025-09-04,...,co_mean kosong,...
```

Nilai kosong tersebut tidak diisi dengan angka buatan.

---

# 4. PostgreSQL Aiven

## 4.1 Tujuan penggunaan PostgreSQL Aiven

PostgreSQL Aiven digunakan sebagai penyimpanan database terpusat untuk data time series. Dengan memindahkan CSV ke database, data dapat diakses oleh KNIME menggunakan koneksi database secara langsung.

Alur tahap ini:

```text
CSV
 ↓
PostgreSQL Aiven
 ↓
Table
```

## 4.2 Koneksi Aiven melalui pgAdmin

Koneksi dilakukan menggunakan pgAdmin.

Parameter utama yang digunakan:

```text
Host       : host PostgreSQL dari Aiven
Port       : 26271
Database   : defaultdb
Username   : avnadmin
Password   : password dari Aiven
SSL        : require
```

Password tidak dicatat di laporan.

Setelah konfigurasi disimpan, pgAdmin berhasil menampilkan:

```text
Servers
└── PSD-Aiven
    └── Databases
        └── defaultdb
```

Hal tersebut menunjukkan bahwa koneksi PostgreSQL Aiven berhasil.

---

# 5. Membuat Tabel PostgreSQL

Tabel yang dibuat:

```text
sidojangkung_air_quality
```

SQL:

```sql
CREATE TABLE public.sidojangkung_air_quality (
    date DATE PRIMARY KEY,

    no2_mean DOUBLE PRECISION,
    no2_min DOUBLE PRECISION,
    no2_max DOUBLE PRECISION,
    no2_stdev DOUBLE PRECISION,
    no2_sample_count INTEGER,
    no2_nodata_count INTEGER,

    co_mean DOUBLE PRECISION,
    co_min DOUBLE PRECISION,
    co_max DOUBLE PRECISION,
    co_stdev DOUBLE PRECISION,
    co_sample_count INTEGER,
    co_nodata_count INTEGER,

    o3_mean DOUBLE PRECISION,
    o3_min DOUBLE PRECISION,
    o3_max DOUBLE PRECISION,
    o3_stdev DOUBLE PRECISION,
    o3_sample_count INTEGER,
    o3_nodata_count INTEGER,

    so2_mean DOUBLE PRECISION,
    so2_min DOUBLE PRECISION,
    so2_max DOUBLE PRECISION,
    so2_stdev DOUBLE PRECISION,
    so2_sample_count INTEGER,
    so2_nodata_count INTEGER
);
```

Tipe data dipilih berdasarkan isi CSV:

- `date` menggunakan `DATE`;
- variabel pengukuran menggunakan `DOUBLE PRECISION`;
- jumlah sample dan jumlah no-data menggunakan `INTEGER`.

`date` dijadikan primary key karena data digunakan pada tingkat harian sehingga setiap tanggal diharapkan hanya memiliki satu baris agregasi.

---

# 6. Import CSV ke PostgreSQL

CSV dimasukkan menggunakan menu:

```text
Table
→ Import/Export Data
```

Konfigurasi penting:

```text
Import/Export : Import
Filename      : D:\Projects\Kuliah\PSD\data\processed\sidojangkung_air_quality_daily.csv
Format        : csv
Encoding      : UTF8
Header        : Yes
Delimiter     : ,
Quote         : "
Escape        : "
```

Header harus diaktifkan agar baris pertama CSV dibaca sebagai nama kolom.

Nilai kosong dibiarkan sebagai nilai kosong sehingga dapat direpresentasikan sebagai `NULL`.

## 6.1 Verifikasi jumlah data

Setelah proses import, data diperiksa menggunakan:

```sql
SELECT COUNT(*) AS jumlah_data
FROM public.sidojangkung_air_quality;
```

Hasil:

```text
364
```

Artinya terdapat 364 baris data harian yang masuk ke database.

## 6.2 Verifikasi isi data

Untuk melihat data:

```sql
SELECT *
FROM public.sidojangkung_air_quality
ORDER BY date
LIMIT 5;
```

Hasil menunjukkan tanggal dan nilai parameter berhasil tersimpan. Nilai yang kosong pada CSV juga terbaca sebagai `NULL`.

Dengan demikian tahap:

> **memindahkan data time series ke PostgreSQL Aiven**

telah selesai.

---

# 7. Menarik Data PostgreSQL ke KNIME

## 7.1 Workflow KNIME

Workflow yang digunakan:

```text
PostgreSQL Connector
        ↓
DB Table Selector
        ↓
DB Reader
        ↓
Column Filter
        ↓
Statistics
```

Workflow tersebut dibuat untuk mengambil data dari PostgreSQL kemudian menganalisis fitur utamanya.

---

# 8. PostgreSQL Connector

Node:

```text
PostgreSQL Connector
```

digunakan untuk membuat koneksi dari KNIME ke database PostgreSQL Aiven menggunakan JDBC.

Konfigurasi koneksi menggunakan informasi yang sama dengan koneksi Aiven:

```text
Hostname       : host Aiven
Port            : 26271
Database name  : defaultdb
Authentication : Username and Password
Username       : avnadmin
Database dialect: PostgreSQL
```

Pada JDBC Parameters ditambahkan:

```text
sslmode = require
```

Setelah konfigurasi diterapkan dan node dieksekusi, `PostgreSQL Connector` berubah menjadi hijau.

---

# 9. DB Table Selector

Node:

```text
DB Table Selector
```

digunakan untuk menentukan tabel database yang akan digunakan.

Tabel yang dipilih:

```text
public.sidojangkung_air_quality
```

Dengan demikian KNIME diarahkan ke tabel hasil import CSV.

---

# 10. DB Reader

Node:

```text
DB Reader
```

digunakan untuk membaca hasil query/database table dan menghasilkan data table yang dapat digunakan oleh node analisis KNIME.

Setelah berhasil dieksekusi, data PostgreSQL sudah tersedia di workflow KNIME.

---

# 11. Column Filter

Karena tabel memiliki 25 kolom, analisis statistik difokuskan pada feature utama.

Kolom yang dipertahankan:

```text
date
no2_mean
co_mean
o3_mean
so2_mean
```

Kolom pendukung tidak digunakan dalam analisis Statistics utama.

Alasan penggunaan Column Filter adalah agar hasil Statistics lebih mudah dibaca dan pembahasan dosen dapat difokuskan pada empat feature polutan utama.

---

# 12. Statistics Node

Node:

```text
Statistics
```

digunakan untuk menghitung ringkasan statistik pada feature numerik.

Output Statistics yang diperoleh:

| Feature | Min | Mean | Median | Max | Std. Dev. | Skewness | Kurtosis | No. Missing | No. +∞ | No. -∞ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no2_mean` | 1.35E-5 | 4.65E-5 | ? | 9.77E-5 | 1.99E-5 | 0.6548 | -0.2377 | 194 | 0 | 0 |
| `co_mean` | 0.0202 | 0.0296 | ? | 0.0452 | 0.0039 | 0.6794 | 1.4574 | 144 | 0 | 0 |
| `o3_mean` | 0.1095 | 0.1158 | ? | 0.1228 | 0.0025 | 0.2925 | -0.0635 | 4 | 0 | 0 |
| `so2_mean` | -0.0013 | 7.73E-5 | ? | 0.0014 | 0.0003 | -0.0062 | 4.07 | 146 | 0 | 0 |

> `?` pada Median merupakan tampilan output pada Statistics node yang digunakan dalam eksperimen ini. Nilai tersebut tidak dianggap sebagai nilai median numerik.

---

# 13. Penjelasan Setiap Statistik

## 13.1 Minimum

### Pengertian

Minimum adalah nilai terkecil pada suatu fitur.

### Rumus

$$
x_{min}=\min(x_1,x_2,\ldots,x_n)
$$

### Contoh

Data:

```text
2, 4, 5, 7, 9
```

Nilai terkecil:

$$
Min=2
$$

### Hasil dataset

Untuk `no2_mean`:

$$
Min=1.35\times10^{-5}
$$

---

## 13.2 Mean

### Pengertian

Mean atau rata-rata menunjukkan nilai rata-rata dari seluruh observasi.

### Rumus

$$
\bar{x}=\frac{\sum_{i=1}^{n}x_i}{n}
$$

### Contoh

Data:

```text
2, 4, 4, 6, 8
```

$$
\begin{aligned}
\bar{x}
&= \frac{2+4+4+6+8}{5} \\
&= \frac{24}{5} \\
&= 4.8
\end{aligned}
$$

### Hasil dataset

Untuk `no2_mean`:

$$
Mean=4.65\times10^{-5}
$$

---

## 13.3 Median

### Pengertian

Median adalah nilai tengah setelah data diurutkan.

Untuk jumlah data ganjil, median adalah satu nilai tengah.

Untuk jumlah data genap:

$$
Median=\frac{x_{tengah1}+x_{tengah2}}{2}
$$

### Contoh ganjil

```text
2, 4, 5, 7, 9
```

$$
Median=5
$$

### Contoh genap

```text
2, 4, 6, 8
```

$$
Median=\frac{4+6}{2}=5
$$

### Hasil KNIME

Pada output Statistics yang digunakan, Median tampil sebagai:

```text
?
```

Oleh karena itu laporan tidak menggunakan `?` sebagai nilai numerik.

---

## 13.4 Maximum

### Pengertian

Maximum adalah nilai terbesar pada suatu fitur.

### Rumus

$$
x_{max}=\max(x_1,x_2,\ldots,x_n)
$$

### Contoh

```text
2, 4, 5, 7, 9
```

$$
Max=9
$$

### Hasil dataset

Untuk `no2_mean`:

$$
Max=9.77\times10^{-5}
$$

---

## 13.5 Standard Deviation

### Pengertian

Standard deviation atau simpangan baku menunjukkan seberapa jauh data menyebar dari nilai rata-rata.

### Rumus sample

$$
s=
\sqrt{
\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}
{n-1}
}
$$

### Contoh

Data:

```text
2, 4, 4, 6, 8
```

Mean:

$$
\bar{x}=4.8
$$

Kuadrat selisih:

$$
(2-4.8)^2=7.84
$$

$$
(4-4.8)^2=0.64
$$

$$
(4-4.8)^2=0.64
$$

$$
(6-4.8)^2=1.44
$$

$$
(8-4.8)^2=10.24
$$

Jumlah:

$$
20.80
$$

Variance:

$$
s^2=\frac{20.80}{4}=5.20
$$

Standard deviation:

$$
s=\sqrt{5.20}\approx2.28
$$

### Hasil dataset

Untuk `no2_mean`:

```text
Std. Dev. = 1.99E-5
```

---

## 13.6 Variance

### Pengertian

Variance mengukur penyebaran nilai terhadap mean berdasarkan kuadrat deviasi.

### Rumus sample

$$
s^2=
\frac{\sum(x_i-\bar{x})^2}
{n-1}
$$

### Contoh

Untuk:

```text
2, 4, 4, 6, 8
```

diperoleh:

$$
s^2=5.20
$$

### Hubungan dengan standard deviation

$$
Variance=(Std.Dev.)^2
$$

Untuk hasil `no2_mean`, dengan Std. Dev. yang ditampilkan sebesar `1.99E-5`:

$$
Variance\approx(1.99\times10^{-5})^2
$$

$$
Variance\approx3.96\times10^{-10}
$$

Nilai tersebut merupakan perkiraan karena Std. Dev. pada tabel KNIME telah dibulatkan.

---

## 13.7 Skewness

### Pengertian

Skewness menunjukkan tingkat kemencengan atau ketidaksimetrisan distribusi.

Interpretasi umum:

```text
Skewness > 0 → cenderung menceng ke kanan
Skewness < 0 → cenderung menceng ke kiri
Skewness ≈ 0 → relatif simetris
```

Salah satu bentuk sample skewness:

$$
G_1=
\frac{n}{(n-1)(n-2)}
\sum
\left(
\frac{x_i-\bar{x}}{s}
\right)^3
$$

### Hasil

| Feature | Skewness |
|---|---:|
| `no2_mean` | 0.6548 |
| `co_mean` | 0.6794 |
| `o3_mean` | 0.2925 |
| `so2_mean` | -0.0062 |

NO₂ dan CO memiliki skewness positif. SO₂ sangat dekat dengan nol berdasarkan nilai skewness.

Skewness tidak menunjukkan penyebab kenaikan/penurunan polutan; statistik ini hanya menggambarkan bentuk distribusi nilai.

---

## 13.8 Kurtosis

### Pengertian

Kurtosis menggambarkan karakteristik bentuk distribusi, terutama perilaku ekor distribusi dan kecenderungan nilai ekstrem.

Salah satu bentuk excess kurtosis sample:

$$
\begin{aligned}
G_2
&=
\frac{n(n+1)}
{(n-1)(n-2)(n-3)}
\sum
\left(
\frac{x_i-\bar{x}}{s}
\right)^4 \\
&\quad -
\frac{3(n-1)^2}
{(n-2)(n-3)}
\end{aligned}
$$

### Hasil

| Feature | Kurtosis |
|---|---:|
| `no2_mean` | -0.2377 |
| `co_mean` | 1.4574 |
| `o3_mean` | -0.0635 |
| `so2_mean` | 4.07 |

SO₂ memiliki nilai kurtosis paling tinggi di antara empat feature utama.

Interpretasi harus tetap dibatasi pada bentuk distribusi statistik dan tidak digunakan untuk menyimpulkan penyebab lingkungan secara langsung.

---

## 13.9 No. Missing

### Pengertian

No. Missing adalah jumlah observasi yang tidak mempunyai nilai pada feature tertentu.

### Rumus

$$
N_{missing}=
\text{jumlah observasi kosong}
$$

### Hasil

| Feature | No. Missing |
|---|---:|
| `no2_mean` | 194 |
| `co_mean` | 144 |
| `o3_mean` | 4 |
| `so2_mean` | 146 |

O₃ mempunyai missing paling sedikit.

---

## 13.10 No. +∞

### Pengertian

No. +∞ menunjukkan jumlah nilai positif tak hingga (`+∞`).

### Hasil

Pada seluruh feature utama:

$$
No.+\infty=0
$$

Artinya tidak terdapat nilai positif tak hingga.

---

## 13.11 No. -∞

### Pengertian

No. -∞ menunjukkan jumlah nilai negatif tak hingga (`-∞`).

### Hasil

Pada seluruh feature utama:

$$
No.-\infty=0
$$

Artinya tidak terdapat nilai negatif tak hingga.

---

## 13.12 Histogram

### Pengertian

Histogram digunakan untuk melihat distribusi frekuensi data.

Data dibagi menjadi beberapa interval atau *bin*, kemudian jumlah observasi pada setiap bin ditampilkan dalam bentuk batang.

Contoh:

| Rentang nilai | Frekuensi |
|---|---:|
| 0–2 | 2 |
| 2–4 | 5 |
| 4–6 | 8 |
| 6–8 | 4 |

Histogram membantu melihat:

- konsentrasi data;
- bentuk distribusi;
- kemencengan distribusi;
- kemungkinan nilai yang jauh dari mayoritas data.

Histogram berbeda dengan bar chart karena histogram digunakan untuk data kuantitatif yang dikelompokkan ke dalam interval.

---

# 14. Interpretasi Hasil Empat Feature

## 14.1 NO₂

Hasil Statistics menunjukkan:

```text
Min          = 1.35E-5
Mean         = 4.65E-5
Max          = 9.77E-5
Std. Dev.    = 1.99E-5
Skewness     = 0.6548
Kurtosis     = -0.2377
Missing      = 194
```

Nilai skewness positif menunjukkan distribusi `no2_mean` cenderung menceng ke kanan. Terdapat 194 missing value. Tidak terdapat nilai `+∞` maupun `-∞`.

---

## 14.2 CO

Hasil:

```text
Min          = 0.0202
Mean         = 0.0296
Max          = 0.0452
Std. Dev.    = 0.0039
Skewness     = 0.6794
Kurtosis     = 1.4574
Missing      = 144
```

Skewness positif menunjukkan distribusi `co_mean` cenderung menceng ke kanan. Terdapat 144 missing value dan tidak ditemukan nilai tak hingga.

---

## 14.3 O₃

Hasil:

```text
Min          = 0.1095
Mean         = 0.1158
Max          = 0.1228
Std. Dev.    = 0.0025
Skewness     = 0.2925
Kurtosis     = -0.0635
Missing      = 4
```

Nilai skewness lebih dekat ke nol dibanding NO₂ dan CO. O₃ memiliki jumlah missing paling sedikit, yaitu 4.

---

## 14.4 SO₂

Hasil:

```text
Min          = -0.0013
Mean         = 7.73E-5
Max          = 0.0014
Std. Dev.    = 0.0003
Skewness     = -0.0062
Kurtosis     = 4.07
Missing      = 146
```

Skewness `-0.0062` sangat dekat dengan nol. Kurtosis `4.07` adalah yang tertinggi dari empat feature utama.

Nilai minimum SO₂ bernilai negatif. Nilai tersebut tidak boleh langsung diubah menjadi nol tanpa dasar metodologis. Nilai retrieval satelit seperti ini perlu dibahas sebagai karakteristik/ketidakpastian hasil pengukuran, bukan otomatis dianggap sebagai data rusak.

---

# 15. Hubungan dengan Materi `Chap2-Data`

Tugas ini berkaitan langsung dengan beberapa bagian materi:

## Temporal data / time-series

Materi menempatkan temporal data sebagai salah satu bentuk ordered data.

Data proyek memiliki struktur:

```text
tanggal → nilai parameter
```

sehingga dapat dianalisis sebagai time series.

## Attributes / features

Materi menjelaskan bahwa attribute atau feature merupakan karakteristik dari suatu data object.

Dalam proyek ini:

```text
NO₂ → no2_mean
CO  → co_mean
O₃  → o3_mean
SO₂ → so2_mean
```

merupakan feature numerik yang dianalisis.

## Central tendency

Mean dan median digunakan untuk menggambarkan kecenderungan pusat data.

## Dispersion

Standard deviation dan variance digunakan untuk menggambarkan penyebaran data.

## Graphic displays

Histogram dapat digunakan untuk melihat distribusi data.

Materi juga membahas boxplot, quartile, dan IQR:

$$
IQR=Q3-Q1
$$

serta aturan umum outlier:

$$
Lower=Q1-1.5(IQR)
$$

$$
Upper=Q3+1.5(IQR)
$$

Bagian tersebut dapat digunakan pada tahap analisis data quality/outlier bila dosen meminta analisis lanjutan.

---

# 16. Ringkasan Hasil

| Feature | Mean | Std. Dev. | Skewness | Kurtosis | Missing |
|---|---:|---:|---:|---:|---:|
| NO₂ | 4.65E-5 | 1.99E-5 | 0.6548 | -0.2377 | 194 |
| CO | 0.0296 | 0.0039 | 0.6794 | 1.4574 | 144 |
| O₃ | 0.1158 | 0.0025 | 0.2925 | -0.0635 | 4 |
| SO₂ | 7.73E-5 | 0.0003 | -0.0062 | 4.07 | 146 |

Dari empat feature utama:

- O₃ memiliki missing value paling sedikit.
- NO₂ memiliki missing value paling banyak.
- NO₂ dan CO menunjukkan skewness positif.
- SO₂ mempunyai skewness paling dekat dengan nol.
- SO₂ mempunyai nilai kurtosis paling tinggi.
- Tidak terdapat nilai `+∞` atau `-∞`.

---

# 17. Kesimpulan

Tahapan tugas lanjutan PSD telah dilakukan melalui tiga proses utama.

Pertama, data time series hasil pengolahan sebelumnya dipindahkan ke PostgreSQL Aiven. Tabel `sidojangkung_air_quality` berhasil dibuat dan CSV berhasil diimpor dengan total 364 baris.

Kedua, data PostgreSQL berhasil ditarik ke KNIME melalui:

```text
PostgreSQL Connector
→ DB Table Selector
→ DB Reader
```

Kemudian `Column Filter` digunakan untuk memfokuskan analisis pada empat feature utama.

Ketiga, node Statistics digunakan untuk memperoleh statistik deskriptif berupa minimum, mean, maximum, standard deviation, skewness, kurtosis, missing value, serta jumlah nilai positif/negatif tak hingga. Masing-masing statistik kemudian dijelaskan menggunakan definisi, rumus, contoh, dan interpretasi.

Tahap ini menghasilkan dasar yang dapat digunakan untuk analisis data lebih lanjut pada proyek PSD, termasuk eksplorasi distribusi, data quality, outlier, dan visualisasi time series.

---

# 18. Bukti yang Sebaiknya Dicantumkan dalam Laporan

Untuk dokumentasi tugas, bukti yang disarankan:

1. Tampilan GeoJSON/batas wilayah Sidojangkung.
2. Isi atau struktur CSV.
3. pgAdmin yang menunjukkan server Aiven terhubung.
4. Struktur tabel `sidojangkung_air_quality`.
5. Proses import CSV.
6. Query `COUNT(*)` yang menghasilkan 364.
7. Query preview data.
8. Workflow KNIME.
9. Konfigurasi PostgreSQL Connector.
10. DB Table Selector yang memilih `public.sidojangkung_air_quality`.
11. DB Reader.
12. Column Filter.
13. Statistics node.
14. Output Statistics.

Screenshot tidak harus menampilkan password atau credential Aiven.

---

# 19. File dan Workflow Proyek

File utama yang digunakan:

```text
data/
├── geojson/
│   └── sidojangkung.geojson
└── processed/
    └── sidojangkung_air_quality_daily.csv
```

Workflow KNIME:

```text
PostgreSQL Connector
        ↓
DB Table Selector
        ↓
DB Reader
        ↓
Column Filter
        ↓
Statistics
```

Database:

```text
Aiven PostgreSQL
└── defaultdb
    └── public
        └── sidojangkung_air_quality
```
