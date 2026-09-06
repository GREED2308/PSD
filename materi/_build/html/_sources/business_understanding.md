# Business Understanding

## 1. Latar Belakang

Kualitas udara merupakan salah satu aspek lingkungan yang perlu diperhatikan karena kondisi atmosfer dapat berubah dari waktu ke waktu. Data science dapat digunakan untuk mengumpulkan, mengeksplorasi, dan memvisualisasikan data lingkungan sehingga perubahan parameter atmosfer dapat dipahami berdasarkan data.

Proyek ini berfokus pada Desa Sidojangkung, Kecamatan Menganti, Kabupaten Gresik, Jawa Timur. Data parameter atmosfer diperoleh dari Copernicus Sentinel-5P/TROPOMI dan wilayah pengamatan dibatasi menggunakan batas administratif Desa Sidojangkung dalam format GeoJSON.

Parameter yang dianalisis meliputi Nitrogen Dioxide (NO₂), Carbon Monoxide (CO), Ozone (O₃), dan Sulfur Dioxide (SO₂).

## 2. Tujuan

Tujuan proyek ini adalah:

1. Mengumpulkan data parameter polutan atmosfer di Desa Sidojangkung.
2. Menganalisis perubahan parameter polutan dari waktu ke waktu.
3. Mengetahui karakteristik dan kelengkapan data yang diperoleh.
4. Mengidentifikasi missing value dan nilai yang tidak biasa.
5. Memvisualisasikan perubahan parameter polutan dalam bentuk time series.

## 3. Manfaat

Manfaat proyek ini adalah:

1. Memberikan gambaran berbasis data mengenai perubahan parameter polutan atmosfer di Desa Sidojangkung.
2. Membantu memahami karakteristik temporal masing-masing parameter.
3. Mengetahui tingkat kelengkapan dan permasalahan pada dataset.
4. Menjadi dasar untuk tahap Data Preparation dan analisis lanjutan pada proyek Sains Data.

## 4. Pertanyaan Bisnis

Pertanyaan yang ingin dijawab dalam proyek ini adalah:

1. Bagaimana perubahan parameter NO₂, CO, O₃, dan SO₂ selama periode pengamatan?
2. Parameter mana yang memiliki kelengkapan data paling tinggi?
3. Apakah terdapat missing value pada data?
4. Apakah terdapat nilai yang tidak biasa atau anomali?
5. Bagaimana pola perubahan masing-masing parameter berdasarkan waktu?

## 5. Ruang Lingkup

Ruang lingkup proyek meliputi:

- Wilayah penelitian: Desa Sidojangkung, Kecamatan Menganti, Kabupaten Gresik.
- Sumber data: Copernicus Sentinel-5P/TROPOMI Level-2.
- Parameter: NO₂, CO, O₃, dan SO₂.
- Periode pengamatan: 1 September 2025 sampai 30 Agustus 2026.
- Interval pengamatan: harian.
- Batas wilayah: polygon GeoJSON Desa Sidojangkung.
- Visualisasi utama: time series dan agregasi bulanan.

## 6. Batasan

Data yang digunakan berasal dari pengamatan satelit Sentinel-5P/TROPOMI sehingga tidak dapat dianggap sebagai pengukuran langsung konsentrasi udara di permukaan seperti pengukuran stasiun pemantauan kualitas udara.

Nilai parameter digunakan untuk menggambarkan variasi atmosfer pada wilayah penelitian. Proyek ini tidak digunakan untuk menentukan secara langsung tingkat risiko kesehatan masyarakat.

Selain itu, penyebab perubahan nilai polutan seperti aktivitas kendaraan, industri, pembakaran, curah hujan, angin, atau faktor meteorologi lainnya tidak disimpulkan hanya berdasarkan dataset ini.