# Dataset kerja PSD v6 — review

## File keluaran
- `dataset_6kelas_v6_review.geojson`: master 353 fitur. Geometri, ID, class_id dan atribut asli dipertahankan; metadata audit ditambahkan.
- `dataset_6kelas_v6_kandidat_representatif_303.geojson`: 303 fitur setelah mengeluarkan 49 anggota duplikat geometri identik satu kelas dan `JTM0311` (konflik label silang kelas).
- `fitur_sentinel2a_l2a_6kelas_v6_review.csv`: seluruh 353 fitur hasil ekstraksi terkini dengan anotasi keputusan v6.
- `fitur_sentinel2a_l2a_6kelas_v6_kandidat_representatif.csv`: fitur untuk 303 kandidat review.
- `log_perubahan_dataset_v6.csv`: pasangan duplikat dan keputusan yang diterapkan.
- `ringkasan_dataset_v6.csv`: jumlah fitur per kelas dan status ekstraksi.

## Keputusan yang diterapkan
1. Semua fitur asli tetap berada di GeoJSON master v6; tidak ada geometri sumber dihapus atau digeser.
2. Untuk 49 pasangan geometri identik yang satu kelas, satu fitur dipilih sebagai perwakilan sementara berdasarkan prioritas status ekstraksi `OK`, jumlah piksel bersih lebih banyak, lalu ID stabil. Anggota pasangan lainnya ditandai sebagai duplikat dan dikeluarkan hanya dari subset 303.
3. `WATER_REVIEW_036` dipertahankan sebagai kandidat kelas 5 (laut/perairan pesisir) berdasarkan review visual pengguna. Ini tetap kandidat dan memerlukan verifikasi independen untuk laporan final.
4. `JTM0311` memiliki geometri identik dengan `WATER_REVIEW_036` namun label awalnya danau. Atribut kelas aslinya dipertahankan di master untuk jejak audit, tetapi fitur tersebut dikeluarkan dari subset perwakilan dan diberi status konflik label.
5. `JTM0051` dan `JTM0150` adalah near-duplicate (bukan identik). Keduanya tetap berada di subset sementara tetapi memiliki `near_duplicate_group_id` yang sama; keduanya jangan dipisahkan antara training/testing sebelum review visual selesai.
6. Semua fitur tetap `usable_for_training=False` dan `training_ready_v6=False`. Ini adalah dataset review, bukan dataset training final.

## Input dan reproduksibilitas
Script membangun output ini tanpa mengubah file input. Pemilihan wakil menggunakan CSV hasil retry terkini. Audit footprint memastikan cakupan geometris saja, bukan bebas awan atau label benar.
