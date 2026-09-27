import pandas as pd
import tsfel

# 1. Memuat file CSV data time series
file_path = 'd:/Projects/Kuliah/PSD/time_series_server_metrics.csv'
df = pd.read_csv(file_path)

# Mengambil kolom data target sebagai array 1D
y = df['Total_Transaksi_DB'].values

print("Memulai ekstraksi fitur statistik dengan TSFEL...")

# 2. Mengambil konfigurasi khusus domain 'statistical' saja
cfg = tsfel.get_features_by_domain("statistical")

# 3. Melakukan ekstraksi fitur statistik
stat_features = tsfel.time_series_features_extractor(cfg, y, fs=1)

# 4. Menampilkan hasil di terminal
print("\nEkstraksi fitur statistik selesai!")
print(f"Total fitur statistik yang dihasilkan: {stat_features.shape[1]}")
print("\nRingkasan nilai fitur:")
print(stat_features.T)

# 5. Menyimpan hasil ekstraksi statistik ke file CSV baru
# output_path = 'd:/Projects/Kuliah/PSD/fitur_statistik_tsfel.csv'
# stat_features.to_csv(output_path, index=False)
# print(f"\nFile berhasil disimpan ke: {output_path}")