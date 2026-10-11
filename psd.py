import pandas as pd
import numpy as np
import inspect
from scipy.interpolate import interp1d
import tsfel.feature_extraction.features as tsfel_features

# ---------- 1. Muat 3 CSV time series milik sendiri ----------
NO2_FILE = 'data/processed/menganti_NO2_daily.csv'
SO2_FILE = 'data/processed/menganti_SO2_daily.csv'
CO_FILE  = 'data/processed/menganti_CO_daily.csv'

df_no2 = pd.read_csv(NO2_FILE)
df_so2 = pd.read_csv(SO2_FILE)
df_co  = pd.read_csv(CO_FILE)

df_no2['date'] = pd.to_datetime(df_no2['date'])
df_so2['date'] = pd.to_datetime(df_so2['date'])
df_co['date'] = pd.to_datetime(df_co['date'])

# Ambil hanya tanggal + polutan
df_no2 = df_no2[['date', 'NO2']]
df_so2 = df_so2[['date', 'SO2']]
df_co  = df_co[['date', 'CO']]

# Gabungkan menjadi satu file
df = (
    df_no2
    .merge(df_so2, on='date', how='outer')
    .merge(df_co, on='date', how='outer')
    .sort_values('date')
    .reset_index(drop=True)
)

print("Kolom:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

pollutants = ['NO2', 'SO2', 'CO']
fs = 1
# ---------- 2. Daftar PERSIS 68 fitur yang diminta ----------
FEATURE_LIST = """abs_energy auc autocorr average_power calc_centroid calc_max calc_mean
calc_median calc_min calc_std calc_var dfa distance ecdf ecdf_percentile ecdf_percentile_count
ecdf_slope entropy fundamental_frequency higuchi_fractal_dimension hist_mode human_range_energy
hurst_exponent interq_range kurtosis lempel_ziv lpcc max_frequency max_power_spectrum
maximum_fractal_length mean_abs_deviation mean_abs_diff mean_diff median_abs_deviation
median_abs_diff median_diff median_frequency mfcc mse negative_turning neighbourhood_peaks
petrosian_fractal_dimension pk_pk_distance positive_turning power_bandwidth rms skewness slope
spectral_centroid spectral_decrease spectral_distance spectral_entropy spectral_kurtosis
spectral_positive_turning spectral_roll_off spectral_roll_on spectral_skewness spectral_slope
spectral_spread spectral_variation spectrogram_mean_coeff sum_abs_diff wavelet_abs_mean
wavelet_energy wavelet_entropy wavelet_std wavelet_var zero_cross""".split()

print(f"Jumlah fitur per polutan: {len(FEATURE_LIST)}")
print(f"Total target fitur keseluruhan: {len(FEATURE_LIST) * len(pollutants)}")


def to_scalar(result):
    if isinstance(result, dict) and "values" in result:
        result = result["values"]
    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float)
        return float(np.nanmean(arr))
    return float(result)


def extract_one(fn_name, signal, fs):
    fn = getattr(tsfel_features, fn_name)
    params = inspect.signature(fn).parameters
    if "fs" in params:
        result = fn(signal, fs)
    else:
        result = fn(signal)
    return to_scalar(result)


# Dictionary untuk menampung seluruh hasil ekstraksi dari semua polutan
combined_row = {}

def polynomial_interpolate(df, date_col, value_col, degree=3):
    data = df[[date_col, value_col]].copy()

    # Pastikan numerik
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')

    # Ubah tanggal menjadi angka agar bisa digunakan sebagai sumbu x
    x = data[date_col].astype('int64').to_numpy()
    y = data[value_col].to_numpy(dtype=float)

    # Ambil data yang valid
    mask_valid = np.isfinite(y)

    x_valid = x[mask_valid]
    y_valid = y[mask_valid]

    if len(x_valid) <= degree:
        raise ValueError(
            f"Data valid untuk {value_col} hanya {len(x_valid)}, "
            f"tidak cukup untuk polynomial degree {degree}."
        )

    # Polynomial interpolation
    interpolator = interp1d(
        x_valid,
        y_valid,
        kind=degree,
        bounds_error=False,
        fill_value="extrapolate"
    )

    # Hanya isi titik yang missing/outlier
    y_filled = y.copy()
    mask_missing = ~mask_valid
    y_filled[mask_missing] = interpolator(x[mask_missing])

    data[value_col] = y_filled

    return data

# ---------- 3. Looping untuk membersihkan dan mengekstraksi tiap polutan ----------
for pollutant in pollutants:
    print(f"\n--- Memproses polutan: {pollutant} ---")
    
    # Buat copy dataframe agar tidak saling menimpa
    df_poly = df[['date', pollutant]].copy()

    # Paksa kolom target jadi numerik
    df_poly[pollutant] = pd.to_numeric(df_poly[pollutant], errors='coerce')
    n_missing_before = df_poly[pollutant].isna().sum()
    print(f"Jumlah NaN/non-numerik pada {pollutant}: {n_missing_before}")

    # Handling outlier dengan IQR
    Q1 = df_poly[pollutant].quantile(0.25)
    Q3 = df_poly[pollutant].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_poly.loc[(df_poly[pollutant] < lower_bound) | (df_poly[pollutant] > upper_bound), pollutant] = np.nan

    # Polynomial interpolation untuk missing value + outlier
    df_clean = polynomial_interpolate(
        df_poly,
        date_col='date',
        value_col=pollutant,
        degree=3
    )

    # Validasi hasil preprocessing
    n_missing_after = df_clean[pollutant].isna().sum()
    print(f"Jumlah NaN setelah polynomial interpolation: {n_missing_after}")

    signal_1d = df_clean[pollutant].astype(float).values

    # Ekstraksi fitur dan beri prefix nama polutan (misal: NO2_abs_energy)
    for fn_name in FEATURE_LIST:
        feature_key = f"{pollutant}_{fn_name}"
        combined_row[feature_key] = extract_one(fn_name, signal_1d, fs)

# ---------- 4. Simpan ke DataFrame final (1 baris, 204 kolom) ----------
extracted_features_final = pd.DataFrame([combined_row])

print(f"\nBerhasil! Total kolom akhir yang dihasilkan: {extracted_features_final.shape[1]}")

# output_filename = 'All-Pollutants-Menganti-TSFEL.csv'
output_filename = 'All-Pollutants-Menganti-TSFEL-Polynomial.csv'
extracted_features_final.to_csv(output_filename, index=False)
print(f"File berhasil disimpan sebagai: {output_filename}")

print("\n=== VALIDASI HASIL ===")

check = pd.read_csv("All-Pollutants-Menganti-TSFEL-Polynomial.csv")

print("Shape:", check.shape)
print("Jumlah kolom:", len(check.columns))
print("Jumlah missing:", check.isna().sum().sum())
print("Jumlah kolom unik:", check.columns.nunique())

print("\n5 kolom pertama:")
print(check.iloc[:, :5])

print("\n5 kolom terakhir:")
print(check.iloc[:, -5:])