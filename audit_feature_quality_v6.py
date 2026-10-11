from pathlib import Path
import sys
import pandas as pd

PROJECT = Path(__file__).resolve().parent
INPUT = PROJECT / 'outputs' / 'dataset_v6_review' / 'fitur_sentinel2a_l2a_6kelas_v6_kandidat_representatif_retried.csv'
OUTDIR = PROJECT / 'outputs' / 'dataset_v6_review'

# Screening thresholds only, for review prioritization; they do not auto-approve training data.
MIN_CLEAR_PIXELS = 10
MIN_CLEAR_FRACTION = 0.50


def find_col(df, options):
    lookup = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lookup:
            return lookup[opt.lower()]
    return None


def main():
    print('=== Audit kualitas fitur Sentinel-2 v6 ===')
    print(f'Proyek : {PROJECT}')
    print(f'Input  : {INPUT}')
    if not INPUT.exists():
        print('GAGAL: CSV input tidak ditemukan. Pastikan script berada di folder proyek PSD.')
        return 2

    df = pd.read_csv(INPUT, dtype={'sample_id': str}, encoding='utf-8-sig')
    required = ['sample_id', 'class_name', 'status', 'clear_pixel_count', 'clear_pixel_fraction_of_polygon']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print('GAGAL: kolom wajib tidak ditemukan:', ', '.join(missing))
        print('Kolom tersedia:', ', '.join(df.columns))
        return 2

    df['clear_pixel_count_num'] = pd.to_numeric(df['clear_pixel_count'], errors='coerce').fillna(0)
    df['clear_fraction_num'] = pd.to_numeric(df['clear_pixel_fraction_of_polygon'], errors='coerce').fillna(0)

    def classify(row):
        if str(row['status']).upper() != 'OK':
            return 'NO_CLEAR_PIXELS_OR_EXTRACTION_FAILED'
        if row['clear_pixel_count_num'] < MIN_CLEAR_PIXELS and row['clear_fraction_num'] < MIN_CLEAR_FRACTION:
            return 'VERY_FEW_PIXELS_AND_LOW_FRACTION'
        if row['clear_pixel_count_num'] < MIN_CLEAR_PIXELS:
            return 'VERY_FEW_CLEAR_PIXELS'
        if row['clear_fraction_num'] < MIN_CLEAR_FRACTION:
            return 'LOW_CLEAR_FRACTION'
        return 'REVIEWABLE_NOT_AUTO_APPROVED'

    df['quality_review_flag'] = df.apply(classify, axis=1)
    df['quality_review_note'] = df['quality_review_flag'].map({
        'NO_CLEAR_PIXELS_OR_EXTRACTION_FAILED': 'Tidak ada fitur spektral yang lolos masker; jangan training sebelum diperbaiki/dikecualikan dengan alasan.',
        'VERY_FEW_PIXELS_AND_LOW_FRACTION': 'Piksel bersih sangat sedikit dan fraksi cakupan rendah; periksa geometri/representativitas.',
        'VERY_FEW_CLEAR_PIXELS': 'Piksel bersih < 10; statistik spektral berpotensi tidak stabil.',
        'LOW_CLEAR_FRACTION': 'Kurang dari 50% piksel poligon lolos masker; pertimbangkan retrial/peninjauan.',
        'REVIEWABLE_NOT_AUTO_APPROVED': 'Memenuhi penyaring numerik awal saja; label dan duplikasi tetap perlu diverifikasi.'
    })

    # Identify possible acquisition/date columns without assuming one particular script schema.
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'datetime', 'acquisition', 'selected_time', 'time_range'])]
    date_summary_rows = []
    for c in date_cols:
        vals = df[c].dropna().astype(str).str.strip()
        vals = vals[vals.ne('')]
        if len(vals) and vals.nunique() <= max(40, int(len(df) * 0.5)):
            for val, count in vals.value_counts().items():
                date_summary_rows.append({'date_column': c, 'date_value': val, 'sample_count': int(count)})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    review_path = OUTDIR / 'audit_kualitas_fitur_v6_per_sampel.csv'
    class_path = OUTDIR / 'audit_kualitas_fitur_v6_per_kelas.csv'
    flags_path = OUTDIR / 'audit_kualitas_fitur_v6_perlu_review.csv'
    dates_path = OUTDIR / 'audit_tanggal_akuisisi_fitur_v6.csv'

    # Keep original columns and add clear, explicitly preliminary review columns.
    df.to_csv(review_path, index=False, encoding='utf-8-sig')
    class_summary = (df.groupby(['class_name', 'quality_review_flag'], dropna=False)
                       .size().reset_index(name='sample_count')
                       .sort_values(['class_name', 'quality_review_flag']))
    class_summary.to_csv(class_path, index=False, encoding='utf-8-sig')
    flagged = df[df['quality_review_flag'] != 'REVIEWABLE_NOT_AUTO_APPROVED'].copy()
    flagged.to_csv(flags_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(date_summary_rows, columns=['date_column', 'date_value', 'sample_count']).to_csv(dates_path, index=False, encoding='utf-8-sig')

    print(f'Jumlah baris : {len(df)}')
    print(f'Jumlah sample_id unik: {df["sample_id"].nunique()}')
    print('\nFlag kualitas awal (bukan keputusan final training):')
    print(df['quality_review_flag'].value_counts(dropna=False).to_string())
    print('\nRingkasan per kelas:')
    print(class_summary.to_string(index=False))
    print('\nKolom tanggal/akuisisi terdeteksi:', ', '.join(date_cols) if date_cols else '(tidak terdeteksi; cek skema CSV)')
    print('\nFile dibuat:')
    print(f' - {review_path}')
    print(f' - {class_path}')
    print(f' - {flags_path}')
    print(f' - {dates_path}')
    print('\nAmbang sementara untuk menandai review: clear pixels < 10 dan/atau fraksi bersih < 0.50.')
    print('Ambang ini hanya untuk prioritas audit; tidak otomatis mengesahkan atau menghapus sampel.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
