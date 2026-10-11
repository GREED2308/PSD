#!/usr/bin/env python3
"""Create a traceable v6 review dataset from the six-class GeoJSON and audits.

Default Windows project layout:
  data/geojson/dataset_6kelas.geojson
  outputs/sentinel2_features_csv/audit_duplikasi_geometri_persis.csv
  outputs/sentinel2_features_csv/audit_overlap_hampir_duplikat_bangunan.csv
  outputs/sentinel2_features_csv/fitur_sentinel2a_l2a_6kelas_retried.csv

Original inputs are read only and never overwritten.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def parse_int(x, default=-1):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def choose_representative(a: str, b: str, feature_by_id: dict[str, dict]) -> tuple[str, str]:
    """Prefer extractable OK sample, then more clear pixels, then stable lower ID."""
    ra, rb = feature_by_id.get(a, {}), feature_by_id.get(b, {})
    ka = (1 if ra.get('status') == 'OK' else 0,
          parse_int(ra.get('clear_pixel_count'), 0),
          -parse_int(ra.get('polygon_pixel_count'), 10**9),
          a)
    kb = (1 if rb.get('status') == 'OK' else 0,
          parse_int(rb.get('clear_pixel_count'), 0),
          -parse_int(rb.get('polygon_pixel_count'), 10**9),
          b)
    # ID tie-break is lower/lexically earlier; invert selection after numerical comparison.
    if ka[:3] != kb[:3]:
        return (a, b) if ka[:3] > kb[:3] else (b, a)
    return (a, b) if a < b else (b, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--source-geojson', type=Path)
    ap.add_argument('--exact-audit', type=Path)
    ap.add_argument('--near-audit', type=Path)
    ap.add_argument('--features-csv', type=Path)
    ap.add_argument('--out-dir', type=Path)
    args = ap.parse_args()
    project = args.project_dir.resolve()
    out_dir = (args.out_dir or (project / 'outputs' / 'dataset_v6_review')).resolve()

    def resolve(p, candidates, label):
        if p:
            q = p.resolve()
            if not q.exists(): raise FileNotFoundError(f'{label} tidak ditemukan: {q}')
            return q
        for cand in candidates:
            if cand.exists(): return cand.resolve()
        raise FileNotFoundError(f'{label} tidak ditemukan. Kandidat: ' + '; '.join(str(c) for c in candidates))

    source_path = resolve(args.source_geojson, [
        project / 'data' / 'geojson' / 'dataset_6kelas.geojson',
        project / 'dataset_6kelas_gabungan_v5_353_kandidat.geojson',
        project / 'data' / 'geojson' / 'dataset_6kelas_gabungan_v5_353_kandidat.geojson'], 'GeoJSON utama')
    review_dir = project / 'outputs' / 'sentinel2_features_csv'
    exact_path = resolve(args.exact_audit, [review_dir / 'audit_duplikasi_geometri_persis.csv', project / 'audit_duplikasi_geometri_persis.csv'], 'Audit geometri duplikat')
    near_path = resolve(args.near_audit, [review_dir / 'audit_overlap_hampir_duplikat_bangunan.csv', project / 'audit_overlap_hampir_duplikat_bangunan.csv'], 'Audit near-duplicate')
    feat_path = resolve(args.features_csv, [review_dir / 'fitur_sentinel2a_l2a_6kelas_retried.csv', project / 'fitur_sentinel2a_l2a_6kelas_retried.csv'], 'CSV fitur hasil retry')

    gj = json.loads(source_path.read_text(encoding='utf-8'))
    features = gj.get('features', [])
    if not features: raise ValueError('GeoJSON tidak berisi features.')
    by_id = {str(f.get('properties', {}).get('sample_id')): f for f in features}
    if len(by_id) != len(features): raise ValueError('sample_id tidak unik pada GeoJSON sumber.')
    exact_rows = read_csv(exact_path)
    near_rows = read_csv(near_path)
    feature_rows = read_csv(feat_path)
    feat_by_id = {r.get('sample_id', ''): r for r in feature_rows}

    # 49 exact same-class pairs; choose one representative per pair.
    exact_groups = []
    group_for_id = {}
    reps = set()
    redundant = set()
    conflict_pairs = []
    group_seq = 0
    for r in exact_rows:
        a, b = r.get('sample_id_a', ''), r.get('sample_id_b', '')
        if not a or not b or a not in by_id or b not in by_id:
            continue
        same = str(r.get('same_class', '')).strip().lower() == 'true'
        if same:
            group_seq += 1
            gid = f'EXACT_DUP_BLDG_{group_seq:03d}'
            rep, drop = choose_representative(a, b, feat_by_id)
            reps.add(rep); redundant.add(drop)
            group_for_id[a] = gid; group_for_id[b] = gid
            exact_groups.append({
                'group_id': gid, 'sample_id_a': a, 'sample_id_b': b,
                'class_a': r.get('class_a', ''), 'class_b': r.get('class_b', ''),
                'status_a_latest': feat_by_id.get(a, {}).get('status', r.get('status_a', '')),
                'clear_pixels_a_latest': feat_by_id.get(a, {}).get('clear_pixel_count', r.get('clear_pixels_a', '')),
                'status_b_latest': feat_by_id.get(b, {}).get('status', r.get('status_b', '')),
                'clear_pixels_b_latest': feat_by_id.get(b, {}).get('clear_pixel_count', r.get('clear_pixels_b', '')),
                'representative_candidate': rep, 'redundant_exact_duplicate': drop,
                'decision': 'keep_one_representative; retain both in master; exclude duplicate from v6 representative subset'
            })
        else:
            conflict_pairs.append(r)

    water_id, lake_id = 'WATER_REVIEW_036', 'JTM0311'
    if water_id not in by_id or lake_id not in by_id:
        raise ValueError('ID konflik WATER_REVIEW_036/JTM0311 tidak ditemukan dalam GeoJSON.')

    near_group_ids = {}
    for i, r in enumerate(near_rows, start=1):
        a, b = r.get('sample_id_a', ''), r.get('sample_id_b', '')
        if a in by_id and b in by_id:
            gid = f'NEAR_DUP_BLDG_{i:03d}'
            near_group_ids[a] = gid
            near_group_ids[b] = gid

    # Derive latest feature info, preserve original attributes, append v6 audit metadata.
    master_features = []
    for f in features:
        props = dict(f.get('properties', {}))
        sid = str(props.get('sample_id', ''))
        latest = feat_by_id.get(sid, {})
        props['label_status_original_v6'] = props.get('label_status', '')
        props['dataset_revision'] = 'v6_review_2026-10-11'
        props['latest_extraction_status'] = latest.get('status', '')
        props['latest_clear_pixel_count'] = parse_int(latest.get('clear_pixel_count'), 0) if latest else None
        props['latest_acquisition_date_requested'] = latest.get('acquisition_date_requested', '') if latest else ''
        props['exact_duplicate_group_id'] = group_for_id.get(sid, '')
        props['near_duplicate_group_id'] = near_group_ids.get(sid, '')
        props['v6_in_representative_review_subset'] = True
        props['v6_review_decision'] = 'REVIEW_LABEL_AND_FEATURE_QUALITY'
        props['v6_review_notes'] = ''

        if sid in reps:
            props['exact_duplicate_role'] = 'REPRESENTATIVE_CANDIDATE'
            props['v6_review_decision'] = 'KEEP_REPRESENTATIVE_PENDING_LABEL_VERIFICATION'
            props['v6_review_notes'] = 'Geometri identik dengan fitur lain dalam grup; dipilih sebagai perwakilan sementara berdasarkan status/piksel fitur, bukan validasi label.'
        elif sid in redundant:
            props['exact_duplicate_role'] = 'REDUNDANT_EXACT_DUPLICATE'
            props['v6_in_representative_review_subset'] = False
            props['v6_review_decision'] = 'EXCLUDE_REDUNDANT_EXACT_DUPLICATE_FROM_REPRESENTATIVE_SUBSET'
            props['v6_review_notes'] = 'Geometri identik; seluruh atribut dan ID tetap disimpan di GeoJSON master untuk keterlacakan.'
        else:
            props['exact_duplicate_role'] = ''

        if sid in near_group_ids:
            props['v6_review_notes'] = (props['v6_review_notes'] + ' ' if props['v6_review_notes'] else '') + 'Bagian dari pasangan bangunan hampir duplikat; jangan membagi pasangan ini antara training dan testing; review visual masih disarankan.'
            if sid not in redundant:
                props['v6_review_decision'] = 'KEEP_BUT_GROUP_FOR_SPATIAL_SPLIT_PENDING_VISUAL_REVIEW'

        if sid == water_id:
            props['exact_duplicate_group_id'] = 'CROSS_CLASS_DUP_WATER_REVIEW_036_JTM0311'
            props['exact_duplicate_role'] = 'CONFLICTING_CLASS_DUPLICATE_REPRESENTATIVE_CANDIDATE'
            props['v6_review_decision'] = 'KEEP_AS_CLASS_5_COASTAL_WATER_CANDIDATE'
            props['v6_review_notes'] = 'Review visual pengguna pada peta pesisir West Kalianget tampak menunjukkan badan air terhubung dengan perairan lebih luas. Pertahankan class_id=5 sebagai kandidat laut/perairan pesisir; ini interpretasi visual dan tetap memerlukan verifikasi independen sebelum training.'
        elif sid == lake_id:
            props['exact_duplicate_group_id'] = 'CROSS_CLASS_DUP_WATER_REVIEW_036_JTM0311'
            props['exact_duplicate_role'] = 'CONFLICTING_CLASS_DUPLICATE_EXCLUDED'
            props['v6_in_representative_review_subset'] = False
            props['v6_review_decision'] = 'EXCLUDE_CONFLICTING_DUPLICATE_FROM_REPRESENTATIVE_SUBSET'
            props['v6_review_notes'] = 'Geometri identik dengan WATER_REVIEW_036 yang dipertahankan sebagai kandidat kelas 5 setelah review visual. Properti class_id=6 dipertahankan di master untuk audit, tetapi fitur ini dikeluarkan dari subset representatif sampai konflik diselesaikan secara independen.'
            props['label_status'] = 'DUPLIKAT_LINTAS_KELAS_DIKECUALIKAN'
            props['usable_for_training'] = False
            props['usable_for_training_now'] = False

        # Explicit guard: the output is a review dataset, not a training-ready dataset.
        props['usable_for_training'] = False
        props['usable_for_training_now'] = False
        props['training_ready_v6'] = False
        master_features.append({**f, 'properties': props})

    rep_features = [f for f in master_features if f['properties'].get('v6_in_representative_review_subset')]
    # Keep the GeoJSON bbox/properties but calculate subset bbox from geometries if available.
    def coords_iter(obj):
        if isinstance(obj, list):
            if len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
                yield (obj[0], obj[1])
            else:
                for x in obj: yield from coords_iter(x)
    def calc_bbox(fs):
        pts = []
        for ft in fs: pts.extend(coords_iter(ft.get('geometry', {}).get('coordinates', [])))
        return [min(x for x,y in pts), min(y for x,y in pts), max(x for x,y in pts), max(y for x,y in pts)] if pts else None
    master_gj = dict(gj); master_gj['name'] = 'PSD dataset 6 kelas v6 review (353 fitur; master dengan penanda audit)'; master_gj['description'] = (gj.get('description','') + ' | v6 review: semua fitur sumber dipertahankan; keputusan audit ditambahkan; bukan dataset training final.').strip(' |')
    master_gj['features'] = master_features
    rep_gj = dict(gj); rep_gj['name'] = 'PSD dataset 6 kelas v6 kandidat representatif (303 fitur)'; rep_gj['description'] = 'Subset kerja: mengeluarkan satu anggota dari setiap pasangan geometri identik satu kelas dan mengeluarkan JTM0311 yang berkonflik label dengan WATER_REVIEW_036. Near-duplicate JTM0051/JTM0150 tetap ada dan ditandai agar tidak dipisahkan antara training/testing. Label belum terverifikasi; tidak siap training.'
    rep_gj['features'] = rep_features
    rep_gj['bbox'] = calc_bbox(rep_features) or rep_gj.get('bbox')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_master = out_dir / 'dataset_6kelas_v6_review.geojson'
    out_subset = out_dir / 'dataset_6kelas_v6_kandidat_representatif_303.geojson'
    out_all_csv = out_dir / 'fitur_sentinel2a_l2a_6kelas_v6_review.csv'
    out_subset_csv = out_dir / 'fitur_sentinel2a_l2a_6kelas_v6_kandidat_representatif.csv'
    out_log = out_dir / 'log_perubahan_dataset_v6.csv'
    out_summary = out_dir / 'ringkasan_dataset_v6.csv'
    out_readme = out_dir / 'README_dataset_v6.md'

    out_master.write_text(json.dumps(master_gj, ensure_ascii=False, indent=2), encoding='utf-8')
    out_subset.write_text(json.dumps(rep_gj, ensure_ascii=False, indent=2), encoding='utf-8')

    # Enrich feature rows using the decisions above.
    meta_by_id = {f['properties']['sample_id']: f['properties'] for f in master_features}
    enriched = []
    for row in feature_rows:
        r = dict(row)
        m = meta_by_id.get(row.get('sample_id', ''), {})
        for col in ['exact_duplicate_group_id','near_duplicate_group_id','exact_duplicate_role','v6_in_representative_review_subset','v6_review_decision','v6_review_notes','training_ready_v6']:
            v = m.get(col, '')
            r[col] = str(v) if isinstance(v, bool) else ('' if v is None else v)
        r['class_id_review_v6'] = m.get('class_id', row.get('class_id',''))
        r['class_name_review_v6'] = m.get('class_name', row.get('class_name',''))
        r['label_status_review_v6'] = m.get('label_status', row.get('label_status',''))
        r['label_status_original_v6'] = m.get('label_status_original_v6', row.get('label_status',''))
        r['training_eligibility_note_v6'] = 'BELUM_SIAP_TRAINING: label kandidat; audit fitur, piksel, duplikasi dan split spasial masih diperlukan.'
        enriched.append(r)
    all_cols = list(dict.fromkeys(k for row in enriched for k in row.keys()))
    write_csv(out_all_csv, enriched, all_cols)
    subset_ids = {f['properties']['sample_id'] for f in rep_features}
    subset_rows = [r for r in enriched if r.get('sample_id') in subset_ids]
    write_csv(out_subset_csv, subset_rows, all_cols)

    log_rows = []
    # one entry per exact duplicate pair
    log_rows.extend(exact_groups)
    log_rows.append({
        'group_id':'CROSS_CLASS_DUP_WATER_REVIEW_036_JTM0311',
        'sample_id_a':water_id, 'sample_id_b':lake_id, 'class_a':'laut', 'class_b':'danau',
        'status_a_latest':feat_by_id.get(water_id,{}).get('status',''),
        'clear_pixels_a_latest':feat_by_id.get(water_id,{}).get('clear_pixel_count',''),
        'status_b_latest':feat_by_id.get(lake_id,{}).get('status',''),
        'clear_pixels_b_latest':feat_by_id.get(lake_id,{}).get('clear_pixel_count',''),
        'representative_candidate':water_id, 'redundant_exact_duplicate':lake_id,
        'decision':'keep WATER_REVIEW_036 as class-5 coastal-water candidate after visual review; exclude JTM0311 from v6 representative subset pending independent validation'
    })
    for r in near_rows:
        log_rows.append({
            'group_id':'NEAR_DUP_BLDG_001', 'sample_id_a':r.get('sample_id_a',''), 'sample_id_b':r.get('sample_id_b',''),
            'class_a':r.get('class_name',''), 'class_b':r.get('class_name',''),
            'status_a_latest':feat_by_id.get(r.get('sample_id_a',''),{}).get('status',r.get('status_a','')),
            'clear_pixels_a_latest':feat_by_id.get(r.get('sample_id_a',''),{}).get('clear_pixel_count',r.get('clear_pixels_a','')),
            'status_b_latest':feat_by_id.get(r.get('sample_id_b',''),{}).get('status',r.get('status_b','')),
            'clear_pixels_b_latest':feat_by_id.get(r.get('sample_id_b',''),{}).get('clear_pixel_count',r.get('clear_pixels_b','')),
            'representative_candidate':'', 'redundant_exact_duplicate':'',
            'decision':'near duplicate, not exact: both remain in subset for now; assign same spatial split group or complete visual review before modelling'
        })
    write_csv(out_log, log_rows)

    # Summary by class and latest feature status for master and representative subset.
    summ = []
    for label, fs in [('master_v6_review', master_features), ('representative_review_subset', rep_features)]:
        cnt = Counter((str(f['properties'].get('class_id')), str(f['properties'].get('class_name'))) for f in fs)
        for (cid, cname), n in sorted(cnt.items(), key=lambda x: int(x[0][0])):
            status_ct = Counter(feat_by_id.get(f['properties']['sample_id'], {}).get('status', 'MISSING') for f in fs if str(f['properties'].get('class_id'))==cid)
            summ.append({'dataset':label,'class_id':cid,'class_name':cname,'feature_count':n,'latest_OK':status_ct.get('OK',0),'latest_NO_CLEAR_PIXELS':status_ct.get('NO_CLEAR_PIXELS',0),'labels_verified_count':sum(1 for f in fs if str(f['properties'].get('class_id'))==cid and 'TERVERIFIKASI' in str(f['properties'].get('label_status','')).upper() and 'BELUM' not in str(f['properties'].get('label_status','')).upper())})
    write_csv(out_summary, summ)
    out_readme.write_text(f'''# Dataset kerja PSD v6 — review

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
''', encoding='utf-8')

    print('=== Dataset v6 review dibuat ===')
    print(f'Sumber GeoJSON : {source_path}')
    print(f'CSV fitur      : {feat_path}')
    print(f'Jumlah master  : {len(master_features)}')
    print(f'Jumlah subset  : {len(rep_features)}')
    print(f'Pasangan exact same-class : {len(exact_groups)}')
    print(f'Konflik label lintas kelas: {len(conflict_pairs)}')
    print(f'Near-duplicate            : {len(near_rows)}')
    print(f'CSV fitur status terbaru: {Counter(r.get("status", "MISSING") for r in feature_rows)}')
    print('\nFile keluaran:')
    for p in [out_master, out_subset, out_all_csv, out_subset_csv, out_log, out_summary, out_readme]: print(' -', p)
    print('\nPENTING: output v6 masih kandidat review dan tidak siap training; seluruh label belum diverifikasi.')

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f'Gagal: {type(e).__name__}: {e}', file=sys.stderr)
        raise SystemExit(1)
