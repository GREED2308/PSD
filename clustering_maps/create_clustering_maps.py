from pathlib import Path
import hashlib
import re
import pandas as pd
import folium
from folium.plugins import MiniMap

BASE = Path(__file__).resolve().parent
LINEAR_CSV = BASE / 'clustering_linear_final.csv'
POLY_CSV = BASE / 'clustering_polynomial_final.csv'
OUT_DIR = BASE / 'clustering_maps'
OUT_DIR.mkdir(exist_ok=True)

COORDS = {
    'baron': (-7.6033, 112.0601),
    'nunukan': (4.1357, 117.6531),
    'sreseh': (-7.1983, 113.0805),
    'manyar': (-7.1192, 112.5975),
    'kamal': (-7.1662, 112.7214),
    'kedungpring': (-7.2181, 112.1931),
    'gresik kota': (-7.1566, 112.6555),
    'waru pamekasan': (-6.9634, 113.5350),
    'waru': (-6.9634, 113.5350),
    'paciran': (-6.8767, 112.3414),
    'kertosono': (-7.5878, 112.1006),
    'jabon': (-7.5419, 112.7686),
    'menganti': (-7.2618, 112.5857),
    'banyu ajuh': (-7.1585, 112.7280),
    'bandung jogoroto': (-7.5683, 112.2856),
    'widang': (-6.9850, 112.1647),
    'wonoayu': (-7.4338, 112.6172),
    'kwanyar': (-7.1604, 112.8569),
    'sambeng': (-7.2882, 112.2783),
    'kalianget': (-7.0519, 113.9408),
    'cerme': (-7.2281, 112.5583),
    'tikala': (1.4820, 124.8540),
    'kerek': (-6.8778, 111.9169),
    'wonokromo': (-7.3006, 112.7383),
    'asemrowo': (-7.2519, 112.7103),
    'kota sumenep': (-7.0086, 113.8617),
    'socah': (-7.0983, 112.7289),
    'pilangkenceng': (-7.4931, 111.6664),
    'tanah merah': (-7.0931, 112.8286),
    'labang': (-7.1294, 112.7936),
    'widodaren': (-7.3918, 111.2081),
    'bangkalan': (-7.0455, 112.7351),
    'warudoyong': (-6.9298, 106.9161),
    'banyuajuh': (-7.1620, 112.7310),
    'dukun': (-6.9903, 112.5186),
    'kecamatan bangkalan': (-7.0325, 112.7450),
}

COLORS = {
    'Cluster 0': '#e53935',
    'Cluster 1': '#1e88e5',
    'Cluster 2': '#43a047',
    'Cluster 3': '#fb8c00',
    'Cluster 4': '#8e24aa',
    'Cluster 5': '#00acc1',
    'Cluster 6': '#6d4c41',
}


def clean(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def resolve_coord(area: str):
    a = clean(area)
    a_norm = re.sub(r'[^a-z0-9]+', ' ', a).strip()
    # Specific phrases first to avoid matching a generic substring too early.
    keys = sorted(COORDS, key=len, reverse=True)
    for k in keys:
        k_norm = re.sub(r'[^a-z0-9]+', ' ', k).strip()
        if k_norm in a_norm:
            return COORDS[k]
    # Normalized fallback for spelling variants such as Asemrowo/Asem Rwo.
    a2 = a_norm.replace(' ', '')
    for k, coord in COORDS.items():
        k2 = re.sub(r'[^a-z0-9]+', '', k)
        if k2 in a2:
            return coord
    raise ValueError(f'Koordinat tidak ditemukan untuk daerah: {area}')


def deterministic_jitter(name: str, idx: int):
    # Small, deterministic offset to prevent overlapping markers from the same area.
    h = hashlib.sha256(f'{name}|{idx}'.encode()).digest()
    u = int.from_bytes(h[:4], 'big') / 2**32
    v = int.from_bytes(h[4:8], 'big') / 2**32
    return (u - 0.5) * 0.012, (v - 0.5) * 0.012


def prepare(path: Path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
    for c in ['nama', 'daerah', 'Cluster']:
        df[c] = df[c].astype(str).str.replace('"', '', regex=False).str.strip()
    df['Cluster_Label'] = df['Cluster'].map(lambda x: f"Cluster {x.split('_')[-1]}" if 'cluster_' in x.lower() else x)
    coords = []
    for i, row in df.iterrows():
        lat, lon = resolve_coord(row['daerah'])
        dlat, dlon = deterministic_jitter(row['nama'], i)
        coords.append((lat + dlat, lon + dlon))
    df['latitude'], df['longitude'] = zip(*coords)
    return df


def make_map(df, title, subtitle, out_path):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles=None, control_scale=True)

    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Street Map'
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Citra Satelit (Esri)'
    ).add_to(m)

    groups = {}
    for cname in sorted(df['Cluster_Label'].unique()):
        groups[cname] = folium.FeatureGroup(name=cname, show=True)
        groups[cname].add_to(m)

    for _, row in df.iterrows():
        cname = row['Cluster_Label']
        color = COLORS.get(cname, '#616161')
        cnum = cname.split()[-1] if cname.startswith('Cluster ') else cname
        popup = folium.Popup(
            f"<div style='font-family:Arial;font-size:12px;min-width:200px;'>"
            f"<b>{row['nama']}</b><hr style='margin:4px 0;'>"
            f"<b>Daerah:</b> {row['daerah']}<br>"
            f"<b>Cluster:</b> {cname}<br>"
            f"<b>Koordinat:</b> {row['latitude']:.5f}, {row['longitude']:.5f}"
            f"</div>", max_width=300
        )
        icon = folium.DivIcon(
            html=(f"<div style='background:{color};color:#fff;border-radius:50%;width:30px;height:30px;"
                  f"display:flex;align-items:center;justify-content:center;font:700 11px Arial;"
                  f"border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.35);'>C{cnum}</div>"),
            icon_size=(30, 30), icon_anchor=(15, 15)
        )
        folium.Marker(
            [row['latitude'], row['longitude']], icon=icon, popup=popup,
            tooltip=f"{row['nama']} — {cname}"
        ).add_to(groups[cname])

    MiniMap(toggle_display=True, position='bottomright').add_to(m)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    counts = df['Cluster_Label'].value_counts().to_dict()
    legend_items = ''.join(
        f"<div style='display:flex;align-items:center;margin:5px 0;'>"
        f"<span style='display:inline-block;width:13px;height:13px;border-radius:50%;background:{COLORS.get(c, '#616161')};margin-right:7px;'></span>"
        f"<span>{c} — {counts.get(c, 0)} mahasiswa</span></div>"
        for c in sorted(counts)
    )
    legend = f"""
    <div style='position:fixed;bottom:45px;left:20px;z-index:9999;background:#fff;padding:14px 18px;
                border-radius:10px;box-shadow:0 3px 14px rgba(0,0,0,.18);font-family:Arial;min-width:240px;'>
      <div style='font-size:16px;font-weight:700;margin-bottom:4px;'>{title}</div>
      <div style='font-size:12px;color:#666;margin-bottom:10px;'>{subtitle}</div>
      {legend_items}
      <div style='font-size:11px;color:#888;margin-top:9px;'>Klik marker untuk detail mahasiswa.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out_path)


def normalized_name(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def make_comparison(lin, poly):
    a = lin.copy(); b = poly.copy()
    a['name_key'] = a['nama'].map(normalized_name)
    b['name_key'] = b['nama'].map(normalized_name)
    comp = a[['id', 'nama', 'daerah', 'Cluster', 'name_key']].merge(
        b[['id', 'nama', 'daerah', 'Cluster', 'name_key']], on='name_key', how='outer', suffixes=('_linear', '_polynomial'), indicator=True
    )
    comp['same_cluster'] = comp['Cluster_linear'].eq(comp['Cluster_polynomial'])
    return comp


def main():
    lin = prepare(LINEAR_CSV)
    poly = prepare(POLY_CSV)

    make_map(
        lin,
        'Peta Clustering Linear',
        'PCA 37 • K=3 • 37 mahasiswa',
        OUT_DIR / 'peta_clustering_linear.html'
    )
    make_map(
        poly,
        'Peta Clustering Polynomial',
        'PCA 37 • K=7 • 37 mahasiswa',
        OUT_DIR / 'peta_clustering_polynomial.html'
    )

    comp = make_comparison(lin, poly)
    comp.to_csv(OUT_DIR / 'perbandingan_cluster_linear_polynomial.csv', index=False)

    print('=== SELESAI ===')
    print('Linear:', lin[['id','nama','daerah','Cluster']].shape, 'clusters=', lin['Cluster'].value_counts().to_dict())
    print('Polynomial:', poly[['id','nama','daerah','Cluster']].shape, 'clusters=', poly['Cluster'].value_counts().to_dict())
    print('Comparison rows:', len(comp), 'same=', int(comp['same_cluster'].sum()), 'different=', int((~comp['same_cluster']).sum()))
    unresolved = []
    for src, df in [('linear', lin), ('polynomial', poly)]:
        for area in df['daerah']:
            try:
                resolve_coord(area)
            except ValueError:
                unresolved.append((src, area))
    print('Unresolved areas:', unresolved)
    print('Outputs:', OUT_DIR)

if __name__ == '__main__':
    main()
