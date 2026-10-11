"""Audit cakupan produk Sentinel-2A terhadap setiap poligon sampel PSD.

Input default:
  data/geojson/dataset_6kelas.geojson
  outputs/sentinel2_search/kandidat_sentinel2a_l2a.csv

Output:
  outputs/sentinel2_search/cakupan_produk_per_sampel.csv
  outputs/sentinel2_search/ringkasan_cakupan_sampel.csv
  outputs/sentinel2_search/ringkasan_cakupan_per_kelas.csv

Tidak mengunduh raster Sentinel-2. Script hanya mengambil metadata STAC untuk
58 item kandidat dan memeriksa irisan geometri sampel dengan footprint produk.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from shapely.geometry import shape
from shapely.prepared import prep

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "data" / "geojson" / "dataset_6kelas.geojson"
CANDIDATES_CSV = PROJECT_DIR / "outputs" / "sentinel2_search" / "kandidat_sentinel2a_l2a.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_search"
TIMEOUT_SECONDS = 45
RETRIES = 3


def read_features(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        raise ValueError(f"Format bukan FeatureCollection/Feature: {path}")
    out = []
    for idx, f in enumerate(features):
        if not f.get("geometry"):
            continue
        props = f.get("properties") or {}
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        out.append({
            "sample_id": str(props.get("sample_id") or props.get("id") or f"FEATURE_{idx+1:04d}"),
            "class_id": props.get("class_id", ""),
            "class_name": props.get("class_name") or props.get("class_display") or "",
            "label_status": props.get("label_status") or props.get("validation_status") or "",
            "usable_for_training": props.get("usable_for_training", props.get("usable_for_training_now", False)),
            "source_file": props.get("source_file", ""),
            "geometry": geom,
        })
    if not out:
        raise ValueError(f"Tidak ada fitur bergeometri di {path}")
    return out


def get_item_json(url: str, session: requests.Session) -> dict[str, Any]:
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            obj = response.json()
            if not obj.get("geometry"):
                raise ValueError("Respons STAC tidak memiliki geometry")
            return obj
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Gagal membaca item STAC {url}: {last_error}")


def tile_from_item_id(item_id: str, fallback: str = "") -> str:
    # IDs such as S2A_MSIL2A_..._T49MFN_...
    match = re.search(r"_(T\d{2}[A-Z]{3})_", str(item_id))
    return match.group(1) if match else (fallback or "")


def main() -> int:
    print("=== Audit cakupan produk Sentinel-2A per sampel ===")
    print(f"Folder proyek : {PROJECT_DIR}")
    print(f"Dataset sampel: {SAMPLE_PATH}")
    print(f"CSV kandidat  : {CANDIDATES_CSV}")

    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            f"CSV kandidat tidak ditemukan: {CANDIDATES_CSV}\n"
            "Jalankan search_sentinel2_cdse_fixed.py terlebih dahulu."
        )
    samples = read_features(SAMPLE_PATH)
    candidates = pd.read_csv(CANDIDATES_CSV, keep_default_na=False)
    if candidates.empty:
        raise ValueError("CSV kandidat kosong")
    if "item_self_link" not in candidates.columns:
        raise ValueError("CSV tidak memiliki kolom item_self_link")

    print(f"Jumlah sampel terbaca: {len(samples)}")
    print(f"Jumlah kandidat produk: {len(candidates)}")
    print("Mengambil metadata geometri item STAC; tidak mengunduh raster...")

    session = requests.Session()
    session.headers.update({"User-Agent": "PSD-EastJava-coverage-audit/1.0"})
    pair_rows: list[dict[str, Any]] = []
    products_ok = 0
    products_failed: list[tuple[str, str]] = []

    for n, row in candidates.iterrows():
        item_id = str(row.get("item_id", ""))
        item_url = str(row.get("item_self_link", "")).strip()
        if not item_url:
            products_failed.append((item_id, "item_self_link kosong"))
            continue
        try:
            item = get_item_json(item_url, session)
            item_geom = shape(item["geometry"])
            if item_geom.is_empty:
                raise ValueError("Footprint kosong")
        except Exception as exc:
            products_failed.append((item_id, str(exc)))
            print(f"[{n+1}/{len(candidates)}] GAGAL metadata: {item_id}: {exc}")
            continue

        products_ok += 1
        props = item.get("properties") or {}
        tile = tile_from_item_id(item_id, str(row.get("mgrs_tile", "")))
        date_text = str(row.get("datetime_utc", props.get("datetime", "")))
        date = date_text[:10]
        cloud_raw = row.get("cloud_cover_pct", props.get("eo:cloud_cover", ""))
        try:
            cloud = float(cloud_raw)
        except (TypeError, ValueError):
            cloud = float("nan")

        prepared_item = prep(item_geom)
        matches = 0
        for sample in samples:
            geom = sample["geometry"]
            # Prepared geometry accelerates footprint intersection tests.
            if prepared_item.intersects(geom):
                matches += 1
                pair_rows.append({
                    "sample_id": sample["sample_id"],
                    "class_id": sample["class_id"],
                    "class_name": sample["class_name"],
                    "label_status": sample["label_status"],
                    "usable_for_training": sample["usable_for_training"],
                    "source_file": sample["source_file"],
                    "item_id": item_id,
                    "datetime_utc": date_text,
                    "acquisition_date": date,
                    "platform": row.get("platform", props.get("platform", "")),
                    "mgrs_tile": tile,
                    "cloud_cover_pct": cloud,
                    "item_self_link": item_url,
                })
        print(f"[{n+1}/{len(candidates)}] {tile or 'tile?' } {date} | awan {cloud:g}% | sampel beririsan: {matches}")

    if not products_ok:
        raise RuntimeError("Tidak ada metadata produk yang berhasil dibaca. Periksa koneksi/API.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_cols = [
        "sample_id", "class_id", "class_name", "label_status", "usable_for_training",
        "source_file", "item_id", "datetime_utc", "acquisition_date", "platform",
        "mgrs_tile", "cloud_cover_pct", "item_self_link"
    ]
    pairs = pd.DataFrame(pair_rows, columns=pair_cols)
    pairs_path = OUTPUT_DIR / "cakupan_produk_per_sampel.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    # Build one row for every original sample, including ones not covered by any candidate.
    per_sample_rows = []
    for sample in samples:
        hits = pairs[pairs["sample_id"] == sample["sample_id"]].copy() if not pairs.empty else pd.DataFrame()
        if not hits.empty:
            hits["cloud_cover_pct"] = pd.to_numeric(hits["cloud_cover_pct"], errors="coerce")
            hits = hits.sort_values(["cloud_cover_pct", "acquisition_date"], ascending=[True, False], na_position="last")
            best = hits.iloc[0]
            best_item = best["item_id"]
            best_date = best["acquisition_date"]
            best_tile = best["mgrs_tile"]
            best_cloud = best["cloud_cover_pct"]
            tiles = ";".join(sorted(set(str(v) for v in hits["mgrs_tile"] if str(v))))
            products = ";".join(hits["item_id"].astype(str).tolist())
            dates = ";".join(sorted(set(hits["acquisition_date"].astype(str))))
        else:
            best_item = best_date = best_tile = tiles = products = dates = ""
            best_cloud = float("nan")
        per_sample_rows.append({
            "sample_id": sample["sample_id"],
            "class_id": sample["class_id"],
            "class_name": sample["class_name"],
            "label_status": sample["label_status"],
            "usable_for_training": sample["usable_for_training"],
            "source_file": sample["source_file"],
            "covered_by_candidate": not hits.empty,
            "candidate_product_count": len(hits),
            "candidate_tiles": tiles,
            "candidate_dates": dates,
            "best_product_min_scene_cloud": best_item,
            "best_product_date": best_date,
            "best_product_tile": best_tile,
            "best_product_cloud_pct": best_cloud,
            "all_candidate_products": products,
        })
    per_sample = pd.DataFrame(per_sample_rows)
    per_sample_path = OUTPUT_DIR / "ringkasan_cakupan_sampel.csv"
    per_sample.to_csv(per_sample_path, index=False, encoding="utf-8-sig")

    per_class = per_sample.groupby(["class_id", "class_name"], dropna=False).agg(
        jumlah_sampel=("sample_id", "count"),
        tercakup=("covered_by_candidate", "sum"),
        belum_tercakup=("covered_by_candidate", lambda s: (~s.astype(bool)).sum()),
        total_pasangan_produk_sampel=("candidate_product_count", "sum"),
    ).reset_index()
    per_class_path = OUTPUT_DIR / "ringkasan_cakupan_per_kelas.csv"
    per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")

    print("\n=== SELESAI ===")
    print(f"Produk metadata berhasil dibaca: {products_ok}/{len(candidates)}")
    print(f"Jumlah irisan produk-sampel: {len(pairs)}")
    print(f"Sampel yang memiliki minimal satu kandidat produk: {int(per_sample['covered_by_candidate'].sum())}/{len(per_sample)}")
    print("\nRingkasan cakupan per kelas:")
    print(per_class.to_string(index=False))
    print("\nFile hasil:")
    print(f"  {pairs_path}")
    print(f"  {per_sample_path}")
    print(f"  {per_class_path}")
    if products_failed:
        print(f"\nPeringatan: metadata gagal dibaca untuk {len(products_failed)} produk; lihat daftar berikut:")
        for item_id, err in products_failed:
            print(f"  - {item_id}: {err}")
    print("\nCatatan: irisan footprint hanya memastikan cakupan geografis, bukan awan lokal atau kebenaran label.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
