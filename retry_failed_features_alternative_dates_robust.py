from __future__ import annotations
"""Retry failed Sentinel-2 sample extractions on alternative dates, with OAuth network retries.

Uses extract_features_to_csv_sentinel2.py helpers so band selection, geometry masking,
feature formulas, and SCL mask are consistent with the baseline extraction.
Never overwrites the original feature CSV. Saves a separate merged checkpoint and retry log.
"""
import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import extract_features_to_csv_sentinel2 as base

# OAuth/CDSE can occasionally time out. Add bounded retry with exponential backoff.
_original_get_token = base.get_token
def _get_token_with_retry(session: requests.Session) -> str:
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            return _original_get_token(session)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as exc:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"OAuth endpoint tidak terjangkau setelah {max_attempts} percobaan. "
                    "Periksa koneksi internet/VPN/DNS lalu jalankan ulang. "
                    "Tidak ada secret/token yang dicetak."
                ) from exc
            wait_s = min(5 * (2 ** (attempt - 1)), 40)
            print(
                f"OAuth timeout/koneksi ({attempt}/{max_attempts}); "
                f"coba lagi dalam {wait_s} detik..."
            )
            time.sleep(wait_s)

base.get_token = _get_token_with_retry

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sentinel2_features_csv"
BASE_FEATURES = OUTPUT_DIR / "fitur_sentinel2a_l2a_6kelas.csv"
MERGED_OUTPUT = OUTPUT_DIR / "fitur_sentinel2a_l2a_6kelas_retried.csv"
RETRY_LOG = OUTPUT_DIR / "retry_fitur_alternatif_diagnostics.csv"
COVERAGE_DETAIL = PROJECT_DIR / "outputs" / "sentinel2_search" / "audit_footprint_coverage_detail.csv"


def as_num(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def load_checkpoint() -> pd.DataFrame:
    # Continue an earlier retry run if it exists; otherwise start from original.
    path = MERGED_OUTPUT if MERGED_OUTPUT.exists() else BASE_FEATURES
    if not path.exists():
        raise FileNotFoundError(f"CSV fitur tidak ditemukan: {BASE_FEATURES}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    if "sample_id" not in df or "status" not in df:
        raise ValueError(f"CSV tidak memiliki kolom sample_id/status: {path}")
    print(f"Checkpoint fitur: {path}")
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def candidates_for_sample(coverage: pd.DataFrame, sid: str, current_date: str, tried_dates: set[str], limit: int):
    rows = coverage[coverage["sample_id"].astype(str) == sid].copy()
    if rows.empty:
        return []
    rows["acquisition_date"] = rows["acquisition_date"].astype(str).str[:10]
    rows["sample_area_covered_pct"] = pd.to_numeric(rows["sample_area_covered_pct"], errors="coerce")
    rows["cloud_cover_pct"] = pd.to_numeric(rows["cloud_cover_pct"], errors="coerce")
    rows = rows[rows["sample_area_covered_pct"] >= 99.5]
    rows = rows[rows["acquisition_date"] != current_date]
    rows = rows[~rows["acquisition_date"].isin(tried_dates)]
    # One Process API request is made for a date; merge candidate items/tiles that day.
    grouped = []
    for d, g in rows.groupby("acquisition_date"):
        grouped.append({
            "date": str(d),
            "cloud": float(g["cloud_cover_pct"].min()) if g["cloud_cover_pct"].notna().any() else 100.0,
            "item_ids": "|".join(g["item_id"].dropna().astype(str).unique()),
            "tiles": "|".join(g["mgrs_tile"].dropna().astype(str).unique()),
            "coverage": float(g["sample_area_covered_pct"].min()),
        })
    grouped.sort(key=lambda x: (x["cloud"], x["date"]))
    return grouped[:limit]


def score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        as_num(row.get("clear_pixel_fraction_of_polygon")),
        as_num(row.get("clear_pixel_count")),
        as_num(row.get("dataMask_valid_pixel_count")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry only failed samples using alternative full-footprint dates.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N failed samples for a trial; default all remaining failed samples.")
    parser.add_argument("--max-alternatives", type=int, default=3, help="Max alternative dates per sample in this run (default 3).")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between Process API calls in seconds.")
    args = parser.parse_args()

    print("=== Retry ekstraksi gagal dengan tanggal alternatif ===")
    print(f"Folder proyek : {PROJECT_DIR}")
    print(f"CSV dasar     : {BASE_FEATURES}")
    print(f"CSV gabungan  : {MERGED_OUTPUT}")
    print(f"Audit footprint: {COVERAGE_DETAIL}")
    print("Mask tetap: dataMask > 0 dan SCL kelas 4/5/6; tidak dilonggarkan otomatis.")

    if not COVERAGE_DETAIL.exists():
        raise FileNotFoundError(f"CSV audit footprint tidak ditemukan: {COVERAGE_DETAIL}")
    df = load_checkpoint()
    samples = base.load_samples()
    sample_map = {s["sample_id"]: s for s in samples}
    coverage = pd.read_csv(COVERAGE_DETAIL, dtype={"sample_id": str})
    required = {"sample_id", "acquisition_date", "mgrs_tile", "item_id", "cloud_cover_pct", "sample_area_covered_pct"}
    missing_cols = required - set(coverage.columns)
    if missing_cols:
        raise ValueError(f"Kolom CSV footprint tidak lengkap: {sorted(missing_cols)}")

    if RETRY_LOG.exists():
        log_df = pd.read_csv(RETRY_LOG, dtype={"sample_id": str, "candidate_date": str})
    else:
        log_df = pd.DataFrame(columns=["sample_id", "candidate_date", "clear_pixel_count", "status", "scl_histogram", "error"])
    tried_by_sample: dict[str, set[str]] = {}
    if not log_df.empty:
        for sid, g in log_df.groupby("sample_id"):
            tried_by_sample[str(sid)] = set(g["candidate_date"].astype(str).str[:10])

    # Keep exact row records in a map so the original CSV remains unchanged.
    rows_by_id = {str(r["sample_id"]): r for r in df.to_dict(orient="records")}
    failed_ids = [sid for sid, r in rows_by_id.items() if str(r.get("status", "")) != "OK"]
    print(f"Jumlah baris checkpoint: {len(rows_by_id)}")
    print(f"Sampel gagal yang masih perlu ditinjau: {len(failed_ids)}")
    if args.limit > 0:
        failed_ids = failed_ids[:args.limit]
        print(f"Mode uji: memproses {len(failed_ids)} sampel gagal pertama")

    session = requests.Session()
    token = base.get_token(session)
    print("OAuth berhasil; token/secret tidak ditampilkan.")

    new_log_rows: list[dict[str, Any]] = []
    attempt_total = 0
    improved = 0
    unchanged = 0
    for idx, sid in enumerate(failed_ids, 1):
        current = rows_by_id[sid]
        sample = sample_map.get(sid)
        if sample is None:
            print(f"[{idx}/{len(failed_ids)}] {sid}: tidak ada di GeoJSON; dilewati")
            continue
        current_date = str(current.get("acquisition_date_requested", ""))[:10]
        tried = tried_by_sample.get(sid, set())
        cand_dates = candidates_for_sample(coverage, sid, current_date, tried, args.max_alternatives)
        if not cand_dates:
            print(f"[{idx}/{len(failed_ids)}] {sid} | {current.get('class_name')} | tidak ada tanggal alternatif baru")
            unchanged += 1
            continue

        best = current
        sample_attempted = 0
        print(f"[{idx}/{len(failed_ids)}] {sid} | {current.get('class_name')} | tanggal awal={current_date or '-'} | kandidat alternatif={len(cand_dates)}")
        for cand in cand_dates:
            assignment = {
                "assigned_item_id": cand["item_ids"],
                "acquisition_date": cand["date"],
                "mgrs_tile": cand["tiles"],
                "scene_cloud_cover_pct": cand["cloud"],
                "sample_footprint_coverage_pct": cand["coverage"],
                "assignment_source": str(COVERAGE_DETAIL),
            }
            result, token = base.extract_one(sample, assignment, session, token)
            result["retry_attempted"] = True
            result["retry_used_alternative_date"] = cand["date"]
            result["retry_note"] = "Alternative date tested; selected only if feature/pixel score improves. Original CSV retained separately."
            sample_attempted += 1
            attempt_total += 1
            new_log_rows.append({
                "sample_id": sid,
                "class_name": current.get("class_name"),
                "original_assigned_date": current_date,
                "candidate_date": cand["date"],
                "candidate_scene_cloud_cover_pct": cand["cloud"],
                "candidate_tiles": cand["tiles"],
                "candidate_item_ids": cand["item_ids"],
                "status": result.get("status"),
                "dataMask_valid_pixel_count": result.get("dataMask_valid_pixel_count"),
                "clear_pixel_count": result.get("clear_pixel_count"),
                "clear_pixel_fraction_of_polygon": result.get("clear_pixel_fraction_of_polygon"),
                "scl_cloud_shadow_pixel_count": result.get("scl_cloud_shadow_pixel_count"),
                "scl_class_2_dark_pixel_count": result.get("scl_class_2_dark_pixel_count"),
                "scl_class_4_vegetation_pixel_count": result.get("scl_class_4_vegetation_pixel_count"),
                "scl_class_5_bare_nonvegetated_pixel_count": result.get("scl_class_5_bare_nonvegetated_pixel_count"),
                "scl_class_6_water_pixel_count": result.get("scl_class_6_water_pixel_count"),
                "error": result.get("error", ""),
            })
            print(f"    {cand['date']} cloud_meta={cand['cloud']:.2f}% | {result.get('status')} | clear={result.get('clear_pixel_count', 0)} | fraction={as_num(result.get('clear_pixel_fraction_of_polygon')):.3f}")
            if score(result) > score(best):
                best = result
            # If every polygon pixel is clean, no alternative can improve on clear fraction/count.
            if as_num(result.get("polygon_pixel_count")) > 0 and as_num(result.get("clear_pixel_count")) >= as_num(result.get("polygon_pixel_count")):
                break
            time.sleep(max(0.0, args.sleep))

        if best is not current and str(best.get("status")) == "OK" and as_num(best.get("clear_pixel_count")) > as_num(current.get("clear_pixel_count")):
            rows_by_id[sid] = best
            improved += 1
            print(f"    DIPILIH: tanggal={best.get('acquisition_date_requested')} clear={best.get('clear_pixel_count')} (baris alternatif disimpan)")
        else:
            unchanged += 1
            print("    Belum ada alternatif yang meningkatkan jumlah/fraksi piksel bersih; baris awal dipertahankan.")

        # Persist merged output after each sample to support restart.
        merged = pd.DataFrame(list(rows_by_id.values()))
        order = {sid0: i for i, sid0 in enumerate([s["sample_id"] for s in samples])}
        merged["_order"] = merged["sample_id"].astype(str).map(order).fillna(10**9)
        merged = merged.sort_values("_order").drop(columns=["_order"])
        save_df(merged, MERGED_OUTPUT)
        if new_log_rows:
            append = pd.DataFrame(new_log_rows)
            all_log = pd.concat([log_df, append], ignore_index=True)
            # Avoid duplicate sample/date rows if a prior log happened to exist.
            all_log = all_log.drop_duplicates(subset=["sample_id", "candidate_date"], keep="last")
            save_df(all_log, RETRY_LOG)
            log_df = all_log
            new_log_rows.clear()
            tried_by_sample[sid] = set(log_df.loc[log_df["sample_id"].astype(str) == sid, "candidate_date"].astype(str).str[:10])

    final = pd.DataFrame(list(rows_by_id.values()))
    if not final.empty:
        status_counts = final["status"].value_counts(dropna=False)
        print("\n=== Ringkasan retry ===")
        print(f"Produk alternatif dicoba : {attempt_total}")
        print(f"Sampel berhasil diperbaiki: {improved}")
        print(f"Sampel gagal/tidak membaik: {int((final['status'].astype(str) != 'OK').sum())}")
        print("Status akhir:")
        print(status_counts.to_string())
    print(f"CSV gabungan hasil retry : {MERGED_OUTPUT}")
    print(f"Log percobaan alternatif : {RETRY_LOG}")
    print("CSV fitur asli tidak ditimpa. Sampel yang tidak membaik tetap ditandai dengan status asalnya.")
    print("Peringatan: baris status OK tetap perlu diperiksa jumlah piksel, kualitas label, dan duplikasi geometri sebelum training.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Gagal: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
