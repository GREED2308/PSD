"""Test OAuth client credentials for Copernicus Data Space Sentinel Hub.

Reads COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET from a local .env
file placed next to this script or in the current project directory. Does not
print the client secret or the access token.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_dir = Path.cwd()

    # Prefer .env in current working directory; fall back to script directory.
    env_path = project_dir / ".env"
    if not env_path.is_file():
        env_path = script_dir / ".env"

    if not env_path.is_file():
        print("ERROR: File .env tidak ditemukan.")
        print(f"Letakkan .env di folder proyek: {project_dir}")
        return 2

    load_dotenv(env_path, override=False)
    client_id = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        print("ERROR: COPERNICUS_CLIENT_ID atau COPERNICUS_CLIENT_SECRET kosong.")
        print(f"Periksa nama variabel di: {env_path}")
        return 2

    print("Menguji OAuth Copernicus Data Space Sentinel Hub...")
    print(f"File .env: {env_path}")
    try:
        response = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"ERROR: Gagal menghubungi endpoint token ({type(exc).__name__}).")
        print("Periksa koneksi internet, proxy, atau firewall, lalu coba lagi.")
        return 1

    if not response.ok:
        print(f"ERROR: Endpoint token mengembalikan HTTP {response.status_code}.")
        try:
            body = response.json()
            error = body.get("error")
            description = body.get("error_description")
            if error:
                print(f"Kode error: {error}")
            if description:
                print(f"Keterangan: {description}")
        except ValueError:
            print("Respons bukan JSON; detail respons tidak ditampilkan demi keamanan.")
        print("Jangan kirim Client Secret ke chat. Pastikan kredensial berasal dari OAuth Client Sentinel Hub Dashboard dan belum kedaluwarsa.")
        return 1

    try:
        payload = response.json()
    except ValueError:
        print("ERROR: Respons token bukan JSON yang valid.")
        return 1

    if not payload.get("access_token"):
        print("ERROR: Respons berhasil tetapi tidak memuat access_token.")
        return 1

    expires_in = payload.get("expires_in", "tidak dinyatakan")
    print("BERHASIL: OAuth token berhasil diperoleh.")
    print(f"Masa berlaku token (detik): {expires_in}")
    print("Access token tidak ditampilkan. .env dan secret tetap lokal.")
    print("Berikutnya kita dapat menguji Process API dengan permintaan raster kecil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
