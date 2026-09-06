"""
Downloads real oceanographic and AIS data for OCEAN-SHIELD.
1. HYCOM GOFS 3.1 surface currents via OPeNDAP/THREDDS
2. Open-Meteo historical wind data (free, no auth)
3. MarineCadastre AIS data (public)
"""

import os
import sys
import json
import urllib.request
import urllib.error
import ssl
import io
import zipfile

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))


def download_hycom_netcdf():
    """Download real HYCOM ocean currents for Gulf of Kachchh region."""
    output_dir = os.path.join(DATASETS_DIR, "ocean_met")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hycom_real_gulf_kachchh.nc")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        print(f"✅ Real HYCOM NetCDF already exists: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path

    urls_to_try = [
        # HYCOM GOFS 3.1 latest analysis via ncss
        ("https://tds.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/sur?"
         "var=water_u&var=water_v"
         "&north=23.5&south=21.5&west=68.0&east=70.5"
         "&time_start=2024-01-01T00:00:00Z&time_end=2024-01-02T00:00:00Z"
         "&accept=netcdf4"),
        # HYCOM Reanalysis
        ("https://tds.hycom.org/thredds/ncss/GLBu0.08/expt_19.1/sur?"
         "var=water_u&var=water_v"
         "&north=23.5&south=21.5&west=68.0&east=70.5"
         "&time_start=2018-09-25T00:00:00Z&time_end=2018-09-27T00:00:00Z"
         "&accept=netcdf4"),
    ]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for i, url in enumerate(urls_to_try):
        print(f"[{i+1}/{len(urls_to_try)}] Trying: {url[:80]}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCEAN-SHIELD/1.0 (SIH26143 Research)"})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(output_path, "wb") as f:
                        f.write(data)
                    print(f"✅ Downloaded real HYCOM NetCDF: {output_path} ({len(data)} bytes)")
                    return output_path
                else:
                    print(f"   Response too small ({len(data)} bytes)")
        except Exception as e:
            print(f"   Failed: {e}")

    print("⚠️ Could not download real HYCOM data.")
    return None


def download_openmeteo_wind():
    """Download real wind data from Open-Meteo (free, no API key)."""
    output_dir = os.path.join(DATASETS_DIR, "ocean_met")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "openmeteo_wind_kachchh.json")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
        print(f"✅ Open-Meteo wind data already exists: {output_path}")
        return output_path

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=22.4&longitude=69.2"
        "&start_date=2024-01-01&end_date=2024-01-03"
        "&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        "&timezone=UTC"
    )

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "OCEAN-SHIELD/1.0"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            data = resp.read()
            if len(data) > 200:
                with open(output_path, "wb") as f:
                    f.write(data)
                result = json.loads(data)
                hours = len(result.get("hourly", {}).get("time", []))
                print(f"✅ Downloaded Open-Meteo wind data: {hours} hourly records")
                return output_path
    except Exception as e:
        print(f"⚠️ Open-Meteo download failed: {e}")

    return None


def download_marinecadastre_ais():
    """Download real AIS data from MarineCadastre / NOAA."""
    output_path = os.path.join(DATASETS_DIR, "marinecadastre_real_ais.csv")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 5000:
        print(f"✅ Real AIS CSV already exists: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path

    urls_to_try = [
        # MarineCadastre AIS data - daily zips
        "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/AIS_2022_01_01.zip",
        "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip",
    ]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for i, url in enumerate(urls_to_try):
        print(f"[{i+1}/{len(urls_to_try)}] Trying AIS: {url[:80]}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OCEAN-SHIELD/1.0 (SIH26143 Research)"})
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                data = resp.read()
                if len(data) > 5000:
                    try:
                        zf = zipfile.ZipFile(io.BytesIO(data))
                        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                        if csv_names:
                            csv_data = zf.read(csv_names[0])
                            lines = csv_data.decode("utf-8", errors="replace").split("\n")
                            subset = "\n".join(lines[:2001])
                            with open(output_path, "w") as f:
                                f.write(subset)
                            print(f"✅ Downloaded real AIS data: {output_path} ({len(subset)} bytes, {min(len(lines), 2000)} rows)")
                            return output_path
                    except Exception as ze:
                        print(f"   Zip extraction failed: {ze}")
                else:
                    print(f"   Response too small ({len(data)} bytes)")
        except Exception as e:
            print(f"   Failed: {e}")

    print("⚠️ Could not download real AIS data from any source.")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("OCEAN-SHIELD: Real Data Acquisition")
    print("=" * 70)

    print("\n--- 1. HYCOM Ocean Currents ---")
    hycom_path = download_hycom_netcdf()

    print("\n--- 2. Open-Meteo Wind Data ---")
    wind_path = download_openmeteo_wind()

    print("\n--- 3. MarineCadastre AIS ---")
    ais_path = download_marinecadastre_ais()

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  HYCOM:      {'✅ ' + hycom_path if hycom_path else '❌ Failed'}")
    print(f"  Wind:       {'✅ ' + wind_path if wind_path else '❌ Failed'}")
    print(f"  AIS:        {'✅ ' + ais_path if ais_path else '❌ Failed'}")
    print("=" * 70)
