"""
OCEAN-SHIELD: Zenodo Sentinel-1 SAR Oil Spill Dataset Downloader
Smart India Hackathon 2026 (SIH26143 / NTRO)

This utility manages downloading and extracting curated subsets from the official
Zenodo Sentinel-1 SAR Oil Spill dataset (Records 8346860, 8253899, 13761290).

Each full scene is a 2048x2048 dual-polarization (VV/VH) TIFF image (~41.5 MB).
Because the full dataset is ~94 GB compressed (>250 GB uncompressed), this utility
provides safe, quota-aware subset downloading that respects local disk limits.
"""

import os
import sys
import shutil
import urllib.request
import py7zr

DATASET_RECORDS = {
    "part1_oil_spill": {
        "title": "Part I: Oil Spill (Train/Val)",
        "zenodo_id": 8346860,
        "mask_file": "01_Train_Val_Oil_Spill_mask.7z",
        "mask_url": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z/content",
        "images_file": "01_Train_Val_Oil_Spill_images.7z",
        "images_url": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_images.7z/content",
        "total_compressed_gb": 40.7
    },
    "part2_lookalike_no_oil": {
        "title": "Part II: Lookalike & No-Oil (Train/Val)",
        "zenodo_id": 8253899,
        "lookalike_mask_url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_mask.7z/content",
        "no_oil_mask_url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_No_Oil_mask.7z/content",
        "lookalike_images_url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_images.7z/content",
        "total_compressed_gb": 43.8
    },
    "part3_test": {
        "title": "Part III: Test Benchmark (Oil, Lookalike, No-Oil)",
        "zenodo_id": 13761290,
        "test_url": "https://zenodo.org/api/records/13761290/files/02_Test_images_and_ground_truth.7z/content",
        "total_compressed_gb": 9.4
    }
}

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "zenodo_sentinel1_sar"))


def check_disk_space():
    total, used, free = shutil.disk_usage(OUTPUT_DIR)
    free_gb = free / (1024 ** 3)
    print(f"📊 Available Disk Space: {free_gb:.2f} GB")
    return free_gb


def download_masks():
    """Downloads all ground truth mask archives (Oil, Lookalike, No-Oil) - takes ~10 MB total."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mask_downloads = [
        ("01_Train_Val_Oil_Spill_mask.7z", DATASET_RECORDS["part1_oil_spill"]["mask_url"]),
        ("01_Train_Val_Lookalike_mask.7z", DATASET_RECORDS["part2_lookalike_no_oil"]["lookalike_mask_url"]),
        ("01_Train_Val_No_Oil_mask.7z", DATASET_RECORDS["part2_lookalike_no_oil"]["no_oil_mask_url"]),
    ]

    for fname, url in mask_downloads:
        dest = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(dest):
            print(f"✅ Already downloaded: {fname}")
            continue
        print(f"⬇️ Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        print(f"   Saved {fname} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")


def extract_sample_masks(count_per_category=20):
    """Extracts ground truth 2048x2048 masks for each category."""
    tasks = [
        ("01_Train_Val_Oil_Spill_mask.7z", "part1_oil_spill_masks", count_per_category),
        ("01_Train_Val_Lookalike_mask.7z", "part2_lookalike_masks", count_per_category),
        ("01_Train_Val_No_Oil_mask.7z", "part2_no_oil_masks", count_per_category),
    ]

    for archive_name, target_folder, count in tasks:
        archive_path = os.path.join(OUTPUT_DIR, archive_name)
        target_path = os.path.join(OUTPUT_DIR, target_folder)
        os.makedirs(target_path, exist_ok=True)

        if not os.path.exists(archive_path):
            print(f"⚠️ Archive not found: {archive_name}. Run mask download first.")
            continue

        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            names = [n for n in z.getnames() if n.endswith(".tif")][:count]
            z.extract(path=target_path, targets=names)
            print(f"✅ Extracted {len(names)} masks into {target_folder}/")


def print_status():
    print("=" * 70)
    print("🛡️ OCEAN-SHIELD — Zenodo Sentinel-1 SAR Dataset Status")
    print("=" * 70)
    check_disk_space()
    print("\n📂 Extracted Mask Directories:")
    for folder in ["part1_oil_spill_masks", "part2_lookalike_masks", "part2_no_oil_masks"]:
        path = os.path.join(OUTPUT_DIR, folder)
        if os.path.exists(path):
            count = sum([len(files) for _, _, files in os.walk(path) if any(f.endswith('.tif') for f in files)])
            print(f"  • {folder}: {count} TIFF ground truth masks ready")
        else:
            print(f"  • {folder}: Not extracted")
    print("=" * 70)


if __name__ == "__main__":
    print_status()
    print("\n[1] Downloading ground truth mask archives...")
    download_masks()
    print("\n[2] Extracting 20 real masks per category...")
    extract_sample_masks(20)
    print("\n[3] Final Status:")
    print_status()
