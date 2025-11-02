"""
Download & prepare PROTOTYPE hyperspectral dataset for Plant 2.1.

- Downloads the Dataverse ZIP archive (file DOI)
- Extracts only useful files (.dat, .hdr, .png, .csv)
- Handles naming discrepancies (plant_2, plant_21, plant_2.5 → plant_2.1)
- Renames .csv files consistently to pos_Plant2.1.csv
- Reconstructs folder structure: DATASET/PROTOTYPE/day_X/plant_2.1/
- Deletes ZIP after extraction to save space
"""

import os
import re
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# CONFIG
FILE_DOI = "doi:10.57745/XAZQ7O"  # File DOI (not dataset DOI)
SERVER = "https://entrepot.recherche.data.gouv.fr"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "DATASET"
EXTRACT_DIR = DATASET_DIR / "PROTOTYPE"
ZIP_PATH = DATASET_DIR / "Prototype_Dataset.zip"

# Discrepancy mapping for plant folder names
PLANT_NAME_MAP = {
    "day_2": "plant_2",
    "day_3": "plant_21",
    "day_15": "plant_2.5",
    "day_16": "plant_2.5",
    "day_17": "plant_2.5",
}


# FUNCTIONS
def download_file(file_doi: str, out_path: Path) -> Path:
    """Download the dataset ZIP file from Dataverse by file DOI."""
    url = f"{SERVER}/api/access/datafile/:persistentId?persistentId={file_doi}"

    if out_path.exists():
        print(f"[Skip download] {out_path.name}")
        return out_path

    print(f"[Downloading ZIP] {url}")
    with requests.get(url, stream=True, timeout=1200) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading ZIP"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"[OK] Downloaded ZIP to {out_path}")
    return out_path


def extract_and_filter(zip_path: Path, extract_dir: Path):
    """Extract only useful plant_2.1 data from the full ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = [m for m in zip_ref.namelist()
                   if m.endswith((".dat", ".hdr", ".png", ".csv"))]

        print(f"[Extracting] {len(members)} useful files found in archive")

        for member in tqdm(members, desc="Filtering relevant files"):
            # Identify the day folder
            day_match = re.search(r"(day_\d+)", member)
            if not day_match:
                continue
            day = day_match.group(1)

            # Determine original plant folder name
            plant_original = PLANT_NAME_MAP.get(day, "plant_2.1")

            # Keep only relevant plant folders
            if f"{day}/{plant_original}/" not in member:
                continue

            # Normalize folder name
            fixed_member = member.replace(plant_original, "plant_2.1")
            out_path = extract_dir / fixed_member

            if out_path.exists():
                continue

            # Ensure directory structure exists
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract and rename csv if needed
            with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            if out_path.suffix == ".csv":
                new_csv = out_path.parent / "pos_Plant2.1.csv"
                out_path.rename(new_csv)

    print(f"[OK] Extraction complete for all days into {extract_dir}")


def main():
    DATASET_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(exist_ok=True)

    # 1. Download the ZIP file
    zip_path = download_file(FILE_DOI, ZIP_PATH)

    # 2. Extract and filter
    extract_and_filter(zip_path, EXTRACT_DIR)

    # 3. Cleanup
    try:
        os.remove(zip_path)
        print(f"[Cleanup] Removed raw archive {zip_path}")
    except Exception as e:
        print(f"[Warn] Could not remove {zip_path}: {e}")

    print(f"\n✅ Prototype dataset ready at: {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
