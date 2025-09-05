import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm   # to show progress bars

# CONFIGURATION
FILE_DOI = "doi:10.57745/YHART9"   # File DOI for March 2021 ZIP only, rest will be downloaded later
SERVER = "https://entrepot.recherche.data.gouv.fr"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "DATASET"
EXTRACT_DIR = DATASET_DIR / "March2021"

# Functions to download files using Dataverse API
def download_file(file_doi: str, out_dir: Path) -> Path:
    # Downloading the March2021 ZIP from Dataverse API by file DOI with progress bar.
    url = f"{SERVER}/api/access/datafile/:persistentId?persistentId={file_doi}"
    out_path = out_dir / "Data_March2021.zip"
    if out_path.exists():
        print(f"[Skip download] {out_path}")
        return out_path

    print(f"[Downloading] {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading ZIP"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):  # divided into chunks of 10 MB for faster download
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return out_path

def extract_and_clean(zip_path: Path, extract_dir: Path):
    # Extracting zip and only keeping .dat, .hdr, .png, .csv files with skip checks.
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = [m for m in zip_ref.namelist()
                   if m.endswith((".dat", ".hdr", ".png", ".csv"))]

        print(f"[Extracting] {len(members)} useful files from {zip_path.name}")
        for i, member in enumerate(members, start=1):
            out_path = extract_dir / member
            if out_path.exists():
                print(f"  [Skip exists] {member}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            zip_ref.extract(member, path=extract_dir)
            print(f"  [{i}/{len(members)}] {member}")

    print(f"[OK] Extraction complete into {extract_dir}")

def main():
    DATASET_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(exist_ok=True)

    # Downloading ZIP by FILE DOI
    zip_path = download_file(FILE_DOI, DATASET_DIR)

    # Extracting useful files only
    extract_and_clean(zip_path, EXTRACT_DIR)

    # Deleting ZIP to save space
    try:
        os.remove(zip_path)
        print(f"[Cleanup] Removed raw archive {zip_path}")
    except Exception as e:
        print(f"[Warn] Could not remove {zip_path}: {e}")

    print(f"\n✅ Dataset ready at: {EXTRACT_DIR}")

if __name__ == "__main__":
    main()
