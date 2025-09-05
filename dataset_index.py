"""
dataset_index.py

Creates a lightweight manifest_raw.csv for the March2021 dataset.

- It scans DATASET/March2021 recursively for .dat files.
- For each .dat file, records:
    base        (canonical like REFLECTANCE_223)
    dat_path    (relative path)
    hdr_path    (relative path if exists)
    png_path    (relative path if exists in same plant folder)
    mask_path   (relative path if exists in same plant folder)
    has_mask    (1 if mask found, else 0)
    plant_num   (extracted from plant folder name, e.g. plant_4 -> 4)
    dpi         (extracted from ancestor folder like day_14 -> 14)
 - Outputs METADATA/manifest_raw.csv for further processing.

"""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "DATASET" / "March2021"
META_DIR = ROOT / "METADATA"
META_DIR.mkdir(exist_ok=True)

# Extracts base name from file name
def canonical_base_from_dat(dat_path: Path):
    base = dat_path.stem
    base = re.sub(r'(?i)^_+', '', base)  # removes leading underscores
    m = re.match(r'(?i)(?:REFLECTANCE[_-]?)(.*)', base)
    if m:
        num = m.group(1)
        mnum = re.search(r'(\d+)', num)
        if mnum:
            return f"REFLECTANCE_{mnum.group(1)}"
        else:
            return f"REFLECTANCE_{num}"
    mnum = re.search(r'(\d+)', base)
    if mnum:
        return f"REFLECTANCE_{mnum.group(1)}"
    return base

def find_png_for_dat(dat_path: Path):
    folder = dat_path.parent
    base = dat_path.stem
    stripped = re.sub(r'(?i)^_?', '', base)
    stripped = re.sub(r'(?i)^REFLECTANCE[_-]?', '', stripped)
    cand = folder / (base + ".png")
    if cand.exists():
        return cand
    cand2 = folder / (stripped + ".png")
    if cand2.exists():
        return cand2
    num_match = re.search(r'(\d+)', stripped)
    if num_match:
        num = num_match.group(1)
        for p in sorted(folder.glob("*.png")):
            if num in p.stem:
                return p
    pngs = sorted(folder.glob("*.png"))
    return pngs[0] if pngs else None

def find_plant_mask_in_folder(folder: Path):
    patterns = ("pos_p*.csv", "pos_*.csv", "mask*.csv", "pos_*.txt", "annotation*.csv")
    for pat in patterns:
        for c in sorted(folder.glob(pat)):
            return c
    return None

def extract_plant_num_from_path(p: Path):
    for part in reversed(p.parts):
        m = re.search(r'(?i)plant[_\s-]*0*([0-9]{1,5})', part)
        if m:
            return int(m.group(1))
    return None

def extract_dpi_from_path(p: Path):
    for part in reversed(p.parts):
        m = re.search(r'(?i)day[_\s-]*0*([0-9]{1,3})', part)
        if m:
            return int(m.group(1))
    return None

def better_candidate(curr: dict, cand: dict):
    if curr is None:
        return True
    # prefer mask
    if cand.get("has_mask",0) and not curr.get("has_mask",0):
        return True
    # prefer png
    if cand.get("png_path") and not curr.get("png_path"):
        return True
    # prefer lower dpi
    cdpi = cand.get("dpi"); curdpi = curr.get("dpi")
    if cdpi is not None and curdpi is None:
        return True
    if cdpi is not None and curdpi is not None and cdpi < curdpi:
        return True
    return False

# This is the main function to build the manifest
def build_manifest():
    candidates = {}
    duplicates = {}
    dat_files = sorted(DATASET_DIR.rglob("*.dat"))
    if not dat_files:
        raise SystemExit(f"No .dat files found under {DATASET_DIR}")

    for dat in dat_files:
        canonical = canonical_base_from_dat(dat)
        hdr = dat.with_suffix(".hdr")
        png = find_png_for_dat(dat)
        mask = find_plant_mask_in_folder(dat.parent)
        plant_num = extract_plant_num_from_path(dat)
        dpi = extract_dpi_from_path(dat)

        row = {
            "base": canonical,
            "dat_path": str(dat.relative_to(ROOT)),
            "hdr_path": str(hdr.relative_to(ROOT)) if hdr.exists() else None,
            "png_path": str(png.relative_to(ROOT)) if png else None,
            "mask_path": str(mask.relative_to(ROOT)) if mask else None,
            "has_mask": 1 if mask else 0,
            "plant_num": int(plant_num) if plant_num is not None else None,
            "dpi": int(dpi) if dpi is not None else None,
        }

        existing = candidates.get(canonical)
        if better_candidate(existing, row):
            if existing is not None:
                duplicates.setdefault(canonical, []).append(existing)
            candidates[canonical] = row
        else:
            duplicates.setdefault(canonical, []).append(row)

    df = pd.DataFrame(list(candidates.values()))
    df['dpi_sort'] = df['dpi'].fillna(10**6).astype(int)
    df['plant_sort'] = df['plant_num'].fillna(10**6).astype(int)
    df = df.sort_values(by=['dpi_sort','plant_sort']).drop(columns=['dpi_sort','plant_sort'])

    out = META_DIR / "manifest_raw.csv"
    df.to_csv(out, index=False)

    print(f"[OK] Manifest saved: {out} ({len(df)} unique bases)")
    print("Summary:")
    print("  total .dat files:", len(dat_files))
    print("  unique bases:", len(df))
    print("  with mask:", int(df['has_mask'].sum()))
    print("  dpi distribution:\n", df['dpi'].value_counts(dropna=False).sort_index().to_string())
    print("\nSample rows:")
    print(df.head(12).to_string(index=False))
    return df

if __name__ == "__main__":
    build_manifest()
