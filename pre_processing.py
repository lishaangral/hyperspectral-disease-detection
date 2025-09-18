"""
Preprocessing module for July2022 hyperspectral dataset (single-leaf per image).

This script contains modular functions to:
 - read ENVI (.hdr/.dat) hyperspectral cubes
 - read RGB preview (.png)
 - generate a single centered-leaf mask (PNG HSV threshold + fallback NDVI)
 - select spectral band range and apply Savitzky-Golay smoothing
 - sample spatial patches inside the leaf mask
 - align patch centers across days (assumes one monitored leaf centered in image)
 - map symptomatic CSV pixel annotations to patches
 - assemble per-patch, per-day rows and export an annotation CSV for downstream training

Design goals:
 - Clear, reusable functions with minimal external assumptions
 - Conservative defaults tuned to SPECIM IQ images (512×512, ~200 bands)
 - Saveable artifacts (annotation CSV; optional per-patch .npy arrays)

Dependencies (install via pip if not present):
  numpy, pandas, opencv-python, imageio, spectral, scipy, scikit-image, tqdm

"""

from pathlib import Path
import os
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.filters import threshold_otsu
import imageio
import cv2
from tqdm import tqdm

# spectral library for ENVI .hdr/.dat reading
try:
    from spectral import open_image
except Exception as e:
    raise ImportError("Install the 'spectral' package (pip install spectral).") from e


# Helper I/O and utilities
def read_envi_cube(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an ENVI-format hyperspectral cube using spectral.open_image.

    Returns:
      cube: ndarray (H, W, B) float32
      wavelengths: ndarray (B,) float of wavelengths in nm (if available), else indices
    """
    img = open_image(str(hdr_path))
    cube = img.load().astype(np.float32)  # shape: (rows, cols, bands) or (bands, rows, cols)
    # spectral.open_image returns shape (rows, cols, bands) typically
    if cube.ndim != 3:
        raise ValueError(f"Unexpected cube shape {cube.shape} from {hdr_path}")

    # try to read wavelength list from metadata
    meta = img.metadata or {}
    wls = None
    if "wavelength" in meta:
        try:
            wls = np.array([float(w) for w in meta["wavelength"]])
        except Exception:
            # sometimes stored as comma-separated string
            try:
                wls = np.array([float(x) for x in meta["wavelength"].split(",")])
            except Exception:
                wls = None

    if wls is None:
        # fallback to band indices
        b = cube.shape[2]
        wls = np.arange(b, dtype=float)
    return cube, wls


def read_png(png_path: Path) -> np.ndarray:
    """Read PNG preview into ndarray (H, W, 3) in uint8 (RGB)."""
    img = imageio.imread(str(png_path))
    # imageio may return RGBA; convert to RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# Mask generation
def mask_from_png_rgb(png_img: np.ndarray,
                      lower_hsv=(25, 40, 40),
                      upper_hsv=(95, 255, 255),
                      min_area=1000) -> np.ndarray:
    """
    Generate a leaf mask from the RGB preview using HSV green thresholding.
    Returns binary mask (H, W) boolean.
    """
    hsv = cv2.cvtColor(png_img, cv2.COLOR_RGB2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper).astype(np.uint8)
    # morphological cleaning
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # remove small blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=-1)
    return (cleaned > 0)


def mask_from_cube_ndvi(cube: np.ndarray, wavelengths: np.ndarray,
                        red_nm=680, nir_nm=800) -> np.ndarray:
    """
    Fallback mask generator using an NDVI-like index derived from hyperspectral cube.
    Returns binary mask (H, W) boolean.
    """
    # find nearest band indices
    def nearest_idx(warr, target):
        idx = np.abs(warr - target).argmin()
        return int(idx)

    red_idx = nearest_idx(wavelengths, red_nm)
    nir_idx = nearest_idx(wavelengths, nir_nm)
    red = cube[:, :, red_idx].astype(np.float32)
    nir = cube[:, :, nir_idx].astype(np.float32)
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    # threshold with Otsu
    try:
        t = threshold_otsu(ndvi)
    except Exception:
        # fallback percentile
        t = np.percentile(ndvi, 50)
    mask = ndvi > t
    # morphological cleanup
    mask = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return (mask > 0)


# Spectral processing
def trim_bands(cube: np.ndarray, wavelengths: np.ndarray,
               min_wl=420.0, max_wl=950.0) -> Tuple[np.ndarray, np.ndarray]:
    """Trim noisy edge bands by wavelength range. Returns (cube_trimmed, wls_trimmed)."""
    idxs = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
    if len(idxs) == 0:
        return cube, wavelengths
    return cube[:, :, idxs], wavelengths[idxs]


def select_band_range(cube: np.ndarray, wavelengths: np.ndarray,
                      sel_min=580.0, sel_max=760.0) -> Tuple[np.ndarray, np.ndarray]:
    """Select a spectral subrange. Returns (subcube, subwls)."""
    idxs = np.where((wavelengths >= sel_min) & (wavelengths <= sel_max))[0]
    if len(idxs) == 0:
        # fallback: return original
        return cube, wavelengths
    return cube[:, :, idxs], wavelengths[idxs]


def smooth_spectra(cube: np.ndarray, window_length=7, polyorder=2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing along the spectral axis (last axis).
    Expects cube shape (H, W, B).
    """
    H, W, B = cube.shape
    flattened = cube.reshape(-1, B)
    # enforce odd window
    wl = window_length if window_length % 2 == 1 else window_length + 1
    if wl >= B:
        wl = B - 1 if (B - 1) % 2 == 1 else B - 2
        if wl < 3:
            return cube
    sm = savgol_filter(flattened, wl, polyorder, axis=1)
    sm_cube = sm.reshape(H, W, B)
    return sm_cube.astype(np.float32)


# Patch extraction & alignment
def sample_patch_centers(mask: np.ndarray, patch_size: int = 32,
                         stride: Optional[int] = None, min_coverage: float = 0.6) -> List[Tuple[int, int]]:
    """
    Sample patch centers inside the mask using a regular grid (stride default = patch_size/2).
    Returns list of (row_center, col_center).
    """
    if stride is None:
        stride = patch_size // 2
    H, W = mask.shape
    centers = []
    half = patch_size // 2
    rows = list(range(half, H - half + 1, stride))
    cols = list(range(half, W - half + 1, stride))
    for r in rows:
        for c in cols:
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half
            patch_mask = mask[r0:r1, c0:c1]
            if patch_mask.size == 0:
                continue
            coverage = float(np.count_nonzero(patch_mask)) / patch_mask.size
            if coverage >= min_coverage:
                centers.append((r, c))
    return centers


def extract_patch(cube: np.ndarray, center: Tuple[int, int], patch_size: int = 32) -> np.ndarray:
    """
    Extract patch given cube (H,W,B) and center (r,c). Returns shape (patch_size, patch_size, B).
    """
    r, c = center
    half = patch_size // 2
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    return cube[r0:r1, c0:c1, :]


# CSV annotation mapping
def read_symptom_csv(csv_path: Path) -> np.ndarray:
    """
    Read CSV of symptomatic pixel positions. Expected two columns: row, col
    Returns array shape (N,2) of ints.
    """
    df = pd.read_csv(csv_path, header=None)
    arr = df.iloc[:, :2].values.astype(int)
    return arr


def patch_contains_symptom(center: Tuple[int, int], patch_size: int,
                           symptom_pixels: np.ndarray) -> bool:
    """Return True if any symptom pixel falls inside the patch centered at center."""
    r, c = center
    half = patch_size // 2
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    if symptom_pixels.size == 0:
        return False
    # check bounding box containment
    within_r = (symptom_pixels[:, 0] >= r0) & (symptom_pixels[:, 0] < r1)
    within_c = (symptom_pixels[:, 1] >= c0) & (symptom_pixels[:, 1] < c1)
    return bool(np.any(within_r & within_c))


# High-level per-leaf processing
def process_leaf_day(hdr_path: Path, png_path: Optional[Path], csv_path: Optional[Path],
                     patch_size: int = 32, stride: Optional[int] = None,
                     sel_band_min=580.0, sel_band_max=760.0,
                     smoothing_window=7) -> Dict:
    """
    Process a single image (one leaf) and return:
      {
        'cube': (H,W,B) or None (if saving disabled),
        'wavelengths': array,
        'mask': boolean (H,W),
        'centers': list of (r,c),
        'patch_count': int,
        'symptom_pixels': ndarray (N,2) or None
      }
    This function does not save patches; it returns structures for higher-level orchestration.
    """
    cube, wls = read_envi_cube(hdr_path)
    # trim noisy edges and select band range
    cube, wls = trim_bands(cube, wls, min_wl=420.0, max_wl=950.0)
    cube, wls = select_band_range(cube, wls, sel_min=sel_band_min, sel_max=sel_band_max)
    # smoothing
    cube = smooth_spectra(cube, window_length=smoothing_window, polyorder=2)

    # mask: prefer PNG-based mask if available
    if png_path and png_path.exists():
        png_img = read_png(png_path)
        mask = mask_from_png_rgb(png_img)
        # fallback to NDVI mask if mask too small
        if mask.sum() < 500:
            mask = mask_from_cube_ndvi(cube, wls)
    else:
        mask = mask_from_cube_ndvi(cube, wls)

    centers = sample_patch_centers(mask, patch_size=patch_size, stride=stride, min_coverage=0.6)

    symptom_pixels = None
    if csv_path and csv_path.exists():
        try:
            symptom_pixels = read_symptom_csv(csv_path)
        except Exception:
            symptom_pixels = np.empty((0, 2), dtype=int)
    else:
        symptom_pixels = np.empty((0, 2), dtype=int)

    return {
        "cube": cube,
        "wavelengths": wls,
        "mask": mask,
        "centers": centers,
        "patch_count": len(centers),
        "symptom_pixels": symptom_pixels
    }


# Build per-leaf-per-day annotation rows
def build_annotations_for_leaf(leaf_day_info: Dict[str, Dict],
                               plant_id: str,
                               leaf_id: str,
                               treatment: str,
                               pre_window: int = 3,
                               patch_size: int = 32) -> List[Dict]:
    """
    Assemble annotation rows for all days of a single (plant, leaf).
    leaf_day_info: dict mapping day_str -> dict returned by process_leaf_day
    Returns list of dict rows (one per patch per day) with fields:
      ['plant_id', 'leaf_id', 'day', 'center_row', 'center_col', 'patch_size',
       'mask_coverage', 'contains_symptom', 'label_source_placeholder']
    Label assignment (disease/water/control) should be performed later when `first_symptom_day` is known.
    """
    rows = []
    # choose a canonical set of patch centers to align across days.
    # Strategy: choose centers from the first available day (earliest day in keys)
    day_keys = sorted(leaf_day_info.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else x)
    if not day_keys:
        return rows
    ref = leaf_day_info[day_keys[0]]
    centers = ref["centers"]

    # precompute mask coverage per center for each day and symptom flags
    for day in day_keys:
        info = leaf_day_info[day]
        mask = info["mask"]
        symptom_pixels = info["symptom_pixels"]
        for center in centers:
            r, c = center
            half = patch_size // 2
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half
            # handle boundary (if patch extends outside due to different day cropping)
            H, W = mask.shape
            if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                mask_cov = 0.0
            else:
                patch_mask = mask[r0:r1, c0:c1]
                mask_cov = float(np.count_nonzero(patch_mask)) / patch_mask.size if patch_mask.size > 0 else 0.0

            contains_sym = patch_contains_symptom(center, patch_size, symptom_pixels)

            row = {
                "plant_id": plant_id,
                "leaf_id": leaf_id,
                "day": day,
                "center_row": int(r),
                "center_col": int(c),
                "patch_size": int(patch_size),
                "mask_coverage": float(mask_cov),
                "contains_symptom": bool(contains_sym)
            }
            rows.append(row)
    return rows


# CSV export helpers
def save_annotation_table(rows: List[Dict], out_csv: Path):
    """Save annotation rows to CSV (appendable)."""
    if not rows:
        print("[WARN] No annotation rows to save.")
        return
    df = pd.DataFrame(rows)
    # consistent column order
    cols = ["plant_id", "leaf_id", "day", "center_row", "center_col",
            "patch_size", "mask_coverage", "contains_symptom"]
    df = df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] Annotations -> {out_csv} ({len(df)} rows)")


# Command-line orchestration
def discover_leaf_folders(root_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Discover leaf folders in a dataset root structure that follows:
      root/day_N/plant_X/  (with REFLECTANCE_*.hdr and png, csv)
    Returns list of tuples (hdr_path, day_str, plant_folder_name)
    """
    items = []
    for day_dir in sorted(root_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        day_str = day_dir.name
        for plant_dir in sorted(day_dir.iterdir()):
            if not plant_dir.is_dir():
                continue
            # try to find hdr file inside plant_dir (common names: REFLECTANCE_*.hdr or *.hdr)
            hdr_candidates = list(plant_dir.glob("*.hdr"))
            if not hdr_candidates:
                continue
            # pick first HDR
            hdr_path = hdr_candidates[0]
            items.append((hdr_path, day_str, plant_dir.name))
    return items


def run_dataset_preprocessing(dataset_root: Path,
                              output_dir: Path,
                              patch_size: int = 32,
                              stride: Optional[int] = None,
                              sel_band_min: float = 580.0,
                              sel_band_max: float = 760.0,
                              smoothing_window: int = 7,
                              pre_window: int = 3,
                              save_annotation_csv: bool = True):
    """
    Top-level pipeline orchestrator. Processes each leaf/day, grouping by plant/leaf,
    building annotation rows and writing a combined CSV.
    """
    dataset_root = Path(dataset_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: discover available hdr files and build a mapping
    discovered = discover_leaf_folders(dataset_root)
    if not discovered:
        raise SystemExit(f"No HDR files found under {dataset_root}")

    # Group by plant folder name (assumes leaf directories are plant-level; further leaf numbering handled by folder naming)
    grouped = {}
    for hdr_path, day_str, plant_folder in discovered:
        # attempt to identify leaf id from plant_folder (e.g., plant_2 or plant_2.1)
        parts = plant_folder.split("_")
        plant_id = parts[1] if len(parts) >= 2 else plant_folder
        # leaf_id: if further like 2.1 in name; else use plant_id as leaf
        leaf_id = plant_folder  # keep folder name as leaf_id (unique)
        key = (plant_id, leaf_id)
        grouped.setdefault(key, {})[day_str] = hdr_path.parent  # store plant_dir path

    # Process each (plant, leaf)
    all_rows = []
    for (plant_id, leaf_id), days_map in grouped.items():
        leaf_day_info = {}
        for day_str in sorted(days_map.keys()):
            plant_dir = days_map[day_str]
            # identify expected filenames inside plant_dir
            # hdr = any *.hdr, png = any *.png, csv = any *.csv
            hdr_files = list(plant_dir.glob("*.hdr"))
            png_files = list(plant_dir.glob("*.png"))
            csv_files = list(plant_dir.glob("*.csv"))
            hdr_path = hdr_files[0] if hdr_files else None
            png_path = png_files[0] if png_files else None
            # pick CSV which likely contains symptomatic coordinates; allow missing
            csv_path = csv_files[0] if csv_files else None
            if hdr_path is None:
                print(f"[SKIP] No HDR in {plant_dir}")
                continue
            info = process_leaf_day(hdr_path, png_path, csv_path,
                                    patch_size=patch_size, stride=stride,
                                    sel_band_min=sel_band_min, sel_band_max=sel_band_max,
                                    smoothing_window=smoothing_window)
            leaf_day_info[day_str] = info
        # Build annotation rows aligned to canonical centers
        rows = build_annotations_for_leaf(leaf_day_info, plant_id=plant_id, leaf_id=leaf_id,
                                          pre_window=pre_window, patch_size=patch_size)
        all_rows.extend(rows)

    # Save combined annotations
    out_csv = out_dir / "annotations_patches.csv"
    if save_annotation_csv:
        save_annotation_table(all_rows, out_csv)

    print(f"[DONE] Processed {len(grouped)} leaf instances, produced {len(all_rows)} rows.")
    return out_csv


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess July2022 dataset into patch annotations.")
    p.add_argument("--dataset", required=True, help="Path to extracted July2022 dataset root (day_x folders).")
    p.add_argument("--out", required=True, help="Output directory to store annotations / optional artifacts.")
    p.add_argument("--patch", type=int, default=32, help="Patch size (default 32).")
    p.add_argument("--bands-min", type=float, default=580.0, help="Band selection min wavelength (nm).")
    p.add_argument("--bands-max", type=float, default=760.0, help="Band selection max wavelength (nm).")
    p.add_argument("--smoothing", type=int, default=7, help="Savitzky-Golay smoothing window (odd).")
    p.add_argument("--pre-window", type=int, default=3, help="Pre-symptomatic window in days for labeling (informational).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    run_dataset_preprocessing(dataset_root, out_dir,
                              patch_size=args.patch,
                              sel_band_min=args.bands_min,
                              sel_band_max=args.bands_max,
                              smoothing_window=args.smoothing,
                              pre_window=args.pre_window)
