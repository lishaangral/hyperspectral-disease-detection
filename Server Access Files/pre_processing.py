"""
Preprocessing module for July2022 hyperspectral dataset (single-leaf per image). This script contains modular functions to: 
- read ENVI (.hdr/.dat) hyperspectral cubes 
- read RGB preview (.png) 
- generate a single centered-leaf mask (PNG HSV threshold + fallback NDVI) 
- select spectral band range and apply Savitzky-Golay smoothing 
- sample spatial patches inside the leaf mask 
- align patch centers across days (assumes one monitored leaf centered in image) 
- map symptomatic CSV pixel annotations to patches 
- assemble per-patch, per-day rows and export an annotation CSV for downstream training Design goals: 
- Clear, reusable functions with minimal external assumptions 
- Conservative defaults tuned to SPECIM IQ images (512×512, ~200 bands) 
- Saveable artifacts (annotation CSV; optional per-patch .npy arrays) 

Dependencies (install via pip if not present): 
numpy, pandas, opencv-python, imageio, spectral, scipy, scikit-image, tqdm

Outputs:
 - annotations_patches.csv written to OUTPUT_DIR
"""

from pathlib import Path
import os
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.filters import threshold_otsu
import imageio.v2 as imageio
import cv2

# spectral library for ENVI .hdr/.dat reading
try:
    from spectral import open_image
except Exception as e:
    raise ImportError("Install the 'spectral' package (pip install spectral).") from e


# CONFIG (edit values here)
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "DATASET" / "July2022" / "data_July2022"        # path to extracted July2022 dataset root
OUTPUT_DIR = PROJECT_ROOT / "PREPROCESSED" / "july2022"         # where annotations will be saved

PATCH_SIZE = 32
STRIDE = None                                             # None -> patch_size // 2
SEL_BAND_MIN = 580.0
SEL_BAND_MAX = 760.0
SMOOTHING_WINDOW = 7
PRE_WINDOW = 3

MIN_MASK_AREA = 1000
MIN_PATCH_COVERAGE = 0.6

# End CONFIG


# Helper I/O and utilities
def read_envi_cube(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an ENVI-format hyperspectral cube using spectral.open_image.
    Returns cube (H, W, B) float32 and wavelengths (B,) if available else band indices.
    """
    img = open_image(str(hdr_path))
    cube = img.load().astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Unexpected cube shape {cube.shape} from {hdr_path}")
    meta = img.metadata or {}
    wls = None
    if "wavelength" in meta:
        try:
            wls = np.array([float(w) for w in meta["wavelength"]])
        except Exception:
            try:
                wls = np.array([float(x) for x in meta["wavelength"].split(",")])
            except Exception:
                wls = None
    if wls is None:
        b = cube.shape[2]
        wls = np.arange(b, dtype=float)
    return cube, wls


def read_png(png_path: Path) -> np.ndarray:
    """Read PNG preview into ndarray (H, W, 3) in uint8 (RGB)."""
    img = imageio.imread(str(png_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# Mask generation
def mask_from_png_rgb(png_img: np.ndarray,
                      lower_hsv=(25, 40, 40),
                      upper_hsv=(95, 255, 255),
                      min_area=MIN_MASK_AREA) -> np.ndarray:
    """Generate a leaf mask from the RGB preview using HSV green thresholding."""
    hsv = cv2.cvtColor(png_img, cv2.COLOR_RGB2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=-1)
    return (cleaned > 0)


def mask_from_cube_ndvi(cube: np.ndarray, wavelengths: np.ndarray,
                        red_nm=680, nir_nm=800) -> np.ndarray:
    """Fallback mask using NDVI-like index derived from hyperspectral cube."""
    def nearest_idx(warr, target):
        return int(np.abs(warr - target).argmin())

    red_idx = nearest_idx(wavelengths, red_nm)
    nir_idx = nearest_idx(wavelengths, nir_nm)
    red = cube[:, :, red_idx].astype(np.float32)
    nir = cube[:, :, nir_idx].astype(np.float32)
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    try:
        t = threshold_otsu(ndvi)
    except Exception:
        t = np.percentile(ndvi, 50)
    mask = ndvi > t
    mask = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return (mask > 0)


# Spectral processing
def trim_bands(cube: np.ndarray, wavelengths: np.ndarray,
               min_wl=420.0, max_wl=950.0) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
    if len(idxs) == 0:
        return cube, wavelengths
    return cube[:, :, idxs], wavelengths[idxs]


def select_band_range(cube: np.ndarray, wavelengths: np.ndarray,
                      sel_min=SEL_BAND_MIN, sel_max=SEL_BAND_MAX) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.where((wavelengths >= sel_min) & (wavelengths <= sel_max))[0]
    if len(idxs) == 0:
        return cube, wavelengths
    return cube[:, :, idxs], wavelengths[idxs]


def smooth_spectra(cube: np.ndarray, window_length=SMOOTHING_WINDOW, polyorder=2) -> np.ndarray:
    H, W, B = cube.shape
    flattened = cube.reshape(-1, B)
    wl = window_length if window_length % 2 == 1 else window_length + 1
    if wl >= B:
        wl = B - 1 if (B - 1) % 2 == 1 else B - 2
        if wl < 3:
            return cube
    sm = savgol_filter(flattened, wl, polyorder, axis=1)
    sm_cube = sm.reshape(H, W, B)
    return sm_cube.astype(np.float32)


# Patch extraction & alignment
def sample_patch_centers(mask: np.ndarray, patch_size: int = PATCH_SIZE,
                         stride: Optional[int] = STRIDE, min_coverage: float = MIN_PATCH_COVERAGE) -> List[Tuple[int, int]]:
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


def extract_patch(cube: np.ndarray, center: Tuple[int, int], patch_size: int = PATCH_SIZE) -> np.ndarray:
    r, c = center
    half = patch_size // 2
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    return cube[r0:r1, c0:c1, :]


# CSV annotation mapping
def read_symptom_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    arr = df.iloc[:, :2].values.astype(int)
    return arr


def patch_contains_symptom(center: Tuple[int, int], patch_size: int,
                           symptom_pixels: np.ndarray) -> bool:
    r, c = center
    half = patch_size // 2
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    if symptom_pixels.size == 0:
        return False
    within_r = (symptom_pixels[:, 0] >= r0) & (symptom_pixels[:, 0] < r1)
    within_c = (symptom_pixels[:, 1] >= c0) & (symptom_pixels[:, 1] < c1)
    return bool(np.any(within_r & within_c))


# High-level per-leaf processing
def process_leaf_day(hdr_path: Path, png_path: Optional[Path], csv_path: Optional[Path],
                     patch_size: int = PATCH_SIZE, stride: Optional[int] = STRIDE,
                     sel_band_min: float = SEL_BAND_MIN, sel_band_max: float = SEL_BAND_MAX,
                     smoothing_window: int = SMOOTHING_WINDOW) -> Dict:
    cube, wls = read_envi_cube(hdr_path)
    cube, wls = trim_bands(cube, wls, min_wl=420.0, max_wl=950.0)
    cube, wls = select_band_range(cube, wls, sel_min=sel_band_min, sel_max=sel_band_max)
    cube = smooth_spectra(cube, window_length=smoothing_window, polyorder=2)

    if png_path and png_path.exists():
        png_img = read_png(png_path)
        mask = mask_from_png_rgb(png_img)
        if mask.sum() < 500:
            mask = mask_from_cube_ndvi(cube, wls)
    else:
        mask = mask_from_cube_ndvi(cube, wls)

    centers = sample_patch_centers(mask, patch_size=patch_size, stride=stride, min_coverage=MIN_PATCH_COVERAGE)

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
                               pre_window: int = PRE_WINDOW,
                               patch_size: int = PATCH_SIZE) -> List[Dict]:
    rows = []
    day_keys = sorted(leaf_day_info.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else x)
    if not day_keys:
        return rows
    ref = leaf_day_info[day_keys[0]]
    centers = ref["centers"]
    for day in day_keys:
        info = leaf_day_info[day]
        mask = info["mask"]
        symptom_pixels = info["symptom_pixels"]
        for center in centers:
            r, c = center
            half = patch_size // 2
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half
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
    if not rows:
        print("[WARN] No annotation rows to save.")
        return
    df = pd.DataFrame(rows)
    cols = ["plant_id", "leaf_id", "day", "center_row", "center_col",
            "patch_size", "mask_coverage", "contains_symptom"]
    df = df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] Annotations -> {out_csv} ({len(df)} rows)")


# Discovery & orchestration
def discover_leaf_folders(root_dir: Path) -> List[Tuple[Path, str, str]]:
    items = []
    for day_dir in sorted(root_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        day_str = day_dir.name
        for plant_dir in sorted(day_dir.iterdir()):
            if not plant_dir.is_dir():
                continue
            hdr_candidates = list(plant_dir.glob("*.hdr"))
            if not hdr_candidates:
                continue
            hdr_path = hdr_candidates[0]
            items.append((hdr_path, day_str, plant_dir.name))
    return items


def run_dataset_preprocessing(dataset_root: Path = DATASET_ROOT,
                              output_dir: Path = OUTPUT_DIR,
                              patch_size: int = PATCH_SIZE,
                              stride: Optional[int] = STRIDE,
                              sel_band_min: float = SEL_BAND_MIN,
                              sel_band_max: float = SEL_BAND_MAX,
                              smoothing_window: int = SMOOTHING_WINDOW,
                              pre_window: int = PRE_WINDOW,
                              save_annotation_csv: bool = True):
    dataset_root = Path(dataset_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_leaf_folders(dataset_root)
    if not discovered:
        raise SystemExit(f"No HDR files found under {dataset_root}")

    grouped = {}
    for hdr_path, day_str, plant_folder in discovered:
        parts = plant_folder.split("_")
        plant_id = parts[1] if len(parts) >= 2 else plant_folder
        leaf_id = plant_folder
        key = (plant_id, leaf_id)
        grouped.setdefault(key, {})[day_str] = hdr_path.parent

    all_rows = []
    for (plant_id, leaf_id), days_map in grouped.items():
        leaf_day_info = {}
        for day_str in sorted(days_map.keys()):
            plant_dir = days_map[day_str]
            hdr_files = list(plant_dir.glob("*.hdr"))
            png_files = list(plant_dir.glob("*.png"))
            csv_files = list(plant_dir.glob("*.csv"))
            hdr_path = hdr_files[0] if hdr_files else None
            png_path = png_files[0] if png_files else None
            csv_path = csv_files[0] if csv_files else None
            if hdr_path is None:
                print(f"[SKIP] No HDR in {plant_dir}")
                continue
            info = process_leaf_day(hdr_path, png_path, csv_path,
                                    patch_size=patch_size, stride=stride,
                                    sel_band_min=sel_band_min, sel_band_max=sel_band_max,
                                    smoothing_window=smoothing_window)
            leaf_day_info[day_str] = info
        rows = build_annotations_for_leaf(leaf_day_info, plant_id=plant_id, leaf_id=leaf_id,
                                          treatment="unknown", pre_window=pre_window, patch_size=patch_size)
        all_rows.extend(rows)

    out_csv = out_dir / "annotations_patches.csv"
    if save_annotation_csv:
        save_annotation_table(all_rows, out_csv)

    print(f"[DONE] Processed {len(grouped)} leaf instances, produced {len(all_rows)} rows.")
    return out_csv


# Run (hard-coded)
if __name__ == "__main__":
    print(f"[INFO] Dataset root: {DATASET_ROOT}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    result_csv = run_dataset_preprocessing()
    print(f"[RESULT] Annotations CSV: {result_csv.resolve()}")
