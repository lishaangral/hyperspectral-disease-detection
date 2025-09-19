#!/usr/bin/env python3
"""
Automated similarity-based labeler for July2022 hyperspectral dataset.

This is a robust, self-contained labeler. Configuration is hard-coded in the
CONFIG block below — edit those values and run:

    python labeler_similarity_auto.py

Outputs:
 - labels_patches.csv (per-patch per-day labels + similarity/confidence columns)
 - labels_summary.json (counts + thresholds used)
 - diagnostics/ per-leaf similarity series CSVs (for manual inspection)

Key fixes included:
 - Normalize plant_id immediately after loading annotations (converts 11.0 -> "11")
 - Force water reference to REF_PLANT_WATER ("11") if that plant appears anywhere
   in the normalized annotations (guarantees use of plant 11 when present)
 - More robust HDR lookup and safer handling of missing data / exceptions
 - Clear diagnostic prints to confirm normalized IDs and chosen references
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import warnings
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from math import acos

try:
    from spectral import open_image
except Exception as e:
    raise ImportError("Install 'spectral' package (pip install spectral)") from e

# CONFIG (edit if needed)
PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATIONS_CSV = PROJECT_ROOT / "PREPROCESSED" / "july2022" / "annotations_patches.csv"
DATASET_ROOT = PROJECT_ROOT / "DATASET" / "July2022"

OUTPUT_DIR = PROJECT_ROOT / "PREPROCESSED" / "july2022" / "labels_auto"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = OUTPUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Preferred explicit references
REF_PLANT_DISEASE = "2"
REF_PLANT_WATER = "11"

# spectral & processing params
BAND_MIN = 580.0
BAND_MAX = 760.0
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
DEFAULT_PATCH_SIZE = 32

# labeling thresholds & margins
SYMPTOM_PERCENTILE = 25
PRE_PERCENTILE = 50
MARGIN = 0.05
PRE_WINDOW = 3
SIM_AGG_FUNC = np.nanmedian
VERBOSE = True


# Utility functions
def read_envi_cube(hdr_path: Path):
    """Return (cube HxWxB float32, wavelengths B float)"""
    img = open_image(str(hdr_path))
    cube = img.load().astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Unexpected cube shape {cube.shape} for {hdr_path}")
    meta = img.metadata or {}
    wls = None
    if "wavelength" in meta:
        try:
            wls = np.array([float(x) for x in meta["wavelength"]])
        except Exception:
            try:
                wls = np.array([float(x) for x in meta["wavelength"].split(",")])
            except Exception:
                wls = None
    if wls is None:
        wls = np.arange(cube.shape[2], dtype=float)
    return cube, wls


def trim_and_select_bands(cube, wavelengths, low=BAND_MIN, high=BAND_MAX):
    idxs = np.where((wavelengths >= low) & (wavelengths <= high))[0]
    return (cube[:, :, idxs], wavelengths[idxs]) if idxs.size > 0 else (cube, wavelengths)


def smooth_spectra_vectorized(cube, window=SMOOTH_WINDOW, polyorder=SMOOTH_POLY):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    wl = window if (window % 2 == 1) else window + 1
    if wl >= B:
        wl = B - 1 if (B - 1) % 2 == 1 else B - 2
        if wl < 3:
            return cube
    sm = savgol_filter(flat, wl, polyorder, axis=1)
    return sm.reshape(H, W, sm.shape[1]).astype(np.float32)


def mean_patch_spectrum(cube, center_row, center_col, patch_size):
    half = patch_size // 2
    r0, r1 = center_row - half, center_row + half
    c0, c1 = center_col - half, center_col + half
    H, W, B = cube.shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return None
    patch = cube[r0:r1, c0:c1, :]
    if patch.size == 0:
        return None
    return patch.reshape(-1, patch.shape[2]).mean(axis=0).astype(np.float32)


def l2_normalize(vec):
    v = vec.astype(np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    an = l2_normalize(a); bn = l2_normalize(b)
    return float(np.clip(float(np.dot(an, bn)), -1.0, 1.0))


def spectral_angle(a, b):
    if a is None or b is None:
        return float("inf")
    an = l2_normalize(a); bn = l2_normalize(b)
    dot = float(np.clip(float(np.dot(an, bn)), -1.0, 1.0))
    try:
        return float(acos(dot))
    except Exception:
        return float("inf")


# File discovery helpers
def find_hdr_for_annotation(row, dataset_root: Path):
    """
    Heuristic to locate a .hdr for the annotation row.
    Tries day-level folder matches then fallback to plant/leaf matches.
    """
    day_str = str(row["day"])
    leaf_id = str(row["leaf_id"])
    plant_id = str(row["plant_id"])
    # Prefer exact day folder match
    for cand in dataset_root.rglob(f"*{day_str}*"):
        if cand.is_dir():
            # search subfolders under day folder
            for sub in cand.iterdir():
                if not sub.is_dir():
                    continue
                if leaf_id in sub.name or f"plant_{plant_id}" in sub.name or plant_id in sub.name:
                    hdrs = list(sub.glob("*.hdr"))
                    if hdrs:
                        return hdrs[0]
            hd = list(cand.glob("*.hdr"))
            if hd:
                return hd[0]
    # fallback: search any hdr with matching parent name
    for hdr in dataset_root.rglob("*.hdr"):
        p = hdr.parent.name
        if leaf_id in p or f"plant_{plant_id}" in p or plant_id in p:
            return hdr
    # last fallback: first hdr found
    all_hdrs = list(dataset_root.rglob("*.hdr"))
    return all_hdrs[0] if all_hdrs else None


# Reference selection / building
def normalize_plant_id_val(x):
    """Normalize plant_id field: 11.0 -> '11', 2.1 -> '2.1' (keeps decimals for leaf-level strings)."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) or isinstance(x, np.floating):
        # convert integral floats to integer string
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    s = str(x).strip()
    # common form '11.0' -> '11'
    m = re.match(r"^(\d+)\.0+$", s)
    if m:
        return m.group(1)
    # otherwise keep as-is (e.g., '2.3' or 'plant_2')
    return s


def auto_select_reference_plants(ann_df: pd.DataFrame):
    """
    Auto-select disease & water refs.
    This version:
      - prefers configured REF_PLANT_DISEASE if present among symptomatic plants,
      - forces REF_PLANT_WATER if present anywhere in the normalized dataset,
      - ensures disease != water when possible.
    """
    symp = ann_df[ann_df["contains_symptom"] == True]
    if symp.empty:
        return None, None, []
    counts = symp["plant_id"].astype(str).value_counts()
    ranked = list(counts.index)

    # disease preference
    disease = str(REF_PLANT_DISEASE) if str(REF_PLANT_DISEASE) in ranked else ranked[0]

    # water preference: force REF_PLANT_WATER if present anywhere in annotations
    all_plants = set(ann_df["plant_id"].astype(str).unique())
    if str(REF_PLANT_WATER) in all_plants:
        water = str(REF_PLANT_WATER)
        if water == disease:
            # pick next different symptomatic plant if available
            water = next((p for p in ranked if p != disease), water)
    else:
        water = next((p for p in ranked if p != disease), disease)

    if VERBOSE:
        print(f"[AUTO-REF] symptomatic plants ranked: {ranked[:8]}")
        print(f"[AUTO-REF] chosen disease ref: {disease}, chosen water ref: {water}")
    return disease, water, ranked


def build_reference_spectrum(annotations_df: pd.DataFrame, ref_plant_id: str, dataset_root: Path):
    """
    Build reference spectrum for given plant by aggregating spectra from patches
    where contains_symptom == True for that plant.
    """
    ref_rows = annotations_df[
        (annotations_df["plant_id"].astype(str) == str(ref_plant_id)) & (annotations_df["contains_symptom"] == True)
    ]
    if ref_rows.empty:
        if VERBOSE:
            print(f"[WARN] No symptomatic CSV rows found for reference plant {ref_plant_id}.")
        return None, []
    spectra = []
    for _, r in ref_rows.iterrows():
        hdr = find_hdr_for_annotation(r, dataset_root)
        if hdr is None:
            continue
        try:
            cube, wls = read_envi_cube(hdr)
            cube, wls = trim_and_select_bands(cube, wls, BAND_MIN, BAND_MAX)
            cube = smooth_spectra_vectorized(cube, window=SMOOTH_WINDOW)
            patch_size = int(r.get("patch_size", DEFAULT_PATCH_SIZE))
            center_r = int(r["center_row"]); center_c = int(r["center_col"])
            spec = mean_patch_spectrum(cube, center_r, center_c, patch_size)
            if spec is not None:
                spectra.append(spec)
        except Exception as e:
            if VERBOSE:
                print(f"[WARN] building ref spec failed for {hdr}: {e}")
    if not spectra:
        return None, []
    stacked = np.stack(spectra, axis=0)
    ref = np.median(stacked, axis=0)
    return ref.astype(np.float32), spectra


# Main pipeline
def generate_similarity_labels_auto():
    if not ANNOTATIONS_CSV.exists():
        raise SystemExit(f"[ERR] annotations not found: {ANNOTATIONS_CSV}")
    ann = pd.read_csv(ANNOTATIONS_CSV)

    # --- Normalize plant_id immediately and ensure contains_symptom boolean ---
    ann['plant_id'] = ann['plant_id'].apply(normalize_plant_id_val)
    # ensure contains_symptom is boolean
    if 'contains_symptom' in ann.columns:
        ann['contains_symptom'] = ann['contains_symptom'].astype(bool)
    else:
        ann['contains_symptom'] = False

    print(f"[INFO] Loaded annotations: {len(ann)} rows")
    if VERBOSE:
        print("[DEBUG] unique normalized plants (sample):", sorted(list(pd.unique(ann['plant_id'].astype(str))))[:60])
        print("[DEBUG] symptom counts per plant (top 20):")
        print(ann[ann['contains_symptom']].plant_id.value_counts().head(20))

    # Configure refs: try configured then auto-select if needed.
    disease_ref = str(REF_PLANT_DISEASE)
    water_ref = str(REF_PLANT_WATER)

    # Force water_ref to REF_PLANT_WATER if that plant exists in normalized annotations
    if str(REF_PLANT_WATER) in set(ann['plant_id'].astype(str).unique()):
        water_ref = str(REF_PLANT_WATER)
        if VERBOSE:
            print(f"[OVERRIDE] Forcing water_ref = {water_ref} because it exists in annotations")

    rd, rd_spectra = build_reference_spectrum(ann, disease_ref, DATASET_ROOT)
    rw, rw_spectra = build_reference_spectrum(ann, water_ref, DATASET_ROOT)

    # If either ref missing, auto-select robustly
    if rd is None or rw is None:
        disease_auto, water_auto, ranked = auto_select_reference_plants(ann)
        if disease_auto is None:
            raise SystemExit("[ERROR] No plants with CSV-derived symptoms found; cannot auto-select references.")
        if rd is None:
            disease_ref = disease_auto
            rd, rd_spectra = build_reference_spectrum(ann, disease_ref, DATASET_ROOT)
        if rw is None:
            # prefer forced water_ref if present and different
            if str(REF_PLANT_WATER) in ranked and str(REF_PLANT_WATER) != disease_ref:
                water_ref = str(REF_PLANT_WATER)
            else:
                water_ref = next((p for p in ranked if p != disease_ref), disease_ref)
            rw, rw_spectra = build_reference_spectrum(ann, water_ref, DATASET_ROOT)

    if rd is None and rw is None:
        raise SystemExit("[ERROR] Could not build reference spectra after auto-selection.")
    if VERBOSE:
        print(f"[INFO] Using disease_ref={disease_ref}, water_ref={water_ref}")

    # Compute thresholds
    disease_sims = [cosine_similarity(s, rd) for s in rd_spectra] if rd is not None else []
    water_sims = [cosine_similarity(s, rw) for s in rw_spectra] if rw is not None else []
    thresh = {}
    thresh["disease_symp_cosine"] = float(np.percentile(disease_sims, 100 - SYMPTOM_PERCENTILE)) if disease_sims else 0.92
    thresh["disease_pre_cosine"] = float(np.percentile(disease_sims, 100 - PRE_PERCENTILE)) if disease_sims else 0.85
    thresh["water_symp_cosine"] = float(np.percentile(water_sims, 100 - SYMPTOM_PERCENTILE)) if water_sims else 0.92
    thresh["water_pre_cosine"] = float(np.percentile(water_sims, 100 - PRE_PERCENTILE)) if water_sims else 0.85
    if VERBOSE:
        print("[INFO] thresholds:", thresh)

    # Compute similarities per annotation row
    rows_out = []
    leaf_day_sims: Dict[Tuple[str, str], Dict[str, List[Tuple[float, float]]]] = {}

    for idx, row in tqdm(ann.iterrows(), total=len(ann), desc="Compute patch similarities"):
        plant = str(row["plant_id"]); leaf = str(row["leaf_id"]); day = str(row["day"])
        center_r = int(row["center_row"]); center_c = int(row["center_col"])
        patch_size = int(row.get("patch_size", DEFAULT_PATCH_SIZE))
        contains_sym = bool(row.get("contains_symptom", False))
        hdr = find_hdr_for_annotation(row, DATASET_ROOT)
        sim_d = sim_w = 0.0
        sam_d = sam_w = float("inf")
        conf = 0.0
        if hdr is not None:
            try:
                cube, wls = read_envi_cube(hdr)
                cube, wls = trim_and_select_bands(cube, wls, BAND_MIN, BAND_MAX)
                cube = smooth_spectra_vectorized(cube, window=SMOOTH_WINDOW)
                spec = mean_patch_spectrum(cube, center_r, center_c, patch_size)
                if spec is not None:
                    if rd is not None:
                        sim_d = cosine_similarity(spec, rd)
                        sam_d = spectral_angle(spec, rd)
                    if rw is not None:
                        sim_w = cosine_similarity(spec, rw)
                        sam_w = spectral_angle(spec, rw)
                    best = max(sim_d, sim_w)
                    second = min(sim_d, sim_w)
                    conf = float(max(0.0, best - second))
            except Exception as e:
                if VERBOSE:
                    print(f"[WARN] sim compute failed for hdr {hdr}: {e}")
        leaf_day_sims.setdefault((plant, leaf), {}).setdefault(day, []).append((sim_d, sim_w))
        rows_out.append({
            "idx": int(idx),
            "plant_id": plant,
            "leaf_id": leaf,
            "day": day,
            "center_row": int(center_r),
            "center_col": int(center_c),
            "patch_size": int(patch_size),
            "mask_coverage": float(row.get("mask_coverage", 0.0)),
            "contains_symptom": bool(contains_sym),
            "sim_disease": float(sim_d),
            "sim_water": float(sim_w),
            "sam_disease": float(sam_d),
            "sam_water": float(sam_w),
            "sim_confidence": float(conf)
        })

    # Aggregate per-leaf time-series and decide first symptomatic day & assigned class
    leaf_decisions = {}
    for (plant, leaf), day_map in leaf_day_sims.items():
        def day_to_int(s):
            m = re.search(r"(\d+)", str(s))
            return int(m.group(1)) if m else 0
        day_keys = sorted(list(day_map.keys()), key=day_to_int)
        disease_scores = []; water_scores = []; days_ordered = []
        for d in day_keys:
            sims = np.array(day_map[d])
            if sims.size:
                disease_scores.append(float(SIM_AGG_FUNC(sims[:, 0])))
                water_scores.append(float(SIM_AGG_FUNC(sims[:, 1])))
            else:
                disease_scores.append(0.0); water_scores.append(0.0)
            days_ordered.append(d)
        disease_scores = np.array(disease_scores); water_scores = np.array(water_scores)

        first_symp_day = None; assigned_class = "healthy"
        # strong symptomatic detection first
        for i, d in enumerate(days_ordered):
            sd = disease_scores[i]; sw = water_scores[i]
            if rd is not None and sd >= thresh["disease_symp_cosine"] and (sd - sw) >= MARGIN:
                first_symp_day = d; assigned_class = "disease"; break
            if rw is not None and sw >= thresh["water_symp_cosine"] and (sw - sd) >= MARGIN:
                first_symp_day = d; assigned_class = "water"; break
        # pre-symptomatic check
        if first_symp_day is None:
            for i, d in enumerate(days_ordered):
                sd = disease_scores[i]; sw = water_scores[i]
                if rd is not None and sd >= thresh["disease_pre_cosine"] and (sd - sw) >= MARGIN:
                    first_symp_day = d; assigned_class = "disease_pre"; break
                if rw is not None and sw >= thresh["water_pre_cosine"] and (sw - sd) >= MARGIN:
                    first_symp_day = d; assigned_class = "water_pre"; break

        leaf_decisions[(plant, leaf)] = {
            "days": days_ordered,
            "disease_scores": disease_scores.tolist(),
            "water_scores": water_scores.tolist(),
            "first_symptom_day": first_symp_day,
            "assigned_class": assigned_class
        }

        pd.DataFrame({
            "day": days_ordered,
            "disease_sim": disease_scores,
            "water_sim": water_scores
        }).to_csv(DIAG_DIR / f"diag_plant{plant}_leaf{leaf}.csv", index=False)

    # Assign final labels per-row using CSV evidence and leaf_decisions
    out_rows_final = []
    for r in rows_out:
        plant = r["plant_id"]; leaf = r["leaf_id"]; day = r["day"]
        contains_sym = r["contains_symptom"]; sim_d = r["sim_disease"]; sim_w = r["sim_water"]
        dd = leaf_decisions.get((plant, leaf), {})
        first_day = dd.get("first_symptom_day", None)
        assigned_class = dd.get("assigned_class", "healthy")
        label = 0; label_source = "auto_default"

        # Priority 1: explicit CSV symptomatic point in this patch
        if contains_sym:
            if rd is not None and sim_d >= (sim_w + MARGIN):
                label = 2; label_source = "csv_symptom_vs_ref_disease"
            elif rw is not None and sim_w >= (sim_d + MARGIN):
                label = 4; label_source = "csv_symptom_vs_ref_water"
            else:
                if assigned_class.startswith("disease"):
                    label = 2; label_source = "csv_symptom_leafdec_disease"
                elif assigned_class.startswith("water"):
                    label = 4; label_source = "csv_symptom_leafdec_water"
                else:
                    label = 2; label_source = "csv_symptom_unknown_pref_disease"
            out_rows_final.append({**r, "label": label, "label_source": label_source, "first_symptom_day": first_day})
            continue

        # Non-CSV rows: leaf-wide decision & temporal position
        def day_to_int_try(s):
            m = re.search(r"(\d+)", str(s)); return int(m.group(1)) if m else None
        d_int = day_to_int_try(day); fsd_int = day_to_int_try(first_day) if first_day is not None else None

        if first_day is None:
            label = 0; label_source = "no_symptom_evidence"
        else:
            if d_int is None or fsd_int is None:
                label = 0; label_source = "day_parse_failed"
            else:
                if d_int >= fsd_int:
                    if assigned_class.startswith("disease"):
                        label = 2; label_source = "leaf_symp_auto_disease"
                    elif assigned_class.startswith("water"):
                        label = 4; label_source = "leaf_symp_auto_water"
                    else:
                        label = 0; label_source = "leaf_symp_auto_unknown"
                elif fsd_int - PRE_WINDOW <= d_int < fsd_int:
                    if assigned_class.startswith("disease"):
                        label = 1; label_source = "prewindow_leafwide_auto_disease"
                    elif assigned_class.startswith("water"):
                        label = 3; label_source = "prewindow_leafwide_auto_water"
                    else:
                        label = 0; label_source = "prewindow_auto_unknown"
                else:
                    label = 0; label_source = "outside_prewindow"
        out_rows_final.append({**r, "label": int(label), "label_source": label_source, "first_symptom_day": first_day})

    # Save outputs
    out_df = pd.DataFrame(out_rows_final)
    out_path = OUTPUT_DIR / "labels_patches.csv"
    out_df.to_csv(out_path, index=False)

    counts = out_df["label"].value_counts().to_dict()
    for k in [0, 1, 2, 3, 4]:
        counts.setdefault(k, 0)
    summary = {
        "total_rows": int(len(out_df)),
        "counts": counts,
        "thresholds": thresh,
        "ref_plants": {"disease": disease_ref, "water": water_ref}
    }
    with open(OUTPUT_DIR / "labels_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] labels written to:", out_path)
    print("[INFO] summary:", summary)
    print("[INFO] diagnostics:", DIAG_DIR)
    return out_df, summary


# Run
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_similarity_labels_auto()
