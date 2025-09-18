#!/usr/bin/env python3
"""
Labeling module for July2022 hyperspectral dataset.

Transforms patch-level annotation rows (from preprocessing) + plant/leaf metadata
into final per-patch, per-day supervised labels for training the 3D-CNN + LSTM model.

Key features:
 - Ingests `annotations_patches.csv` (one row per patch/day produced by preprocessing)
 - Ingests optional `plant_metadata.csv` or infers treatment groups from directory names
 - Computes `first_symptom_day` per (plant_id, leaf_id) from available CSV-derived flags
 - Assigns one of five labels per patch/day:
       0 = healthy
       1 = pre_symp_disease
       2 = symp_disease
       3 = pre_symp_water
       4 = symp_water
 - Supports two pre-symptomatic labeling modes:
     * "leaf_wide": label all patches on the leaf within pre_window as pre-symp
     * "spatially_precise": label only patches that later contain symptom pixels
 - Outputs final `labels_patches.csv` ready for model training and a small summary JSON.

"""

from pathlib import Path
import argparse
import json
import math
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Label definitions & utils
LABEL_HEALTHY = 0
LABEL_PRE_SYMP_DISEASE = 1
LABEL_SYMP_DISEASE = 2
LABEL_PRE_SYMP_WATER = 3
LABEL_SYMP_WATER = 4

DAY_PARSE_HINTS = ("day", "d", "t")  # fallback tokens when parsing day string -> int


def parse_day_str(day_str: str) -> Optional[int]:
    """
    Convert folder/day string into integer day index.
    Handles formats like 'day_3', 'Day-07', '3', or 'd3'.
    Returns None if parsing fails.
    """
    if pd.isna(day_str):
        return None
    s = str(day_str).lower()
    # direct integer attempt
    try:
        return int(s)
    except Exception:
        pass
    # try to find integers in the string
    import re
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


# Core labeler logic
def compute_first_symptom_day_per_leaf(df: pd.DataFrame) -> Dict[Tuple[str, str], Optional[int]]:
    """
    Given annotations dataframe with columns ['plant_id','leaf_id','day','contains_symptom'],
    return mapping (plant_id,leaf_id) -> earliest day (int) where contains_symptom == True.
    If no symptom rows exist, value is None.
    """
    mapping = {}
    grouped = df.groupby(["plant_id", "leaf_id"])
    for (plant, leaf), g in grouped:
        # parse day strings to ints, keep those where contains_symptom true
        symptomatic = g[g["contains_symptom"] == True]
        if symptomatic.empty:
            mapping[(plant, leaf)] = None
            continue
        days = [parse_day_str(x) for x in symptomatic["day"].values]
        days = [d for d in days if d is not None]
        mapping[(plant, leaf)] = min(days) if days else None
    return mapping


def infer_treatment_groups_from_manifest(manifest_path: Optional[Path]) -> Dict[str, str]:
    """
    Read a plant manifest CSV if provided with columns ['plant_id','treatment'].
    If manifest not provided, returns empty dict and the caller may provide manual mapping.
    """
    mapping = {}
    if manifest_path is None:
        return mapping
    if not manifest_path.exists():
        return mapping
    df = pd.read_csv(manifest_path)
    if "plant_id" not in df.columns or "treatment" not in df.columns:
        return mapping
    for _, row in df.iterrows():
        pid = str(row["plant_id"])
        mapping[pid] = str(row["treatment"]).lower()
    return mapping


def assign_labels(df: pd.DataFrame,
                  treatment_map: Dict[str, str],
                  first_symptom_map: Dict[Tuple[str, str], Optional[int]],
                  pre_window: int = 3,
                  mode: str = "leaf_wide") -> pd.DataFrame:
    """
    Assign final labels to each patch/day row in df.
    df must include: ['plant_id','leaf_id','day','mask_coverage','contains_symptom'].
    mode:
      - 'leaf_wide' : any patch on the leaf during pre-window gets pre_symp label
      - 'spatially_precise' : only patches that later contain symptom pixels get pre_symp
    Returns a new DataFrame with an added 'label' (int) and 'label_source' columns.
    """
    rows = []
    # ensure day_int column
    df = df.copy()
    df["day_int"] = df["day"].apply(parse_day_str)
    # precompute mapping of which (plant,leaf,center)->future_symptom_days
    # also build a quick lookup of patches that ever contain symptom (for spatially_precise)
    df["center_key"] = df["center_row"].astype(str) + "_" + df["center_col"].astype(str)
    ever_symp = set(df[df["contains_symptom"] == True].apply(
        lambda r: (r["plant_id"], r["leaf_id"], r["center_key"]), axis=1).tolist())

    for _, row in df.iterrows():
        plant = str(row["plant_id"])
        leaf = str(row["leaf_id"])
        day_int = row["day_int"]
        center_key = row["center_key"]
        contains_sym = bool(row["contains_symptom"])
        mask_cov = float(row.get("mask_coverage", 0.0))

        # default label = healthy
        label = LABEL_HEALTHY
        label_source = "auto_default"

        # treatment prior (if available) - 'inoculated', 'water_stress', 'control' expected
        treatment = treatment_map.get(plant, None)

        # first known symptom day for this leaf (from CSV-derived mapping)
        first_day = first_symptom_map.get((plant, leaf), None)

        # symptomatic assignment: if patch contains_symptom on this day -> symp
        if contains_sym:
            if treatment == "water_stress":
                label = LABEL_SYMP_WATER
                label_source = "csv_symptom_pixel"
            else:
                # default to disease if treatment unknown/inoculated
                label = LABEL_SYMP_DISEASE
                label_source = "csv_symptom_pixel"
            rows.append({**row.to_dict(), "label": label, "label_source": label_source})
            continue

        # if no first_day known, we cannot assign pre-window confidently; remain healthy
        if first_day is None or day_int is None:
            rows.append({**row.to_dict(), "label": label, "label_source": label_source})
            continue

        # compute if this day is in pre-window
        if (first_day - pre_window) <= day_int < first_day:
            # pre-symptomatic logic differs by mode
            if mode == "leaf_wide":
                if treatment == "water_stress":
                    label = LABEL_PRE_SYMP_WATER
                else:
                    label = LABEL_PRE_SYMP_DISEASE
                label_source = "prewindow_leafwide"
            elif mode == "spatially_precise":
                # assign pre_symp only if this center is one that ever becomes symptomatic later
                key_trip = (plant, leaf, center_key)
                if key_trip in ever_symp:
                    if treatment == "water_stress":
                        label = LABEL_PRE_SYMP_WATER
                    else:
                        label = LABEL_PRE_SYMP_DISEASE
                    label_source = "prewindow_spatial"
                else:
                    # remain healthy if this spatial location never becomes symptomatic
                    label = LABEL_HEALTHY
                    label_source = "prewindow_spatial_not_near_future_symptom"
            else:
                # unknown mode -> default to leaf_wide
                if treatment == "water_stress":
                    label = LABEL_PRE_SYMP_WATER
                else:
                    label = LABEL_PRE_SYMP_DISEASE
                label_source = "prewindow_default"
        else:
            # not in pre-window and no contains_symptom -> healthy
            label = LABEL_HEALTHY
            label_source = "outside_prewindow_or_no_symptom"

        rows.append({**row.to_dict(), "label": int(label), "label_source": label_source})

    out_df = pd.DataFrame(rows)
    # Drop helper columns if needed, but keep center_key/day_int for traceability
    return out_df


# Output helpers & summary
def save_labels(out_df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[SAVED] Labeled patches: {out_csv} ({len(out_df)} rows)")


def summary_stats(labeled_df: pd.DataFrame) -> Dict:
    counts = labeled_df["label"].value_counts().to_dict()
    # make sure every label key exists
    for k in [LABEL_HEALTHY, LABEL_PRE_SYMP_DISEASE, LABEL_SYMP_DISEASE, LABEL_PRE_SYMP_WATER, LABEL_SYMP_WATER]:
        counts.setdefault(k, 0)
    unique_leaves = labeled_df[["plant_id", "leaf_id"]].drop_duplicates().shape[0]
    return {
        "total_rows": int(len(labeled_df)),
        "unique_leaves": int(unique_leaves),
        "counts": counts
    }


# CLI glue
def parse_args():
    p = argparse.ArgumentParser(description="Assign 5-class labels to preprocessed patch annotations.")
    p.add_argument("--annotations", required=True, help="Path to annotations_patches.csv from preprocessing.")
    p.add_argument("--manifest", required=False, default=None,
                   help="Optional plant manifest CSV with columns ['plant_id','treatment'].")
    p.add_argument("--out-dir", required=True, help="Directory to write labeled CSV and summary.")
    p.add_argument("--pre-window", type=int, default=3, help="Number of days before first symptom to label as pre-symp.")
    p.add_argument("--mode", choices=("leaf_wide", "spatially_precise"), default="leaf_wide",
                   help="How to assign pre-symptomatic labels spatially.")
    return p.parse_args()


def main():
    args = parse_args()
    ann_path = Path(args.annotations)
    manifest_path = Path(args.manifest) if args.manifest else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise SystemExit(f"[ERROR] annotations file not found: {ann_path}")

    ann_df = pd.read_csv(ann_path)
    print(f"[INFO] Loaded annotations ({len(ann_df)} rows)")

    # treatment map (optional). If manifest missing, fallback to empty mapping (user can edit later).
    treatment_map = infer_treatment_groups_from_manifest(manifest_path)
    if treatment_map:
        print(f"[INFO] Loaded treatment manifest for {len(treatment_map)} plants")
    else:
        print(f"[WARN] No manifest provided or invalid manifest. Treatments must be filled or inferred later.")

    # compute first symptom day per leaf from CSV-derived contains_symptom flag
    first_symp_map = compute_first_symptom_day_per_leaf(ann_df)
    n_with_symp = sum(1 for v in first_symp_map.values() if v is not None)
    print(f"[INFO] Found {n_with_symp} leaf instances with symptomatic CSV-derived days")

    # assign labels
    labeled = assign_labels(ann_df, treatment_map, first_symp_map,
                            pre_window=args.pre_window, mode=args.mode)

    out_csv = out_dir / "labels_patches.csv"
    save_labels(labeled, out_csv)

    # save a small summary JSON
    summary = summary_stats(labeled)
    summary_path = out_dir / "labels_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Summary -> {summary_path}")
    print(f"[DONE] Labeling complete. Stats: {summary}")


if __name__ == "__main__":
    main()
