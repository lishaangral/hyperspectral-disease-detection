#!/usr/bin/env python3
"""
PyTorch Dataset & DataLoader for temporal patch sequences from July2022 hyperspectral dataset.

Responsibilities:
- Load `labels_patches.csv` produced by the labeling pipeline.
- Group rows by (plant_id, leaf_id, center_row, center_col) -> one sequence per spatial location.
- For each timestep in the sequence, read the ENVI cube (.hdr/.dat) for that day/plant,
  extract the patch cube (patch_size x patch_size x bands), optionally apply smoothing/band-slice,
  normalize bands using provided mean/std or compute on the fly (not recommended).
- Return sequence tensors and per-timestep labels/masks for supervised training.

Notes:
- This module reads ENVI via `spectral.open_image`. Install `spectral` (pip install spectral).
- The module favors clarity and safe defaults. Tune caching and normalization for your environment.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import os
import math
import json
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from spectral import open_image

# Utilities
def parse_day_str(day_str: str) -> Optional[int]:
    """Convert folder/day string to integer day index. Returns None if fails."""
    if pd.isna(day_str):
        return None
    s = str(day_str)
    # quick integer parse
    try:
        return int(s)
    except Exception:
        import re
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    return None


def load_envi_cube(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ENVI image using spectral.open_image
    Returns: cube (H,W,B) float32, wavelengths (B,) float (if present else band indices)
    """
    img = open_image(str(hdr_path))
    cube = img.load().astype(np.float32)
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


def nearest_band_indices(wavelengths: np.ndarray, min_wl: float, max_wl: float) -> np.ndarray:
    """Return band indices where wavelengths are within [min_wl, max_wl]."""
    idxs = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
    if idxs.size == 0:
        # fallback: return all bands
        return np.arange(len(wavelengths), dtype=int)
    return idxs


def extract_patch_from_cube(cube: np.ndarray, center: Tuple[int, int], patch_size: int) -> np.ndarray:
    """Extract patch cube (H_patch, W_patch, B) given center (r,c)."""
    r, c = center
    half = patch_size // 2
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    # If patch goes out of bounds, pad with zeros to maintain consistent size
    H, W, B = cube.shape
    pad_top = max(0, -r0)
    pad_left = max(0, -c0)
    pad_bottom = max(0, r1 - H)
    pad_right = max(0, c1 - W)
    r0_clamped = max(0, r0)
    r1_clamped = min(H, r1)
    c0_clamped = max(0, c0)
    c1_clamped = min(W, c1)
    patch = cube[r0_clamped:r1_clamped, c0_clamped:c1_clamped, :]
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
    return patch


# Dataset class
class PatchSequenceDataset(Dataset):
    """
    Dataset that yields sequences of patches for a spatial location across days.

    Constructor args:
      annotations_csv: CSV created by labeler (labels_patches.csv) or annotations_patches.csv (if you prefer)
      dataset_root: path to folder that contains day subfolders and plant folders
      patch_size: spatial patch size (int)
      band_min/band_max: spectral slice (nm)
      normalize_stats: dict with 'mean' and 'std' arrays for per-band normalization (shape: [S,])
      cache_mode: 'none' (always read), 'memory' (cache cubes in RAM), 'disk' (save per-patch npz files under cache_dir)
      cache_dir: path for disk caching (if cache_mode == 'disk')
    """

    def __init__(
        self,
        annotations_csv: str,
        dataset_root: str,
        patch_size: int = 32,
        band_min: float = 580.0,
        band_max: float = 760.0,
        normalize_stats: Optional[Dict[str, np.ndarray]] = None,
        cache_mode: str = "memory",
        cache_dir: Optional[str] = None,
        min_mask_coverage: float = 0.6,
    ):
        super().__init__()
        self.annotations_csv = Path(annotations_csv)
        self.dataset_root = Path(dataset_root)
        self.patch_size = int(patch_size)
        self.band_min = float(band_min)
        self.band_max = float(band_max)
        self.normalize_stats = normalize_stats
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_mask_coverage = float(min_mask_coverage)

        if self.cache_mode not in ("none", "memory", "disk"):
            raise ValueError("cache_mode must be one of 'none','memory','disk'")

        if self.cache_mode == "disk" and self.cache_dir is None:
            raise ValueError("cache_dir must be provided when cache_mode == 'disk'")

        # read CSV
        df = pd.read_csv(self.annotations_csv)
        required_cols = {"plant_id", "leaf_id", "day", "center_row", "center_col", "patch_size", "mask_coverage", "contains_symptom", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Annotations CSV missing required columns: {missing}")

        # convert types
        df["center_key"] = df["center_row"].astype(str) + "_" + df["center_col"].astype(str)
        df["day_int"] = df["day"].apply(parse_day_str)
        if df["day_int"].isnull().any():
            warnings.warn("Some day strings could not be parsed to integers; those rows may be excluded during grouping.")

        # group rows into sequences by (plant_id, leaf_id, center_key)
        self.seq_groups = []
        grouped = df.groupby(["plant_id", "leaf_id", "center_key"])
        for (plant, leaf, center_key), g in grouped:
            # sort by day_int
            g_sorted = g.sort_values("day_int")
            # build entries: each entry holds reference to plant/leaf folder for that day
            seq_rows = []
            for _, row in g_sorted.iterrows():
                # determine plant folder path for this day: look under dataset_root/<day>/<plant_folder> (best-effort)
                # We assume directory structure: dataset_root/<day_folder>/<plant_folder>/...hdr
                # Day folder may be 'day_3' or '3' etc. We'll search for folder that contains the day int.
                day_int = int(row["day_int"]) if not pd.isna(row["day_int"]) else None
                seq_rows.append({
                    "plant": str(plant),
                    "leaf": str(leaf),
                    "day_str": str(row["day"]),
                    "day_int": day_int,
                    "center_row": int(row["center_row"]),
                    "center_col": int(row["center_col"]),
                    "mask_coverage": float(row["mask_coverage"]),
                    "contains_symptom": bool(row["contains_symptom"]),
                    "label": int(row["label"]) if "label" in row and not pd.isna(row["label"]) else -1,
                    # placeholder for hdr_path to be resolved lazily
                    "hdr_path": None,
                    "plant_dir_name": None,
                })
            # store group
            self.seq_groups.append(((str(plant), str(leaf), center_key), seq_rows))

        # lazy mapping from day+plant -> hdr path; we'll discover on-demand
        self._hdr_cache = {}   # key: (day_str, plant_foldername) -> hdr_path
        self._cube_memory_cache = {}  # key: hdr_path -> (cube,wls)
        if self.cache_mode == "disk":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-resolve HDR paths for each seq_rows entry by searching dataset tree for matching day folder
        # Build map of day folder names for quick lookup
        self._day_folder_names = [p.name for p in sorted(self.dataset_root.iterdir()) if p.is_dir()]

        # Attempt to find plant folder name inside each day folder that contains the plant id substring (plant id might be '2' or 'plant_2' etc.)
        for key, seq_rows in self.seq_groups:
            for entry in seq_rows:
                day_str = entry["day_str"]
                # we will look for a day folder containing the day_str integer if parseable
                if entry["day_int"] is not None:
                    # try exact match patterns
                    matched_day_folder = None
                    for dfname in self._day_folder_names:
                        if str(entry["day_int"]) in dfname:
                            matched_day_folder = dfname
                            break
                    if matched_day_folder is None:
                        # fallback to the first day folder (best-effort)
                        matched_day_folder = self._day_folder_names[0]
                else:
                    matched_day_folder = self._day_folder_names[0]

                # now find plant folder inside that day folder
                day_folder_path = self.dataset_root / matched_day_folder
                # try to find folder that contains plant id substring
                plant_candidates = []
                for cand in day_folder_path.iterdir():
                    if cand.is_dir() and (str(entry["plant"]) in cand.name or f"plant_{entry['plant']}" in cand.name):
                        plant_candidates.append(cand)
                if plant_candidates:
                    chosen = plant_candidates[0]
                else:
                    # fallback to first dir in day folder (best-effort)
                    subdirs = [d for d in day_folder_path.iterdir() if d.is_dir()]
                    chosen = subdirs[0] if subdirs else None

                if chosen is None:
                    entry["hdr_path"] = None
                    entry["plant_dir_name"] = None
                else:
                    # find first .hdr inside chosen
                    hdrs = list(chosen.glob("*.hdr"))
                    entry["hdr_path"] = str(hdrs[0]) if hdrs else None
                    entry["plant_dir_name"] = chosen.name

        # final: filter out sequences with no valid hdr_path in any row
        validated_groups = []
        for key, seq_rows in self.seq_groups:
            # if at least one row has hdr_path not None, keep sequence (we'll skip missing timesteps at retrieval)
            if any(r["hdr_path"] for r in seq_rows):
                validated_groups.append((key, seq_rows))
        self.seq_groups = validated_groups

    def __len__(self):
        return len(self.seq_groups)

    def _load_cube_cached(self, hdr_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ENVI cube and return (H,W,B), wavelengths.
        Caching behavior depends on cache_mode.
        """
        if hdr_path is None:
            raise FileNotFoundError("hdr_path is None while trying to load cube")

        if self.cache_mode == "memory":
            if hdr_path in self._cube_memory_cache:
                return self._cube_memory_cache[hdr_path]
            cube, wls = load_envi_cube(Path(hdr_path))
            self._cube_memory_cache[hdr_path] = (cube, wls)
            return cube, wls
        elif self.cache_mode == "disk":
            # disk caching of full cube can be very large; instead cache per-patch NPZ when extracting
            cube, wls = load_envi_cube(Path(hdr_path))
            return cube, wls
        else:
            # no caching
            cube, wls = load_envi_cube(Path(hdr_path))
            return cube, wls

    def _process_patch_from_entry(self, entry: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Given one row-entry, load the corresponding hdr cube and extract the patch,
        slice bands to [band_min,band_max], transpose to (S,H,W) and return float32 numpy.
        Returns None if hdr/path missing.
        """
        hdr_path = entry.get("hdr_path", None)
        if hdr_path is None:
            return None
        # if disk caching per-patch: check npz path
        if self.cache_mode == "disk" and self.cache_dir is not None:
            # create unique filename based on hdr path + center coords
            basename = f"{Path(hdr_path).stem}_r{entry['center_row']}_c{entry['center_col']}.npz"
            cache_file = self.cache_dir / basename
            if cache_file.exists():
                arr = np.load(cache_file)["arr"]
                return arr.astype(np.float32)
        # load cube
        cube, wls = self._load_cube_cached(hdr_path)
        # choose bands
        band_idxs = nearest_band_indices(wls, self.band_min, self.band_max)
        subcube = cube[:, :, band_idxs]  # (H,W,S)
        # extract patch
        patch = extract_patch_from_cube(subcube, (entry["center_row"], entry["center_col"]), self.patch_size)  # (H_p, W_p, S)
        # reorder to (S, H, W)
        patch = np.transpose(patch, (2, 0, 1)).astype(np.float32)
        # optional normalization (per-band)
        if self.normalize_stats is not None:
            mean = self.normalize_stats.get("mean")  # shape (S,)
            std = self.normalize_stats.get("std")
            if mean is not None and std is not None:
                # mean/std assumed shape (S,)
                # expand to (S,1,1)
                m = mean.reshape(-1, 1, 1)
                s = std.reshape(-1, 1, 1)
                patch = (patch - m) / (s + 1e-8)
        # cache per-patch if disk mode
        if self.cache_mode == "disk" and self.cache_dir is not None:
            np.savez_compressed(cache_file, arr=patch)
        return patch

    def __getitem__(self, idx: int):
        """
        Return:
          seq_tensor: torch.FloatTensor of shape (T, 1, S, H, W)  (T = number of timesteps for this sequence)
          labels: torch.LongTensor shape (T,) with -1 for unlabeled timesteps
          label_mask: torch.BoolTensor shape (T,) True where label != -1
          meta: dict with keys 'plant','leaf','center_key','days' etc.
        """
        key, seq_rows = self.seq_groups[idx]
        patches = []
        labels = []
        day_ints = []
        mask_flags = []

        for entry in seq_rows:
            try:
                patch = self._process_patch_from_entry(entry)
            except Exception as e:
                patch = None
            if patch is None:
                # represent missing timestep as zeros of expected shape
                # we still want to keep temporal position to preserve T length
                # infer S from normalize_stats or fallback
                if self.normalize_stats is not None and "mean" in self.normalize_stats:
                    S = len(self.normalize_stats["mean"])
                else:
                    # attempt to infer from a cached cube
                    # naive fallback: set zeros with (1, patch_size, patch_size)
                    S = 1
                patch = np.zeros((S, self.patch_size, self.patch_size), dtype=np.float32)
                labeled = -1
                mask_flag = False
            else:
                labeled = int(entry.get("label", -1))
                mask_flag = (labeled != -1)
            patches.append(torch.from_numpy(patch))  # each is (S,H,W)
            labels.append(int(labeled))
            day_ints.append(entry.get("day_int", None))
            mask_flags.append(bool(mask_flag))

        # stack: T x (S,H,W) -> we want (T, 1, S, H, W)
        patches = [p.unsqueeze(0) for p in patches]  # each -> (1,S,H,W)
        seq_tensor = torch.stack(patches, dim=0)  # (T,1,S,H,W)
        labels = torch.LongTensor(labels)
        label_mask = torch.BoolTensor(mask_flags)

        meta = {
            "plant": key[0],
            "leaf": key[1],
            "center_key": key[2],
            "days": day_ints
        }
        return seq_tensor, labels, label_mask, meta


# Collate function for DataLoader
def patch_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor, Dict]]):
    """
    Collate a batch of variable-length sequences.
    Inputs: list of (seq_tensor (T,1,S,H,W), labels (T,), mask (T,), meta)
    Outputs:
      batch_seq: FloatTensor (B, T_max, 1, S, H, W)
      batch_labels: LongTensor (B, T_max) with -1 padded for missing
      batch_label_mask: BoolTensor (B, T_max)
      lengths: LongTensor (B,) actual lengths
      metas: list of meta dicts
    """
    seqs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    metas = [item[3] for item in batch]

    lengths = torch.LongTensor([s.shape[0] for s in seqs])
    T_max = lengths.max().item()

    B = len(seqs)
    # shapes
    _, _, S, H, W = seqs[0].shape

    # create tensors
    batch_seq = torch.zeros((B, T_max, 1, S, H, W), dtype=seqs[0].dtype)
    batch_labels = torch.full((B, T_max), fill_value=-1, dtype=torch.long)
    batch_mask = torch.zeros((B, T_max), dtype=torch.bool)

    for i, (s, lab, m) in enumerate(zip(seqs, labels, masks)):
        t = s.shape[0]
        batch_seq[i, :t] = s  # s is (T,1,S,H,W)
        batch_labels[i, :t] = lab
        batch_mask[i, :t] = m

    return {
        "seq": batch_seq,           # (B, T_max, 1, S, H, W)
        "labels": batch_labels,     # (B, T_max)
        "label_mask": batch_mask,   # (B, T_max)
        "lengths": lengths,         # (B,)
        "meta": metas
    }
