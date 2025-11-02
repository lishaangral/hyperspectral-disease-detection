#!/usr/bin/env python3
"""
Training script for 3D-CNN + LSTM hyperspectral early-detection model.

This version uses an in-file configuration block (edit these variables directly)
instead of CLI arguments. Edit the CONFIG section below and run:

    python train_hybrid.py

Expectations:
 - `dataset_sequence.py` provides PatchSequenceDataset and patch_collate.
 - `model_hybrid.py` provides Hybrid3DConvLSTM.
 - `labels_patches.csv` (output of labeler) exists and is prepared.
 - Split your annotations into train/val CSVs beforehand (recommended) and point
   TRAIN_ANNOTATIONS / VAL_ANNOTATIONS to those files.

The training loop uses masked per-timestep cross-entropy and computes validation
metrics including early-detection statistics. Adjust hyperparameters in the
CONFIG block to experiment.
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project modules — adapt filenames/locations if you renamed files
from dataset_sequence import PatchSequenceDataset, patch_collate
from model_hybrid import Hybrid3DConvLSTM

# CONFIG (edit these values)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_ANNOTATIONS = PROJECT_ROOT / "preproc" / "labels_patches_train.csv"   # <- set to your train CSV
VAL_ANNOTATIONS = PROJECT_ROOT / "preproc" / "labels_patches_val.csv"       # <- set to your val CSV
DATASET_ROOT = PROJECT_ROOT / "DATASET" / "July2022"                       # dataset root (day folders)
OUT_DIR = PROJECT_ROOT / "runs" / "exp_manual_config"                       # where checkpoints & logs are saved

# Data / patch settings
PATCH_SIZE = 32
BAND_MIN = 580.0
BAND_MAX = 760.0
CACHE_MODE = "memory"   # "none" | "memory" | "disk"
CACHE_DIR = PROJECT_ROOT / "cache"    # used only if CACHE_MODE == "disk"
MIN_MASK_COVERAGE = 0.6

# Training hyperparameters
EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda"      # or "cpu"
NUM_WORKERS = 4

# Model hyperparameters
ENCODER_PARAMS = {"in_channels": 1, "encoder_dim": 256, "base_channels": 16}
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
BIDIRECTIONAL = True
CLASSIFIER_HIDDEN = 128
NUM_CLASSES = 5

# Loss / optimization flags
CLASS_WEIGHTS = None   # e.g. [1.0, 2.0, 5.0, 2.0, 3.0] or None
USE_FOCAL = False
FOCAL_GAMMA = 2.0
MIXED_PRECISION = False

# Checkpoint / scheduler
SAVE_EVERY = 1    # save checkpoint every N epochs
LR_SCHEDULER = True
PATIENCE = 4

# Misc
SEED = 42
PRINT_EVERY = 10

# End CONFIG

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
if CACHE_MODE == "disk":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Utilities & Losses

class MaskedCrossEntropy(nn.Module):
    """
    Cross-entropy applied per timestep with a boolean mask (True = include).
    Expects predictions shape (B, T, C), targets shape (B, T) with label ints.
    Mask is (B, T) boolean.
    """
    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        B, T, C = logits.shape
        logits_flat = logits.reshape(B * T, C)
        targets_flat = targets.reshape(B * T)
        losses = self.ce(logits_flat, targets_flat)  # (B*T,)
        losses = losses.reshape(B, T)
        mask_f = mask.float()
        denom = mask_f.sum()
        if denom.item() == 0:
            return logits.new_tensor(0.0), 0.0
        loss = (losses * mask_f).sum() / denom
        return loss, denom.item()


def focal_loss_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                      gamma: float = 2.0, weight=None):
    """
    Simple focal loss wrapper for multi-class logits with mask.
    logits: (B,T,C), targets (B,T), mask (B,T)
    """
    ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    targets_flat = targets.reshape(B * T)
    losses = ce(logits_flat, targets_flat)  # (B*T,)
    probs = torch.softmax(logits_flat, dim=-1)
    pt = probs[torch.arange(probs.shape[0]), targets_flat]
    focal = ((1 - pt) ** gamma) * losses
    focal = focal.reshape(B, T)
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    loss = (focal * mask_f).sum() / denom
    return loss


def compute_early_detection_metrics(batch_probs: np.ndarray, batch_labels: np.ndarray,
                                    batch_label_mask: np.ndarray, seq_meta: List[Dict],
                                    pre_symp_label_idxs=(1,), symp_label_idxs=(2,),
                                    threshold: float = 0.5) -> Dict[str, Any]:
    """
    Lightweight early-detection metrics aggregated per-batch.
    Uses sequence-level ground-truth symptom indices in meta or infers from labels.
    Returns dict with avg_lead_time, detection_rate, etc.
    """
    B, T, C = batch_probs.shape
    results = {"n_sequences": B, "n_detected_total": 0, "lead_times": [], "detection_offsets": [], "per_seq": []}

    for i in range(B):
        probs = batch_probs[i]
        labels = batch_labels[i]
        mask = batch_label_mask[i]
        meta = seq_meta[i]
        # infer first symptomatic index
        fsd = meta.get("first_symptom_day", None)
        if fsd is None:
            symp_idxs = np.where((labels == 2) | (labels == 4))[0]
            if symp_idxs.size == 0:
                results["per_seq"].append({"detected": False, "reason": "no_gt_symptom"})
                continue
            fsd_idx = int(symp_idxs[0])
        else:
            days = meta.get("days", None)
            if days and fsd in days:
                fsd_idx = days.index(fsd)
            else:
                fsd_idx = int(fsd)

        idxs = list(pre_symp_label_idxs) + list(symp_label_idxs)
        agg_series = probs[:, idxs].sum(axis=1)
        detected_idxs = np.where(agg_series >= threshold)[0]
        if detected_idxs.size == 0:
            results["per_seq"].append({"detected": False, "fsd_idx": fsd_idx})
            continue
        det_idx = int(detected_idxs[0])
        lead = fsd_idx - det_idx
        offset = det_idx - fsd_idx
        results["n_detected_total"] += 1
        results["lead_times"].append(lead)
        results["detection_offsets"].append(offset)
        results["per_seq"].append({"detected": True, "det_idx": det_idx, "fsd_idx": fsd_idx, "lead": lead})

    if results["lead_times"]:
        results["avg_lead_time"] = float(np.mean(results["lead_times"]))
        results["median_lead_time"] = float(np.median(results["lead_times"]))
    else:
        results["avg_lead_time"] = None
        results["median_lead_time"] = None
    results["detection_rate"] = results["n_detected_total"] / max(1, results["n_sequences"])
    return results


# Train / Validation loops

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device, loss_fn: MaskedCrossEntropy, epoch: int,
                use_focal: bool = False, focal_gamma: float = 2.0, scaler=None):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_idx, batch in enumerate(dataloader, start=1):
        seq = batch["seq"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(seq, label_mask=mask)
                logits = outputs["logits"]
                if use_focal:
                    loss = focal_loss_logits(logits, labels, mask, gamma=focal_gamma)
                else:
                    loss, denom = loss_fn(logits, labels, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(seq, label_mask=mask)
            logits = outputs["logits"]
            if use_focal:
                loss = focal_loss_logits(logits, labels, mask, gamma=focal_gamma)
            else:
                loss, denom = loss_fn(logits, labels, mask)
            loss.backward()
            optimizer.step()

        batch_n = seq.shape[0]
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n
        if (batch_idx % PRINT_EVERY) == 0:
            print(f"[Train] Epoch {epoch}  batch {batch_idx}/{len(dataloader)}  loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, total_examples)
    print(f"[Train] Epoch {epoch}: avg_loss={avg_loss:.4f}")
    return avg_loss


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device,
                   epoch: int, threshold: float = 0.5):
    model.eval()
    all_logits = []
    all_labels = []
    all_masks = []
    metas = []
    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)
            outputs = model(seq, label_mask=mask)
            logits = outputs["logits"].cpu().numpy()
            probs = outputs["probs"].cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            metas.extend(batch["meta"])

    if not all_logits:
        return {}

    all_logits = np.concatenate(all_logits, axis=0)    # (N, T, C)
    all_probs = np.concatenate([np.exp(l) / np.exp(l).sum(axis=-1, keepdims=True) for l in all_logits], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)    # (N, T)
    all_masks = np.concatenate(all_masks, axis=0)      # (N, T)

    # Flatten per-timestep for classification metrics using masked entries
    flat_preds = []
    flat_targets = []
    for i in range(all_labels.shape[0]):
        for t in range(all_labels.shape[1]):
            if all_masks[i, t]:
                probs_t = all_probs[i, t]
                pred_cls = int(np.argmax(probs_t))
                flat_preds.append(pred_cls)
                flat_targets.append(int(all_labels[i, t]))

    if len(flat_targets) == 0:
        print("[Val] No labeled timesteps found in validation set.")
        return {}

    labels_unique = sorted(list(set(flat_targets)))
    p, r, f, _ = precision_recall_fscore_support(flat_targets, flat_preds, labels=labels_unique, zero_division=0)
    stats = {"per_class": {}}
    for lab, pp, rr, ff in zip(labels_unique, p, r, f):
        stats["per_class"][str(lab)] = {"precision": float(pp), "recall": float(rr), "f1": float(ff)}
    stats["macro_f1"] = float(np.mean(f))
    print(f"[Val] Epoch {epoch} macro_f1={stats['macro_f1']:.4f}")

    # Early-detection metrics (sequence-level)
    early = compute_early_detection_metrics(all_probs, all_labels, all_masks, metas, threshold=threshold)
    stats["early_detection"] = early
    return stats


# Main orchestration

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load datasets (assumes train/val CSVs exist). If you have only one CSV, split by plant_id manually first.
    if not TRAIN_ANNOTATIONS.exists():
        raise SystemExit(f"[ERROR] Train annotations not found: {TRAIN_ANNOTATIONS}")
    if not VAL_ANNOTATIONS.exists():
        print(f"[WARN] Val annotations not found: {VAL_ANNOTATIONS} -- using train CSV for validation (NOT recommended).")
    train_ds = PatchSequenceDataset(annotations_csv=str(TRAIN_ANNOTATIONS),
                                    dataset_root=str(DATASET_ROOT),
                                    patch_size=PATCH_SIZE,
                                    band_min=BAND_MIN,
                                    band_max=BAND_MAX,
                                    normalize_stats=None,
                                    cache_mode=CACHE_MODE,
                                    cache_dir=str(CACHE_DIR) if CACHE_MODE == "disk" else None,
                                    min_mask_coverage=MIN_MASK_COVERAGE)
    val_ds = PatchSequenceDataset(annotations_csv=str(VAL_ANNOTATIONS if VAL_ANNOTATIONS.exists() else TRAIN_ANNOTATIONS),
                                  dataset_root=str(DATASET_ROOT),
                                  patch_size=PATCH_SIZE,
                                  band_min=BAND_MIN,
                                  band_max=BAND_MAX,
                                  normalize_stats=None,
                                  cache_mode="none",
                                  cache_dir=None,
                                  min_mask_coverage=MIN_MASK_COVERAGE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=patch_collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=patch_collate, num_workers=max(1, NUM_WORKERS // 2))

    # Model
    model = Hybrid3DConvLSTM(encoder_params=ENCODER_PARAMS,
                             lstm_hidden=LSTM_HIDDEN,
                             lstm_layers=LSTM_LAYERS,
                             bidirectional=BIDIRECTIONAL,
                             classifier_hidden=CLASSIFIER_HIDDEN,
                             num_classes=NUM_CLASSES)
    model.to(device)

    # Loss & optimizer
    class_weights_tensor = None
    if CLASS_WEIGHTS is not None:
        class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    loss_fn = MaskedCrossEntropy(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=PATIENCE, factor=0.5) if LR_SCHEDULER else None

    scaler = torch.cuda.amp.GradScaler() if (MIXED_PRECISION and device.type == "cuda") else None

    best_val = -math.inf
    best_ckpt = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn, epoch,
                                 use_focal=USE_FOCAL, focal_gamma=FOCAL_GAMMA, scaler=scaler)
        val_stats = validate_epoch(model, val_loader, device, epoch=epoch, threshold=0.5)
        metric = val_stats.get("macro_f1", 0.0) if val_stats else 0.0
        if scheduler is not None:
            scheduler.step(metric)

        # Save checkpoint
        if (epoch % SAVE_EVERY) == 0:
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_stats": val_stats}
            ckpt_path = OUT_DIR / f"ckpt_epoch_{epoch:03d}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[SAVE] Checkpoint -> {ckpt_path}")

        # Track best
        if metric > best_val:
            best_val = metric
            best_ckpt = OUT_DIR / "best_model.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_stats": val_stats}, best_ckpt)
            print(f"[BEST] New best model at epoch {epoch}: macro_f1={metric:.4f}")

        t1 = time.time()
        print(f"[EPOCH] {epoch} finished in {t1 - t0:.1f}s\n")

    print(f"[DONE] Training complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
