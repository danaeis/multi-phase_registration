"""
deformable_registration_v2.py  —  B-spline Deformable Registration
===================================================================
Four selectable metrics (one per ablation row):

  --metric grad_ncc    Central-diff |∇HU| + ANTS NCC         [HEADLINE]
                       Phase-invariant sharp-edge alignment.
  --metric sobel_ncc   Sobel |∇HU| (σ=0.75mm) + ANTS NCC    [ABLATION]
                       Pre-smoothed edges; less noise-sensitive than grad_ncc.
  --metric mmi         Mattes MI on raw HU                    [ABLATION]
                       Traditional; phase-variant HU may hurt.
  --metric seg_dist    Organ signed-distance maps + MSE        [ABLATION]
                       Pure geometry; contrast-invariant but seg-quality-limited.
  --metric mind_ncc    MIND 6-offset descriptor + ANTS NCC    [ABLATION]
                       Self-similarity descriptor (Heinrich 2012); modality-
                       independent, robust to phase-variant HU. NCC on collapsed
                       scalar MIND feature map ∈ (0,1].
  --metric sobel_binary  Per-organ Otsu binary edges + MSE    [ABLATION]
                       Sobel magnitude binarised with per-organ Otsu threshold;
                       MSE drives exact boundary overlap. Threshold adapts per
                       organ label and per study from local edge distribution.
  --metric mind_sobel  MIND × (1 + sobel_binary) + ANTS NCC  [ABLATION]
                       Boundary-boosted MIND: MIND scalar values are doubled at
                       organ edge voxels (where binary Sobel = 1), unchanged
                       elsewhere. Combines MIND's phase-invariant structural
                       signal with hard boundary emphasis in one NCC objective.

Bugs fixed vs the previous version:
  1. LBFGSB function-eval budget: maximumNumberOfFunctionEvaluations was
     num_iterations*2=300. LBFGSB needs ~5-10 f-evals per gradient step for line
     search → budget exhausted after ~30-60 effective iterations per pyramid level,
     optimizer never converged. Fixed to num_iterations*20.
  2. NaN identity trick: `neg == neg` → `not np.isnan(neg)`.
  3. LOW_WEIGHT_THRESHOLD=0.6 was below all ORGAN_WEIGHTS (min=0.8) → low-dilation
     bucket was always empty. Raised to 1.2 so aorta/IVC/ribs/iliopsoas get 4mm
     dilation instead of 8mm.
  4. ORGAN_WEIGHTS missing labels 60,61,62 (ribs_L) and 72,73,74 (ribs_R) that are
     present in register_fixed.ALL_STABLE_LABELS. Now synced.
  5. run_deformable() returned a bare 4-tuple; optimizer_iters + stop_reason were
     printed but not returned. Now returns a meta-dict with all optimizer stats.
  6. build_organ_mask() returned only the mask. Now returns (mask, vox, pct) so
     callers can log mask-coverage quality.

New features:
  - sobel_ncc: SmoothingRecursiveGaussian(σ=0.75mm) + GradientMagnitude + ANTS NCC.
    Key difference from grad_ncc: Sobel adds a light orthogonal-smoothing pass
    (mimicking the separable Sobel kernel) before the central-difference step, making
    edge maps less sensitive to single-voxel noise.
  - DeformableLogger: structured run logger.
      · Console: human-readable, same channel as print() for seamless reading.
      · Per-study JSON: {study_out}/deformable_log_{metric}.json — contains optimizer
        stats, timing, mask coverage, mesh size per phase.
      · Batch CSV: {output_dir}/batch_summary_{metric}.csv — one row per phase,
        all studies; ready for pandas analysis / ablation table.
  - Per-phase wall-time with cumulative ETA for batch runs.
  - Mask-coverage warning: <1% of volume is a red flag for empty/bad seg_reg.
  - Folding severity tiers: 0–1% OK | 1–5% WARN | >5% SEVERE (with grid-spacing hint).
  - --baseline flag: select fixed_baseline_rigid_registered input vs aligned.
  - --iters / --sampling / --no-field CLI knobs.

Folding control note:
  SimpleITK ImageRegistrationMethod has no bending-energy penalty term. Smoothness
  comes from grid spacing + Gaussian pyramid only. %|J|<=0 is REPORTED (not gated).
  If folding is severe, either increase --grid spacing or switch to Elastix with
  BendingEnergyPenalty.

Output postfix note (config.py compatibility):
  This script writes *_deformable.nii.gz. config.py A6_bspline expects *_bspline.nii.gz
  in experiments/results/A6_bspline/. When evaluating with evaluate_registration.py
  pass --vol_postfix _deformable.nii.gz --seg_postfix _deformable_seg_reg.nii.gz
  directly instead of going through config.py CONDITIONS.

Usage:
    # Single study (default metric, example ID):
    python deformable_registration_v2.py

    # Single study, explicit ID and metric:
    python deformable_registration_v2.py <study_id> --metric grad_ncc

    # Batch — headline metric:
    python deformable_registration_v2.py --metric grad_ncc --all [--skip]

    # Batch — Sobel ablation:
    python deformable_registration_v2.py --metric sobel_ncc --all --skip

    # Batch — MMI ablation on baseline input:
    python deformable_registration_v2.py --metric mmi --all --baseline --skip

    # Coarser grid (less folding, fewer DOF):
    python deformable_registration_v2.py --metric grad_ncc --all --grid 40

    # No field saves (faster, smaller disk):
    python deformable_registration_v2.py --metric sobel_ncc --all --no-field
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from configs import MAIN_PATH

# ── Paths ──────────────────────────────────────────────────────────────────────
# Aligned input: register_fixed.py --aligned output
INPUT_DIR_ALIGNED  = Path(MAIN_PATH + "fixed_aligned_rigid_registered")
# Baseline input: register_fixed.py --baseline output
INPUT_DIR_BASELINE = Path(MAIN_PATH + "fixed_baseline_rigid_registered")
OUTPUT_DIR_BASE    = Path(MAIN_PATH + "deformable_registered")  # _<metric>[_baseline] appended
LABELS_CSV         = MAIN_PATH + "labels.csv"

# Postfixes — match rigid Pass 2 output (both aligned and baseline pipelines)
VOL_POSTFIX      = "_rigid1.nii.gz"
SEG_REG_POSTFIX  = "_rigid1_seg_reg.nii.gz"
SEG_FULL_POSTFIX = "_rigid1_seg_full.nii.gz"

# ── B-spline hyperparameters ───────────────────────────────────────────────────
GRID_SPACING_MM    = 25.0    # control-point grid spacing (mm). 40=coarse / 25=default / 15=fine
GRAD_CLAMP_MAX     = 200.0   # |∇HU| clamp: keeps soft-tissue edges on-scale vs bone/air spikes
SOBEL_SIGMA_MM     = 0.75    # pre-smooth σ for sobel_ncc; 0.75mm ≈ 0.5 vox at 1.5mm iso
NCC_RADIUS         = 4       # ANTS NCC neighborhood window half-radius (voxels)

# Folding thresholds for console reporting (not used as gates)
FOLDING_WARN_PCT   = 1.0
FOLDING_SEVERE_PCT = 5.0

METRICS          = ("grad_ncc", "sobel_ncc", "mmi", "seg_dist", "mind_ncc", "sobel_binary", "mind_sobel")
EXAMPLE_STUDY_ID = "1.2.840.113619.2.359.3.2831208971.108.1589585466.773"

# ── Organ weights ──────────────────────────────────────────────────────────────
# Synced to register_fixed.ALL_STABLE_LABELS (must be kept in sync manually;
# direct import avoided to prevent circular dependency).
#
# Excluded intentionally:
#   13 (iliac_artery), 14 (iliac_vein) — phase-variant HU (bright in arterial,
#   dark in NC) → noisy metric gradient. Same reason as rigid REFINEMENT_LABELS.
#
# Fixed vs previous version:
#   Added 60,61,62 (ribs_L) and 72,73,74 (ribs_R) — were in ALL_STABLE_LABELS
#   but missing here despite the docstring claiming sync.
ORGAN_WEIGHTS: Dict[int, float] = {
    # ── Bone / geometry anchors ───────────────────────────────────────────────
    40: 3.0,                                               # spinal_cord
    21: 2.0, 22: 2.0, 23: 2.0, 24: 2.0, 25: 2.0,         # L1–L5
    26: 2.0, 27: 2.0, 28: 2.0,                             # T10–T12
    38: 1.5, 39: 1.5,                                      # sacrum, rib_R2
    45: 1.0,                                               # hip
    60: 0.8, 61: 0.8, 62: 0.8,                            # ribs_left  [ADDED]
    72: 0.8, 73: 0.8, 74: 0.8,                            # ribs_right [ADDED]
    # ── Stable soft organs ────────────────────────────────────────────────────
    1:  3.0,   # liver       — large, stable, high Dice signal
    2:  2.0,   # spleen
    3:  2.0,   # kidney_left
    4:  2.0,   # kidney_right
    9:  1.0,   # aorta
    10: 1.0,   # IVC
    16: 0.8,   # iliopsoas_left
    17: 0.8,   # iliopsoas_right
}

# Labels with weight < threshold go to the low-dilation bucket (4mm vs 8mm).
# FIXED: was 0.6, below the minimum weight (0.8) → low bucket always empty.
# Now 1.2 → labels 45/9/10/60-62/72-74/16/17 (w<1.2) get the tighter 4mm dilation.
LOW_WEIGHT_THRESHOLD = 1.2


# ===========================================================================
# LOGGER
# ===========================================================================

class DeformableLogger:
    """
    Structured run logger for one metric/mode batch.

    · Console   : human-readable via Python logging (same stdout as print()).
    · Study JSON: {study_out}/deformable_log_{metric}.json — one file per study,
                  contains per-phase optimizer stats + timing.
    · Batch CSV : {output_dir}/batch_summary_{metric}.csv — one row per
                  (study × phase), written at end of batch.
    """

    def __init__(self, output_dir: str, metric: str, grid_mm: float):
        self.output_dir = output_dir
        self.metric     = metric
        self.grid_mm    = grid_mm
        self._batch_rows: List[dict] = []

        # Attach to a named logger so batch and single-study share the same instance.
        name = f"deform.{metric}"
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(h)
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False  # avoid duplicate output if root logger is active

    # ── Console helpers ────────────────────────────────────────────────────────
    def info(self, msg: str)    -> None: self._log.info(msg)
    def warning(self, msg: str) -> None: self._log.warning(f"  ⚠  {msg}")
    def error(self, msg: str)   -> None: self._log.error(f"  ❌ {msg}")

    # ── Phase result accumulation ──────────────────────────────────────────────
    def accumulate(self, study_id: str, series_id: str, phase: str,
                   t_s: float, meta: dict) -> None:
        """Add one phase row to the in-memory batch accumulator."""
        self._batch_rows.append({
            "study_id":        study_id,
            "series_id":       series_id,
            "phase":           phase,
            "metric":          self.metric,
            "grid_mm":         self.grid_mm,
            "t_deform_s":      round(t_s, 1),
            "final_metric":    meta.get("final_metric"),
            "neg_jac_pct":     meta.get("neg_jac_pct"),
            "optimizer_iters": meta.get("optimizer_iters"),
            "stop_reason":     meta.get("stop_reason"),
            "mask_vox":        meta.get("mask_vox"),
            "mask_pct":        meta.get("mask_pct"),
            "mesh":            str(meta.get("mesh")),
            "status":          meta.get("status", "ok"),
        })

    # ── Per-study JSON ─────────────────────────────────────────────────────────
    def save_study_json(self, study_id: str, study_out: str,
                        phase_results: List[dict], t_total_s: float) -> None:
        """Write {study_out}/deformable_log_{metric}.json."""
        doc = {
            "study_id":  study_id,
            "metric":    self.metric,
            "grid_mm":   self.grid_mm,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "t_total_s": round(t_total_s, 1),
            "phases":    phase_results,
        }
        path = os.path.join(study_out, f"deformable_log_{self.metric}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        print(f"  📄 Study log → {os.path.basename(path)}")

    # ── Batch summary CSV ──────────────────────────────────────────────────────
    def save_batch_csv(self) -> Optional[str]:
        """Write {output_dir}/batch_summary_{metric}.csv. Returns path or None."""
        if not self._batch_rows:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"batch_summary_{self.metric}.csv")
        pd.DataFrame(self._batch_rows).to_csv(path, index=False)
        print(f"\n  ✓ Batch summary → {path}")
        return path


# ===========================================================================
# FILE DISCOVERY
# ===========================================================================

def scan_input_directory(
        input_dir: str,
        vol_postfix: str, seg_reg_postfix: str, seg_full_postfix: str,
) -> Dict[str, Dict[str, dict]]:
    """Return {study_id: {series_id: {image, seg_reg, seg_full}}}."""
    catalog: Dict[str, Dict[str, dict]] = {}
    if not os.path.exists(input_dir):
        print(f"  ❌ Input dir not found: {input_dir}")
        return catalog

    study_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d))]
    print(f"\n  Scanning {input_dir} ...")
    print(f"  postfixes: vol={vol_postfix}  seg={seg_reg_postfix}")
    print(f"  Study dirs: {len(study_dirs)}")

    for study_id in study_dirs:
        study_path = os.path.join(input_dir, study_id)
        catalog[study_id] = {}
        for vol_path in glob.glob(os.path.join(study_path, f"*{vol_postfix}")):
            bn = os.path.basename(vol_path)
            if "seg" in bn or not bn.startswith(study_id + "_"):
                continue
            series_id = bn[len(study_id) + 1:].replace(vol_postfix, "")
            seg_reg   = vol_path.replace(vol_postfix, seg_reg_postfix)
            seg_full  = vol_path.replace(vol_postfix, seg_full_postfix)
            if not os.path.exists(seg_reg):
                print(f"  ⚠  Missing seg_reg: {series_id[:40]}... — skipped")
                continue
            catalog[study_id][series_id] = {
                "image":    vol_path,
                "seg_reg":  seg_reg,
                "seg_full": seg_full if os.path.exists(seg_full) else None,
            }

    total = sum(len(v) for v in catalog.values())
    print(f"  Total usable series: {total}")
    return catalog


# ===========================================================================
# MASK UTILITIES
# ===========================================================================

def build_organ_mask(
        seg_reg_sitk: sitk.Image,
        organ_weights: Dict[int, float] = ORGAN_WEIGHTS,
        low_weight_threshold: float = LOW_WEIGHT_THRESHOLD,
        primary_dilation_mm: float = 8.0,
        low_weight_dilation_mm: float = 4.0,
) -> Tuple[sitk.Image, int, float]:
    """
    Build a binary registration mask from the seg_reg label map.

    Returns:
        (mask, mask_vox_count, mask_pct_of_volume)

    High-weight organs (w >= low_weight_threshold) are dilated by primary_dilation_mm;
    low-weight organs (w < low_weight_threshold) by low_weight_dilation_mm.

    With LOW_WEIGHT_THRESHOLD=1.2:
      High (8mm): liver(3.0), spleen(2.0), kidneys(2.0), spine(2–3.0), sacrum(1.5)
      Low  (4mm): hip(1.0), aorta(1.0), IVC(1.0), ribs(0.8), iliopsoas(0.8)
    """
    seg_np  = np.round(sitk.GetArrayFromImage(seg_reg_sitk)).astype(np.int32)
    spacing = seg_reg_sitk.GetSpacing()  # (x, y, z)
    high    = np.zeros_like(seg_np, np.uint8)
    low     = np.zeros_like(seg_np, np.uint8)

    for label, w in organ_weights.items():
        region = seg_np == label
        if region.sum() < 50:
            continue
        if w >= low_weight_threshold:
            high[region] = 1
        else:
            low[region] = 1

    def _dilate(arr: np.ndarray, mm: float) -> np.ndarray:
        if arr.sum() == 0:
            return arr
        tmp = sitk.GetImageFromArray(arr)
        tmp.SetSpacing(spacing)
        radii = [max(1, int(round(mm / s))) for s in spacing]
        return sitk.GetArrayFromImage(sitk.BinaryDilate(tmp, radii))

    combined = np.clip(
        _dilate(high, primary_dilation_mm) + _dilate(low, low_weight_dilation_mm),
        0, 1,
    ).astype(np.uint8)

    out = sitk.GetImageFromArray(combined)
    out.CopyInformation(seg_reg_sitk)
    out = sitk.BinaryFillhole(out)

    vox = int(sitk.GetArrayFromImage(out).sum())
    pct = 100.0 * vox / seg_np.size
    print(f"  Organ mask: {vox:,} vox  ({pct:.1f}% of volume)")
    return out, vox, round(pct, 2)


def resample_mask_to_moving(
        fixed_mask: sitk.Image,
        moving_img: sitk.Image,
        extra_dilation_mm: float = 12.0,
) -> sitk.Image:
    """
    Resample fixed mask into moving-image space with extra dilation.
    The dilation absorbs residual rigid-registration error so the B-spline
    optimizer sees moving organ neighborhoods even when there's a few-mm offset.
    Uses identity transform (images are already roughly aligned post-rigid).
    """
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(moving_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    r.SetTransform(sitk.Transform())  # identity — rigid already done
    mov = r.Execute(fixed_mask)
    if extra_dilation_mm > 0:
        sp    = moving_img.GetSpacing()
        radii = [max(1, int(round(extra_dilation_mm / s))) for s in sp]
        mov   = sitk.BinaryDilate(mov, radii)
    return sitk.Cast(mov, sitk.sitkUInt8)


# ===========================================================================
# METRIC IMAGE PREPROCESSING
# ===========================================================================

def _grad_magnitude(img_sitk: sitk.Image,
                    clamp_max: float = GRAD_CLAMP_MAX) -> sitk.Image:
    """
    Central-difference gradient magnitude (no pre-smoothing).
    Sharpest phase-invariant edges — sensitive to single-voxel noise but
    maximally responsive to true tissue boundaries.  Used for grad_ncc.
    """
    g = sitk.GradientMagnitude(sitk.Cast(img_sitk, sitk.sitkFloat32))
    return sitk.Clamp(g, lowerBound=0.0, upperBound=clamp_max)


def _sobel_magnitude(img_sitk: sitk.Image,
                     sigma_mm: float = SOBEL_SIGMA_MM,
                     clamp_max: float = GRAD_CLAMP_MAX) -> sitk.Image:
    """
    Sobel-approximated edge magnitude: σ=SOBEL_SIGMA_MM Gaussian pre-smooth
    followed by central-difference GradientMagnitude.

    The pre-smooth step mimics the orthogonal-smoothing component of a separable
    3D Sobel kernel. At 1.5mm isotropic, σ=0.75mm ≈ 0.5 voxels — light enough
    to preserve anatomical edges while suppressing single-voxel CT noise.

    vs grad_ncc: sobel_ncc produces smoother, less noise-driven edge maps.
    Practical impact: sobel_ncc typically gives more stable LBFGSB gradients
    near thin structures (ribs, vessel walls).
    """
    img_f = sitk.Cast(img_sitk, sitk.sitkFloat32)
    # Light isotropic Gaussian — mimics Sobel's orthogonal smoothing
    img_s = sitk.SmoothingRecursiveGaussian(img_f, sigma=sigma_mm)
    # Central-difference gradient on the smoothed image
    g = sitk.GradientMagnitude(img_s)
    return sitk.Clamp(g, lowerBound=0.0, upperBound=clamp_max)


# ── MIND cardinal offsets: (axis, shift) for ±Z, ±Y, ±X ────────────────────
_MIND_OFFSETS: List[Tuple[int, int]] = [
    (0, +1), (0, -1),   # ±Z
    (1, +1), (1, -1),   # ±Y
    (2, +1), (2, -1),   # ±X
]


def _otsu_1d(values: np.ndarray) -> float:
    """
    Compute Otsu threshold for a 1-D float array using 256 histogram bins.
    Maximises between-class variance in one pass; returns value in original range.
    """
    n = len(values)
    if n < 100:
        return float(np.median(values))
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = float(n) - w0
    mu0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-8)
    mu1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-8)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return float(centers[len(centers) // 2])
    return float(centers[np.argmax(np.where(valid, sigma_b, -1.0))])


def _mind_feature(img_sitk: sitk.Image) -> sitk.Image:
    """
    6-offset MIND descriptor collapsed to a scalar feature map for NCC.

    Algorithm (Heinrich et al. 2012 — MIND: Modality Independent Neighbourhood
    Descriptor):
      1. For each offset r in {±Z, ±Y, ±X}:
            D(x, r) = (I(x) − I(x+r))²
      2. Local variance:  V(x) = mean_r D(x, r) + ε
      3. MIND(x, r)    = exp(−D(x, r) / 2V(x))
      4. Collapse       : mind_scalar(x) = mean_r MIND(x, r)   → scalar ∈ (0, 1]

    NCC is then applied to mind_scalar: structurally similar neighbourhoods
    produce consistent self-similarity patterns regardless of absolute HU, making
    the metric phase-invariant.

    Memory note: two sequential passes over the volume avoid holding 6 diff arrays
    simultaneously. Peak usage ≈ 3× volume (img_np, V, mind_sum + one temp diff).
    """
    img_f  = sitk.Cast(img_sitk, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(img_f).astype(np.float32)   # (Z, Y, X)

    # Pass 1 — variance estimate V(x) = mean of 6 squared differences
    diff_sq_sum = np.zeros_like(img_np)
    for ax, sh in _MIND_OFFSETS:
        diff_sq_sum += (img_np - np.roll(img_np, sh, axis=ax)) ** 2
    V = diff_sq_sum / 6.0 + 1e-6
    del diff_sq_sum

    # Pass 2 — MIND values, summed and collapsed
    mind_sum = np.zeros_like(img_np)
    for ax, sh in _MIND_OFFSETS:
        mind_sum += np.exp(-(img_np - np.roll(img_np, sh, axis=ax)) ** 2 / (2.0 * V))
    mind_scalar = (mind_sum / 6.0).astype(np.float32)
    del mind_sum, V

    out = sitk.GetImageFromArray(mind_scalar)
    out.CopyInformation(img_f)
    return out


def _sobel_binary(
        img_sitk:     sitk.Image,
        seg_sitk:     sitk.Image,
        organ_weights: Dict[int, float] = ORGAN_WEIGHTS,
        sigma_mm:     float = SOBEL_SIGMA_MM,
) -> sitk.Image:
    """
    Per-organ Otsu-thresholded binary edge map for sobel_binary metric.

    For each organ label with ≥ 50 voxels:
      1. Extract Sobel magnitude values inside the organ mask.
      2. Compute Otsu threshold from those values (adaptive per organ per study).
      3. Mark voxels where Sobel > threshold as edge (1.0), others 0.0.
    Final binary map = union of all per-organ edge regions.

    Using per-organ Otsu rather than a global threshold ensures that low-contrast
    structures (iliopsoas, IVC) get an appropriate threshold without being drowned
    out by high-gradient bone edges.

    MSE on these binary maps drives exact organ-boundary coincidence; the loss is
    maximally interpretable — every non-zero residual is a missed or extra edge voxel.
    """
    sobel_sitk = _sobel_magnitude(img_sitk, sigma_mm=sigma_mm)
    sobel_np   = sitk.GetArrayFromImage(sobel_sitk).astype(np.float32)   # (Z, Y, X)
    seg_np     = np.round(sitk.GetArrayFromImage(seg_sitk)).astype(np.int32)

    binary = np.zeros_like(sobel_np, dtype=np.float32)
    labels_found = 0

    for label, _ in organ_weights.items():
        mask = seg_np == label
        if int(mask.sum()) < 50:
            continue
        labels_found += 1
        thr = _otsu_1d(sobel_np[mask])
        binary[mask & (sobel_np > thr)] = 1.0

    if labels_found == 0:
        # Fallback: global Otsu on full volume (only if seg is empty/mismatched)
        print("  ⚠  sobel_binary: no organ labels found — falling back to global Otsu")
        thr    = _otsu_1d(sobel_np.ravel())
        binary = (sobel_np > thr).astype(np.float32)

    edge_pct = 100.0 * binary.mean()
    print(f"  sobel_binary: {labels_found} organs thresholded  "
          f"edge_vox={int(binary.sum()):,}  ({edge_pct:.1f}% of vol)")

    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(img_sitk)
    return out


def _mind_sobel_feature(
        img_sitk: sitk.Image,
        seg_sitk: sitk.Image,
) -> sitk.Image:
    """
    Boundary-boosted MIND composite feature for mind_sobel metric.

    Formula:  combined(x) = mind_scalar(x) × (1 + sobel_binary(x))

    Effect:
      - At organ edge voxels (sobel_binary = 1): MIND value is doubled → the NCC
        window sees twice the contrast at boundaries, pulling the gradient signal
        towards correct boundary alignment.
      - Interior / background voxels (sobel_binary = 0): pure MIND signal,
        preserving phase-invariant structural matching across the whole organ.

    This is strictly stronger than either individual metric:
      - vs mind_ncc: adds hard boundary emphasis without discarding interior structure.
      - vs sobel_binary: retains continuous structural signal rather than working
        only on sparse binary edges.

    The output is ∈ (0, 2] (MIND ∈ (0,1] × factor ∈ {1,2}); NCC is applied since
    the composite is a bounded, smooth-ish field rather than a pure binary signal.
    """
    mind_np  = sitk.GetArrayFromImage(_mind_feature(img_sitk)).astype(np.float32)
    sobel_np = sitk.GetArrayFromImage(_sobel_binary(img_sitk, seg_sitk)).astype(np.float32)
    combined = (mind_np * (1.0 + sobel_np)).astype(np.float32)

    out = sitk.GetImageFromArray(combined)
    out.CopyInformation(img_sitk)
    return out


def _seg_distance_map(seg_sitk: sitk.Image,
                      labels: Dict[int, float] = ORGAN_WEIGHTS) -> sitk.Image:
    """
    Signed Maurer distance map (mm) to the union of organ label regions.

    Convention (insideIsPositive=False):
      voxel inside any organ → value ≤ 0 (distance inward)
      voxel outside all organs → value > 0 (Euclidean distance to nearest surface)

    MSE on signed-distance maps is a natural shape-registration loss: minimising
    (d_fixed - d_moving)² drives organ surfaces to coincide regardless of HU.
    """
    seg_np = np.round(sitk.GetArrayFromImage(seg_sitk)).astype(np.int32)
    union  = np.isin(seg_np, list(labels.keys())).astype(np.uint8)
    m = sitk.GetImageFromArray(union)
    m.CopyInformation(seg_sitk)
    dist = sitk.SignedMaurerDistanceMap(
        m, insideIsPositive=False, squaredDistance=False, useImageSpacing=True,
    )
    return sitk.Cast(dist, sitk.sitkFloat32)


def _metric_images(
        metric: str,
        fixed_img:  sitk.Image, moving_img:  sitk.Image,
        fixed_seg:  sitk.Image, moving_seg:  sitk.Image,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Return (fixed_metric_img, moving_metric_img) fed to the optimizer.
    These may differ from the raw CT (e.g. edge magnitude, distance maps).
    """
    if metric == "mmi":
        return (sitk.Cast(fixed_img, sitk.sitkFloat32),
                sitk.Cast(moving_img, sitk.sitkFloat32))
    if metric == "grad_ncc":
        return _grad_magnitude(fixed_img), _grad_magnitude(moving_img)
    if metric == "sobel_ncc":
        return _sobel_magnitude(fixed_img), _sobel_magnitude(moving_img)
    if metric == "seg_dist":
        return _seg_distance_map(fixed_seg), _seg_distance_map(moving_seg)
    if metric == "mind_ncc":
        return _mind_feature(fixed_img), _mind_feature(moving_img)
    if metric == "sobel_binary":
        return _sobel_binary(fixed_img, fixed_seg), _sobel_binary(moving_img, moving_seg)
    if metric == "mind_sobel":
        return (_mind_sobel_feature(fixed_img, fixed_seg),
                _mind_sobel_feature(moving_img, moving_seg))
    raise ValueError(f"Unknown metric '{metric}'. Choose from: {METRICS}")


# ===========================================================================
# B-SPLINE REGISTRATION
# ===========================================================================

def setup_bspline(
        metric: str,
        fixed_metric_img: sitk.Image,
        fixed_mask:  Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
        grid_spacing_mm: float,
        num_iterations:  int,
        sampling_pct:    float,
) -> Tuple[sitk.ImageRegistrationMethod, sitk.BSplineTransform, List[int]]:
    """
    Configure B-spline registration method.

    Returns:
        (registration_method, initial_transform, mesh_size)

    Optimizer: LBFGSB (bounded quasi-Newton). Stable for B-spline DOF, unlike the
    gradient-descent variant that caused rigid Pass 2 divergence.

    BUG FIX — maximumNumberOfFunctionEvaluations:
      Previous value: num_iterations * 2 = 300.
      LBFGSB does ~5-10 f-evals per iteration (Wolfe line search). With budget=300
      the optimizer could complete only ~30-60 effective gradient steps per pyramid
      level → never converged, stopping on function-eval limit every time.
      Fixed value: num_iterations * 20 (gives each level ample f-eval headroom).

    Pyramid: 3 levels [4×, 2×, 1×]. Each level runs up to num_iterations steps.
    Smoothing: [2mm, 1mm, 0mm] (SmoothingSigmasAreSpecifiedInPhysicalUnitsOn).
    """
    mesh = [
        max(1, int(round(sz * sp / grid_spacing_mm)))
        for sz, sp in zip(fixed_metric_img.GetSize(), fixed_metric_img.GetSpacing())
    ]
    print(f"  B-spline mesh: {mesh}  grid={grid_spacing_mm}mm  metric={metric}")

    init_tx = sitk.BSplineTransformInitializer(
        fixed_metric_img, transformDomainMeshSize=mesh, order=3,
    )

    reg = sitk.ImageRegistrationMethod()

    # ── Metric ────────────────────────────────────────────────────────────────
    if metric == "mmi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric in ("grad_ncc", "sobel_ncc", "mind_ncc", "mind_sobel"):
        # NCC on edge-magnitude or collapsed MIND scalar maps.
        # mind_ncc: MIND descriptor ∈ (0,1] — NCC is phase-invariant and robust
        # to the near-uniform background values in MIND feature maps.
        reg.SetMetricAsANTSNeighborhoodCorrelation(radius=NCC_RADIUS)
    elif metric in ("seg_dist", "sobel_binary"):
        # MSE: natural for distance fields and binary {0,1} edge maps.
        # sobel_binary: every residual pixel is a missing/extra edge → L2 exact.
        reg.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct, seed=42)

    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
        print("  Fixed mask  ✓")
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
        print("  Moving mask ✓")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=num_iterations * 20,  # FIX: was *2 → too low
        costFunctionConvergenceFactor=1e7,
    )

    # ── Multi-resolution pyramid ───────────────────────────────────────────────
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg, init_tx, mesh


def _neg_jac_pct(field: sitk.Image, body_mask: Optional[sitk.Image]) -> float:
    """
    %|J| ≤ 0 inside body_mask — local folding indicator.

    Reported but NOT gated (a small % of folding voxels at organ boundaries
    is common in B-spline; gating would silently revert to rigid).
    If > FOLDING_SEVERE_PCT consider increasing grid spacing.
    """
    jac = sitk.DisplacementFieldJacobianDeterminant(field)
    j   = sitk.GetArrayFromImage(jac)          # numpy (Z,Y,X)
    if body_mask is not None:
        bm = sitk.GetArrayFromImage(
            sitk.Resample(body_mask, jac, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0)
        ).astype(bool)
        if bm.shape == j.shape and bm.any():
            j = j[bm]
    j = j[np.isfinite(j)]
    return float(100.0 * (j <= 0).mean()) if j.size else float("nan")


def run_deformable(
        metric: str,
        fixed_img:  sitk.Image, moving_img:  sitk.Image,
        fixed_seg:  sitk.Image, moving_seg:  sitk.Image,
        fixed_mask:  Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
        grid_spacing_mm: float = GRID_SPACING_MM,
        num_iterations:  int   = 150,
        sampling_pct:    float = 0.15,
        final_interpolator: int = sitk.sitkLanczosWindowedSinc,
        save_field:  bool = False,
        field_path:  Optional[str] = None,
) -> Tuple[sitk.Image, sitk.Transform, dict]:
    """
    Execute B-spline registration for one (fixed, moving) pair.

    The transform is fitted on metric images (edge maps / distance maps / raw HU)
    but applied to the ORIGINAL CT for output.

    Returns:
        (registered_volume, transform, result_meta)

    result_meta keys:
        final_metric    : float  — optimizer objective value at convergence
        neg_jac_pct     : float  — %|J|≤0 in fixed_mask (NaN if field unavailable)
        optimizer_iters : int    — iterations completed
        stop_reason     : str    — optimizer stop-condition description
        mesh            : list   — B-spline mesh size [Nx, Ny, Nz]
    """
    f_met, m_met = _metric_images(metric, fixed_img, moving_img, fixed_seg, moving_seg)
    reg, init_tx, mesh = setup_bspline(
        metric, f_met, fixed_mask, moving_mask,
        grid_spacing_mm, num_iterations, sampling_pct,
    )
    reg.SetInitialTransform(init_tx, inPlace=True)

    final_tx    = reg.Execute(f_met, m_met)
    final_met   = float(reg.GetMetricValue())
    opt_iters   = int(reg.GetOptimizerIteration())
    stop_reason = reg.GetOptimizerStopConditionDescription()

    print(f"  Optimizer: metric={final_met:.6f}  iters={opt_iters}  stop='{stop_reason}'")

    # Build displacement field — needed for folding report + optional save
    field = sitk.TransformToDisplacementField(
        final_tx, sitk.sitkVectorFloat32,
        fixed_img.GetSize(), fixed_img.GetOrigin(),
        fixed_img.GetSpacing(), fixed_img.GetDirection(),
    )
    neg = _neg_jac_pct(field, fixed_mask)

    # FIX: was `neg == neg` (NaN identity trick) → use np.isnan
    if not np.isnan(neg):
        if neg > FOLDING_SEVERE_PCT:
            flag = f"  🔴 SEVERE — consider --grid {int(grid_spacing_mm * 1.5)}"
        elif neg > FOLDING_WARN_PCT:
            flag = "  ⚠ HIGH"
        else:
            flag = ""
        print(f"  %|J|≤0 (in mask): {neg:.3f}%{flag}")
    else:
        print("  %|J|≤0: n/a")

    # Apply transform to original CT (not metric image)
    registered = sitk.Resample(
        moving_img, fixed_img, final_tx,
        final_interpolator, -1024.0, moving_img.GetPixelID(),
    )

    if save_field and field_path:
        sitk.WriteImage(field, field_path)
        print(f"  ✓ DVF: {os.path.basename(field_path)} "
              f"({os.path.getsize(field_path)/1e6:.1f} MB)")

    return registered, final_tx, {
        "final_metric":    final_met,
        "neg_jac_pct":     neg,
        "optimizer_iters": opt_iters,
        "stop_reason":     stop_reason,
        "mesh":            mesh,
    }


# ===========================================================================
# SAVE / APPLY HELPERS
# ===========================================================================

def _save(img: sitk.Image, path: str, label: str) -> bool:
    try:
        sitk.WriteImage(img, path)
        print(f"  ✓ {label}: {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


def _apply_seg(seg: sitk.Image, reference: sitk.Image,
               transform: sitk.Transform) -> sitk.Image:
    """Warp a segmentation mask with nearest-neighbour interpolation."""
    return sitk.Resample(seg, reference, transform,
                         sitk.sitkNearestNeighbor, 0, seg.GetPixelID())


# ===========================================================================
# PER-STUDY
# ===========================================================================

def register_study_deformable(
        study_id:    str,
        catalog:     Dict,
        labels_df:   pd.DataFrame,
        output_dir:  str,
        metric:      str,
        logger:      DeformableLogger,
        grid_spacing_mm:    float = GRID_SPACING_MM,
        num_iterations:     int   = 150,
        sampling_pct:       float = 0.15,
        final_interpolator: int   = sitk.sitkLanczosWindowedSinc,
        save_field:         bool  = True,
        skip_existing:      bool  = True,
) -> dict:
    """
    Register all non-NC phases of one study to the NC (fixed) reference.

    Returns: {"status", "study_id", "successful": [...], "failed": [...], "skipped": [...]}
    """
    if study_id not in catalog or not catalog[study_id]:
        return {"status": "skipped", "reason": "not_in_catalog"}

    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    nc_rows    = study_rows[study_rows["Label"] == "Non-contrast"]
    if nc_rows.empty:
        return {"status": "skipped", "reason": "no_nc_label"}
    nc_sid = nc_rows.iloc[0]["SeriesInstanceUID"]
    if nc_sid not in catalog[study_id]:
        return {"status": "skipped", "reason": "nc_files_missing"}

    study_out = os.path.join(output_dir, study_id)
    os.makedirs(study_out, exist_ok=True)

    print(f"\n{'='*80}\nDeformable [{metric}]: {study_id[:60]}\n"
          f"  grid={grid_spacing_mm}mm  iters={num_iterations}  "
          f"sampling={sampling_pct:.0%}\n{'='*80}")

    t0_study = time.time()

    # ── Load NC (fixed reference) ──────────────────────────────────────────────
    nc         = catalog[study_id][nc_sid]
    nc_img     = sitk.ReadImage(nc["image"])
    nc_seg_reg = sitk.ReadImage(nc["seg_reg"])
    nc_seg_full = sitk.ReadImage(nc["seg_full"]) if nc["seg_full"] else None

    sz  = nc_img.GetSize()
    sp  = [round(s, 2) for s in nc_img.GetSpacing()]
    print(f"\n  Fixed (NC): size={sz}  spacing={sp}  "
          f"series={nc_sid[:40]}...")

    print("  Building fixed organ mask...")
    fixed_mask, mask_vox, mask_pct = build_organ_mask(nc_seg_reg)
    if mask_pct < 1.0:
        logger.warning(
            f"Fixed mask covers only {mask_pct:.1f}% of volume — "
            "seg_reg may be empty/mismatched. Registration may be unreliable."
        )

    # Copy NC files to output (they are the fixed reference, no transform needed)
    nc_prefix = os.path.join(study_out, f"{study_id}_{nc_sid}")
    _save(nc_img,     f"{nc_prefix}_deformable.nii.gz",         "NC vol (ref)")
    _save(nc_seg_reg, f"{nc_prefix}_deformable_seg_reg.nii.gz", "NC seg_reg (ref)")
    if nc_seg_full:
        _save(nc_seg_full, f"{nc_prefix}_deformable_seg_full.nii.gz", "NC seg_full (ref)")

    results: dict = {
        "status":     "complete",
        "study_id":   study_id,
        "successful": [],
        "failed":     [],
        "skipped":    [],
    }
    phase_log: List[dict] = []  # for per-study JSON

    for _, row in study_rows.iterrows():
        sid, phase = row["SeriesInstanceUID"], row["Label"]
        if sid == nc_sid:
            continue
        if sid not in catalog[study_id]:
            results["failed"].append({"series": sid, "phase": phase,
                                      "reason": "not_in_catalog"})
            continue

        prefix    = os.path.join(study_out, f"{study_id}_{sid}")
        out_vol   = f"{prefix}_deformable.nii.gz"
        out_field = f"{prefix}_deformable_field.nii.gz"

        if skip_existing and os.path.exists(out_vol):
            print(f"\n  ⏭  [{phase}] already exists — skip")
            results["skipped"].append(sid)
            continue

        files = catalog[study_id][sid]
        print(f"\n  {'─'*60}\n  [{phase}]  {sid[:50]}...")

        t0_phase = time.time()
        try:
            mov_img      = sitk.ReadImage(files["image"])
            mov_seg_reg  = sitk.ReadImage(files["seg_reg"])
            mov_seg_full = sitk.ReadImage(files["seg_full"]) if files["seg_full"] else None

            mv_sz = mov_img.GetSize()
            mv_sp = [round(s, 2) for s in mov_img.GetSpacing()]
            print(f"  Moving: size={mv_sz}  spacing={mv_sp}")

            # Moving mask: fixed mask projected into moving space + 12mm extra dilation
            # to absorb any residual rigid offset before B-spline takes over.
            print("  Building moving mask (fixed mask + 12mm dilation in moving space)...")
            moving_mask = resample_mask_to_moving(fixed_mask, mov_img, extra_dilation_mm=12.0)

            reg_vol, final_tx, meta = run_deformable(
                metric, nc_img, mov_img, nc_seg_reg, mov_seg_reg,
                fixed_mask, moving_mask,
                grid_spacing_mm=grid_spacing_mm,
                num_iterations=num_iterations,
                sampling_pct=sampling_pct,
                final_interpolator=final_interpolator,
                save_field=save_field,
                field_path=out_field if save_field else None,
            )

            t_s = time.time() - t0_phase
            print(f"  ⏱  Phase: {t_s:.1f}s")

            # Save warped volume and masks
            _save(reg_vol, out_vol, f"[{phase}] deformable vol")
            _save(_apply_seg(mov_seg_reg, nc_img, final_tx),
                  f"{prefix}_deformable_seg_reg.nii.gz",  f"[{phase}] seg_reg")
            if mov_seg_full:
                _save(_apply_seg(mov_seg_full, nc_img, final_tx),
                      f"{prefix}_deformable_seg_full.nii.gz", f"[{phase}] seg_full")

            # Annotate meta with mask coverage (same fixed mask for all phases)
            meta.update({"mask_vox": mask_vox, "mask_pct": mask_pct, "status": "ok"})

            logger.accumulate(study_id, sid, phase, t_s, meta)
            results["successful"].append({
                "series":       sid,
                "phase":        phase,
                "final_metric": meta["final_metric"],
                "neg_jac_pct":  meta["neg_jac_pct"],
            })
            phase_log.append({
                "series_id":     sid,
                "phase":         phase,
                "fixed_size":    list(nc_img.GetSize()),
                "moving_size":   list(mov_img.GetSize()),
                **meta,
                "t_s":           round(t_s, 1),
            })

        except Exception as exc:
            import traceback; traceback.print_exc()
            t_s = time.time() - t0_phase
            logger.error(f"[{phase}] after {t_s:.1f}s: {exc}")
            err_meta = {"status": "error", "error": str(exc)}
            logger.accumulate(study_id, sid, phase, t_s, err_meta)
            results["failed"].append({"series": sid, "phase": phase, "reason": str(exc)})
            phase_log.append({
                "series_id": sid, "phase": phase,
                "status": "error", "error": str(exc), "t_s": round(t_s, 1),
            })

    t_total = time.time() - t0_study
    logger.save_study_json(study_id, study_out, phase_log, t_total)

    print(f"\n  {'='*60}\n  Study done in {t_total:.1f}s: "
          f"✓{len(results['successful'])}  "
          f"⏭{len(results['skipped'])}  ✗{len(results['failed'])}")
    return results


# ===========================================================================
# BATCH
# ===========================================================================

def register_all_studies_deformable(
        input_dir:       str,
        output_dir:      str,
        labels_csv:      str,
        metric:          str,
        vol_postfix:     str   = VOL_POSTFIX,
        seg_reg_postfix: str   = SEG_REG_POSTFIX,
        seg_full_postfix:str   = SEG_FULL_POSTFIX,
        grid_spacing_mm: float = GRID_SPACING_MM,
        num_iterations:  int   = 150,
        sampling_pct:    float = 0.15,
        save_field:      bool  = True,
        skip_existing:   bool  = True,
        study_ids=None,
) -> None:
    labels_df = pd.read_csv(labels_csv)
    catalog   = scan_input_directory(
        input_dir, vol_postfix, seg_reg_postfix, seg_full_postfix,
    )
    if not catalog:
        print("❌ No studies found."); return

    valid  = sorted(set(catalog) & set(labels_df["StudyInstanceUID"].unique()))
    if study_ids is not None:
        valid = [s for s in valid if s in study_ids]
        print(f"  study_ids filter: {len(valid)} studies selected")
    logger = DeformableLogger(output_dir, metric, grid_spacing_mm)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}\nBATCH DEFORMABLE [{metric}]\n"
          f"  input  : {input_dir}\n"
          f"  output : {output_dir}\n"
          f"  grid   : {grid_spacing_mm}mm   iters={num_iterations}   "
          f"sampling={sampling_pct:.0%}\n"
          f"  studies: {len(valid)}\n{'='*80}")

    counts       = {"ok": 0, "skipped": 0, "failed": 0}
    study_times: List[float] = []
    neg_all:     List[float] = []

    for idx, sid in enumerate(valid, 1):
        # Rolling ETA from completed studies
        if study_times:
            eta_s   = (len(valid) - idx + 1) * float(np.mean(study_times))
            eta_str = str(timedelta(seconds=int(eta_s)))
        else:
            eta_str = "?"
        print(f"\n[{idx}/{len(valid)}]  ETA {eta_str}  {sid[:55]}...")

        t0 = time.time()
        try:
            res = register_study_deformable(
                sid, catalog, labels_df, output_dir, metric, logger,
                grid_spacing_mm=grid_spacing_mm,
                num_iterations=num_iterations,
                sampling_pct=sampling_pct,
                save_field=save_field,
                skip_existing=skip_existing,
            )
            study_times.append(time.time() - t0)

            if res["status"] == "complete" and res["successful"]:
                counts["ok"] += 1
                neg_all += [
                    s["neg_jac_pct"]
                    for s in res["successful"]
                    if not np.isnan(s.get("neg_jac_pct", float("nan")))  # FIX: was nan==nan
                ]
            elif res["status"] == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1

        except Exception as exc:
            import traceback; traceback.print_exc()
            study_times.append(time.time() - t0)
            print(f"  ❌ Study failed: {exc}")
            counts["failed"] += 1

    logger.save_batch_csv()

    print(f"\n{'='*80}\nDONE [{metric}]  "
          f"✓{counts['ok']}  ⏭{counts['skipped']}  ✗{counts['failed']}  / {len(valid)}")
    if neg_all:
        print(f"  Folding %|J|≤0 across all warps: "
              f"median={np.median(neg_all):.3f}%  "
              f"p95={np.percentile(neg_all, 95):.3f}%  "
              f"max={np.max(neg_all):.3f}%  n={len(neg_all)}")
    if study_times:
        avg_m = np.mean(study_times) / 60
        print(f"  Avg study time: {avg_m:.1f}min  "
              f"  Total: {sum(study_times)/60:.0f}min")
    print(f"{'='*80}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="B-spline deformable registration — multi-metric ablation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "study_id", nargs="?", default=EXAMPLE_STUDY_ID,
        help="Study ID for single-study mode (default: example study)",
    )
    ap.add_argument(
        "--metric", choices=METRICS, default="grad_ncc",
        help=(
            "Registration metric:\n"
            "  grad_ncc     : central-diff |∇HU| + ANTS NCC  [HEADLINE]\n"
            "  sobel_ncc    : Sobel |∇HU| (σ=0.75mm) + ANTS NCC  [ABLATION]\n"
            "  mmi          : raw HU + Mattes MI  [ABLATION]\n"
            "  seg_dist     : organ signed-dist maps + MSE  [ABLATION]\n"
            "  mind_ncc     : MIND 6-offset descriptor + ANTS NCC  [ABLATION]\n"
            "  sobel_binary : per-organ Otsu binary edges + MSE  [ABLATION]\n"
            "  mind_sobel   : MIND × (1 + sobel_binary) + ANTS NCC  [ABLATION]"
        ),
    )
    ap.add_argument("--all",       action="store_true", help="Batch over all studies")
    ap.add_argument("--skip",      action="store_true", help="Skip already-processed series")
    ap.add_argument(
        "--baseline", action="store_true",
        help="Use fixed_baseline_rigid_registered input (default: fixed_aligned_rigid_registered)",
    )
    ap.add_argument(
        "--grid", type=float, default=GRID_SPACING_MM,
        help=f"B-spline grid spacing in mm (default={GRID_SPACING_MM}). "
             "Larger → coarser, fewer DOF, less folding risk.",
    )
    ap.add_argument(
        "--iters", type=int, default=150,
        help="LBFGSB max iterations per pyramid level (default=150)",
    )
    ap.add_argument(
        "--sampling", type=float, default=0.15,
        help="Random metric sampling fraction (default=0.15 = 15%)",
    )
    ap.add_argument(
        "--no-field", action="store_true",
        help="Do NOT save displacement field (saves ~150-300 MB per phase)",
    )
    ap.add_argument(
        "--split", choices=["all", "train", "test"], default="all",
        help="Restrict to the train/test split from compare_registrtaion/generate_split.py.",
    )
    args = ap.parse_args()

    # Resolve split filter
    _study_filter = None
    if args.split != "all":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compare_registrtaion"))
        from generate_split import load_split as _load_split
        _study_filter = _load_split(args.split)
        print(f"  --split {args.split}: restricting to {len(_study_filter)} studies")

    input_dir = str(INPUT_DIR_BASELINE if args.baseline else INPUT_DIR_ALIGNED)
    mode      = "baseline" if args.baseline else "aligned"
    out_dir   = str(OUTPUT_DIR_BASE) + f"_{args.metric}"
    if args.baseline:
        out_dir += "_baseline"

    print(
        f"metric={args.metric}  grid={args.grid}mm  mode={mode}\n"
        f"out={out_dir}\n"
        f"all={args.all}  skip={args.skip}  save_field={not args.no_field}\n"
        f"iters={args.iters}  sampling={args.sampling:.0%}"
    )

    if args.all:
        register_all_studies_deformable(
            input_dir, out_dir, LABELS_CSV, args.metric,
            grid_spacing_mm=args.grid,
            num_iterations=args.iters,
            sampling_pct=args.sampling,
            save_field=not args.no_field,
            skip_existing=args.skip,
            study_ids=_study_filter,
        )
    else:
        labels_df = pd.read_csv(LABELS_CSV)
        catalog   = scan_input_directory(
            input_dir, VOL_POSTFIX, SEG_REG_POSTFIX, SEG_FULL_POSTFIX,
        )
        logger = DeformableLogger(out_dir, args.metric, args.grid)
        os.makedirs(out_dir, exist_ok=True)
        register_study_deformable(
            args.study_id, catalog, labels_df, out_dir, args.metric, logger,
            grid_spacing_mm=args.grid,
            num_iterations=args.iters,
            sampling_pct=args.sampling,
            save_field=not args.no_field,
            skip_existing=args.skip,
        )
        logger.save_batch_csv()