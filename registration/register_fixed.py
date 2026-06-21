
from __future__ import annotations

import os, sys, glob, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.optimize import minimize

from configs import MAIN_PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_logger import get_stage_logger

# ---------------------------------------------------------------------------
# GPU: import torch if available, fall back to numpy silently
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ALIGNED_INPUT_DIR        = Path(MAIN_PATH + "aligned_volumes")
ALIGNED_OUTPUT_DIR       = Path(MAIN_PATH + "fixed_aligned_rigid_registered")
ALIGNED_VOL_POSTFIX      = "_aligned.nii.gz"
ALIGNED_SEG_REG_POSTFIX  = "_aligned_seg_reg.nii.gz"
ALIGNED_SEG_FULL_POSTFIX = "_aligned_seg_full.nii.gz"

BASELINE_INPUT_DIR        = Path(MAIN_PATH + "baseline_volumes")
BASELINE_OUTPUT_DIR       = Path(MAIN_PATH + "fixed_baseline_rigid_registered")
BASELINE_VOL_POSTFIX      = "_baseline.nii.gz"
BASELINE_SEG_REG_POSTFIX  = "_baseline_seg_reg.nii.gz"
BASELINE_SEG_FULL_POSTFIX = "_baseline_seg_full.nii.gz"

LABELS_CSV       = MAIN_PATH + "labels.csv"
EXAMPLE_STUDY_ID = "1.2.392.200036.9116.2.5.1.37.2418751871.1573442981.375377"

# ---------------------------------------------------------------------------
# Organ label weights — native TotalSegmentator IDs
# ---------------------------------------------------------------------------
BONE_LABELS: Dict[int, float] = {
    40: 3.0,
    21: 2.0, 22: 2.0, 23: 2.0, 24: 2.0, 25: 2.0,
    26: 2.0, 27: 2.0, 28: 2.0,
    38: 1.5, 39: 1.5, 45: 1.0,
    60: 0.8, 61: 0.8, 62: 0.8,
    72: 0.8, 73: 0.8, 74: 0.8,
}

ALL_STABLE_LABELS: Dict[int, float] = {
    **BONE_LABELS,
    1: 3.0, 2: 2.0, 3: 2.0, 4: 2.0,
    # iliac arteries kept for Kabsch (centroid is phase-stable enough for SVD init)
    # but removed from REFINEMENT_LABELS (phase-variant HU → noisy Dice)
    13: 2.0, 14: 1.5,
    9: 1.0, 10: 1.0,
    # stomach(8) and esophagus(7) removed: deformable, NC ≠ phase position,
    # contributed ~40% of Dice loss signal pointing the wrong direction
    16: 0.8, 17: 0.8,
}

REFINEMENT_LABELS: Dict[int, float] = {
    # Soft organs: large, phase-stable position, reliable Dice signal
    1: 3.0,   # liver
    2: 2.0,   # spleen
    3: 2.0,   # kidney_left
    4: 2.0,   # kidney_right
    # Spine: rigid anchors; weights raised to compensate for removed gut organs
    40: 2.0,  # spinal_cord
    21: 1.5, 22: 1.5, 23: 1.5, 24: 1.5, 25: 1.5,   # L1–L5
    # Vessels: phase-stable centroids, raised weight
    9:  1.5,  # aorta
    10: 1.5,  # IVC
    # REMOVED (corrupt loss gradient):
    #  7  esophagus  — moves 20-50mm between phases (peristalsis)
    #  8  stomach    — moves 30-60mm between phases (fill state), was 40% of total loss
    #  13 iliac_art  — enhances in arterial, dark in NC → noisy Dice
    #  14 iliac_vein — same as 13
    #  38 sacrum     — labelled as rib_R1 in code comment; not sacrum; noisy erosion
}

# Erosion is now computed per-organ in _organ_erosion_mm().
# This constant is kept as a fallback for any label not in ORGAN_EROSION_MM.
DICE_EROSION_MM_DEFAULT = 2.0
DICE_LAMBDA          = 10.0
MIN_ORGAN_VOXELS     = 200
MAX_COORDS_PER_ORGAN = 5000   # subsample for fast Dice estimate (<1% variance)


ORGAN_EROSION_MM: Dict[int, float] = {
    1:  6.00,   # liver          (soft_large:  8%  of 128mm eff_d, clamped to 6.0)
    2:  4.58,   # spleen         (soft_med:    7%  of  65mm)
    3:  4.23,   # kidney_left    (soft_med:    7%  of  60mm)
    4:  4.07,   # kidney_right   (soft_med:    7%  of  58mm)
    9:  3.50,   # aorta          (tubular:     5%  of  73mm, clamped to 3.5)
    10: 3.50,   # IVC            (tubular:     5%  of  72mm, clamped to 3.5)
    21: 1.50,   # L1             (bone:        3%  of  44mm, clamped to 1.5)
    22: 1.50,   # L2             (bone:        3%  of  46mm)
    23: 1.50,   # L3             (bone:        3%  of  48mm)
    24: 1.50,   # L4             (bone:        3%  of  48mm)
    25: 1.50,   # L5             (bone:        3%  of  49mm)
    26: 1.50,   # T12            (bone:        3%  of  40mm)
    27: 1.50,   # T11            (bone:        3%  of  41mm)
    28: 1.50,   # T10            (bone:        3%  of  42mm)
    38: 1.88,   # sacrum/rib_R1  (bone:        3%  of  63mm)
    39: 1.50,   # rib_R2         (bone:        3%  of  41mm)
    40: 1.50,   # spinal_cord    (bone:        3%  of  43mm; min_dim=26.5mm)
}

def _organ_erosion_mm(label: int) -> float:
    """Per-organ erosion radius. Falls back to DICE_EROSION_MM_DEFAULT."""
    return ORGAN_EROSION_MM.get(label, DICE_EROSION_MM_DEFAULT)


# ===========================================================================
# FILE DISCOVERY
# ===========================================================================

def scan_input_directory(
        input_dir: str, vol_postfix: str,
        seg_reg_postfix: str, seg_full_postfix: str,
) -> Dict[str, Dict[str, dict]]:
    catalog: Dict[str, Dict[str, dict]] = {}
    if not os.path.exists(input_dir):
        print(f"  Missing: {input_dir}"); return catalog

    study_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d))]
    print(f"\n  Scanning {input_dir}... ({len(study_dirs)} studies)")

    for study_id in study_dirs:
        sp = os.path.join(input_dir, study_id)
        catalog[study_id] = {}
        for img_path in glob.glob(os.path.join(sp, f"*{vol_postfix}")):
            if "seg" in os.path.basename(img_path): continue
            bn = os.path.basename(img_path)
            if not bn.startswith(study_id + "_"): continue
            sid = bn[len(study_id)+1:].replace(vol_postfix, "")
            srp = img_path.replace(vol_postfix, seg_reg_postfix)
            sfp = img_path.replace(vol_postfix, seg_full_postfix)
            if not os.path.exists(srp): continue
            catalog[study_id][sid] = {
                "image":    img_path,
                "seg_reg":  srp,
                "seg_full": sfp if os.path.exists(sfp) else None,
            }
    print(f"  Total series: {sum(len(v) for v in catalog.values())}")
    return catalog


# ===========================================================================
# PRECOMPUTED ORGAN DATA  —  computed ONCE per study, reused every iteration
# ===========================================================================

@dataclass
class OrganPrecomp:
    """All static per-organ data for the Dice+centroid loss."""
    label:   int
    weight:  float

    # Fixed (NC) data — never changes during optimization
    fixed_centroid_mm:  np.ndarray          # (3,) physical XYZ
    fixed_eroded_mask:  np.ndarray          # (Z,Y,X) bool — used for Dice lookup
    fixed_eroded_count: int                 # |eroded fixed mask|

    # Moving organ voxel coordinates — extracted once, transformed per iter
    # Shape (N, 3), physical XYZ mm, subsampled to MAX_COORDS_PER_ORGAN
    moving_eroded_coords_mm:  np.ndarray
    moving_eroded_count_full: int           # full eroded count (for Dice denom)
    moving_centroid_mm:       np.ndarray    # (3,) physical XYZ


@dataclass
class BatchPrecomp:
    """All organs concatenated for a single batched matmul per iteration."""
    # (Total_N, 3) — all organ eroded coords stacked
    all_coords_mm:   np.ndarray
    # (Total_N,)  — which organ index each coord belongs to
    organ_idx:       np.ndarray
    # per-organ arrays (n_organs,) matching the same organ ordering as organs list
    organs:          List[OrganPrecomp]
    # fixed space geometry for coordinate→voxel conversion
    fixed_origin:    np.ndarray    # (3,)  XYZ mm
    fixed_spacing:   np.ndarray    # (3,)  XYZ mm/vox
    fixed_dir_inv:   np.ndarray    # (3,3) inverse direction matrix
    fixed_shape_zyx: Tuple[int, int, int]
    # center of rotation (geometric center of fixed volume)
    center_mm:       np.ndarray    # (3,)  XYZ mm
    # GPU tensors — set only if torch+CUDA available
    gpu_coords:      Optional[object] = None   # torch.Tensor (Total_N, 3)


def _erode_np(mask: np.ndarray, spacing_xyz, erosion_mm: float) -> np.ndarray:
    """BinaryErode on numpy array — called only during precomputation."""
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk_mask.SetSpacing(spacing_xyz)
    radius = [max(1, int(round(erosion_mm / s))) for s in spacing_xyz]
    return sitk.GetArrayFromImage(sitk.BinaryErode(sitk_mask, radius)).astype(bool)


def _centroid_physical(seg_np, label, spacing_xyz, origin_xyz, direction):
    mask = seg_np == label
    if mask.sum() < MIN_ORGAN_VOXELS:
        return None
    cz, cy, cx = ndi.center_of_mass(mask)
    sx, sy, sz = spacing_xyz
    vox_scaled = np.array([cx*sx, cy*sy, cz*sz])
    return np.array(origin_xyz) + np.array(direction).reshape(3,3) @ vox_scaled


def _eroded_coords_physical(eroded_mask_zyx: np.ndarray,
                             spacing_xyz, origin_xyz, direction,
                             label: int = 0) -> np.ndarray:
    """
    Extract physical XYZ coordinates of all eroded voxels, subsampled.
    Returns (N, 3) array, N <= MAX_COORDS_PER_ORGAN.
    seed=label ensures reproducibility per organ without systematic spatial bias.
    """
    zyx_indices = np.argwhere(eroded_mask_zyx)           # (N, 3) ZYX
    if len(zyx_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if len(zyx_indices) > MAX_COORDS_PER_ORGAN:
        rng = np.random.default_rng(label)               # FIX 3: seed by label
        zyx_indices = zyx_indices[
            rng.choice(len(zyx_indices), MAX_COORDS_PER_ORGAN, replace=False)
        ]

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    D = np.array(direction).reshape(3, 3)
    # vox_scaled: (N, 3) XYZ
    vox_scaled = np.column_stack([
        zyx_indices[:, 2] * sx,
        zyx_indices[:, 1] * sy,
        zyx_indices[:, 0] * sz,
    ])
    # physical XYZ: origin + D @ vox_scaled.T  →  (N, 3)
    return (np.array([ox, oy, oz]) + (D @ vox_scaled.T).T).astype(np.float32)


def precompute_organs(
        nc_seg_sitk:  sitk.Image,
        mov_seg_sitk: sitk.Image,
        nc_img:       sitk.Image,
        label_weights: Dict[int, float],
        erosion_mm:    float = None,
        use_gpu:       bool  = _TORCH_AVAILABLE,
) -> BatchPrecomp:
    """
    Pre-extract all static organ data ONCE before the optimizer loop.
    This replaces per-iteration sitk.Resample + BinaryErode + center_of_mass.
    """
    nc_np   = np.round(sitk.GetArrayFromImage(nc_seg_sitk)).astype(np.int32)
    mov_np  = np.round(sitk.GetArrayFromImage(mov_seg_sitk)).astype(np.int32)
    sp      = nc_seg_sitk.GetSpacing()    # XYZ
    origin  = nc_seg_sitk.GetOrigin()
    direc   = nc_seg_sitk.GetDirection()

    D     = np.array(direc).reshape(3, 3)
    D_inv = np.linalg.inv(D)

    sz_xyz    = np.array(nc_img.GetSize(), dtype=float)
    center_mm = np.array(nc_img.GetOrigin()) + 0.5 * sz_xyz * np.array(nc_img.GetSpacing())
    center_mm = (np.array(nc_img.GetOrigin())
                 + 0.5 * (sz_xyz - 1.0) * np.array(nc_img.GetSpacing()))

    organs: List[OrganPrecomp] = []

    for label, weight in label_weights.items():
        # ── Fixed organ ────────────────────────────────────────────────
        fixed_mask = nc_np == label
        if fixed_mask.sum() < MIN_ORGAN_VOXELS:
            continue
        label_erosion = erosion_mm if erosion_mm is not None else _organ_erosion_mm(label)
        fixed_eroded = _erode_np(fixed_mask, sp, label_erosion)
        if fixed_eroded.sum() == 0:
            continue
        fixed_centroid = _centroid_physical(nc_np, label, sp, origin, direc)

        # ── Moving organ ───────────────────────────────────────────────
        mov_mask = mov_np == label
        if mov_mask.sum() < MIN_ORGAN_VOXELS:
            continue
        # Use moving seg's own spacing/origin/direction
        sp_m   = mov_seg_sitk.GetSpacing()
        or_m   = mov_seg_sitk.GetOrigin()
        dir_m  = mov_seg_sitk.GetDirection()

        moving_eroded = _erode_np(mov_mask, sp_m, label_erosion)
        if moving_eroded.sum() == 0:
            continue
        moving_eroded_count = int(moving_eroded.sum())
        moving_centroid = _centroid_physical(mov_np, label, sp_m, or_m, dir_m)
        moving_coords   = _eroded_coords_physical(moving_eroded, sp_m, or_m, dir_m,
                                                    label=label)

        if moving_centroid is None or fixed_centroid is None:
            continue

        organs.append(OrganPrecomp(
            label=label, weight=weight,
            fixed_centroid_mm=fixed_centroid.astype(np.float32),
            fixed_eroded_mask=fixed_eroded,
            fixed_eroded_count=int(fixed_eroded.sum()),
            moving_eroded_coords_mm=moving_coords,
            moving_eroded_count_full=moving_eroded_count,
            moving_centroid_mm=moving_centroid.astype(np.float32),
        ))

    if not organs:
        raise ValueError("No common organs found for refinement.")

    # ── Concatenate all organ coordinates into one big array ───────────
    all_coords = np.vstack([o.moving_eroded_coords_mm for o in organs])  # (Total_N, 3)
    organ_idx  = np.concatenate([
        np.full(len(o.moving_eroded_coords_mm), i, dtype=np.int32)
        for i, o in enumerate(organs)
    ])

    mode = "fixed" if erosion_mm is not None else "adaptive"
    print(f"    Precomputed {len(organs)} organs ({mode} erosion), "
          f"{len(all_coords):,} total coordinate points")
    gpu_coords = None
    if use_gpu and _TORCH_AVAILABLE:
        import torch
        gpu_coords = torch.from_numpy(all_coords).cuda()  # (Total_N, 3)
        print(f"    GPU tensors allocated ({gpu_coords.shape[0]:,} pts on CUDA)")

    return BatchPrecomp(
        all_coords_mm=all_coords,
        organ_idx=organ_idx,
        organs=organs,
        fixed_origin=np.array(origin, dtype=np.float32),
        fixed_spacing=np.array(sp, dtype=np.float32),
        fixed_dir_inv=D_inv.astype(np.float32),
        fixed_shape_zyx=(nc_np.shape[0], nc_np.shape[1], nc_np.shape[2]),
        center_mm=center_mm.astype(np.float32),
        gpu_coords=gpu_coords,
    )


# ===========================================================================
# TRANSFORM UTILITIES
# ===========================================================================

def euler_to_rotation_zyx(rx: float, ry: float, rz: float) -> np.ndarray:
    """R = Rz @ Ry @ Rx — matches SimpleITK Euler3DTransform(SetComputeZYX=True)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def forward_transform_coords(
        coords_mm: np.ndarray,      # (N, 3) moving physical XYZ
        R: np.ndarray,              # (3, 3) from euler_to_rotation_zyx
        t: np.ndarray,              # (3,)  translation [tx, ty, tz]
        center: np.ndarray,         # (3,)  center of rotation XYZ mm
) -> np.ndarray:
    """
    Apply the FORWARD transform (moving→fixed physical space).

    SimpleITK stores the INVERSE transform: x_moving = R @ (x_fixed - c) + c + t
    Forward (moving→fixed):                x_fixed  = R^T @ (x_moving - c - t) + c

    This is exact — no approximation.
    """
    # shifted = coords_mm - center - t    shape (N, 3)
    shifted = coords_mm - center[None, :] - t[None, :]
    # x_fixed = R^T @ shifted^T + center
    return (R.T @ shifted.T).T + center[None, :]   # (N, 3)


def forward_transform_coords_gpu(
        gpu_coords,        # torch.Tensor (N, 3) on CUDA
        R: np.ndarray,
        t: np.ndarray,
        center: np.ndarray,
) -> "torch.Tensor":
    import torch
    R_t  = torch.from_numpy(R.T).float().cuda()       # (3, 3)
    t_th = torch.from_numpy(t + center).float().cuda() # broadcast trick
    c_th = torch.from_numpy(center).float().cuda()
    shifted = gpu_coords - c_th - torch.from_numpy(t).float().cuda()
    return (R_t @ shifted.T).T + c_th                 # (N, 3)


def coords_to_voxel_zyx(
        pts_xyz: np.ndarray,   # (N, 3) physical XYZ
        origin:  np.ndarray,   # (3,)
        spacing: np.ndarray,   # (3,)
        dir_inv: np.ndarray,   # (3, 3)
) -> np.ndarray:
    """Convert physical XYZ → integer voxel indices in ZYX order for numpy indexing."""
    shifted = pts_xyz - origin[None, :]                # (N, 3)
    vox_xyz = (dir_inv @ shifted.T).T / spacing[None, :]  # (N, 3)
    return np.round(vox_xyz[:, ::-1]).astype(np.int32)     # (N, 3) ZYX


def coords_to_voxel_zyx_gpu(pts_xyz, origin, spacing, dir_inv):
    """GPU version — all tensors on CUDA."""
    import torch
    origin_t  = torch.from_numpy(origin).float().cuda()
    spacing_t = torch.from_numpy(spacing).float().cuda()
    dir_inv_t = torch.from_numpy(dir_inv).float().cuda()
    shifted   = pts_xyz - origin_t
    vox_xyz   = (dir_inv_t @ shifted.T).T / spacing_t
    vox_zyx   = torch.flip(vox_xyz, dims=[1]).round().long()
    return vox_zyx.cpu().numpy()                         # back to CPU for mask lookup


# ===========================================================================
# FAST LOSS FUNCTION  —  no Resample, no BinaryErode inside loop
# ===========================================================================

def compute_loss_fast(
        params:     np.ndarray,     # [tx, ty, tz, rx, ry, rz]
        batch:      BatchPrecomp,
        dice_lambda: float = DICE_LAMBDA,
) -> float:
    """
    L = Σ_organs w_i * [ centroid_dist_mm_i  +  dice_lambda * (1 - eroded_dice_i) ]
        / Σ w_i

    Key optimizations vs original compute_loss:
      - No sitk.Resample (full 3D volume): replaced by coordinate transform
      - No per-organ BinaryErode per iter: fixed masks precomputed, moving masks
        also precomputed (erosion commutes with rigid transforms)
      - No per-organ center_of_mass per iter: centroid transformed analytically
      - One batched matmul for ALL organ coordinates combined
      - Optional GPU matmul for the coordinate transform step
    """
    tx, ty, tz, rx, ry, rz = params
    R = euler_to_rotation_zyx(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float32)

    # ── Transform ALL organ coordinates in ONE batched call ────────────
    if batch.gpu_coords is not None:
        transformed_xyz = coords_to_voxel_zyx_gpu(
            forward_transform_coords_gpu(batch.gpu_coords, R, t, batch.center_mm),
            batch.fixed_origin, batch.fixed_spacing, batch.fixed_dir_inv,
        )
    else:
        transformed_pts = forward_transform_coords(
            batch.all_coords_mm, R, t, batch.center_mm
        )
        transformed_xyz = coords_to_voxel_zyx(
            transformed_pts,
            batch.fixed_origin, batch.fixed_spacing, batch.fixed_dir_inv,
        )
    # transformed_xyz: (Total_N, 3) ZYX integer indices in fixed space

    total_loss   = 0.0
    total_weight = 0.0
    Z, Y, X = batch.fixed_shape_zyx

    for i, organ in enumerate(batch.organs):
        # ── Centroid distance (analytical — no volume operation) ───────
        c_fixed   = organ.fixed_centroid_mm
        c_transformed = forward_transform_coords(
            organ.moving_centroid_mm[None, :], R, t, batch.center_mm
        )[0]
        c_dist = float(np.linalg.norm(c_fixed - c_transformed))

        # ── Eroded Dice via coordinate lookup ─────────────────────────
        mask_idx  = batch.organ_idx == i
        vox       = transformed_xyz[mask_idx]            # (N_i, 3) ZYX

        # Bounds filter
        in_bounds = (
            (vox[:, 0] >= 0) & (vox[:, 0] < Z) &
            (vox[:, 1] >= 0) & (vox[:, 1] < Y) &
            (vox[:, 2] >= 0) & (vox[:, 2] < X)
        )
        v = vox[in_bounds]

        if len(v) > 0:
            hits        = organ.fixed_eroded_mask[v[:, 0], v[:, 1], v[:, 2]].sum()
            n_moving    = organ.moving_eroded_count_full
            n_fixed     = organ.fixed_eroded_count
            # FIX 2: scale by in-bounds count len(v), not pre-filter len(vox).
            # Using len(vox) underestimates intersection when the organ is
            # partially outside the fixed volume (many points fail bounds check),
            # producing artificially low Dice and inflated loss.
            scale        = n_moving / max(len(v), 1)
            intersection = hits * scale
            denom        = n_fixed + n_moving
            dice         = float(2.0 * intersection / denom) if denom > 0 else 0.0
        else:
            dice = 0.0

        organ_loss    = c_dist + dice_lambda * (1.0 - dice)
        total_loss   += organ.weight * organ_loss
        total_weight += organ.weight

    return total_loss / total_weight if total_weight > 0 else 1e6


# ===========================================================================
# PASS 0 — KABSCH SVD CENTROID ALIGNMENT (unchanged)
# ===========================================================================

def organ_centroid_physical(seg_np, spacing_xyz, origin_xyz, direction, label):
    mask = seg_np == label
    if mask.sum() < MIN_ORGAN_VOXELS:
        return None
    cz, cy, cx = ndi.center_of_mass(mask)
    sx, sy, sz = spacing_xyz
    vox_scaled = np.array([cx*sx, cy*sy, cz*sz])
    return np.array(origin_xyz) + np.array(direction).reshape(3,3) @ vox_scaled


def extract_centroids(seg_sitk, label_weights):
    seg_np  = np.round(sitk.GetArrayFromImage(seg_sitk)).astype(np.int32)
    sp, or_, di = seg_sitk.GetSpacing(), seg_sitk.GetOrigin(), seg_sitk.GetDirection()
    result = {}
    for label, weight in label_weights.items():
        c = organ_centroid_physical(seg_np, sp, or_, di, label)
        if c is not None:
            result[label] = (c, weight)
    return result


def kabsch_align(fixed_c, moving_c):
    common = sorted(set(fixed_c) & set(moving_c))
    if len(common) < 3:
        raise ValueError(f"Need ≥3 common organs for Kabsch, got {len(common)}")
    pts_f = np.array([fixed_c[l][0]  for l in common])
    pts_m = np.array([moving_c[l][0] for l in common])
    wts   = np.array([fixed_c[l][1] * moving_c[l][1] for l in common])
    wts  /= wts.sum()
    mu_f  = (wts[:, None] * pts_f).sum(0)
    mu_m  = (wts[:, None] * pts_m).sum(0)
    H     = (wts[:, None] * (pts_m - mu_m)).T @ (pts_f - mu_f)
    U, _, Vt = np.linalg.svd(H)
    d     = np.linalg.det(Vt.T @ U.T)
    D     = np.diag([1.0, 1.0, float(np.sign(d)) if d != 0 else 1.0])
    R     = Vt.T @ D @ U.T
    t     = mu_f - R @ mu_m
    errs  = [np.linalg.norm(fixed_c[l][0] - (R @ moving_c[l][0] + t)) for l in common]
    return R, t, common, float(np.mean(errs))


def rotation_to_euler_zyx(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        rx = math.atan2(R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2(R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0.0
    return rx, ry, rz


def make_euler3d(R, t, center):
    """Build SimpleITK Euler3DTransform (stored as inverse for Resample)."""
    R_inv = R.T
    t_inv = -(R_inv @ t)
    rx, ry, rz = rotation_to_euler_zyx(R_inv)
    tx = sitk.Euler3DTransform()
    tx.SetComputeZYX(True)
    tx.SetCenter(center.tolist())
    tx.SetRotation(rx, ry, rz)
    t_total = t_inv + (R_inv - np.eye(3)) @ center
    tx.SetTranslation(t_total.tolist())
    return tx


def centroid_alignment(nc_seg, mov_seg, nc_img, label_weights, phase="?"):
    fixed_c  = extract_centroids(nc_seg,  label_weights)
    moving_c = extract_centroids(mov_seg, label_weights)
    R, t, common, err = kabsch_align(fixed_c, moving_c)
    print(f"    Kabsch: {len(common)} organs, residual={err:.2f}mm")
    sz     = np.array(nc_img.GetSize(), dtype=float)
    center = np.array(nc_img.GetOrigin()) + 0.5 * sz * np.array(nc_img.GetSpacing())
    tx     = make_euler3d(R, t, center)
    return tx, err


# ===========================================================================
# PASS 1 — OPTIMIZED DICE + CENTROID REFINEMENT
# ===========================================================================

def dice_centroid_refinement(
        nc_img:       sitk.Image,
        nc_seg_sitk:  sitk.Image,
        mov_seg_sitk: sitk.Image,
        init_transform: sitk.Euler3DTransform,
        label_weights:  Dict[int, float],
        dice_lambda:    float = DICE_LAMBDA,
        erosion_mm:     Optional[float] = None,
        max_iter:       int   = 200,
        use_gpu:        bool  = _TORCH_AVAILABLE,
) -> Tuple[sitk.Euler3DTransform, float]:
    """
    Nelder-Mead minimisation of centroid_dist + dice_lambda*(1-Dice).
    Precomputes all organ data once; each iteration does only:
      - one batched matmul (or GPU matmul)
      - integer array lookups on precomputed boolean masks
    """
    print(f"    Precomputing organ data (once)...")
    batch = precompute_organs(nc_seg_sitk, mov_seg_sitk, nc_img,
                              label_weights, erosion_mm, use_gpu=use_gpu)
    use_gpu_actual = batch.gpu_coords is not None
    print(f"    Backend: {'CUDA GPU' if use_gpu_actual else 'CPU numpy'}")

    # FIX 11: verify that the init_transform's center of rotation agrees with
    # batch.center_mm (both computed from nc_img geometry). A mismatch would
    # silently shift the starting point of Pass 1 by the difference.
    init_center = np.array(init_transform.GetCenter(), dtype=np.float32)
    center_gap  = float(np.linalg.norm(init_center - batch.center_mm))
    if center_gap > 1.0:   # >1mm mismatch is a bug, not rounding
        print(f"    WARNING: init_transform center {init_center} differs from "
              f"batch.center_mm {batch.center_mm} by {center_gap:.2f}mm")

    # Extract initial [tx, ty, tz, rx, ry, rz] from init_transform.
    # SimpleITK stores INVERSE: x_moving = R @ (x_fixed - c) + c + t_sitk
    # Our forward_transform_coords uses: x_fixed = R^T @ (x_moving - c - t) + c
    # so t_fwd = t_sitk when using the same center c.
    t0  = list(init_transform.GetTranslation())
    R0  = np.array(init_transform.GetMatrix()).reshape(3, 3)
    rx0, ry0, rz0 = rotation_to_euler_zyx(R0)
    x0  = np.array([t0[0], t0[1], t0[2], rx0, ry0, rz0])

    loss0 = compute_loss_fast(x0, batch, dice_lambda)
    print(f"    Initial loss: {loss0:.4f}")

    res = minimize(
        compute_loss_fast, x0,
        args=(batch, dice_lambda),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 0.1,
                 "fatol": 0.001,    # FIX 5: tightened from 0.01
                 "disp": False},
    )
    tx_mm, ty_mm, tz_mm, rx, ry, rz = res.x
    print(f"    Final loss: {res.fun:.4f}  iters={res.nit}  ok={res.success}")

    final_tx = sitk.Euler3DTransform()
    final_tx.SetComputeZYX(True)
    final_tx.SetCenter(batch.center_mm.tolist())
    final_tx.SetRotation(rx, ry, rz)
    final_tx.SetTranslation([tx_mm, ty_mm, tz_mm])
    return final_tx, float(res.fun)


# ===========================================================================
# PASS 2 — GRADIENT-MAGNITUDE POLISH  (phase-invariant edge alignment)
# ===========================================================================
#
# WHY GRADIENT MAGNITUDE instead of MMI on raw HU:
#   NC / Arterial / Venous share the same anatomical EDGES (organ boundaries,
#   bone cortex, vessel walls) even though their absolute HU values differ
#   dramatically (aorta: ~40 HU in NC, ~300 HU in Arterial).  Raw-intensity
#   metrics (MMI, NCC) try to align intensity distributions that are
#   fundamentally different across phases, which introduces a systematic pull
#   toward incorrect configurations.  |∇HU| is largely phase-independent:
#   a liver-spleen boundary produces a sharp gradient in all phases.
#
# IMPLEMENTATION:
#   1. Compute |∇HU| for both fixed (NC) and moving via Sobel magnitude.
#   2. Clamp to [0, 200] so bone/air interfaces (very large gradients) don't
#      dominate over soft-tissue boundaries.
#   3. Register gradient images with NCC (local normalized cross-correlation,
#      radius 4 vox) inside the organ mask.  NCC on |∇HU| is equivalent to
#      matching edge strength and direction locally.
#   4. Multi-scale pyramid [4×, 2×, 1×] with Gaussian smoothing [2, 1, 0 mm].
#   5. Fixed + moving masks applied (moving mask = fixed mask resampled into
#      moving space + 10mm dilation to absorb residual misalignment).
#   6. Acceptance gate: compare pre/post centroid errors using seg_reg.
#      If Pass 2 increases mean centroid error by >5%, revert to Pass 1.

def _compute_gradient_magnitude(img_sitk: sitk.Image,
                                 clamp_max: float = 200.0) -> sitk.Image:
    """
    Compute voxel-wise gradient magnitude |∇HU|, clamped to [0, clamp_max].

    Sobel is applied in physical space (uses image spacing automatically via
    sitk.SobelEdgeDetection). Result is float32.

    clamp_max=200 keeps soft-tissue boundaries (liver/spleen ~30-80 HU/mm)
    and vessel walls (~80-120 HU/mm) on the same scale while preventing
    bone/air interfaces (>500 HU/mm) from dominating the metric.
    """
    img_f32 = sitk.Cast(img_sitk, sitk.sitkFloat32)
    # GradientMagnitude applies finite-difference in each direction with
    # correct physical spacing, then returns the Euclidean norm.
    grad = sitk.GradientMagnitude(img_f32)
    # Clamp: values > clamp_max → clamp_max
    grad = sitk.Clamp(grad, lowerBound=0.0, upperBound=clamp_max)
    return grad


def _build_organ_mask(seg_reg_sitk: sitk.Image,
                       label_weights: Dict[int, float],
                       dilation_mm: float = 6.0) -> Optional[sitk.Image]:
    """
    Binary mask covering all organs in label_weights, dilated by dilation_mm.
    Returns None if no organ voxels found.
    """
    seg_np  = np.round(sitk.GetArrayFromImage(seg_reg_sitk)).astype(np.int32)
    spacing = seg_reg_sitk.GetSpacing()
    combined = np.zeros_like(seg_np, dtype=np.uint8)
    for label in label_weights:
        region = (seg_np == label).astype(np.uint8)
        if region.sum() < 50:
            continue
        combined |= region
    if combined.sum() == 0:
        return None
    tmp = sitk.GetImageFromArray(combined)
    tmp.SetSpacing(spacing)
    radius = [max(1, int(round(dilation_mm / s))) for s in spacing]
    out = sitk.BinaryDilate(tmp, radius)
    out.CopyInformation(seg_reg_sitk)
    return sitk.Cast(out, sitk.sitkUInt8)


def _resample_mask_to_moving(fixed_mask: sitk.Image,
                              moving_ref: sitk.Image,
                              extra_dilation_mm: float = 10.0) -> sitk.Image:
    """
    Bring the fixed organ mask into moving image space + extra dilation.
    The extra dilation absorbs residual rigid misalignment from Pass 1
    so the moving mask doesn't exclude the correct target region.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())   # identity — just resample to grid
    mov_mask = resampler.Execute(fixed_mask)

    if extra_dilation_mm > 0:
        spacing = moving_ref.GetSpacing()
        radius  = [max(1, int(round(extra_dilation_mm / s))) for s in spacing]
        mov_mask = sitk.BinaryDilate(mov_mask, radius)
    return sitk.Cast(mov_mask, sitk.sitkUInt8)


def _mean_centroid_error_mm(nc_seg: sitk.Image,
                             mov_seg_transformed: sitk.Image,
                             label_weights: Dict[int, float]) -> float:
    """
    Compute weighted mean centroid displacement (mm) between fixed (NC) seg
    and a transformed moving seg.  Used as the acceptance gate criterion.
    Returns a large value (1e6) if no common organs found.
    """
    nc_np  = np.round(sitk.GetArrayFromImage(nc_seg)).astype(np.int32)
    mov_np = np.round(sitk.GetArrayFromImage(mov_seg_transformed)).astype(np.int32)
    sp     = nc_seg.GetSpacing()
    or_    = nc_seg.GetOrigin()
    di     = nc_seg.GetDirection()

    total_err = 0.0
    total_w   = 0.0
    for label, w in label_weights.items():
        cf = _centroid_physical(nc_np,  label, sp, or_, di)
        cm = _centroid_physical(mov_np, label, sp, or_, di)
        if cf is None or cm is None:
            continue
        total_err += w * float(np.linalg.norm(cf - cm))
        total_w   += w
    return (total_err / total_w) if total_w > 0 else 1e6


def gradient_edge_polish(
        fixed:      sitk.Image,
        moving:     sitk.Image,
        fixed_mask: Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
        init_transform: sitk.Transform,
        num_iterations:     int   = 150,
        learning_rate:      float = 0.05,    # FIX 8: reduced from 0.2
        sampling_pct:       float = 0.30,
        gradient_clamp_max: float = 200.0,
        ncc_radius:         int   = 4,
        final_interpolator: int   = sitk.sitkLanczosWindowedSinc,
) -> Tuple[sitk.Image, sitk.Transform, float]:
    """
    FIX 6, 7, 8: Gradient-magnitude NCC registration with multi-scale pyramid.

    Args:
        fixed / moving      : raw CT volumes
        fixed_mask          : binary organ mask for fixed (NC) image
        moving_mask         : permissive organ mask in moving image space
        init_transform      : starting point (output of Pass 1)
        num_iterations      : max GD iterations per pyramid level
        learning_rate       : step size (small — this is a polish, not coarse search)
        sampling_pct        : fraction of masked voxels sampled per iteration
        gradient_clamp_max  : |∇HU| clamped to this value before registration
        ncc_radius          : local NCC window radius in voxels
        final_interpolator  : interpolator for resampling the output volume

    Returns:
        (registered_volume, final_transform, final_metric_value)
    """
    # FIX 6: compute gradient-magnitude images (phase-invariant)
    fixed_grad  = _compute_gradient_magnitude(fixed,  gradient_clamp_max)
    moving_grad = _compute_gradient_magnitude(moving, gradient_clamp_max)

    reg = sitk.ImageRegistrationMethod()

    # NCC on gradient magnitude — captures edge alignment without HU bias
    reg.SetMetricAsANTSNeighborhoodCorrelation(radius=ncc_radius)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct, seed=42)

    # FIX 10: apply both fixed and moving masks
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init_transform, inPlace=False)

    # FIX 7: 3-level pyramid — coarse-to-fine, prevents chasing local texture
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_tx     = reg.Execute(fixed_grad, moving_grad)
    final_metric = reg.GetMetricValue()

    # Resample the original CT volume (not the gradient image)
    registered = sitk.Resample(moving, fixed, final_tx,
                                final_interpolator, -1024.0, moving.GetPixelID())

    print(f"    Pass2 grad-NCC metric={final_metric:.6f}  "
          f"iter={reg.GetOptimizerIteration()}  "
          f"stop={reg.GetOptimizerStopConditionDescription()}")
    return registered, final_tx, final_metric


# ===========================================================================
# SAVE HELPERS
# ===========================================================================

def _save(img, path, label):
    try:
        sitk.WriteImage(img, path)
        mb = os.path.getsize(path) / 1024**2
        print(f"  ok {label}: {os.path.basename(path)} ({mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  fail {label}: {e}"); return False


def _apply_seg(seg, reference, transform):
    return sitk.Resample(seg, reference, transform,
                         sitk.sitkNearestNeighbor, 0, seg.GetPixelID())


def _resample_vol(moving, reference, transform, interp=sitk.sitkLanczosWindowedSinc):
    return sitk.Resample(moving, reference, transform,
                         interp, -1024.0, moving.GetPixelID())


# ===========================================================================
# PER-STUDY REGISTRATION
# ===========================================================================

def register_study(
        study_id: str, catalog, labels_df, output_dir,
        skip_existing=True, pass1_max_iter=200, pass2_iterations=100,
        final_interpolator=sitk.sitkLanczosWindowedSinc,
        use_gpu: bool = _TORCH_AVAILABLE,
) -> dict:
    if study_id not in catalog or not catalog[study_id]:
        return {"status": "skipped", "reason": "not_in_catalog"}

    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    nc_rows    = study_rows[study_rows["Label"] == "Non-contrast"]
    if nc_rows.empty:
        return {"status": "skipped", "reason": "no_noncontrast_label"}

    nc_sid = nc_rows.iloc[0]["SeriesInstanceUID"]
    if nc_sid not in catalog[study_id]:
        return {"status": "skipped", "reason": "nc_files_missing"}

    study_out = os.path.join(output_dir, study_id)
    os.makedirs(study_out, exist_ok=True)

    log = get_stage_logger("registration", study_id=study_id)
    log.info(f"Starting registration: {study_id}")
    
    print(f"\n{'='*80}")
    print(f"Registration (Kabsch→Dice→grad-edge): {study_id}")
    print(f"GPU: {'yes (CUDA)' if (use_gpu and _TORCH_AVAILABLE) else 'no (CPU numpy)'}")
    print(f"{'='*80}")

    nc_files    = catalog[study_id][nc_sid]
    nc_img      = sitk.ReadImage(nc_files["image"])
    nc_seg_reg  = sitk.ReadImage(nc_files["seg_reg"])
    nc_seg_full = sitk.ReadImage(nc_files["seg_full"]) if nc_files["seg_full"] else None

    # Build fixed organ mask (NC) — used for Pass 2 fixed mask
    fixed_mask = _build_organ_mask(nc_seg_reg, ALL_STABLE_LABELS, dilation_mm=6.0)

    # FIX 1: NC is saved once under both rigid0 and rigid1 tags (rigid2 saved after gate)
    nc_prefix = os.path.join(study_out, f"{study_id}_{nc_sid}")
    for tag in ("rigid0", "rigid1", "rigid2"):
        _save(nc_img,     f"{nc_prefix}_{tag}.nii.gz",         f"NC ({tag})")
        _save(nc_seg_reg, f"{nc_prefix}_{tag}_seg_reg.nii.gz", f"NC seg_reg ({tag})")
        if nc_seg_full:
            _save(nc_seg_full, f"{nc_prefix}_{tag}_seg_full.nii.gz", f"NC seg_full ({tag})")

    results = {"status": "complete", "study_id": study_id,
               "successful": [], "failed": [], "skipped": []}

    for _, row in study_rows.iterrows():
        sid, phase = row["SeriesInstanceUID"], row["Label"]
        if sid == nc_sid: continue
        if sid not in catalog[study_id]:
            results["failed"].append({"series": sid, "phase": phase,
                                      "reason": "not_in_catalog"}); continue

        prefix = os.path.join(study_out, f"{study_id}_{sid}")
        if skip_existing and os.path.exists(f"{prefix}_rigid2.nii.gz"):
            print(f"\n  [{phase}] already done — skip")
            results["skipped"].append(sid); continue

        files = catalog[study_id][sid]
        print(f"\n  [{phase}]  {sid[:50]}...")
        try:
            mov_img      = sitk.ReadImage(files["image"])
            mov_seg_reg  = sitk.ReadImage(files["seg_reg"])
            mov_seg_full = sitk.ReadImage(files["seg_full"]) if files["seg_full"] else None

            # ── Pass 0 — Kabsch SVD on ALL_STABLE_LABELS ─────────────────
            # FIX 1: single SVD pass (was: redundant bone-only 0a then full 0b).
            # ALL_STABLE_LABELS includes bones + soft organs so this is already
            # strictly better than bone-only init as a starting point for Pass 1.
            print(f"\n  Pass 0 — Kabsch SVD (all stable organs)...")
            tx0, err0 = centroid_alignment(nc_seg_reg, mov_seg_reg, nc_img,
                                           ALL_STABLE_LABELS, phase)
            _save(_resample_vol(mov_img, nc_img, tx0),
                  f"{prefix}_rigid0.nii.gz", f"[{phase}] pass0")
            _save(_apply_seg(mov_seg_reg, nc_img, tx0),
                  f"{prefix}_rigid0_seg_reg.nii.gz", f"[{phase}] pass0 seg_reg")
            if mov_seg_full:
                _save(_apply_seg(mov_seg_full, nc_img, tx0),
                      f"{prefix}_rigid0_seg_full.nii.gz", f"[{phase}] pass0 seg_full")
             # Measure Pass 0 error on REFINEMENT_LABELS (comparable to err1)
            err0_centroid = _mean_centroid_error_mm(
                nc_seg_reg,
                _apply_seg(mov_seg_reg, nc_img, tx0),
                REFINEMENT_LABELS,
            )
            log.metric("pass0_kabsch", {
                "series_id":       sid,
                "phase":           phase,
                "err_all_stable":  round(float(err0), 3),        # SVD residual, ALL_STABLE
                "err_refinement":  round(float(err0_centroid), 3), # for comparison with Pass1
                "n_organs":        len(ALL_STABLE_LABELS),
            })
            if err0_centroid > 20.0:
                log.warning(
                    f"[{phase}] Pass 0 Kabsch refinement-set residual {err0_centroid:.1f}mm > 20mm"
                )
            tx1, loss1 = dice_centroid_refinement(
                nc_img, nc_seg_reg, mov_seg_reg, tx0,
                REFINEMENT_LABELS, DICE_LAMBDA, None,
                pass1_max_iter, use_gpu=use_gpu,
            )
            mov_seg_reg_pass1 = _apply_seg(mov_seg_reg, nc_img, tx1)
            _save(_resample_vol(mov_img, nc_img, tx1),
                  f"{prefix}_rigid1.nii.gz", f"[{phase}] pass1")
            _save(mov_seg_reg_pass1,
                  f"{prefix}_rigid1_seg_reg.nii.gz", f"[{phase}] pass1 seg_reg")
            if mov_seg_full:
                _save(_apply_seg(mov_seg_full, nc_img, tx1),
                      f"{prefix}_rigid1_seg_full.nii.gz", f"[{phase}] pass1 seg_full")

            # Measure Pass 1 centroid error (for acceptance gate)
            err1 = _mean_centroid_error_mm(nc_seg_reg, mov_seg_reg_pass1, REFINEMENT_LABELS)
            log.metric("pass1_nelder_mead", {
                "series_id": sid, "phase": phase,
                "loss": round(float(loss1), 4), "err_mm": round(float(err1), 3),
            })
            if err1 > err0_centroid * 1.1:   # ← both now use REFINEMENT_LABELS ✓
                log.warning(
                    f"[{phase}] Pass 1 degraded centroid vs Kabsch: "
                    f"{err1:.1f}mm vs {err0_centroid:.1f}mm"
                )
 
            gate = max(err0_centroid * 1.10, err0_centroid + 3.0)
            if err1 > gate:
                print(f"    Pass 1 REJECTED ...")
                tx1 = tx0
                mov_seg_reg_pass1 = _apply_seg(mov_seg_reg, nc_img, tx0)
                err1 = err0_centroid
                
            # # ── Pass 2 — gradient-magnitude edge polish ────────────────────
            print(f"\n  Pass 2 — gradient-edge polish (phase-invariant)...")

            # FIX 10: moving mask = fixed mask resampled into moving space + 10mm dilation
            moving_mask = _resample_mask_to_moving(fixed_mask, mov_img,
                                                    extra_dilation_mm=10.0)

            reg2, tx2, m2 = gradient_edge_polish(
                nc_img, mov_img,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                init_transform=tx1,
                num_iterations=pass2_iterations,
            )

            # FIX 9: acceptance gate — only keep Pass 2 if centroid error doesn't rise
            mov_seg_reg_pass2 = _apply_seg(mov_seg_reg, nc_img, tx2)
            err2 = _mean_centroid_error_mm(nc_seg_reg, mov_seg_reg_pass2,
                                           REFINEMENT_LABELS)
            print(f"    Pass 2 mean centroid error: {err2:.2f}mm  "
                  f"(pass1={err1:.2f}mm)")

            if err2 <= err1 * 1.05:    # accept if within 5% of Pass 1
                print(f"    Pass 2 accepted (Δerr={err2-err1:+.2f}mm)")
                final_vol     = reg2
                final_tx      = tx2
                final_seg_reg = mov_seg_reg_pass2
                final_seg_full = (_apply_seg(mov_seg_full, nc_img, tx2)
                                  if mov_seg_full else None)
                p2_accepted   = True
                log.metric("pass2_gradient_edge", {
                    "series_id": sid, "phase": phase,
                    "err_mm": round(float(err2), 3), "accepted": True,
                    "metric": round(float(m2), 6),
                })
            else:
                print(f"    Pass 2 REJECTED (Δerr={err2-err1:+.2f}mm) — reverting to Pass 1")
                final_vol     = _resample_vol(mov_img, nc_img, tx1)
                final_tx      = tx1
                final_seg_reg = mov_seg_reg_pass1
                final_seg_full = (_apply_seg(mov_seg_full, nc_img, tx1)
                                  if mov_seg_full else None)
                p2_accepted   = False
                log.metric("pass2_gradient_edge", {
                    "series_id": sid, "phase": phase,
                    "err_mm": round(float(err2), 3), "accepted": False,
                    "metric": round(float(m2), 6),
                })
                log.warning(
                    f"[{phase}] Pass 2 rejected: centroid {err2:.1f}mm > "
                    f"Pass1 {err1:.1f}mm — reverted."
                )

            _save(final_vol,     f"{prefix}_rigid2.nii.gz",          f"[{phase}] pass2")
            _save(final_seg_reg, f"{prefix}_rigid2_seg_reg.nii.gz",  f"[{phase}] pass2 seg_reg")
            if final_seg_full:
                _save(final_seg_full, f"{prefix}_rigid2_seg_full.nii.gz",
                      f"[{phase}] pass2 seg_full")

            results["successful"].append({
                "series":           sid,
                "phase":            phase,
                "pass0_err_mm":     err0,
                "pass1_loss":       loss1,
                "pass1_err_mm":     err1,
                "pass2_metric":     m2,
                "pass2_err_mm":     err2,
                "pass2_accepted":   p2_accepted,
            })
            log.stage_summary({
                "series_id":       sid,
                "phase":           phase,
                "status":          "ok",
                "pass0_err_mm":    round(float(err0), 3),
                "pass1_err_mm":    round(float(err1), 3),
                "pass2_err_mm":    round(float(err2), 3),
                "pass2_accepted":  p2_accepted,
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            log.error(
                f"[{phase}] Registration failed: {e}", exc_info=True
            )
            log.stage_summary({
                "series_id": sid,
                "phase":     phase,
                "status":    "failed",
                "error":     str(e),
            })
            results["failed"].append({"series": sid, "phase": phase,
                                      "reason": str(e)})

    print(f"\n  Done: ok={len(results['successful'])}  "
          f"skip={len(results['skipped'])}  fail={len(results['failed'])}")
    return results


# ===========================================================================
# BATCH
# ===========================================================================

def register_all_studies(
        input_dir, output_dir, labels_csv,
        vol_postfix, seg_reg_postfix, seg_full_postfix,
        skip_existing=True, pass1_max_iter=200, pass2_iterations=100,
        use_gpu=_TORCH_AVAILABLE,
):
    labels_df = pd.read_csv(labels_csv)
    catalog   = scan_input_directory(input_dir, vol_postfix,
                                      seg_reg_postfix, seg_full_postfix)
    if not catalog: print("No studies found."); return
    valid = sorted(set(catalog) & set(labels_df["StudyInstanceUID"].unique()))
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"BATCH  studies={len(valid)}  gpu={use_gpu and _TORCH_AVAILABLE}")
    print(f"{'='*80}")
    counts = {"ok":0,"skipped":0,"failed":0}
    for idx, sid in enumerate(valid, 1):
        print(f"\n[{idx}/{len(valid)}] {sid[:55]}...")
        try:
            res = register_study(sid, catalog, labels_df, output_dir,
                                 skip_existing, pass1_max_iter, pass2_iterations,
                                 use_gpu=use_gpu)
            key = "ok" if (res["status"]=="complete" and res["successful"]) \
                  else ("skipped" if res["status"]=="skipped" else "failed")
            counts[key] += 1
        except Exception as e:
            import traceback; traceback.print_exc(); counts["failed"] += 1
    print(f"\nDONE  ok={counts['ok']}  skip={counts['skipped']}  "
          f"fail={counts['failed']}  / {len(valid)}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    args     = sys.argv[1:]
    baseline = "--baseline" in args
    skip     = "--skip"     in args
    no_gpu   = "--no-gpu"   in args
    use_gpu  = _TORCH_AVAILABLE and not no_gpu

    if baseline:
        idir, odir = str(BASELINE_INPUT_DIR), str(BASELINE_OUTPUT_DIR)
        vp, srp, sfp = BASELINE_VOL_POSTFIX, BASELINE_SEG_REG_POSTFIX, BASELINE_SEG_FULL_POSTFIX
    else:
        idir, odir = str(ALIGNED_INPUT_DIR), str(ALIGNED_OUTPUT_DIR)
        vp, srp, sfp = ALIGNED_VOL_POSTFIX, ALIGNED_SEG_REG_POSTFIX, ALIGNED_SEG_FULL_POSTFIX

    print(f"mode={'baseline' if baseline else 'aligned'}  "
          f"skip={skip}  gpu={use_gpu}  cuda={_TORCH_AVAILABLE}")

    if "--all" in args:
        register_all_studies(idir, odir, LABELS_CSV, vp, srp, sfp,
                             skip_existing=skip, use_gpu=use_gpu)
    else:
        print("in single organ")
        sid       = next((a for a in args if not a.startswith("--")), EXAMPLE_STUDY_ID)
        labels_df = pd.read_csv(LABELS_CSV)
        catalog   = scan_input_directory(idir, vp, srp, sfp)
        register_study(sid, catalog, labels_df, odir, skip_existing=skip,
                       use_gpu=use_gpu)