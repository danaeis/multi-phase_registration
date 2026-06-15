"""
evaluate_registration.py
========================
Per-organ registration quality metrics for each pipeline stage.

Metrics
-------
1. Eroded Dice          — shape overlap after eroding mask edges by N mm
                          (robust to segmentation boundary errors)
2. NCC within mask      — normalized cross-correlation of CT intensities
                          inside the eroded organ region
                          (measures actual tissue alignment, not just mask overlap)
3. Centroid displacement (mm) — distance between organ centroids in fixed vs
                          registered-moving space
                          (naturally insensitive to boundary noise, clinically interpretable)

Usage
-----
# Evaluate a single study at one pipeline stage:
python evaluate_registration.py \\
    --study_dir  /data/registered_aligned/STUDY_ID \\
    --labels_csv /data/labels.csv \\
    --seg_postfix _registered_seg.nii.gz \\
    --vol_postfix _registered.nii.gz \\
    --ref_phase   Non-contrast \\
    --erosion_mm  6.0

# Batch over all studies:
python evaluate_registration.py --all \\
    --base_dir    /data/registered_aligned \\
    --labels_csv  /data/labels.csv \\
    --seg_postfix _registered_seg.nii.gz \\
    --vol_postfix _registered.nii.gz \\
    --out_csv     /data/eval_rigid.csv
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Organ label map  (TotalSegmentator labels used in this project)
# ---------------------------------------------------------------------------
ORGAN_LABELS: Dict[int, str] = {
    1:  "liver",
    2:  "spleen",
    3:  "kidney_left",
    4:  "kidney_right",
    5:  "pancreas",
    6:  "gallbladder",
    13: "aorta",
    14: "inferior_vena_cava",
    21: "vertebrae_L1",
    22: "vertebrae_L2",
    23: "vertebrae_L3",
}

# For NCC we only use organs with consistent parenchymal texture across phases
NCC_ORGANS = {1, 2, 3, 4, 13, 14}  # liver, spleen, kidneys, vessels


from scipy.ndimage import distance_transform_edt

def mean_surface_distance_mm(mask_fixed, mask_moving, spacing_zyx):
    """One-sided mean surface distance: mean distance from moving surface to fixed."""
    if mask_fixed.sum() == 0 or mask_moving.sum() == 0:
        return None
    # Distance map from fixed surface
    dist_fixed = distance_transform_edt(~mask_fixed.astype(bool),
                                        sampling=spacing_zyx)
    # Moving surface voxels
    from scipy.ndimage import binary_erosion
    surf_moving = mask_moving.astype(bool) & ~binary_erosion(mask_moving.astype(bool))
    if surf_moving.sum() == 0:
        return None
    return float(dist_fixed[surf_moving].mean())

# ===========================================================================
# MASK UTILITIES
# ===========================================================================

def erode_mask(binary_mask: np.ndarray,
               spacing_mm: Tuple[float, float, float],
               erosion_mm: float = 6.0) -> np.ndarray:
    """
    Erode a binary mask by erosion_mm in physical space.

    Args:
        binary_mask : (Z, Y, X) bool/uint8 array
        spacing_mm  : (z, y, x) voxel spacing in mm  [note: numpy Z-first]
        erosion_mm  : erosion radius in mm

    Returns:
        Eroded binary mask as uint8.
    """
    # Convert to sitk for accurate mm-based morphology
    mask_sitk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    # sitk spacing is (x, y, z) — reverse of numpy
    mask_sitk.SetSpacing(tuple(reversed(spacing_mm)))

    radius_vox = [max(1, int(round(erosion_mm / s)))
                  for s in mask_sitk.GetSpacing()]

    eroded = sitk.BinaryErode(mask_sitk, radius_vox)
    return sitk.GetArrayFromImage(eroded).astype(np.uint8)


def get_spacing_zyx(sitk_img: sitk.Image) -> Tuple[float, float, float]:
    """Return spacing as (z, y, x) to match numpy array axis order."""
    sp = sitk_img.GetSpacing()   # sitk: (x, y, z)
    return (sp[2], sp[1], sp[0])


# ===========================================================================
# METRIC FUNCTIONS
# ===========================================================================

def eroded_dice(mask_fixed: np.ndarray,
                mask_moving: np.ndarray,
                spacing_zyx: Tuple[float, float, float],
                erosion_mm: float) -> Optional[float]:
    """
    Dice coefficient computed on eroded versions of both masks.

    Returns None if either eroded mask is empty (organ too small to erode).
    """
    ef = erode_mask(mask_fixed,  spacing_zyx, erosion_mm)
    em = erode_mask(mask_moving, spacing_zyx, erosion_mm)

    if ef.sum() == 0 or em.sum() == 0:
        return None

    intersection = int((ef & em).sum())
    return 2.0 * intersection / (int(ef.sum()) + int(em.sum()))


def ncc_within_mask(vol_fixed: np.ndarray,
                    vol_moving: np.ndarray,
                    mask_fixed: np.ndarray,
                    spacing_zyx: Tuple[float, float, float],
                    erosion_mm: float) -> Optional[float]:
    """
    Normalized cross-correlation of CT intensities inside the eroded mask region.

    Uses the fixed-image eroded mask as the sampling region (fixed anatomy = reference).
    Returns None if the mask region is too small (<100 voxels after erosion).
    """
    eroded = erode_mask(mask_fixed, spacing_zyx, erosion_mm)
    if eroded.sum() < 100:
        return None

    region = eroded.astype(bool)
    f = vol_fixed[region].astype(np.float64)
    m = vol_moving[region].astype(np.float64)

    f -= f.mean();  f_std = f.std()
    m -= m.mean();  m_std = m.std()

    if f_std < 1e-6 or m_std < 1e-6:
        return None

    return float(np.mean((f / f_std) * (m / m_std)))


def centroid_displacement_mm(mask_fixed: np.ndarray,
                             mask_moving: np.ndarray,
                             spacing_zyx: Tuple[float, float, float]) -> Optional[float]:
    """
    Euclidean distance between organ centroids in mm.

    Uses full (non-eroded) masks — centroid averages over the whole volume
    so it is naturally insensitive to boundary noise.
    Returns None if either mask is empty.
    """
    if mask_fixed.sum() == 0 or mask_moving.sum() == 0:
        return None

    def centroid(mask):
        coords = np.argwhere(mask)          # (N, 3) in (z, y, x) voxel indices
        return coords.mean(axis=0)          # (z, y, x) voxel centroid

    cf = centroid(mask_fixed)
    cm = centroid(mask_moving)

    # Convert voxel distance to mm using per-axis spacing
    diff_mm = (cf - cm) * np.array(spacing_zyx)
    return float(np.linalg.norm(diff_mm))


# ---------------------------------------------------------------------------
# NEW METRICS (added for the ablation/baseline comparison)
# ---------------------------------------------------------------------------

def iqr(series) -> float:
    """Interquartile range (Q3 - Q1). NaN-safe; used for `median ± IQR` reporting."""
    a = np.asarray(series, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    q1, q3 = np.percentile(a, [25, 75])
    return float(q3 - q1)


def hd95_mm(mask_fixed: np.ndarray,
            mask_moving: np.ndarray,
            spacing_zyx: Tuple[float, float, float]) -> Optional[float]:
    """
    Symmetric 95th-percentile Hausdorff distance in mm.

    Replaces one-sided mean surface distance (MSD) as the boundary metric.
    Robust to a small number of surface outliers, which the raw HD is not.

    NOTE on sampling: the masks come from sitk.GetArrayFromImage → axis order
    is (Z, Y, X), and `spacing_zyx` is already (z, y, x). `distance_transform_edt`
    `sampling` must therefore be `spacing_zyx` *un-reversed* (matching
    mean_surface_distance_mm above). Passing spacing_zyx[::-1] would silently
    swap the Z and X physical scales.
    """
    from scipy.ndimage import binary_erosion

    mf = mask_fixed.astype(bool)
    mm = mask_moving.astype(bool)
    if mf.sum() == 0 or mm.sum() == 0:
        return None

    # Surface = voxels in mask but not in its erosion
    surf_f = mf & ~binary_erosion(mf)
    surf_m = mm & ~binary_erosion(mm)
    if surf_f.sum() == 0 or surf_m.sum() == 0:
        return None

    # Distance (mm) from every voxel to the nearest fixed / moving surface
    dt_f = distance_transform_edt(~surf_f, sampling=spacing_zyx)
    dt_m = distance_transform_edt(~surf_m, sampling=spacing_zyx)

    d_m_to_f = dt_f[surf_m]   # moving-surface → fixed-surface
    d_f_to_m = dt_m[surf_f]   # fixed-surface  → moving-surface

    both = np.concatenate([d_m_to_f, d_f_to_m])
    return float(np.percentile(both, 95))


def sobel_ncc(vol_fixed: np.ndarray,
              vol_moving: np.ndarray,
              mask_fixed: np.ndarray,
              spacing_zyx: Tuple[float, float, float],
              erosion_mm: float) -> Optional[float]:
    """
    |∇HU|-NCC : NCC of gradient-magnitude images inside the eroded organ region.

    Phase-invariant image-similarity metric. Plain intensity NCC is invalid
    across contrast phases (the same anatomy has different HU once contrast is
    injected). Edge *location*, however, is largely phase-stable, so correlating
    |∇HU| measures true tissue alignment without being fooled by enhancement.

    This is the same signal Pass 2 optimises, exposed here as an eval metric.
    """
    from scipy.ndimage import sobel

    vf = vol_fixed.astype(np.float64)
    vm = vol_moving.astype(np.float64)

    grad_f = np.sqrt(sum(sobel(vf, axis=i) ** 2 for i in range(3)))
    grad_m = np.sqrt(sum(sobel(vm, axis=i) ** 2 for i in range(3)))

    return ncc_within_mask(grad_f, grad_m, mask_fixed, spacing_zyx, erosion_mm)


def grad_magnitude(vol_np: np.ndarray) -> np.ndarray:
    """Whole-volume |∇HU| (Sobel gradient magnitude), computed once per phase."""
    from scipy.ndimage import sobel
    v = vol_np.astype(np.float64)
    return np.sqrt(sum(sobel(v, axis=i) ** 2 for i in range(3)))


def neg_jacobian_pct(dvf_path: str,
                     displacement_in_mm: bool = True,
                     body_mask: Optional[np.ndarray] = None) -> Optional[float]:
    """
    Study-level folding metric: percentage of voxels with det(J) <= 0.

    J = I + ∂u/∂x for the transformation φ(x) = x + u(x). A non-positive
    determinant means the deformation folds space onto itself (non-invertible),
    which is anatomically impossible and a standard red flag for deformable
    methods. Rigid/affine transforms have a constant positive determinant, so
    this is reported only for deformable conditions (mark "—" for rigid).

    Args:
        dvf_path           : NIfTI displacement field. Accepts (Z,Y,X,3),
                             (3,Z,Y,X), or SimpleITK-style (Z,Y,X,1,3).
        displacement_in_mm : True if the field stores physical-mm displacements
                             (SimpleITK DisplacementFieldTransform, ANTs warps).
                             False if displacements are in voxels (some learned
                             methods). When True we divide the spatial gradient
                             by voxel spacing so ∂u/∂x is dimensionless.
        body_mask          : optional (Z,Y,X) bool mask to restrict the metric
                             to the body region (recommended — air folds are
                             meaningless). If None, the whole field is used.

    Returns:
        Percentage in [0, 100], or None if the field can't be parsed.
    """
    import nibabel as nib

    nii = nib.load(dvf_path)
    dvf = np.asarray(nii.get_fdata(), dtype=np.float64)
    dvf = np.squeeze(dvf)

    # Normalise to (Z, Y, X, 3): find the size-3 component axis and move it last
    comp_axes = [ax for ax, n in enumerate(dvf.shape) if n == 3]
    if not comp_axes:
        return None
    comp_axis = comp_axes[-1] if dvf.shape[-1] == 3 else comp_axes[0]
    dvf = np.moveaxis(dvf, comp_axis, -1)            # (Z, Y, X, 3)
    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        return None

    # Voxel spacing (z, y, x) from the affine zooms (nibabel zooms are x,y,z)
    zooms = nii.header.get_zooms()[:3]
    spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    # ∂u_c/∂axis_k  via central differences along the 3 spatial axes
    # Build the 3x3 Jacobian of u at every voxel, then add identity.
    jac = np.empty(dvf.shape[:3] + (3, 3), dtype=np.float64)
    for c in range(3):                                # component of displacement
        grads = np.gradient(dvf[..., c], *spacing_zyx, edge_order=1)
        for k in range(3):                            # spatial axis
            g = grads[k]
            if not displacement_in_mm:
                # voxel displacement → multiply by spacing to get mm, but since
                # ∂(vox)/∂(vox) is already dimensionless we skip the spacing div.
                g = np.gradient(dvf[..., c], axis=k, edge_order=1)
            jac[..., c, k] = g
    # Add identity (φ = x + u)
    for d in range(3):
        jac[..., d, d] += 1.0

    det = np.linalg.det(jac)

    if body_mask is not None:
        bm = body_mask.astype(bool)
        if bm.shape != det.shape:
            warnings.warn("body_mask shape != DVF shape; ignoring mask")
        else:
            det = det[bm]

    det = det[np.isfinite(det)]
    if det.size == 0:
        return None
    return float(100.0 * (det <= 0).mean())


# ===========================================================================
# PER-ORGAN EVALUATION
# ===========================================================================

def evaluate_organ_pair(
        label: int,
        organ_name: str,
        seg_fixed_np: np.ndarray,
        seg_moving_np: np.ndarray,
        vol_fixed_np: np.ndarray,
        vol_moving_np: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
        erosion_mm: float,
        grad_fixed_np: Optional[np.ndarray] = None,
        grad_moving_np: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    Compute all per-organ metrics for one organ label.

    grad_fixed_np / grad_moving_np : optional precomputed |∇HU| volumes
        (whole-volume gradient magnitude). If provided, the Sobel-NCC is
        computed from them instead of re-running sobel() per organ.

    Returns None if the organ is absent in either mask (skip silently).
    """
    mask_f = (seg_fixed_np  == label).astype(np.uint8)
    mask_m = (seg_moving_np == label).astype(np.uint8)

    # Require at least 200 voxels in both masks to be meaningful
    if mask_f.sum() < 200 or mask_m.sum() < 200:
        return None

    dice = eroded_dice(mask_f, mask_m, spacing_zyx, erosion_mm)

    hd95 = hd95_mm(mask_f, mask_m, spacing_zyx)

    # |∇HU|-NCC: use precomputed gradient volumes when available
    if grad_fixed_np is not None and grad_moving_np is not None:
        sncc = ncc_within_mask(grad_fixed_np, grad_moving_np,
                               mask_f, spacing_zyx, erosion_mm)
    else:
        sncc = sobel_ncc(vol_fixed_np, vol_moving_np,
                         mask_f, spacing_zyx, erosion_mm)

    ncc = None
    if label in NCC_ORGANS:
        ncc = ncc_within_mask(
            vol_fixed_np, vol_moving_np, mask_f, spacing_zyx, erosion_mm
        )

    centroid = centroid_displacement_mm(mask_f, mask_m, spacing_zyx)

    return {
        "label":        label,
        "organ":        organ_name,
        "eroded_dice":  dice,
        "hd95_mm":      hd95,
        "sobel_ncc":    sncc,
        "ncc":          ncc,          # legacy intensity NCC (excluded from summary)
        "centroid_mm":  centroid,
    }


# ===========================================================================
# STUDY-LEVEL EVALUATION
# ===========================================================================

def find_series_files(
        study_dir: str,
        study_id: str,
        series_id: str,
        seg_postfix: str,
        vol_postfix: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate seg and volume files for one series.
    Tries {study_id}_{series_id}{postfix} inside study_dir.
    """
    base = os.path.join(study_dir, f"{study_id}_{series_id}")
    seg_path = base + seg_postfix
    vol_path = base + vol_postfix

    seg_path = seg_path if os.path.exists(seg_path) else None
    vol_path = vol_path if os.path.exists(vol_path) else None
    return seg_path, vol_path


def evaluate_study(
        study_id: str,
        study_dir: str,
        labels_df: pd.DataFrame,
        seg_postfix: str,
        vol_postfix: str,
        ref_phase: str = "Non-contrast",
        erosion_mm: float = 6.0,
        organ_labels: Dict[int, str] = ORGAN_LABELS,
) -> List[dict]:
    """
    Evaluate registration quality for all non-reference phases in one study.

    Args:
        study_id    : StudyInstanceUID
        study_dir   : directory containing this study's files
        labels_df   : labels CSV as DataFrame
        seg_postfix : e.g. '_registered_seg.nii.gz' or '_aligned_seg_reg.nii.gz'
        vol_postfix : e.g. '_registered.nii.gz'    or '_aligned.nii.gz'
        ref_phase   : label of the fixed/reference series
        erosion_mm  : erosion radius applied before Dice and NCC
        organ_labels: {label_int: name} to evaluate

    Returns:
        List of per-organ-per-phase result dicts.
    """
    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    results: List[dict] = []

    # ── Find reference series ─────────────────────────────────────────────
    ref_rows = study_rows[study_rows["Label"] == ref_phase]
    if ref_rows.empty:
        print(f"  ⚠  No '{ref_phase}' phase for {study_id[:40]}... — skip")
        return results

    ref_sid = ref_rows.iloc[0]["SeriesInstanceUID"]
    ref_seg_path, ref_vol_path = find_series_files(
        study_dir, study_id, ref_sid, seg_postfix, vol_postfix
    )

    if ref_seg_path is None or ref_vol_path is None:
        print(f"  ⚠  Reference files missing for {study_id[:40]}... — skip")
        return results

    ref_seg_sitk = sitk.ReadImage(ref_seg_path)
    ref_vol_sitk = sitk.ReadImage(ref_vol_path)
    ref_seg_np   = np.round(sitk.GetArrayFromImage(ref_seg_sitk)).astype(np.int32)
    ref_vol_np   = sitk.GetArrayFromImage(ref_vol_sitk).astype(np.float32)
    spacing_zyx  = get_spacing_zyx(ref_vol_sitk)
    ref_grad_np  = grad_magnitude(ref_vol_np)   # |∇HU| of fixed, reused per organ

    # ── Evaluate each moving phase ────────────────────────────────────────
    for _, row in study_rows.iterrows():
        sid   = row["SeriesInstanceUID"]
        phase = row["Label"]

        if phase == ref_phase:
            continue

        seg_path, vol_path = find_series_files(
            study_dir, study_id, sid, seg_postfix, vol_postfix
        )

        if seg_path is None or vol_path is None:
            print(f"  ⚠  [{phase}] files missing — skip")
            continue

        mov_seg_np = np.round(
            sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        ).astype(np.int32)
        mov_vol_np = sitk.GetArrayFromImage(
            sitk.ReadImage(vol_path)
        ).astype(np.float32)

        # Shape sanity check
        if mov_seg_np.shape != ref_seg_np.shape:
            print(f"  ⚠  [{phase}] shape mismatch "
                  f"{mov_seg_np.shape} vs {ref_seg_np.shape} — skip")
            continue

        print(f"  Evaluating [{phase}]...")

        mov_grad_np = grad_magnitude(mov_vol_np)   # |∇HU| of this moving phase

        for label, organ_name in organ_labels.items():
            res = evaluate_organ_pair(
                label, organ_name,
                ref_seg_np, mov_seg_np,
                ref_vol_np, mov_vol_np,
                spacing_zyx, erosion_mm,
                grad_fixed_np=ref_grad_np,
                grad_moving_np=mov_grad_np,
            )
            if res is None:
                continue

            results.append({
                "study_id":    study_id,
                "phase":       phase,
                "seg_postfix": seg_postfix,
                **res,
            })

    return results


# ===========================================================================
# BATCH EVALUATION
# ===========================================================================

def evaluate_all(
        base_dir: str,
        labels_csv: str,
        seg_postfix: str,
        vol_postfix: str,
        ref_phase: str = "Non-contrast",
        erosion_mm: float = 6.0,
        out_csv: Optional[str] = None,
        organ_labels: Dict[int, str] = ORGAN_LABELS,
) -> pd.DataFrame:
    """
    Run evaluation over all study subdirectories in base_dir.

    Args:
        base_dir    : directory with one subfolder per study
        labels_csv  : path to labels CSV
        seg_postfix : filename postfix for segmentation files
        vol_postfix : filename postfix for CT volume files
        ref_phase   : fixed/reference phase label
        erosion_mm  : erosion radius in mm
        out_csv     : if given, save results DataFrame here
        organ_labels: organ label dict

    Returns:
        DataFrame with one row per (study, phase, organ).
    """
    labels_df = pd.read_csv(labels_csv)
    study_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION")
    print(f"  base_dir    : {base_dir}")
    print(f"  seg_postfix : {seg_postfix}")
    print(f"  vol_postfix : {vol_postfix}")
    print(f"  ref_phase   : {ref_phase}")
    print(f"  erosion_mm  : {erosion_mm}")
    print(f"  studies     : {len(study_dirs)}")
    print(f"{'='*80}")

    all_results: List[dict] = []

    for idx, study_id in enumerate(study_dirs, 1):
        study_dir = os.path.join(base_dir, study_id)
        print(f"\n[{idx}/{len(study_dirs)}] {study_id[:55]}...")

        try:
            rows = evaluate_study(
                study_id=study_id,
                study_dir=study_dir,
                labels_df=labels_df,
                seg_postfix=seg_postfix,
                vol_postfix=vol_postfix,
                ref_phase=ref_phase,
                erosion_mm=erosion_mm,
                organ_labels=organ_labels,
            )
            all_results.extend(rows)
            print(f"  ✓ {len(rows)} organ-phase pairs evaluated")
        except Exception as e:
            import traceback
            print(f"  ✗ Error: {e}")
            traceback.print_exc()

    df = pd.DataFrame(all_results)

    if df.empty:
        print("\n⚠  No results collected — check postfixes and directory structure.")
        return df

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY  (median ± IQR across studies, per organ)")
    print(f"{'='*80}")

    metric_cols = ["eroded_dice", "hd95_mm", "sobel_ncc", "centroid_mm"]
    summary = (
        df.groupby(["organ", "phase"])[metric_cols]
        .agg(["median", iqr])
        .round(4)
    )
    print(summary.to_string())

    def med_iqr(col, fmt):
        s = df[col]
        return f"{s.median():{fmt}} ± {iqr(s):{fmt}}"

    print(f"\nOverall (all organs, all phases, median ± IQR):")
    print(f"  Eroded Dice  : {med_iqr('eroded_dice', '.3f')}")
    print(f"  HD95 (mm)    : {med_iqr('hd95_mm', '.1f')}")
    print(f"  |∇HU|-NCC    : {med_iqr('sobel_ncc', '.3f')}")
    print(f"  Centroid (mm): {med_iqr('centroid_mm', '.1f')}")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\n✓ Results saved → {out_csv}")

        # Also save per-organ summary
        summary_path = out_csv.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path)
        print(f"✓ Summary saved → {summary_path}")

    return df


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate registration quality at any pipeline stage."
    )

    p.add_argument("--all", action="store_true",
                   help="Batch over all study subdirs in --base_dir")
    p.add_argument("--base_dir",  default=None,
                   help="Root dir with one subdir per study (batch mode)")
    p.add_argument("--study_dir", default=None,
                   help="Single study directory (single-study mode)")
    p.add_argument("--study_id",  default=None,
                   help="Study ID (required in single-study mode)")
    p.add_argument("--labels_csv", default="../ncct_cect/vindr_ds/labels.csv",
                   help="Path to labels.csv")

    # ── Postfixes — the key parameters that change per pipeline stage ──
    p.add_argument("--seg_postfix", required=True,
                   help=(
                       "Filename postfix for segmentation files.\n"
                       "Examples:\n"
                       "  after alignment  : _aligned_seg_reg.nii.gz\n"
                       "  after alignment  : _aligned_seg_full.nii.gz\n"
                       "  after rigid reg  : _registered_seg.nii.gz\n"
                       "  after deformable : _deformable_seg.nii.gz"
                   ))
    p.add_argument("--vol_postfix", required=True,
                   help=(
                       "Filename postfix for CT volume files.\n"
                       "Examples:\n"
                       "  after alignment  : _aligned.nii.gz\n"
                       "  after rigid reg  : _registered.nii.gz\n"
                       "  after deformable : _deformable.nii.gz"
                   ))

    p.add_argument("--ref_phase",  default="Non-contrast",
                   help="Phase label used as fixed reference (default: Non-contrast)")
    p.add_argument("--erosion_mm", type=float, default=6.0,
                   help="Erosion radius in mm before Dice/NCC (default: 6.0)")
    p.add_argument("--out_csv",    default=None,
                   help="Path to save results CSV (optional)")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.all:
        if args.base_dir is None:
            raise ValueError("--base_dir is required with --all")
        evaluate_all(
            base_dir=args.base_dir,
            labels_csv=args.labels_csv,
            seg_postfix=args.seg_postfix,
            vol_postfix=args.vol_postfix,
            ref_phase=args.ref_phase,
            erosion_mm=args.erosion_mm,
            out_csv=args.out_csv,
        )

    else:
        if args.study_dir is None or args.study_id is None:
            raise ValueError("--study_dir and --study_id are required in single-study mode")

        labels_df = pd.read_csv(args.labels_csv)
        rows = evaluate_study(
            study_id=args.study_id,
            study_dir=args.study_dir,
            labels_df=labels_df,
            seg_postfix=args.seg_postfix,
            vol_postfix=args.vol_postfix,
            ref_phase=args.ref_phase,
            erosion_mm=args.erosion_mm,
        )
        df = pd.DataFrame(rows)
        print(f"\n{df.to_string(index=False)}")

        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"\n✓ Saved → {args.out_csv}")