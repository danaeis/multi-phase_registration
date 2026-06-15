"""
deformable_registration.py  —  B-spline Deformable Registration
================================================================
Applies after rigid registration (or directly on aligned volumes if
you want to measure the effect of skipping rigid).

Metric strategy
---------------
Same as rigid Pass 1: MMI inside the organ mask.  For deformable
registration MMI is preferred over gradient-NCC because:
  - B-spline has many more DOF than rigid; gradient-NCC can overfit
    to local texture differences between phases
  - MMI captures the full joint histogram relationship which is more
    stable over large deformation fields
  - The organ mask already excludes the HU-noisy regions (bowel, padding)

Mask strategy
-------------
Fixed mask only (NC seg_reg, most reliable).  The moving mask is the
fixed mask resampled into moving space + permissive dilation — same
rationale as rigid registration.  This prevents deforming the padding
regions at the image boundary.

Grid spacing
------------
Parametric.  Suggested starting points for 1.5mm isotropic volumes:
  - 40mm: coarse, good for large residual misalignment (>15mm centroid error)
  - 25mm: medium, good starting point after rigid (~10mm residual)
  - 15mm: fine, refinement after 25mm pass or when rigid was accurate

Input postfixes are parametric so this script can be run on:
  - aligned_volumes (skip rigid entirely)
  - rigid_registered/pass1 output
  - rigid_registered/pass2 output
  ...and results can be compared with the evaluator.

Input layout:
    {input_dir}/{study_id}/{study_id}_{series_id}{vol_postfix}
    {input_dir}/{study_id}/{study_id}_{series_id}{seg_reg_postfix}
    {input_dir}/{study_id}/{study_id}_{series_id}{seg_full_postfix}  (optional)

Output layout:
    {output_dir}/{study_id}/{study_id}_{series_id}_deformable.nii.gz
    {output_dir}/{study_id}/{study_id}_{series_id}_deformable_seg_reg.nii.gz
    {output_dir}/{study_id}/{study_id}_{series_id}_deformable_seg_full.nii.gz
    {output_dir}/{study_id}/{study_id}_{series_id}_deformable_field.nii.gz  (optional)

Usage
-----
python deformable_registration.py                     # single example study
python deformable_registration.py <study_id>          # specific study
python deformable_registration.py --all               # batch
python deformable_registration.py --all --skip        # skip existing
"""

from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from configs import MAIN_PATH

# ---------------------------------------------------------------------------
# Paths — edit these for each experiment
# ---------------------------------------------------------------------------
INPUT_DIR  = Path(MAIN_PATH + "rigid_registered")   # or aligned_volumes
OUTPUT_DIR = Path(MAIN_PATH + "deformable_registered")
LABELS_CSV = MAIN_PATH + "labels.csv"

# Input filename postfixes — change to match the pipeline stage you feed in
VOL_POSTFIX      = "_rigid2.nii.gz"          # CT volume
SEG_REG_POSTFIX  = "_rigid2_seg_reg.nii.gz"  # registration-quality seg
SEG_FULL_POSTFIX = "_rigid2_seg_full.nii.gz" # full seg (optional, propagated)

# Grid spacing options (mm) — run with different values and compare metrics
# 40mm: coarse  |  25mm: medium (recommended start)  |  15mm: fine
GRID_SPACING_MM  = 25.0

EXAMPLE_STUDY_ID = "1.2.840.113619.2.359.3.2831208971.108.1589585466.773"

# ---------------------------------------------------------------------------
# Organ label weights (label int = position in REG_ORGANS, 1-indexed)
# ---------------------------------------------------------------------------
ORGAN_WEIGHTS: Dict[int, float] = {
    1:  3.0,   # liver
    2:  2.0,   # spleen
    3:  2.0,   # kidney_left
    4:  2.0,   # kidney_right
    5:  0.5,   # pancreas
    6:  0.5,   # gallbladder
    7:  0.5,   # adrenal_gland_left
    8:  0.5,   # adrenal_gland_right
    9:  2.0,   # aorta
    10: 1.5,   # inferior_vena_cava
    11: 1.0,   # portal_vein_and_splenic_vein
    12: 1.0,   # iliac_artery_left
    13: 1.0,   # iliac_artery_right
    14: 1.5,   # autochthon_left
    15: 1.5,   # autochthon_right
    16: 1.0,   # iliopsoas_left
    17: 1.0,   # iliopsoas_right
    18: 1.5,   # spinal_cord
    19: 1.5,   # vertebrae_L1
    20: 1.5,   # vertebrae_L2
    21: 1.5,   # vertebrae_L3
    22: 1.5,   # vertebrae_L4
    23: 1.5,   # vertebrae_L5
    24: 1.5,   # vertebrae_S1
    25: 1.5,   # sacrum
}

LOW_WEIGHT_THRESHOLD = 0.6


# ===========================================================================
# FILE DISCOVERY
# ===========================================================================

def scan_input_directory(
        input_dir: str,
        vol_postfix: str,
        seg_reg_postfix: str,
        seg_full_postfix: str,
) -> Dict[str, Dict[str, dict]]:
    """
    Scan input directory for volumes matching the given postfixes.

    Returns:
        catalog[study_id][series_id] = {image, seg_reg, seg_full}
    """
    catalog: Dict[str, Dict[str, dict]] = {}

    if not os.path.exists(input_dir):
        print(f"  ❌ Input directory not found: {input_dir}")
        return catalog

    study_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d))]

    print(f"\n  Scanning {input_dir} ...")
    print(f"  vol_postfix     : {vol_postfix}")
    print(f"  seg_reg_postfix : {seg_reg_postfix}")
    print(f"  seg_full_postfix: {seg_full_postfix}")
    print(f"  Study dirs found: {len(study_dirs)}")

    for study_id in study_dirs:
        study_path = os.path.join(input_dir, study_id)
        catalog[study_id] = {}

        pattern = os.path.join(study_path, f"*{vol_postfix}")
        vol_files = glob.glob(pattern)

        for vol_path in vol_files:
            basename = os.path.basename(vol_path)
            prefix   = f"{study_id}_"
            if not basename.startswith(prefix):
                continue
            series_id = basename[len(prefix):].replace(vol_postfix, "")

            seg_reg_path  = vol_path.replace(vol_postfix, seg_reg_postfix)
            seg_full_path = vol_path.replace(vol_postfix, seg_full_postfix)

            if not os.path.exists(seg_reg_path):
                print(f"  ⚠  Missing seg_reg for {series_id[:35]}... — skip")
                continue

            catalog[study_id][series_id] = {
                "image":    vol_path,
                "seg_reg":  seg_reg_path,
                "seg_full": seg_full_path if os.path.exists(seg_full_path) else None,
            }

    total = sum(len(v) for v in catalog.values())
    print(f"  Total usable series: {total}")
    return catalog


# ===========================================================================
# MASK UTILITIES  (same logic as register_v2.py)
# ===========================================================================

def build_organ_mask(
        seg_reg_sitk: sitk.Image,
        organ_weights: Dict[int, float] = ORGAN_WEIGHTS,
        low_weight_threshold: float = LOW_WEIGHT_THRESHOLD,
        primary_dilation_mm: float = 8.0,
        low_weight_dilation_mm: float = 4.0,
) -> sitk.Image:
    """
    Binary registration mask from seg_reg.

    High-weight organs (>= threshold): dilated by primary_dilation_mm.
    Low-weight organs (< threshold):   dilated by low_weight_dilation_mm.
    Unknown labels: excluded.
    """
    seg_np  = np.round(sitk.GetArrayFromImage(seg_reg_sitk)).astype(np.int32)
    spacing = seg_reg_sitk.GetSpacing()

    high_mask = np.zeros_like(seg_np, dtype=np.uint8)
    low_mask  = np.zeros_like(seg_np, dtype=np.uint8)

    for label, weight in organ_weights.items():
        region = (seg_np == label).astype(np.uint8)
        if region.sum() < 50:
            continue
        if weight >= low_weight_threshold:
            high_mask |= region
        else:
            low_mask  |= region

    def _dilate(arr, dilation_mm):
        if arr.sum() == 0:
            return arr
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        radius = [max(1, int(round(dilation_mm / s))) for s in spacing]
        return sitk.GetArrayFromImage(sitk.BinaryDilate(img, radius))

    combined = np.clip(
        _dilate(high_mask, primary_dilation_mm) +
        _dilate(low_mask,  low_weight_dilation_mm),
        0, 1
    ).astype(np.uint8)

    combined_sitk = sitk.GetImageFromArray(combined)
    combined_sitk.CopyInformation(seg_reg_sitk)
    combined_sitk = sitk.BinaryFillhole(combined_sitk)

    vox_count = int(sitk.GetArrayFromImage(combined_sitk).sum())
    print(f"  Organ mask: {vox_count:,} voxels "
          f"({vox_count / seg_np.size * 100:.1f}% of volume)")
    return combined_sitk


def resample_mask_to_moving(
        fixed_mask: sitk.Image,
        moving_img: sitk.Image,
        extra_dilation_mm: float = 12.0,
) -> sitk.Image:
    """
    Derive a permissive moving mask by resampling the fixed mask.

    Extra dilation is larger than for rigid (12mm vs 10mm) because
    deformable registration starts with residual misalignment from
    the rigid step (or full misalignment if rigid is skipped).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    mov_mask = resampler.Execute(fixed_mask)

    if extra_dilation_mm > 0:
        spacing = moving_img.GetSpacing()
        radius  = [max(1, int(round(extra_dilation_mm / s))) for s in spacing]
        mov_mask = sitk.BinaryDilate(mov_mask, radius)

    return sitk.Cast(mov_mask, sitk.sitkUInt8)


# ===========================================================================
# B-SPLINE REGISTRATION
# ===========================================================================

def setup_bspline(
        fixed_img: sitk.Image,
        fixed_mask: Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
        grid_spacing_mm: float,
        num_iterations: int,
        sampling_pct: float,
) -> Tuple[sitk.ImageRegistrationMethod, sitk.BSplineTransform]:
    """
    Configure B-spline registration method with MMI metric.

    Returns:
        (registration_method, initial_bspline_transform)
    """
    # Compute mesh size from physical grid spacing
    mesh_size = [
        max(1, int(round(sz * sp / grid_spacing_mm)))
        for sz, sp in zip(fixed_img.GetSize(), fixed_img.GetSpacing())
    ]
    print(f"  B-spline mesh size: {mesh_size}  (spacing={grid_spacing_mm}mm)")

    init_tx = sitk.BSplineTransformInitializer(
        fixed_img,
        transformDomainMeshSize=mesh_size,
        order=3,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct, seed=42)

    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
        print(f"  Fixed mask applied")
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
        print(f"  Moving mask applied")

    reg.SetInterpolator(sitk.sitkLinear)

    # LBFGSB converges faster than gradient descent for B-spline
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=num_iterations * 2,
        costFunctionConvergenceFactor=1e7,
    )

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg, init_tx


def run_deformable(
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
        grid_spacing_mm: float = 25.0,
        num_iterations: int = 150,
        sampling_pct: float = 0.15,
        final_interpolator: int = sitk.sitkLanczosWindowedSinc,
        save_field: bool = False,
        field_path: Optional[str] = None,
) -> Tuple[sitk.Image, sitk.Transform, float]:
    """
    Run B-spline deformable registration.

    Args:
        fixed / moving      : CT volumes (any pixel type, cast internally)
        fixed_mask          : binary organ mask for fixed image
        moving_mask         : permissive binary organ mask for moving image
        grid_spacing_mm     : B-spline control point spacing
        num_iterations      : max LBFGSB iterations
        sampling_pct        : fraction of masked voxels sampled per iteration
        final_interpolator  : interpolator for output volume resampling
        save_field          : whether to save the displacement field
        field_path          : path to save displacement field (if save_field)

    Returns:
        (registered_volume, transform, final_metric_value)
    """
    fixed_f32  = sitk.Cast(fixed,  sitk.sitkFloat32)
    moving_f32 = sitk.Cast(moving, sitk.sitkFloat32)

    reg, init_tx = setup_bspline(
        fixed_f32, fixed_mask, moving_mask,
        grid_spacing_mm, num_iterations, sampling_pct,
    )
    reg.SetInitialTransform(init_tx, inPlace=True)

    final_tx     = reg.Execute(fixed_f32, moving_f32)
    final_metric = reg.GetMetricValue()

    print(f"  Deformable done: metric={final_metric:.6f}  "
          f"iter={reg.GetOptimizerIteration()}  "
          f"stop={reg.GetOptimizerStopConditionDescription()}")

    registered = sitk.Resample(
        moving, fixed, final_tx,
        final_interpolator, -1000.0, moving.GetPixelID()
    )

    if save_field and field_path:
        field = sitk.TransformToDisplacementField(
            final_tx, sitk.sitkVectorFloat32,
            fixed.GetSize(), fixed.GetOrigin(),
            fixed.GetSpacing(), fixed.GetDirection(),
        )
        sitk.WriteImage(field, field_path)
        mb = os.path.getsize(field_path) / 1024 / 1024
        print(f"  ✓ Displacement field: {os.path.basename(field_path)} ({mb:.1f} MB)")

    return registered, final_tx, final_metric


# ===========================================================================
# SAVE HELPER
# ===========================================================================

def _save(img: sitk.Image, path: str, label: str) -> bool:
    try:
        sitk.WriteImage(img, path)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✓ {label}: {os.path.basename(path)} ({mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


def _apply_seg_transform(
        seg: sitk.Image,
        reference: sitk.Image,
        transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        seg, reference, transform,
        sitk.sitkNearestNeighbor, 0, seg.GetPixelID()
    )


# ===========================================================================
# PER-STUDY DEFORMABLE REGISTRATION
# ===========================================================================

def register_study_deformable(
        study_id: str,
        catalog: Dict[str, Dict[str, dict]],
        labels_df: pd.DataFrame,
        output_dir: str,
        grid_spacing_mm: float = GRID_SPACING_MM,
        num_iterations: int = 150,
        sampling_pct: float = 0.15,
        final_interpolator: int = sitk.sitkLanczosWindowedSinc,
        save_field: bool = True,
        skip_existing: bool = True,
) -> dict:
    """
    Deformable registration for all phases in one study.

    Args:
        study_id        : StudyInstanceUID
        catalog         : output of scan_input_directory()
        labels_df       : labels CSV
        output_dir      : root output directory
        grid_spacing_mm : B-spline control point spacing in mm
        num_iterations  : max LBFGSB iterations
        sampling_pct    : fraction of masked voxels sampled per iteration
        final_interpolator : resampling interpolator for saved volumes
        save_field      : save displacement field
        skip_existing   : skip series whose deformable output already exists

    Returns:
        result dict with status, successful, failed, skipped
    """
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

    print(f"\n{'='*80}")
    print(f"Deformable Registration: {study_id}")
    print(f"  grid_spacing_mm : {grid_spacing_mm}")
    print(f"{'='*80}")

    # ── Load NC reference ─────────────────────────────────────────────────
    nc_files   = catalog[study_id][nc_sid]
    nc_img     = sitk.ReadImage(nc_files["image"])
    nc_seg_reg = sitk.ReadImage(nc_files["seg_reg"])
    nc_seg_full = (sitk.ReadImage(nc_files["seg_full"])
                   if nc_files["seg_full"] else None)

    print(f"\n  Building fixed organ mask from NC seg_reg...")
    fixed_mask = build_organ_mask(nc_seg_reg)

    # Save NC reference (copy — no transform applied to reference)
    nc_prefix = os.path.join(study_out, f"{study_id}_{nc_sid}")
    _save(nc_img,     f"{nc_prefix}_deformable.nii.gz",          "NC vol (ref)")
    _save(nc_seg_reg, f"{nc_prefix}_deformable_seg_reg.nii.gz",  "NC seg_reg (ref)")
    if nc_seg_full:
        _save(nc_seg_full, f"{nc_prefix}_deformable_seg_full.nii.gz",
              "NC seg_full (ref)")

    results = {"status": "complete", "study_id": study_id,
               "successful": [], "failed": [], "skipped": []}

    # ── Process each moving series ────────────────────────────────────────
    for _, row in study_rows.iterrows():
        sid   = row["SeriesInstanceUID"]
        phase = row["Label"]

        if sid == nc_sid:
            continue
        if sid not in catalog[study_id]:
            print(f"\n  ⚠  [{phase}] not in catalog — skip")
            results["failed"].append({"series": sid, "phase": phase,
                                      "reason": "not_in_catalog"})
            continue

        prefix    = os.path.join(study_out, f"{study_id}_{sid}")
        out_vol   = f"{prefix}_deformable.nii.gz"
        out_field = f"{prefix}_deformable_field.nii.gz"

        if skip_existing and os.path.exists(out_vol):
            print(f"\n  ⏭  [{phase}] already processed — skip")
            results["skipped"].append(sid)
            continue

        files = catalog[study_id][sid]
        print(f"\n  {'─'*60}")
        print(f"  [{phase}]  {sid[:50]}...")

        try:
            mov_img      = sitk.ReadImage(files["image"])
            mov_seg_reg  = sitk.ReadImage(files["seg_reg"])
            mov_seg_full = (sitk.ReadImage(files["seg_full"])
                            if files["seg_full"] else None)

            # Moving mask: fixed mask resampled + extra dilation
            print(f"  Building moving mask (resampled fixed + 12mm dilation)...")
            moving_mask = resample_mask_to_moving(
                fixed_mask, mov_img, extra_dilation_mm=12.0
            )

            # ── Deformable registration ───────────────────────────────────
            reg_vol, tx, metric = run_deformable(
                nc_img, mov_img,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                grid_spacing_mm=grid_spacing_mm,
                num_iterations=num_iterations,
                sampling_pct=sampling_pct,
                final_interpolator=final_interpolator,
                save_field=save_field,
                field_path=out_field if save_field else None,
            )

            # Save volume
            _save(reg_vol, out_vol, f"[{phase}] deformable volume")

            # Save propagated seg_reg
            reg_seg_reg = _apply_seg_transform(mov_seg_reg, nc_img, tx)
            _save(reg_seg_reg, f"{prefix}_deformable_seg_reg.nii.gz",
                  f"[{phase}] deformable seg_reg")

            # Save propagated seg_full
            if mov_seg_full:
                reg_seg_full = _apply_seg_transform(mov_seg_full, nc_img, tx)
                _save(reg_seg_full, f"{prefix}_deformable_seg_full.nii.gz",
                      f"[{phase}] deformable seg_full")

            results["successful"].append({
                "series": sid, "phase": phase, "metric": metric,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ [{phase}] error: {e}")
            results["failed"].append({"series": sid, "phase": phase,
                                      "reason": str(e)})

    print(f"\n  {'='*60}")
    print(f"  Study done: ✓{len(results['successful'])}  "
          f"⏭{len(results['skipped'])}  ✗{len(results['failed'])}")
    return results


# ===========================================================================
# BATCH
# ===========================================================================

def register_all_studies_deformable(
        input_dir: str,
        output_dir: str,
        labels_csv: str,
        vol_postfix: str = VOL_POSTFIX,
        seg_reg_postfix: str = SEG_REG_POSTFIX,
        seg_full_postfix: str = SEG_FULL_POSTFIX,
        grid_spacing_mm: float = GRID_SPACING_MM,
        num_iterations: int = 150,
        sampling_pct: float = 0.15,
        save_field: bool = True,
        skip_existing: bool = True,
):
    labels_df = pd.read_csv(labels_csv)
    catalog   = scan_input_directory(
        input_dir, vol_postfix, seg_reg_postfix, seg_full_postfix
    )

    if not catalog:
        print("❌ No studies found.")
        return

    valid = sorted(set(catalog.keys()) & set(labels_df["StudyInstanceUID"].unique()))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"BATCH DEFORMABLE REGISTRATION")
    print(f"  input           : {input_dir}")
    print(f"  output          : {output_dir}")
    print(f"  vol_postfix     : {vol_postfix}")
    print(f"  seg_reg_postfix : {seg_reg_postfix}")
    print(f"  grid_spacing_mm : {grid_spacing_mm}")
    print(f"  studies         : {len(valid)}")
    print(f"{'='*80}")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for idx, study_id in enumerate(valid, 1):
        print(f"\n[{idx}/{len(valid)}] {study_id[:55]}...")
        try:
            res = register_study_deformable(
                study_id=study_id,
                catalog=catalog,
                labels_df=labels_df,
                output_dir=output_dir,
                grid_spacing_mm=grid_spacing_mm,
                num_iterations=num_iterations,
                sampling_pct=sampling_pct,
                save_field=save_field,
                skip_existing=skip_existing,
            )
            if res["status"] == "complete" and res["successful"]:
                counts["ok"] += 1
            elif res["status"] == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Study failed: {e}")
            counts["failed"] += 1

    print(f"\n{'='*80}")
    print(f"DONE  ✓{counts['ok']}  ⏭{counts['skipped']}  ✗{counts['failed']}"
          f"  / {len(valid)}")
    print(f"{'='*80}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    skip = "--skip" in args

    if "--all" in args:
        register_all_studies_deformable(
            input_dir=str(INPUT_DIR),
            output_dir=str(OUTPUT_DIR),
            labels_csv=LABELS_CSV,
            vol_postfix=VOL_POSTFIX,
            seg_reg_postfix=SEG_REG_POSTFIX,
            seg_full_postfix=SEG_FULL_POSTFIX,
            grid_spacing_mm=GRID_SPACING_MM,
            skip_existing=skip,
        )
    else:
        sid = next((a for a in args if not a.startswith("--")), EXAMPLE_STUDY_ID)
        labels_df = pd.read_csv(LABELS_CSV)
        catalog   = scan_input_directory(
            str(INPUT_DIR), VOL_POSTFIX, SEG_REG_POSTFIX, SEG_FULL_POSTFIX
        )
        register_study_deformable(
            study_id=sid,
            catalog=catalog,
            labels_df=labels_df,
            output_dir=str(OUTPUT_DIR),
            grid_spacing_mm=GRID_SPACING_MM,
            skip_existing=skip,
        )