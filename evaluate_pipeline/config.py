"""
config.py
=========
Single source of truth for the registration experiment comparison.

Every condition (ablation A1-A6, baseline B1-B4) is described by:
    base_dir     : directory containing one subfolder per study
    vol_postfix  : filename postfix for the warped CT volume
    seg_postfix  : filename postfix for the warped seg_reg mask
    is_deformable: whether %|J|<0 (folding) is meaningful (rigid -> "—")
    dvf_postfix  : filename postfix for the displacement field (deformable only)

File-naming convention for EVERY condition (matches the tested
evaluate_registration.find_series_files):

    {base_dir}/{study_id}/{study_id}_{series_id}{vol_postfix}
    {base_dir}/{study_id}/{study_id}_{series_id}{seg_postfix}
    {base_dir}/{study_id}/{study_id}_{series_id}{dvf_postfix}   (deformable)

NOTE: this deliberately keeps the postfix scheme used by register_fixed.py
instead of the plan's "_warped" sketch, so the existing file-discovery and
evaluation code paths work unchanged for the new baselines too. Baseline
runners must therefore write `{study}_{series}_deeds.nii.gz` etc., not
`{series}_warped.nii.gz`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Root path. Prefer the project's own configs.MAIN_PATH so this stays in sync
# with register_fixed.py / align_data_v3.py. Fall back to an env var.
# ---------------------------------------------------------------------------
try:
    from configs import MAIN_PATH                      # type: ignore
except Exception:                                       # pragma: no cover
    MAIN_PATH = os.environ.get("MAIN_PATH", "../../ncct_cect/vindr_ds/")
if not MAIN_PATH.endswith("/"):
    MAIN_PATH += "/"

MAIN = Path(MAIN_PATH)

# Stage directories (mirror register_fixed.py / align_data_v3.py constants)
BASELINE_CROP_DIR  = MAIN / "baseline_volumes"                 # A1: crop only
ALIGNED_CROP_DIR   = MAIN / "aligned_volumes"                  # A2: + z-align
RIGID_ALIGNED_DIR  = MAIN / "fixed_aligned_rigid_registered"   # A3/A4/A5
RESULTS_DIR        = MAIN / "experiments" / "results"          # A6 + all baselines


RIGID_BASELINE_DIR = MAIN / "fixed_baseline_rigid_registered"


LABELS_CSV = MAIN / "labels.csv"
REF_PHASE  = "Non-contrast"
EROSION_MM = 6.0




@dataclass(frozen=True)
class Condition:
    tag:           str
    base_dir:      Path
    vol_postfix:   str
    seg_postfix:   str
    is_deformable: bool = False
    dvf_postfix:   Optional[str] = None   # only for deformable conditions

    @property
    def detail_csv(self) -> Path:
        return RESULTS_DIR / self.tag / "eval_detail.csv"

    @property
    def summary_csv(self) -> Path:
        return RESULTS_DIR / self.tag / "eval_summary.csv"


def _C(tag, base, vp, sp, deformable=False, dvf=None) -> Condition:
    return Condition(tag, base, vp, sp, deformable, dvf)


# ---------------------------------------------------------------------------
# Condition registry. Order defines the order of rows in the final table.
# A1/A2 read from the crop dirs; A3-A5 from the rigid-registered dir
# (register_fixed.py already emits _rigid0/_rigid1/_rigid2). A6 + baselines
# read from results/{tag}/ written by their respective runners.
# ---------------------------------------------------------------------------
CONDITIONS: Dict[str, Condition] = {c.tag: c for c in [
    #    tag             base_dir            vol_postfix              seg_postfix
    _C("A1_crop_only",  BASELINE_CROP_DIR,  "_baseline.nii.gz",      "_baseline_seg_reg.nii.gz"),
    _C("A2_zalign",     ALIGNED_CROP_DIR,   "_aligned.nii.gz",       "_aligned_seg_reg.nii.gz"),
    _C("A3_pass0",      RIGID_ALIGNED_DIR,  "_rigid0.nii.gz",        "_rigid0_seg_reg.nii.gz"),
    _C("A4_pass01",     RIGID_ALIGNED_DIR,  "_rigid1.nii.gz",        "_rigid1_seg_reg.nii.gz"),
    _C("A5_pass012",    RIGID_ALIGNED_DIR,  "_rigid2.nii.gz",        "_rigid2_seg_reg.nii.gz"),
    _C("A6_bspline",    RESULTS_DIR / "A6_bspline", "_bspline.nii.gz",
        "_bspline_seg_reg.nii.gz", deformable=True, dvf="_bspline_dvf.nii.gz"),
    # _C("B1_elastix",    RESULTS_DIR / "B1_elastix", "_elastix.nii.gz",
    #     "_elastix_seg_reg.nii.gz"),
    # _C("B2_deeds",      RESULTS_DIR / "B2_deeds",   "_deeds.nii.gz",
    #     "_deeds_seg_reg.nii.gz",   deformable=True, dvf="_deeds_dvf.nii.gz"),
    # _C("B3_ants",       RESULTS_DIR / "B3_ants",    "_ants.nii.gz",
    #     "_ants_seg_reg.nii.gz",    deformable=True, dvf="_ants_dvf.nii.gz"),
    # _C("B4_vxm",        RESULTS_DIR / "B4_vxm",     "_vxm.nii.gz",
    #     "_vxm_seg_reg.nii.gz",     deformable=True, dvf="_vxm_dvf.nii.gz"),

    _C("A3b_pass0_noalign",  RIGID_BASELINE_DIR, "_rigid0.nii.gz", "_rigid0_seg_reg.nii.gz"),
    _C("A4b_pass01_noalign", RIGID_BASELINE_DIR, "_rigid1.nii.gz", "_rigid1_seg_reg.nii.gz"),
    _C("A5b_pass012_noalign",    RIGID_BASELINE_DIR,  "_rigid2.nii.gz", "_rigid2_seg_reg.nii.gz"),

]}


# Pretty labels for the final table
ROW_LABELS: Dict[str, str] = {
    "A1_crop_only": "A1  Crop only",
    "A2_zalign":    "A2  + Z-align",
    "A3_pass0":     "A3  + Rigid P0 (Kabsch)",
    "A4_pass01":    "A4  + Rigid P0+1 (Nelder-Mead)",
    "A5_pass012":   "A5  + Rigid P0+1+2 (Sobel gate)",
    "A6_bspline":   "A6  Full (+ B-spline)",
    "B1_elastix":   "B1  Elastix rigid (MMI)",
    "B2_deeds":     "B2  DEEDS",
    "B3_ants":      "B3  ANTs-SyN",
    "B4_vxm":       "B4  VoxelMorph",
}