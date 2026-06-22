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
    MAIN_PATH = os.environ.get("MAIN_PATH", "../ncct_cect/vindr_ds/")
if not MAIN_PATH.endswith("/"):
    MAIN_PATH += "/"

MAIN = Path(MAIN_PATH)

# Stage directories (mirror register_fixed.py / align_data_v3.py constants)
BASELINE_CROP_DIR   = MAIN / "baseline_volumes"                  # A1: crop only
ALIGNED_CROP_DIR    = MAIN / "aligned_volumes"                   # A2: + z-align
RIGID_ALIGNED_DIR   = MAIN / "fixed_aligned_rigid_registered"    # A3/A4/A5
RIGID_BASELINE_DIR  = MAIN / "fixed_baseline_rigid_registered"   # A3b/A4b/A5b
RESULTS_DIR         = MAIN / "all_baseline_algorithms"           # A6 + all baselines

LABELS_CSV = MAIN / "labels.csv"
REF_PHASE  = "Non-contrast"
EROSION_MM = 6.0


@dataclass(frozen=True)
class InputSource:
    """A starting point for the comparison algorithms."""
    key:         str
    dir:         Path
    vol_postfix: str
    seg_postfix: str


# The two starting points the baselines are run from. Mirrors what you are
# already doing with register_fixed.py (running it on both aligned + baseline).
#   aligned  = z-aligned crops  (the full pipeline path; algos replace A3-A6)
#   baseline = crop-only        (no z-align; isolates the z-align contribution)
RAW_VOL_DIR = MAIN / "nifti_unprocessed_volumes"   # DICOM→NIfTI, no crop/align

INPUTS: Dict[str, InputSource] = {
    "aligned":  InputSource("aligned",  ALIGNED_CROP_DIR,
                            "_aligned.nii.gz",  "_aligned_seg_reg.nii.gz"),
    "baseline": InputSource("baseline", BASELINE_CROP_DIR,
                            "_baseline.nii.gz", "_baseline_seg_reg.nii.gz"),
    # raw: standardized NIfTIs with NO cropping and NO z-alignment.
    # Running the pipeline on this input isolates the contribution of A1+A2.
    "raw":      InputSource("raw",      RAW_VOL_DIR,
                            "_standardized.nii.gz", "_standardized_seg_reg.nii.gz"),
}


@dataclass(frozen=True)
class Condition:
    tag:           str
    base_dir:      Path
    vol_postfix:   str
    seg_postfix:   str
    is_deformable: bool = False
    dvf_postfix:   Optional[str] = None   # only for deformable conditions
    # For baseline algos: which input source they consumed (None for A-series)
    input_key:     Optional[str] = None

    @property
    def detail_csv(self) -> Path:
        return RESULTS_DIR / self.tag / "eval_detail.csv"

    @property
    def summary_csv(self) -> Path:
        return RESULTS_DIR / self.tag / "eval_summary.csv"

    @property
    def input_source(self) -> Optional[InputSource]:
        return INPUTS[self.input_key] if self.input_key else None


def _C(tag, base, vp, sp, deformable=False, dvf=None, input_key=None) -> Condition:
    return Condition(tag, base, vp, sp, deformable, dvf, input_key)


# ---------------------------------------------------------------------------
# A-series: the in-house pipeline ablation (fixed input per stage).
# A1/A2 read from the crop dirs; A3-A5 from the rigid-registered dir
# (register_fixed.py already emits _rigid0/_rigid1/_rigid2). A6 reads results/.
# ---------------------------------------------------------------------------
_A_SERIES = [
    # ── Aligned pipeline (full path) ──────────────────────────────────────
    _C("A1_crop_only",  BASELINE_CROP_DIR,  "_baseline.nii.gz",  "_baseline_seg_reg.nii.gz"),
    _C("A2_zalign",     ALIGNED_CROP_DIR,   "_aligned.nii.gz",   "_aligned_seg_reg.nii.gz"),
    _C("A3_pass0",      RIGID_ALIGNED_DIR,  "_rigid0.nii.gz",    "_rigid0_seg_reg.nii.gz"),
    _C("A4_pass01",     RIGID_ALIGNED_DIR,  "_rigid1.nii.gz",    "_rigid1_seg_reg.nii.gz"),
    _C("A5_pass012",    RIGID_ALIGNED_DIR,  "_rigid2.nii.gz",    "_rigid2_seg_reg.nii.gz"),
    _C("A6_bspline",    RESULTS_DIR / "A6_bspline", "_bspline.nii.gz",
        "_bspline_seg_reg.nii.gz", deformable=True, dvf="_bspline_dvf.nii.gz"),
    # ── Baseline pipeline (crop only, no z-align) — isolates z-align contribution
    _C("A3b_pass0_noalign",   RIGID_BASELINE_DIR, "_rigid0.nii.gz", "_rigid0_seg_reg.nii.gz"),
    _C("A4b_pass01_noalign",  RIGID_BASELINE_DIR, "_rigid1.nii.gz", "_rigid1_seg_reg.nii.gz"),
    _C("A5b_pass012_noalign", RIGID_BASELINE_DIR, "_rigid2.nii.gz", "_rigid2_seg_reg.nii.gz"),
    _C("A6_bspline_baseline", RESULTS_DIR / "A6_bspline_baseline", "_bspline.nii.gz",
        "_bspline_seg_reg.nii.gz", deformable=True, dvf="_bspline_dvf.nii.gz"),
]

# ---------------------------------------------------------------------------
# B-series: external comparison algorithms. Each is defined ONCE here and
# instantiated against BOTH input sources -> "{tag}__aligned" and
# "{tag}__baseline". Output lives in results/{tag}__{input}/.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineAlgo:
    tag:           str          # e.g. "B1_elastix"
    name:          str          # pretty label
    vol_postfix:   str
    seg_postfix:   str
    is_deformable: bool
    dvf_postfix:   Optional[str]
    # whether running on the "baseline" (crop-only) input is meaningful.
    # Rigid-only methods from crop-only input mostly test "can rigid recover
    # without z-align" — still informative, so default True.
    run_on_baseline: bool = True


BASELINE_ALGOS: Dict[str, BaselineAlgo] = {a.tag: a for a in [
    # ── Controlled rigid comparison ───────────────────────────────────────
    # Identical Euler3D model + centroid-seeded initialisation.
    # The ONLY variable is the optimizer objective.
    # All four are evaluated on the SAME ruler: centroid + eDice + Sobel-NCC.
    #
    #  R_organ : centroid + eroded Dice            (anatomy, needs seg at inference)
    #  R_sobel : Sobel-NCC only                    (phase-invariant, mask-free)
    #  R_full  : centroid + eDice + Sobel-NCC      (proposed method)
    #  R_mmi   : Mattes MI                         (standard intensity baseline)
    BaselineAlgo("R_organ", "Rigid · centroid+eDice",
                 "_rorgan.nii.gz", "_rorgan_seg_reg.nii.gz", False, None),
    BaselineAlgo("R_sobel", "Rigid · Sobel-NCC (mask-free)",
                 "_rsobel.nii.gz", "_rsobel_seg_reg.nii.gz", False, None),
    BaselineAlgo("R_full",  "Rigid · centroid+eDice+Sobel (proposed)",
                 "_rfull.nii.gz",  "_rfull_seg_reg.nii.gz",  False, None),
    BaselineAlgo("R_mmi",   "Rigid · MMI",
                 "_rmmi.nii.gz",   "_rmmi_seg_reg.nii.gz",   False, None),
    # ── Off-the-shelf SOTA, native objectives, same eval ruler ───────────
    BaselineAlgo("B2_deeds", "DEEDS",
                 "_deeds.nii.gz",  "_deeds_seg_reg.nii.gz",  True, "_deeds_dvf.nii.gz"),
    BaselineAlgo("B3_ants",  "ANTs-SyN",
                 "_ants.nii.gz",   "_ants_seg_reg.nii.gz",   True, "_ants_dvf.nii.gz"),
    BaselineAlgo("B4_vxm",   "VoxelMorph",
                 "_vxm.nii.gz",    "_vxm_seg_reg.nii.gz",    True, "_vxm_dvf.nii.gz"),
    BaselineAlgo("B5_unigradicon", "uniGradICON",
                 "_unigradicon.nii.gz", "_unigradicon_seg_reg.nii.gz",
                 True, "_unigradicon_dvf.nii.gz"),
    # B5 with instance optimisation (50 gradient steps after network forward pass).
    # Same network weights, same evaluation ruler — isolates the IO contribution.
    BaselineAlgo("B5_unigradicon_io", "uniGradICON + IO-50",
                 "_unigradicon_io.nii.gz", "_unigradicon_io_seg_reg.nii.gz",
                 True, "_unigradicon_io_dvf.nii.gz"),
]}


def condition_tag(algo_tag: str, input_key: str) -> str:
    return f"{algo_tag}__{input_key}"


def _make_baseline_conditions() -> list:
    conds = []
    for algo in BASELINE_ALGOS.values():
        for ikey in INPUTS:
            if ikey == "baseline" and not algo.run_on_baseline:
                continue
            tag = condition_tag(algo.tag, ikey)
            conds.append(_C(tag, RESULTS_DIR / tag,
                            algo.vol_postfix, algo.seg_postfix,
                            algo.is_deformable, algo.dvf_postfix, input_key=ikey))
    return conds


# Full registry. A-series first, then B-series (algo × input).
CONDITIONS: Dict[str, Condition] = {
    c.tag: c for c in (_A_SERIES + _make_baseline_conditions())
}


# Pretty labels for the final table
ROW_LABELS: Dict[str, str] = {
    # Aligned pipeline
    "A1_crop_only":          "A1   Crop only",
    "A2_zalign":             "A2   + Z-align",
    "A3_pass0":              "A3   + Rigid P0 (Kabsch)",
    "A4_pass01":             "A4   + Rigid P0+1 (Nelder-Mead)",
    "A5_pass012":            "A5   + Rigid P0+1+2 (Sobel gate)",
    "A6_bspline":            "A6   Full (+ B-spline)",
    # Baseline pipeline — no z-align (isolates z-align contribution)
    "A3b_pass0_noalign":     "A3b  Rigid P0        [no z-align]",
    "A4b_pass01_noalign":    "A4b  Rigid P0+1      [no z-align]",
    "A5b_pass012_noalign":   "A5b  Rigid P0+1+2    [no z-align]",
    "A6_bspline_baseline":   "A6b  Full deformable [no z-align]",
}
for _algo in BASELINE_ALGOS.values():
    for _ikey in INPUTS:
        if _ikey == "baseline" and not _algo.run_on_baseline:
            continue
        ROW_LABELS[condition_tag(_algo.tag, _ikey)] = \
            f"{_algo.tag.split('_')[0]}  {_algo.name}  [{_ikey}]"