"""
baselines/_common.py
====================
Shared contract for every baseline runner (B1 Elastix, B2 DEEDS, B3 ANTs,
B4 VoxelMorph). Centralising this removes the bugs that otherwise recur in all
four:

  1. Fixed = NC, moving = each contrast phase. All methods warp moving INTO NC
     space, and the warped-moving seg is evaluated against the NC seg.
  2. Fair starting point: every baseline reads the SAME z-aligned crops
     (`_aligned*`) the rigid ablation starts from, so B-methods are a clean
     swap for the A3-A6 transform stack — not advantaged/disadvantaged by a
     different input grid.
  3. Volume warp uses linear interpolation; LABEL warp uses NEAREST-NEIGHBOUR.
     (Linear-interpolating a label map silently invents fractional labels.)
  4. Displacement fields are exported as (Z,Y,X,3) in PHYSICAL mm with the
     component order reordered to (z,y,x) to match numpy axis order, which is
     what evaluate_registration.neg_jacobian_pct assumes. SimpleITK vectors are
     (x,y,z); writing them unreordered makes det(J) meaningless.

Output naming obeys config.py exactly:
    results/{tag}/{study}/{study}_{series}{vol_postfix}
    results/{tag}/{study}/{study}_{series}{seg_postfix}
    results/{tag}/{study}/{study}_{series}{dvf_postfix}   (deformable only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib

import compare_config as C

try:
    from profiling import profile_stage as _profile_stage
    _PROFILING = True
except ImportError:
    _PROFILING = False

PROFILING_DIR = C.RESULTS_DIR / "profiling"


# ---------------------------------------------------------------------------
# Iterate (study, fixed_nc, moving_phase) work items from the aligned crops
# ---------------------------------------------------------------------------

def iter_pairs(labels_df: pd.DataFrame,
               input_dir: Path = C.ALIGNED_CROP_DIR,
               vol_postfix: str = "_aligned.nii.gz",
               seg_postfix: str = "_aligned_seg_reg.nii.gz",
               ref_phase: str = C.REF_PHASE,
               seg_dir: Optional[Path] = None,
               flat: bool = False) -> Iterator[dict]:
    """
    Yield one dict per (study, moving phase) with resolved fixed/moving paths.

    Two on-disk layouts are supported:

      nested (flat=False, the default — aligned/baseline crops):
          input_dir/{study}/{study}_{series}{vol_postfix}
          (seg_dir or input_dir)/{study}/{study}_{series}{seg_postfix}

      flat (flat=True — raw input):
          input_dir/{study}_{series}{vol_postfix}              (no study subdir)
          seg_dir/{study}_{series}{seg_postfix}                (no study subdir,
                                                                 separate dir tree,
                                                                 e.g. ts_segmentation/)

    `seg_dir` defaults to `input_dir` (segs alongside vols — the nested case).
    Pass a different directory for layouts where segmentations live in a
    separate tree; required for `raw` input (sibling `ts_segmentation/`).
    """
    if seg_dir is None:
        seg_dir = input_dir

    if flat:
        # Studies aren't directories here — derive the study list from the
        # labels CSV itself, since filenames are {study}_{series}{postfix}.
        study_dirs = sorted(labels_df["StudyInstanceUID"].unique())
    else:
        study_dirs = sorted(d for d in os.listdir(input_dir)
                            if (input_dir / d).is_dir())

    def _vol_path(study: str, sid: str) -> Path:
        if flat:
            return input_dir / f"{study}_{sid}{vol_postfix}"
        return input_dir / study / f"{study}_{sid}{vol_postfix}"

    def _seg_path(study: str, sid: str) -> Path:
        if flat:
            return seg_dir / f"{study}_{sid}{seg_postfix}"
        return seg_dir / study / f"{study}_{sid}{seg_postfix}"

    for study in study_dirs:
        rows = labels_df[labels_df["StudyInstanceUID"] == study]
        nc = rows[rows["Label"] == ref_phase]
        if nc.empty:
            continue
        nc_sid = nc.iloc[0]["SeriesInstanceUID"]
        nc_vol = _vol_path(study, nc_sid)
        nc_seg = _seg_path(study, nc_sid)
        if not (nc_vol.exists() and nc_seg.exists()):
            continue
        for _, r in rows.iterrows():
            sid, phase = r["SeriesInstanceUID"], r["Label"]
            if phase == ref_phase:
                continue
            mv_vol = _vol_path(study, sid)
            mv_seg = _seg_path(study, sid)
            if not (mv_vol.exists() and mv_seg.exists()):
                continue
            yield {
                "study": study, "phase": phase,
                "nc_sid": nc_sid, "mv_sid": sid,
                "fixed_vol": str(nc_vol), "fixed_seg": str(nc_seg),
                "moving_vol": str(mv_vol), "moving_seg": str(mv_seg),
            }


# ---------------------------------------------------------------------------
# Warping (transform-based methods: Elastix, ANTs affine, SimpleITK B-spline)
# ---------------------------------------------------------------------------

def warp_volume(moving: sitk.Image, fixed: sitk.Image,
                transform: sitk.Transform,
                default_value: float = -1024.0) -> sitk.Image:
    """Resample moving CT into the fixed grid with linear interpolation."""
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear,
                         default_value, moving.GetPixelID())


def warp_label(moving_seg: sitk.Image, fixed: sitk.Image,
               transform: sitk.Transform) -> sitk.Image:
    """Resample a label map into the fixed grid with NEAREST-NEIGHBOUR."""
    return sitk.Resample(moving_seg, fixed, transform, sitk.sitkNearestNeighbor,
                         0, moving_seg.GetPixelID())


def transform_to_mm_dvf_array(transform: sitk.Transform,
                              fixed: sitk.Image) -> np.ndarray:
    """
    Convert any SimpleITK transform to a dense displacement field sampled on the
    fixed grid, returned as (Z, Y, X, 3) in physical mm with component order
    (z, y, x) to match numpy axis order (what neg_jacobian_pct expects).
    """
    disp_filter = sitk.TransformToDisplacementFieldFilter()
    disp_filter.SetReferenceImage(fixed)
    disp_img = disp_filter.Execute(transform)              # vector image, mm, (x,y,z)
    arr = sitk.GetArrayFromImage(disp_img)                 # (Z, Y, X, 3) comp=(x,y,z)
    arr = arr[..., ::-1]                                   # -> comp=(z,y,x)
    return np.ascontiguousarray(arr.astype(np.float32))


def voxel_dvf_to_mm(dvf_vox_zyx3: np.ndarray,
                    spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    """
    Convert a (Z,Y,X,3) displacement field whose components are in VOXELS,
    ordered (z,y,x), into physical mm. Use this for DEEDS / VoxelMorph outputs
    before saving, so EVERY on-disk DVF is uniform: (Z,Y,X,3), (z,y,x), mm.
    """
    out = np.asarray(dvf_vox_zyx3, dtype=np.float32).copy()
    for c in range(3):
        out[..., c] *= float(spacing_zyx[c])
    return out


# ---------------------------------------------------------------------------
# Saving (obeys the config postfix contract)
# ---------------------------------------------------------------------------

def save_outputs(item: dict, tag: str,
                 warped_vol: sitk.Image, warped_seg: sitk.Image,
                 dvf_zyx3_mm: Optional[np.ndarray] = None) -> None:
    cond = C.CONDITIONS[tag]
    out_dir = cond.base_dir / item["study"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = out_dir / f"{item['study']}_{item['mv_sid']}"

    sitk.WriteImage(warped_vol, str(pre) + cond.vol_postfix)
    sitk.WriteImage(sitk.Cast(warped_seg, sitk.sitkUInt16),
                    str(pre) + cond.seg_postfix)

    if dvf_zyx3_mm is not None:
        if cond.dvf_postfix is None:
            raise ValueError(f"{tag} is not configured as deformable but a DVF was given")
        # affine zooms must be (x,y,z); fixed spacing is (x,y,z)
        sp = warped_vol.GetSpacing()
        aff = np.diag([sp[0], sp[1], sp[2], 1.0])
        nib.save(nib.Nifti1Image(dvf_zyx3_mm, aff), str(pre) + cond.dvf_postfix)


def also_save_warped_nc(item: dict, tag: str) -> None:
    """
    Copy the NC fixed vol+seg into the condition dir under the same postfixes so
    evaluate_registration finds a reference series for this condition.
    """
    cond = C.CONDITIONS[tag]
    out_dir = cond.base_dir / item["study"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = out_dir / f"{item['study']}_{item['nc_sid']}"
    if not os.path.exists(str(pre) + cond.vol_postfix):
        sitk.WriteImage(sitk.ReadImage(item["fixed_vol"]), str(pre) + cond.vol_postfix)
        sitk.WriteImage(sitk.ReadImage(item["fixed_seg"]), str(pre) + cond.seg_postfix)


# ---------------------------------------------------------------------------
# Generic driver
# ---------------------------------------------------------------------------

@dataclass
class RegResult:
    """
    What a method's register_fn returns. Two ways to provide the result:

      (a) transform-based  : set `transform` (a sitk.Transform). The driver then
          warps the volume (linear) and seg (NN) and, for deformable conditions,
          exports the mm-DVF from the transform.

      (b) field/array-based: set `warped_vol`, `warped_seg` and (deformable)
          `dvf_zyx3_mm` directly. Use this for methods whose native output is a
          warped image + a field already in numpy (DEEDS, VoxelMorph, ANTs-SyN
          composite). `dvf_zyx3_mm` MUST be (Z,Y,X,3), component order (z,y,x),
          physical mm (see neg_jacobian_pct / transform_to_mm_dvf_array).
    """
    transform:   Optional[sitk.Transform] = None
    warped_vol:  Optional[sitk.Image] = None
    warped_seg:  Optional[sitk.Image] = None
    dvf_zyx3_mm: Optional[np.ndarray] = None


def run_baseline(algo_tag: str,
                 input_key: str,
                 register_fn,
                 labels_df: pd.DataFrame,
                 studies: Optional[list] = None,
                 skip_existing: bool = True) -> None:
    """
    Drive one comparison algorithm over one input source.

    register_fn(fixed_img, moving_img, moving_seg, item) -> RegResult
    """
    tag = C.condition_tag(algo_tag, input_key)
    cond = C.CONDITIONS[tag]
    src = C.INPUTS[input_key]

    # `raw` input is laid out flat: volumes directly under src.dir as
    # {study}_{series}_standardized.nii.gz, with segmentations in a SEPARATE
    # sibling tree, ts_segmentation/, also flat, as
    # {study}_{series}_seg_reg.nii.gz (always the "reg" variant, never "full").
    # compare_config.py may already define these explicitly on the InputSpec
    # (seg_dir / flat / seg_postfix) — prefer those if present, and only fall
    # back to the convention below when they're absent, so this keeps working
    # if config.py is updated later.
    is_raw = (input_key == "raw")
    flat = getattr(src, "flat", is_raw)
    seg_dir = getattr(src, "seg_dir", None)
    seg_postfix = getattr(src, "seg_postfix", None)
    if is_raw:
        if seg_dir is None:
            seg_dir = Path(src.dir).parent / "ts_segmentation"
        if seg_postfix is None:
            seg_postfix = "_seg_reg.nii.gz"
    else:
        if seg_dir is None:
            seg_dir = src.dir
        if seg_postfix is None:
            seg_postfix = src.seg_postfix

    print(f"\n{'='*80}\n{tag}\n  input : {src.dir}\n  output: {cond.base_dir}\n{'='*80}")
    if is_raw:
        print(f"  (raw layout: flat vols in {src.dir}, flat segs in {seg_dir})")

    n_ok = n_skip = n_fail = 0
    for item in iter_pairs(labels_df, src.dir, src.vol_postfix, seg_postfix,
                           seg_dir=seg_dir, flat=flat):
        if studies and item["study"] not in studies:
            continue

        out_pre = cond.base_dir / item["study"] / f"{item['study']}_{item['mv_sid']}"
        if skip_existing and os.path.exists(str(out_pre) + cond.vol_postfix):
            n_skip += 1
            continue

        print(f"  [{item['phase']:<12}] {item['study'][:40]}…", flush=True)
        try:
            fixed  = sitk.ReadImage(item["fixed_vol"])
            moving = sitk.ReadImage(item["moving_vol"])
            mov_sg = sitk.ReadImage(item["moving_seg"])

            if _PROFILING:
                _ctx = _profile_stage(
                    algo_tag, input_key,
                    item["study"], item["phase"],
                    PROFILING_DIR,
                )
            else:
                import contextlib
                _ctx = contextlib.nullcontext()

            with _ctx:
                res = register_fn(fixed, moving, mov_sg, item)

            if res.warped_vol is None and res.transform is not None:
                res.warped_vol = warp_volume(moving, fixed, res.transform)
                res.warped_seg = warp_label(mov_sg, fixed, res.transform)
                if cond.is_deformable and res.dvf_zyx3_mm is None:
                    res.dvf_zyx3_mm = transform_to_mm_dvf_array(res.transform, fixed)

            if res.warped_vol is None or res.warped_seg is None:
                raise RuntimeError("register_fn returned neither a transform nor warped images")

            save_outputs(item, tag, res.warped_vol, res.warped_seg,
                         res.dvf_zyx3_mm if cond.is_deformable else None)
            also_save_warped_nc(item, tag)
            n_ok += 1
        except Exception as e:                          # pragma: no cover
            import traceback; traceback.print_exc()
            print(f"    ✗ {e}")
            n_fail += 1

    print(f"\n  done {tag}: ok={n_ok} skip={n_skip} fail={n_fail}")


def make_cli(algo_tag: str, register_fn_factory):
    """
    Shared CLI for every runner. register_fn_factory(args) -> register_fn.
    Adds --input {aligned,baseline,both}, --studies, --labels_csv, --no-skip.
    """
    import argparse
    p = argparse.ArgumentParser(description=f"{algo_tag} comparison runner")
    p.add_argument("--input", choices=["aligned", "baseline", "raw", "both", "all"],
                   default="both",
                   help="aligned (z-aligned crops), baseline (crop only), "
                        "raw (no crop/align). both=aligned+baseline, "
                        "all=aligned+baseline+raw.")
    p.add_argument("--studies", nargs="*", default=None,
                   help="Optional subset of study IDs.")
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--no-skip", action="store_true",
                   help="Recompute even if output exists.")
    args, _ = p.parse_known_args()

    labels_df = pd.read_csv(args.labels_csv)
    register_fn = register_fn_factory(args)

    if args.input == "both":
        inputs = ["aligned", "baseline"]
    elif args.input == "all":
        inputs = ["aligned", "baseline", "raw"]
    else:
        inputs = [args.input]
    algo = C.BASELINE_ALGOS[algo_tag]
    for ikey in inputs:
        if ikey == "baseline" and not algo.run_on_baseline:
            print(f"  ({algo_tag} not configured for baseline input — skipping)")
            continue
        run_baseline(algo_tag, ikey, register_fn, labels_df,
                     studies=args.studies, skip_existing=not args.no_skip)




# export DEEDS_BIN=/media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration/compare_registrtaion/deedsBCV/deedsBCV
# export LINEAR_BIN=/media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration/compare_registrtaion/deedsBCV/linearBCV
# export APPLYFLOAT_BIN=/media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration/compare_registrtaion/deedsBCV/applyBCVfloat

# python run_deeds.py --input all --studies ONE_STUDY_ID   # smoke-test on one first