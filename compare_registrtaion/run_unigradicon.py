"""
baselines/run_unigradicon.py  —  B5: uniGradICON deformable registration.

uniGradICON is a pretrained foundation model for medical image registration
(MICCAI 2024).  No training is required — weights download automatically.

Requirements:
    pip install unigradicon

Usage:
    # Aligned + baseline inputs (default):
    python run_unigradicon.py --input both

    # All inputs including raw:
    python run_unigradicon.py --input all

    # Instance optimization (50 fine-tune steps, ~2–5× slower, slightly better):
    python run_unigradicon.py --input aligned --io_steps 50

    # Smoke test on one study:
    python run_unigradicon.py --input aligned --studies <STUDY_ID>

Paper: https://arxiv.org/abs/2403.05780
Repo:  https://github.com/uncbiag/uniGradICON
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ..registration import _common as K
except ImportError:
    import _common as K

import compare_config as C

ALGO_TAG  = "B5_unigradicon"
IO_STEPS  = None   # default: no instance optimisation (fast). Set to int via CLI.

# ---------------------------------------------------------------------------
# Lazy imports — itk and unigradicon are heavy; import only when needed
# ---------------------------------------------------------------------------

def _check_deps():
    missing = []
    for pkg in ("itk", "unigradicon", "icon_registration"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}\n"
            "Install with:  pip install unigradicon\n"
            "(itk and icon_registration are pulled in as dependencies)"
        )


# ---------------------------------------------------------------------------
# Model singleton — load once per process
# ---------------------------------------------------------------------------

_NET = None

def _get_net():
    global _NET
    if _NET is None:
        _check_deps()
        from unigradicon import get_unigradicon
        print("    Loading uniGradICON model (weights download on first run)…",
              flush=True)
        _NET = get_unigradicon()
        _NET.eval()
    return _NET


# ---------------------------------------------------------------------------
# ITK ↔ SimpleITK helpers
# ---------------------------------------------------------------------------

def _sitk_to_itk_path(img: sitk.Image, path: str, pixel_type=None) -> None:
    """Write a SimpleITK image to disk so ITK can read it."""
    if pixel_type is not None:
        img = sitk.Cast(img, pixel_type)
    sitk.WriteImage(img, path)


def _itk_image_to_sitk(itk_img, reference: sitk.Image, tmp_path: str) -> sitk.Image:
    """Write an ITK image to disk and read back as SimpleITK, copying grid info."""
    import itk
    itk.imwrite(itk_img, tmp_path)
    out = sitk.ReadImage(tmp_path)
    out.CopyInformation(reference)
    return out


# ---------------------------------------------------------------------------
# DVF extraction: ITK CompositeTransform → (Z,Y,X,3) numpy mm array (z,y,x order)
# ---------------------------------------------------------------------------

def _phi_to_dvf_zyx_mm(phi_AB, fixed_itk) -> np.ndarray:
    """
    Convert an ITK CompositeTransform to a dense DVF on the fixed grid.

    Returns (Z,Y,X,3) float32 array, component order (z,y,x), physical mm —
    the exact convention expected by neg_jacobian_pct and save_outputs.

    ITK displacement field GetArrayFromImage gives (Z,Y,X,3) with components
    in (x,y,z) order; we flip the last axis to get (z,y,x).
    """
    import itk
    dvf_itk = itk.transform_to_displacement_field_filter(
        phi_AB,
        reference_image=fixed_itk,
        use_reference_image=True,
    )
    arr = itk.GetArrayFromImage(dvf_itk)   # (Z,Y,X,3), comp = (x,y,z), mm
    dvf_zyx = np.ascontiguousarray(arr[..., ::-1].astype(np.float32))  # → (z,y,x)
    return dvf_zyx


# ---------------------------------------------------------------------------
# Core registration function
# ---------------------------------------------------------------------------

def register_unigradicon(
    fixed: sitk.Image,
    moving: sitk.Image,
    moving_seg: sitk.Image,
    item: dict,
    io_steps: int | None = IO_STEPS,
) -> K.RegResult:
    """
    Register moving → fixed using uniGradICON.

    Args:
        fixed / moving / moving_seg  : SimpleITK images (project standard).
        item                         : metadata dict from iter_pairs (unused here
                                       but required by run_baseline signature).
        io_steps                     : None = pure network inference (fast);
                                       int  = additional instance-optimisation
                                       fine-tuning steps (e.g. 50).
    Returns:
        RegResult with warped_vol, warped_seg (sitk.Image) and dvf_zyx3_mm
        (numpy, (Z,Y,X,3), (z,y,x) order, mm).
    """
    import itk
    from unigradicon import preprocess
    import icon_registration.itk_wrapper

    net = _get_net()

    tmp = tempfile.mkdtemp(prefix="unigradicon_")
    try:
        # ── 1. Write sitk → disk so ITK can read ──────────────────────────
        f_path   = os.path.join(tmp, "fixed.nii.gz")
        m_path   = os.path.join(tmp, "moving.nii.gz")
        ms_path  = os.path.join(tmp, "moving_seg.nii.gz")

        _sitk_to_itk_path(fixed,      f_path,  sitk.sitkFloat32)
        _sitk_to_itk_path(moving,     m_path,  sitk.sitkFloat32)
        _sitk_to_itk_path(moving_seg, ms_path, sitk.sitkUInt16)

        fixed_itk   = itk.imread(f_path)
        moving_itk  = itk.imread(m_path)
        mov_seg_itk = itk.imread(ms_path)

        # ── 2. CT preprocessing: clamp [-1000, 1000] HU → normalise [0,1] ──
        fixed_prep  = preprocess(fixed_itk,  "ct")
        moving_prep = preprocess(moving_itk, "ct")

        # ── 3. Register ────────────────────────────────────────────────────
        print(f"    uniGradICON inference (io_steps={io_steps})…", flush=True)
        t0 = time.perf_counter()
        phi_AB, _phi_BA = icon_registration.itk_wrapper.register_pair(
            net,
            moving_prep,    # moving (warped to fixed)
            fixed_prep,     # fixed  (target)
            finetune_steps=io_steps,
        )
        elapsed = time.perf_counter() - t0
        print(f"    uniGradICON wall time: {elapsed:.1f}s", flush=True)

        # ── 4. Warp volume (linear interpolation) ─────────────────────────
        interpolator_lin = itk.LinearInterpolateImageFunction.New(moving_itk)
        warped_itk = itk.resample_image_filter(
            moving_itk,
            transform=phi_AB,
            interpolator=interpolator_lin,
            use_reference_image=True,
            reference_image=fixed_itk,
        )

        # ── 5. Warp seg (nearest-neighbour — preserves integer labels) ────
        interpolator_nn = itk.NearestNeighborInterpolateImageFunction.New(mov_seg_itk)
        warped_seg_itk = itk.resample_image_filter(
            mov_seg_itk,
            transform=phi_AB,
            interpolator=interpolator_nn,
            use_reference_image=True,
            reference_image=fixed_itk,
        )

        # ── 6. Convert warped images back to SimpleITK ────────────────────
        wv_path  = os.path.join(tmp, "warped_vol.nii.gz")
        ws_path  = os.path.join(tmp, "warped_seg.nii.gz")

        warped_vol = _itk_image_to_sitk(warped_itk,     fixed, wv_path)
        warped_seg = _itk_image_to_sitk(warped_seg_itk, fixed, ws_path)
        warped_seg = sitk.Cast(warped_seg, sitk.sitkUInt16)

        # ── 7. DVF at original resolution ────────────────────────────────
        dvf_zyx3_mm = _phi_to_dvf_zyx_mm(phi_AB, fixed_itk)

        return K.RegResult(
            warped_vol=warped_vol,
            warped_seg=warped_seg,
            dvf_zyx3_mm=dvf_zyx3_mm,
        )

    finally:
        # clean up temp dir
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(description="uniGradICON B5 runner")
    p.add_argument("--input",    choices=["aligned", "baseline", "raw", "both", "all"],
                   default="both",
                   help="aligned/baseline/raw input source. "
                        "both=aligned+baseline, all=aligned+baseline+raw.")
    p.add_argument("--studies",  nargs="*", default=None,
                   help="Optional subset of StudyInstanceUIDs.")
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--no-skip",  action="store_true",
                   help="Recompute even if output already exists.")
    p.add_argument("--io_steps", type=int, default=None,
                   help="Instance-optimisation fine-tune steps (default: None = skip IO).")
    args = p.parse_args()

    labels_df = pd.read_csv(args.labels_csv)

    io_steps = args.io_steps

    def _register_fn(fixed, moving, moving_seg, item):
        return register_unigradicon(fixed, moving, moving_seg, item,
                                    io_steps=io_steps)

    if args.input == "both":
        inputs = ["aligned", "baseline"]
    elif args.input == "all":
        inputs = ["aligned", "baseline", "raw"]
    else:
        inputs = [args.input]

    algo = C.BASELINE_ALGOS[ALGO_TAG]
    for ikey in inputs:
        if ikey == "baseline" and not algo.run_on_baseline:
            print(f"  ({ALGO_TAG} not configured for baseline input — skipping)")
            continue
        K.run_baseline(
            ALGO_TAG, ikey, _register_fn, labels_df,
            studies=args.studies,
            skip_existing=not args.no_skip,
        )
