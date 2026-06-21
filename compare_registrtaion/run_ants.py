"""
baselines/run_ants.py  —  B3: ANTs SyN deformable registration.

Requirements:
    pip install antspyx

Key choices for multi-phase CT:
  - Metric = Mattes MI for BOTH affine and SyN stages.
    CC (cross-correlation) is NOT used because it is invalid across contrast
    phases — the same anatomy has different HU after enhancement.
  - Seg propagated with interpolator='nearestNeighbor'.
  - DVF exported via antspyx's composite warp utility, converted to
    (Z,Y,X,3) / (z,y,x) / mm convention for neg_jacobian_pct.

VERIFY ON FIRST STUDY:
    Check that %|J|<0 is 1-5% (typical for SyN). If ~50-100%, the
    composite warp axis order differs — flip the reorder on [FLIP_CHECK].

Usage:
    python -m baselines.run_ants --input both
    python run_ants.py --input aligned --studies STUDY_ID
"""
from __future__ import annotations
import os, sys, tempfile, time
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
import evaluate_pipeline.evaluate_registration as E

ALGO_TAG = "B3_ants"


def _check_antspyx():
    try:
        import ants
        return ants
    except ImportError:
        raise ImportError(
            "antspyx not installed. Run: pip install antspyx\n"
            "Note: antspyx requires ~2GB disk and may take several minutes to install."
        )


def register_ants(fixed: sitk.Image, moving: sitk.Image,
                  moving_seg: sitk.Image, item: dict) -> K.RegResult:
    ants = _check_antspyx()
    tmp = tempfile.mkdtemp(prefix="ants_")
    try:
        f_path = os.path.join(tmp, "fixed.nii.gz")
        m_path = os.path.join(tmp, "moving.nii.gz")
        s_path = os.path.join(tmp, "moving_seg.nii.gz")

        sitk.WriteImage(sitk.Cast(fixed,  sitk.sitkFloat32), f_path)
        sitk.WriteImage(sitk.Cast(moving, sitk.sitkFloat32), m_path)
        sitk.WriteImage(sitk.Cast(moving_seg, sitk.sitkUInt16), s_path)

        fi = ants.image_read(f_path)
        mo = ants.image_read(m_path)
        mo_seg = ants.image_read(s_path)

        print("    running ANTs SyN (MI metric)...", flush=True)
        _t0 = time.perf_counter()
        reg = ants.registration(
            fixed=fi, moving=mo,
            type_of_transform="SyN",
            aff_metric="mattes",      # NOT cc — invalid across contrast phases
            syn_metric="mattes",
            verbose=False,
        )
        _ants_sec = time.perf_counter() - _t0
        fwd = reg["fwdtransforms"]    # list: [warp.nii.gz, affine.mat]
        print(f"    ANTs SyN wall time: {_ants_sec:.1f}s", flush=True)

        # Warp volume (linear interpolation)
        warped_ants = reg["warpedmovout"]
        warped_vol_path = os.path.join(tmp, "warped_vol.nii.gz")
        ants.image_write(warped_ants, warped_vol_path)

        # Warp seg (nearest neighbour — preserves integer labels)
        seg_w = ants.apply_transforms(
            fixed=fi, moving=mo_seg,
            transformlist=fwd,
            interpolator="nearestNeighbor",
        )
        warped_seg_path = os.path.join(tmp, "warped_seg.nii.gz")
        ants.image_write(seg_w, warped_seg_path)

        # Convert warped images back to SimpleITK
        warped_vol = sitk.ReadImage(warped_vol_path)
        warped_vol.CopyInformation(fixed)
        warped_seg_img = sitk.Cast(sitk.ReadImage(warped_seg_path), sitk.sitkUInt16)
        warped_seg_img.CopyInformation(fixed)

        # Composite forward displacement field
        # ants.apply_transforms with compose=prefix writes the field to disk
        dvf_mm = None
        comp_prefix = os.path.join(tmp, "composite_")
        try:
            comp_result = ants.apply_transforms(
                fixed=fi, moving=mo,
                transformlist=fwd,
                compose=comp_prefix,
            )
            # antspyx returns the path string when compose is set
            if isinstance(comp_result, str) and os.path.exists(comp_result):
                comp_path = comp_result
            else:
                # fallback: look for the composed field file
                candidates = [
                    comp_prefix + "comp.nii.gz",
                    comp_prefix + "0.nii.gz",
                ]
                comp_path = next((p for p in candidates if os.path.exists(p)), None)

            if comp_path:
                dvf_sitk = sitk.ReadImage(comp_path)
                arr = sitk.GetArrayFromImage(dvf_sitk)   # (Z,Y,X,3) comp=(x,y,z) mm
                arr = np.squeeze(arr)
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    arr = np.ascontiguousarray(arr[..., ::-1])  # [FLIP_CHECK] → z,y,x
                    dvf_mm = arr.astype(np.float32)
                    print(f"    DVF loaded, shape={dvf_mm.shape}", flush=True)
        except Exception as e:
            print(f"    DVF export failed ({e}) — %|J|<0 will be n/a", flush=True)

        print("    ANTs done", flush=True)
        return K.RegResult(warped_vol=warped_vol, warped_seg=warped_seg_img,
                           dvf_zyx3_mm=dvf_mm)
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    K.make_cli(ALGO_TAG, lambda _args: register_ants)
