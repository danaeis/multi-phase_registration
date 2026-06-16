"""
baselines/run_deeds.py  —  B2: DEEDS deformable registration.

DEEDS (Dense Displacement Estimation via Discrete Sampling, Heinrich et al.)
uses a MIND-based similarity that is contrast-robust — a strong match for
multi-phase CT where absolute HU differs across NC/Arterial/Venous.
Zero training, deterministic, fast on CPU.

Requirements:
    deedsBCV binary — set DEEDS_BIN env var or edit DEEDS_BIN below.
    Download: https://github.com/mattiaspaul/deedsBCV

VERIFY ON FIRST STUDY:
    After running one study, check evaluate_all.py %|J|<0 is small (1-5%).
    If it shows ~50-100%, the flow field component order differs for your
    build — flip line marked [FLIP_CHECK] below.

Usage:
    DEEDS_BIN=/path/to/deedsBCV python -m baselines.run_deeds --input both
    DEEDS_BIN=/path/to/deedsBCV python run_deeds.py --input aligned --studies STUDY_ID
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from . import _common as K
except ImportError:
    import _common as K

import compare_config as C
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluate_pipeline.evaluate_registration as E

ALGO_TAG  = "B2_deeds"
DEEDS_BIN = os.environ.get("DEEDS_BIN", "deedsBCV")

# DEEDS output filenames — confirm against your build.
# Most builds write:  {prefix}_deformed.nii.gz, {prefix}_deformed_seg.nii.gz
# Flow field (optional): {prefix}_flow.nii.gz  — some builds write .dat instead
DEEDS_VOL_SUFFIX  = "_deformed.nii.gz"
DEEDS_SEG_SUFFIX  = "_deformed_seg.nii.gz"
DEEDS_FLOW_SUFFIX = "_flow.nii.gz"


def _check_binary():
    if shutil.which(DEEDS_BIN) is None and not os.path.exists(DEEDS_BIN):
        raise RuntimeError(
            f"deedsBCV not found at '{DEEDS_BIN}'.\n"
            f"Set DEEDS_BIN=/path/to/deedsBCV or install to PATH.\n"
            f"Download: https://github.com/mattiaspaul/deedsBCV"
        )


def _read_like(path: str, like: sitk.Image, as_uint=False) -> sitk.Image:
    """Read image and copy spatial metadata from the fixed volume."""
    img = sitk.ReadImage(path)
    if as_uint:
        img = sitk.Cast(img, sitk.sitkUInt16)
    img.CopyInformation(like)
    return img


def register_deeds(fixed: sitk.Image, moving: sitk.Image,
                   moving_seg: sitk.Image, item: dict) -> K.RegResult:
    _check_binary()
    tmp = tempfile.mkdtemp(prefix="deeds_")
    try:
        f_path = os.path.join(tmp, "fixed.nii.gz")
        m_path = os.path.join(tmp, "moving.nii.gz")
        s_path = os.path.join(tmp, "moving_seg.nii.gz")
        out    = os.path.join(tmp, "out")

        sitk.WriteImage(sitk.Cast(fixed,  sitk.sitkFloat32), f_path)
        sitk.WriteImage(sitk.Cast(moving, sitk.sitkFloat32), m_path)
        sitk.WriteImage(sitk.Cast(moving_seg, sitk.sitkUInt16), s_path)

        print("    running deedsBCV...", flush=True)
        result = subprocess.run(
            [DEEDS_BIN, "-F", f_path, "-M", m_path, "-O", out, "-S", s_path],
            check=True, capture_output=True, text=True, timeout=600
        )
        if result.stdout:
            # print last few lines of deeds output for progress visibility
            lines = result.stdout.strip().split('\n')
            for l in lines[-3:]:
                print(f"    deeds: {l}", flush=True)

        vol_out = out + DEEDS_VOL_SUFFIX
        seg_out = out + DEEDS_SEG_SUFFIX

        if not os.path.exists(vol_out):
            raise RuntimeError(
                f"DEEDS did not produce {vol_out}. "
                f"Files in tmp: {os.listdir(tmp)}"
            )

        warped_vol = _read_like(vol_out, fixed)
        warped_seg = _read_like(seg_out, fixed, as_uint=True)

        # Flow field: (Z,Y,X,3) voxel displacements → convert to physical mm
        # Component order from deedsBCV is typically (x,y,z) voxels.
        # [FLIP_CHECK]: if %|J|<0 ≈ 100%, change arr[...,::-1] to arr below.
        dvf_mm = None
        flow_path = out + DEEDS_FLOW_SUFFIX
        if os.path.exists(flow_path):
            flow = sitk.ReadImage(flow_path)
            arr  = np.squeeze(sitk.GetArrayFromImage(flow))     # (Z,Y,X,3) x,y,z vox
            if arr.ndim == 4 and arr.shape[-1] == 3:
                arr = np.ascontiguousarray(arr[..., ::-1])      # → z,y,x voxels
                sp  = E.get_spacing_zyx(fixed)
                dvf_mm = K.voxel_dvf_to_mm(arr, sp)
                print(f"    flow field loaded, shape={dvf_mm.shape}", flush=True)
            else:
                print(f"    unexpected flow shape {arr.shape}, skipping DVF", flush=True)
        else:
            print(f"    no flow field at {flow_path} — %|J|<0 will be n/a", flush=True)

        print("    deeds done", flush=True)
        return K.RegResult(warped_vol=warped_vol, warped_seg=warped_seg,
                           dvf_zyx3_mm=dvf_mm)
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    K.make_cli(ALGO_TAG, lambda _args: register_deeds)
