"""
baselines/run_deeds.py  —  B2: DEEDS deformable registration.

DEEDS (Heinrich et al.) uses a MIND/SSC self-similarity metric that is
contrast-robust — a strong match for multi-phase CT where absolute HU differs
across NC/Arterial/Venous. Zero training, deterministic, fast on CPU.

Two-stage invocation (the intended DEEDS workflow):
    1. linearBCV  -F fixed -M moving -O <prefix>          → <prefix>_matrix.txt
    2. deedsBCV   -F fixed -M moving -A <prefix>_matrix.txt -O <prefix> -S seg
       → <prefix>_deformed.nii.gz, <prefix>_deformed_seg.nii.gz,
         <prefix>_displacements.dat (control-point field, binary)

Why the linear pre-step matters: deeds' deformable search assumes a reasonable
initial pose. On `raw` input (no crop, no align) it is effectively required;
on `aligned`/`baseline` it still absorbs residual offset and improves results.

Same-dimension requirement: deeds needs fixed and moving on an IDENTICAL grid
and ignores anisotropic spacing. We therefore resample moving (and its seg)
onto the fixed grid up front — a no-op when grids already match (aligned/
baseline), and mandatory for raw, where phases have different extents.

Folding (%|J|<0): deeds writes only `_displacements.dat` (control points), not
a dense NIfTI field. We optionally reconstruct a dense displacement field by
warping three identity coordinate volumes through applyBCVfloat and taking
u(x) = warped_coord(x) - coord(x). Controlled by env var DEEDS_DVF (default on);
on any failure we fall back to dvf=None so the warped vol/seg still save and the
folding column simply shows "—".

Binaries (set via env, else found on PATH):
    DEEDS_BIN       deedsBCV       (required)
    LINEAR_BIN      linearBCV      (required)
    APPLYFLOAT_BIN  applyBCVfloat  (only for dense-DVF reconstruction)
    Download/build: https://github.com/mattiaspaul/deedsBCV

Usage:
    DEEDS_BIN=.../deedsBCV LINEAR_BIN=.../linearBCV \
        python run_deeds.py --input all                 # aligned + baseline + raw
    python run_deeds.py --input raw  --studies STUDY_ID
    DEEDS_DVF=0 python run_deeds.py --input aligned     # skip dense-DVF recon

VERIFY ON FIRST STUDY:
    Check evaluate_all.py reports %|J|<0 ≈ 1-5%. If ~100%, the reconstructed
    field's component order is wrong for your build — see [FLIP_CHECK] below.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile, time
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluate_pipeline.evaluate_registration as E

ALGO_TAG       = "B2_deeds"
DEEDS_BIN      = os.environ.get("DEEDS_BIN", "deedsBCV")
LINEAR_BIN     = os.environ.get("LINEAR_BIN", "linearBCV")
APPLYFLOAT_BIN = os.environ.get("APPLYFLOAT_BIN", "applyBCVfloat")

# Confirmed against the deedsBCV repo:
DEEDS_VOL_SUFFIX = "_deformed.nii.gz"
DEEDS_SEG_SUFFIX = "_deformed_seg.nii.gz"

# Reconstruct a dense DVF for the folding metric? (env override)
RECON_DVF = os.environ.get("DEEDS_DVF", "1") == "1"


def _resolve(binary: str) -> bool:
    return shutil.which(binary) is not None or os.path.exists(binary)


def _check_binaries(need_applyfloat: bool):
    missing = [b for b in (DEEDS_BIN, LINEAR_BIN) if not _resolve(b)]
    if need_applyfloat and not _resolve(APPLYFLOAT_BIN):
        missing.append(APPLYFLOAT_BIN)
    if missing:
        raise RuntimeError(
            f"deeds binary/binaries not found: {missing}.\n"
            f"Set DEEDS_BIN / LINEAR_BIN / APPLYFLOAT_BIN or add them to PATH.\n"
            f"Build: https://github.com/mattiaspaul/deedsBCV"
        )


def _read_like(path: str, like: sitk.Image, as_uint=False) -> sitk.Image:
    img = sitk.ReadImage(path)
    if as_uint:
        img = sitk.Cast(img, sitk.sitkUInt16)
    img.CopyInformation(like)
    return img


def _run(cmd, timeout) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          timeout=timeout)


def _reconstruct_dense_dvf(fixed: sitk.Image, prefix: str, tmp: str):
    """
    Recover a dense forward displacement field from a finished deeds run by
    warping three identity coordinate volumes with applyBCVfloat:

        warped_coord_a(x) = a-coordinate of the source location for fixed voxel x
        u_a(x)            = warped_coord_a(x) - a(x)              [in voxels]

    Returns (Z,Y,X,3) in (z,y,x) component order, physical mm — the convention
    neg_jacobian_pct expects — or None on any failure (folding then shows "—").
    """
    try:
        nx, ny, nz = fixed.GetSize()                 # sitk size is (X, Y, Z)
        # numpy arrays are (Z, Y, X); build per-axis voxel-index volumes.
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                                 indexing="ij")
        coord = {"x": xx.astype(np.float32),
                 "y": yy.astype(np.float32),
                 "z": zz.astype(np.float32)}
        warped = {}
        for ax in ("x", "y", "z"):
            cimg = sitk.GetImageFromArray(coord[ax])  # (Z,Y,X)
            cimg.CopyInformation(fixed)
            cpath = os.path.join(tmp, f"coord_{ax}.nii.gz")
            wpath = os.path.join(tmp, f"warp_{ax}.nii.gz")
            sitk.WriteImage(cimg, cpath)
            # applyBCVfloat -M <image> -O <prefix> -D <output>   (repo-documented)
            _run([APPLYFLOAT_BIN, "-M", cpath, "-O", prefix, "-D", wpath], timeout=300)
            warped[ax] = sitk.GetArrayFromImage(sitk.ReadImage(wpath)).astype(np.float32)

        u_x = warped["x"] - coord["x"]
        u_y = warped["y"] - coord["y"]
        u_z = warped["z"] - coord["z"]
        # [FLIP_CHECK] component order (z,y,x); if folding ≈100%, swap to (x,y,z):
        #   dvf_vox = np.stack([u_x, u_y, u_z], axis=-1)
        dvf_vox = np.stack([u_z, u_y, u_x], axis=-1)            # (Z,Y,X,3), vox
        dvf_mm = K.voxel_dvf_to_mm(dvf_vox, E.get_spacing_zyx(fixed))
        print(f"    dense DVF reconstructed (applyBCVfloat), shape={dvf_mm.shape}",
              flush=True)
        return dvf_mm
    except Exception as e:
        print(f"    ⚠ dense DVF reconstruction failed ({e}) — %|J|<0 will be n/a",
              flush=True)
        return None


def register_deeds(fixed: sitk.Image, moving: sitk.Image,
                   moving_seg: sitk.Image, item: dict) -> K.RegResult:
    _check_binaries(need_applyfloat=RECON_DVF)

    # Same-dims requirement: resample moving + seg onto the fixed grid.
    # No-op when grids already match (aligned/baseline); essential for raw.
    if (moving.GetSize() != fixed.GetSize()
            or moving.GetSpacing() != fixed.GetSpacing()
            or moving.GetOrigin() != fixed.GetOrigin()
            or moving.GetDirection() != fixed.GetDirection()):
        moving = sitk.Resample(moving, fixed, sitk.Transform(),
                               sitk.sitkLinear, -1024.0, moving.GetPixelID())
        moving_seg = sitk.Resample(moving_seg, fixed, sitk.Transform(),
                                   sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID())

    tmp = tempfile.mkdtemp(prefix="deeds_")
    try:
        f_path = os.path.join(tmp, "fixed.nii.gz")
        m_path = os.path.join(tmp, "moving.nii.gz")
        s_path = os.path.join(tmp, "moving_seg.nii.gz")
        # ONE shared prefix so the affine (_matrix.txt) and the deformable
        # (_displacements.dat) live together and applyBCVfloat -O <prefix>
        # can find whatever it needs.
        pre = os.path.join(tmp, "deeds")

        sitk.WriteImage(sitk.Cast(fixed,  sitk.sitkFloat32),  f_path)
        sitk.WriteImage(sitk.Cast(moving, sitk.sitkFloat32),  m_path)
        sitk.WriteImage(sitk.Cast(moving_seg, sitk.sitkUInt16), s_path)

        # ── Stage 1: linear (affine) pre-alignment ─────────────────────────
        print("    linearBCV (affine)...", flush=True)
        _run([LINEAR_BIN, "-F", f_path, "-M", m_path, "-O", pre], timeout=300)
        affine = pre + "_matrix.txt"
        if not os.path.exists(affine):
            raise RuntimeError(
                f"linearBCV produced no affine at {affine}. "
                f"tmp contents: {os.listdir(tmp)}"
            )

        # ── Stage 2: deformable, seeded by the affine ──────────────────────
        print("    deedsBCV (deformable)...", flush=True)
        _t0 = time.perf_counter()
        res = _run([DEEDS_BIN, "-F", f_path, "-M", m_path,
                    "-A", affine, "-O", pre, "-S", s_path], timeout=600)
        print(f"    deedsBCV wall time: {time.perf_counter()-_t0:.1f}s", flush=True)
        if res.stdout:
            for l in res.stdout.strip().split("\n")[-3:]:
                print(f"    deeds: {l}", flush=True)

        vol_out = pre + DEEDS_VOL_SUFFIX
        seg_out = pre + DEEDS_SEG_SUFFIX
        if not os.path.exists(vol_out):
            raise RuntimeError(
                f"DEEDS did not produce {vol_out}. tmp: {os.listdir(tmp)}"
            )

        warped_vol = _read_like(vol_out, fixed)
        warped_seg = _read_like(seg_out, fixed, as_uint=True)

        dvf_mm = _reconstruct_dense_dvf(fixed, pre, tmp) if RECON_DVF else None

        print("    deeds done", flush=True)
        return K.RegResult(warped_vol=warped_vol, warped_seg=warped_seg,
                           dvf_zyx3_mm=dvf_mm)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    K.make_cli(ALGO_TAG, lambda _args: register_deeds)