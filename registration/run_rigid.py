"""
baselines/run_rigid.py  —  Controlled rigid comparison: four objectives, one model.

Same transform: Euler3D rigid.
Same initialisation: centroid-seeded translation (organ-centroid offset), zero rotation.
Same inputs: aligned OR baseline crops.
Same eval ruler: centroid_mm + eroded_Dice + Sobel-NCC (in evaluate_all.py).

The ONLY variable is what the Nelder-Mead optimizer minimises:

  --metric organ  centroid_mm + λ·(1−eroded_Dice)
                  Pure anatomy loss. Needs seg_reg masks at inference.
                  Ablation: "what if we only use segmentation-based terms?"

  --metric sobel  −NCC(|∇HU_fixed|, |∇HU_moving|)  inside a whole-organ union mask
                  Phase-invariant, mask-free after computing the gradient.
                  Ablation: "can edges alone drive registration across phases?"

  --metric full   centroid_mm + λ_d·(1−eroded_Dice) + λ_s·(−Sobel-NCC)
                  Proposed method. Anatomy anchors coarse alignment;
                  Sobel refines boundary sharpness across the phase gap.

  --metric mmi    Mattes Mutual Information (SimpleITK ImageRegistrationMethod,
                  gradient-based). Standard intensity baseline for multi-phase CT.
                  NOT Nelder-Mead — uses SimpleITK's RSGD which is faster and
                  more appropriate for a smooth differentiable metric.

All four write outputs that are evaluated by evaluate_all.py on the same ruler,
so the table rows are directly comparable.

Usage:
    # All four objectives × both input sources (recommended for the paper):
    python -m baselines.run_rigid --metric organ --input both
    python -m baselines.run_rigid --metric sobel --input both
    python -m baselines.run_rigid --metric full  --input both
    python -m baselines.run_rigid --metric mmi   --input both

    # Single study debug:
    python -m baselines.run_rigid --metric full --input aligned --studies STUDY_ID
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import sobel as nd_sobel
from scipy.optimize import minimize

try:
    from . import _common as K
except ImportError:
    import _common as K

import compare_config as C
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluate_pipeline.evaluate_registration as E

# ---------------------------------------------------------------------------
# Loss hyper-parameters (match register_fixed.py Pass-1 conventions)
# ---------------------------------------------------------------------------
ORGAN_WEIGHTS = {
    1: 3.0,  2: 2.0,  3: 2.0,  4: 2.0,   # liver, spleen, kidneys L/R
    13: 1.5, 14: 1.5,                      # aorta, IVC
    21: 1.0, 22: 1.0, 23: 1.0,            # L1-L3
}
LAMBDA_DICE   = 10.0   # scale (1-Dice) into mm-commensurate units
LAMBDA_SOBEL  = 5.0    # scale -NCC (∈[-1,0]) into positive mm-commensurate units
EROSION_MM    = 4.0    # erosion for dice term inside optimizer (fast, not the full 6mm)
NM_MAXITER    = 300
MIN_ORGAN_VOX = 200    # skip organ if fewer voxels in either mask


# ---------------------------------------------------------------------------
# Helpers shared across objectives
# ---------------------------------------------------------------------------

def _phys_centroid(seg_img: sitk.Image, arr: np.ndarray, label: int):
    """Physical (x,y,z) mm centroid for one label, honouring origin+direction."""
    idx = np.argwhere(arr == label)
    if idx.shape[0] == 0:
        return None
    cz, cy, cx = idx.mean(axis=0)
    return np.array(seg_img.TransformContinuousIndexToPhysicalPoint(
        (float(cx), float(cy), float(cz))))


def _centroid_seed(fixed_seg: sitk.Image, fixed_np: np.ndarray,
                   moving_seg: sitk.Image, moving_np: np.ndarray,
                   labels: list) -> np.ndarray:
    """
    Weighted organ-centroid translation seed (x,y,z mm).
    For a fixed-frame transform φ: x_fixed → x_moving,
    t = c_moving − c_fixed aligns centroids under pure translation.
    """
    num = np.zeros(3); den = 0.0
    for l in labels:
        cf = _phys_centroid(fixed_seg,  fixed_np,  l)
        cm = _phys_centroid(moving_seg, moving_np, l)
        if cf is None or cm is None:
            continue
        w = ORGAN_WEIGHTS.get(l, 1.0)
        num += w * (cm - cf); den += w
    return num / max(den, 1e-6)


def _init_simplex(x0: np.ndarray) -> np.ndarray:
    """
    Explicit Nelder-Mead simplex. Without this, scipy uses a default step of
    0.00025 from x0 — far too small to explore mm/radian shifts.
    Steps: 0.05 rad (~3°) for rotations, 8 mm for translations.
    """
    steps = np.array([0.05, 0.05, 0.05, 8.0, 8.0, 8.0])
    return np.vstack([x0] + [x0 + np.eye(6)[i] * steps[i] for i in range(6)])


def _active_labels(fixed_np: np.ndarray, moving_np: np.ndarray) -> list:
    return [l for l in ORGAN_WEIGHTS
            if (fixed_np == l).sum() >= MIN_ORGAN_VOX
            and (moving_np == l).sum() >= MIN_ORGAN_VOX]


def _warp_seg_np(moving_seg: sitk.Image, fixed: sitk.Image,
                 center, params: np.ndarray) -> np.ndarray:
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center)
    tx.SetParameters([float(p) for p in params])
    warped = sitk.Resample(moving_seg, fixed, tx,
                           sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID())
    return np.round(sitk.GetArrayFromImage(warped)).astype(np.int32)


def _grad_mag(vol_np: np.ndarray) -> np.ndarray:
    """|∇HU| via Sobel — same as evaluate_registration.sobel_ncc."""
    v = vol_np.astype(np.float64)
    return np.sqrt(sum(nd_sobel(v, axis=i) ** 2 for i in range(3)))


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Normalised cross-correlation of a and b inside mask. Returns -1..1."""
    a, b = a[mask > 0].ravel(), b[mask > 0].ravel()
    if a.size < 10:
        return 0.0
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0


def _union_mask(fixed_np: np.ndarray, warped_np: np.ndarray,
                labels: list) -> np.ndarray:
    """Binary union of all organ regions in both masks — used for Sobel-NCC."""
    m = np.zeros_like(fixed_np, dtype=np.uint8)
    for l in labels:
        m[(fixed_np == l) | (warped_np == l)] = 1
    return m


def _init_transform(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    return sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        sitk.Cast(fixed,  sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    ))


# ---------------------------------------------------------------------------
# Objective: organ  (centroid + eroded Dice)
# ---------------------------------------------------------------------------

def _loss_organ(fixed_np, fixed_seg, labels, spacing_zyx):
    def _loss(warped_np):
        num = den = 0.0
        for l in labels:
            mf = (fixed_np  == l).astype(np.uint8)
            mm = (warped_np == l).astype(np.uint8)
            if mm.sum() < 50:
                term = 50.0
            else:
                c = E.centroid_displacement_mm(mf, mm, spacing_zyx) or 100.0
                d = E.eroded_dice(mf, mm, spacing_zyx, EROSION_MM) or 0.0
                term = c + LAMBDA_DICE * (1.0 - d)
            num += ORGAN_WEIGHTS.get(l, 1.0) * term
            den += ORGAN_WEIGHTS.get(l, 1.0)
        return num / max(den, 1e-6)
    return _loss


# ---------------------------------------------------------------------------
# Objective: sobel  (−NCC of gradient magnitudes inside organ union)
# ---------------------------------------------------------------------------

def _loss_sobel(fixed_grad, moving_grad, fixed_np, labels):
    def _loss(warped_seg_np):
        # union mask of fixed + warped organ regions
        mask = _union_mask(fixed_np, warped_seg_np, labels)
        ncc = _ncc(fixed_grad, moving_grad, mask)
        return -ncc    # minimise → maximise NCC
    return _loss


# ---------------------------------------------------------------------------
# Objective: full  (centroid + eDice + Sobel-NCC)
# ---------------------------------------------------------------------------

def _loss_full(fixed_np, fixed_seg, labels, spacing_zyx, fixed_grad, moving_grad):
    organ_fn = _loss_organ(fixed_np, fixed_seg, labels, spacing_zyx)
    sobel_fn = _loss_sobel(fixed_grad, moving_grad, fixed_np, labels)
    def _loss(warped_np):
        return organ_fn(warped_np) + LAMBDA_SOBEL * sobel_fn(warped_np)
    return _loss


# ---------------------------------------------------------------------------
# Shared Nelder-Mead driver for organ / sobel / full
# ---------------------------------------------------------------------------

def _register_nm(fixed, moving, moving_seg, item,
                 loss_from_warped_np) -> "K.RegResult":
    """
    Run Nelder-Mead with a centroid-seeded translation start.
    loss_from_warped_np : callable(warped_seg_np) -> float
    """
    spacing_zyx = E.get_spacing_zyx(fixed)
    fixed_seg_img = sitk.ReadImage(item["fixed_seg"])
    fixed_np  = np.round(sitk.GetArrayFromImage(fixed_seg_img)).astype(np.int32)
    moving_np = np.round(sitk.GetArrayFromImage(moving_seg)).astype(np.int32)

    labels = _active_labels(fixed_np, moving_np)
    if not labels:
        return K.RegResult(transform=_init_transform(fixed, moving))

    center = _init_transform(fixed, moving).GetCenter()
    t0 = _centroid_seed(fixed_seg_img, fixed_np, moving_seg, moving_np, labels)
    x0 = np.array([0.0, 0.0, 0.0, t0[0], t0[1], t0[2]], dtype=np.float64)

    loss_fn = loss_from_warped_np(fixed_np, fixed_seg_img, labels,
                                   spacing_zyx, center, moving_seg, fixed, moving)

    def objective(params):
        warped_np = _warp_seg_np(moving_seg, fixed, center, params)
        return loss_fn(warped_np)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": NM_MAXITER, "xatol": 1e-3, "fatol": 1e-3,
                            "initial_simplex": _init_simplex(x0)})

    tx = sitk.Euler3DTransform()
    tx.SetCenter(center)
    tx.SetParameters([float(p) for p in res.x])
    return K.RegResult(transform=tx)


# ---------------------------------------------------------------------------
# Public register_* functions (signature fixed so make_cli can pass them)
# ---------------------------------------------------------------------------

def register_organ(fixed, moving, moving_seg, item) -> "K.RegResult":
    def _build(fixed_np, fixed_seg, labels, spacing_zyx, center, mov_seg, fx, mv):
        return _loss_organ(fixed_np, fixed_seg, labels, spacing_zyx)
    return _register_nm(fixed, moving, moving_seg, item, _build)


def register_sobel(fixed, moving, moving_seg, item) -> "K.RegResult":
    fixed_np_vol  = sitk.GetArrayFromImage(fixed).astype(np.float32)
    moving_np_vol = sitk.GetArrayFromImage(moving).astype(np.float32)
    fixed_grad    = _grad_mag(fixed_np_vol)
    moving_grad   = _grad_mag(moving_np_vol)

    def _build(fixed_np, fixed_seg, labels, spacing_zyx, center, mov_seg, fx, mv):
        return _loss_sobel(fixed_grad, moving_grad, fixed_np, labels)
    return _register_nm(fixed, moving, moving_seg, item, _build)


def register_full(fixed, moving, moving_seg, item) -> "K.RegResult":
    fixed_np_vol  = sitk.GetArrayFromImage(fixed).astype(np.float32)
    moving_np_vol = sitk.GetArrayFromImage(moving).astype(np.float32)
    fixed_grad    = _grad_mag(fixed_np_vol)
    moving_grad   = _grad_mag(moving_np_vol)

    def _build(fixed_np, fixed_seg, labels, spacing_zyx, center, mov_seg, fx, mv):
        return _loss_full(fixed_np, fixed_seg, labels, spacing_zyx,
                          fixed_grad, moving_grad)
    return _register_nm(fixed, moving, moving_seg, item, _build)


def register_mmi(fixed, moving, moving_seg, item) -> "K.RegResult":
    """Mattes-MI rigid via SimpleITK RSGD (gradient-based, faster than NM for smooth metric)."""
    f = sitk.Cast(fixed,  sitk.sitkFloat32)
    m = sitk.Cast(moving, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10, seed=1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(_init_transform(fixed, moving), inPlace=False)
    return K.RegResult(transform=reg.Execute(f, m))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_METRICS = {
    "organ": ("R_organ", register_organ),
    "sobel": ("R_sobel", register_sobel),
    "full":  ("R_full",  register_full),
    "mmi":   ("R_mmi",   register_mmi),
}

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--metric", choices=list(_METRICS), required=True)
    known, _ = pre.parse_known_args()
    algo_tag, register_fn = _METRICS[known.metric]
    K.make_cli(algo_tag, lambda args: register_fn)