"""
baselines/run_rigid.py  —  Four-objective rigid comparison, fast implementation.

ROOT CAUSE OF PREVIOUS SLOWNESS: the inner loop called sitk.Resample on the
full 3D volume at every Nelder-Mead iteration (300 iters × 18M voxels = hang).
Fix: replicate register_fixed.py's BatchPrecomp approach — extract organ voxel
coordinates ONCE before the optimizer, then apply the transform as a batched
matmul per iteration. sitk.Resample only runs ONCE at the very end to produce
the final warped output.

Four objectives, same Euler3D model, same centroid-seeded init:

  organ  centroid_mm + λ·(1−eroded_Dice)            anatomy only
  sobel  −NCC(|∇HU|_fixed, |∇HU|_moving)            phase-invariant edges
  full   centroid + eDice + λ_s·(−Sobel-NCC)         proposed method
  mmi    Mattes MI via SimpleITK RSGD                standard intensity baseline

Usage:
    python -m baselines.run_rigid --metric organ --input both
    python -m baselines.run_rigid --metric full  --input aligned --studies STUDY_ID
    python run_rigid.py --metric mmi --input baseline   (direct invocation)
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import sobel as nd_sobel
from scipy.optimize import minimize

# allow both `python -m baselines.run_rigid` and `python run_rigid.py`
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from . import _common as K
except ImportError:
    import _common as K

import compare_config as C

# ---------------------------------------------------------------------------
# Hyper-parameters  (mirror register_fixed.py where applicable)
# ---------------------------------------------------------------------------
ORGAN_WEIGHTS = {
    1: 3.0,  2: 2.0,  3: 2.0,  4: 2.0,   # liver, spleen, kidneys L/R
    13: 1.5, 14: 1.5,                      # aorta, IVC
    21: 1.0, 22: 1.0, 23: 1.0,            # L1-L3
}
LAMBDA_DICE   = 10.0
LAMBDA_SOBEL  = 5.0
EROSION_MM    = 4.0     # erosion inside optimizer (fast pass)
NM_MAXITER    = 300
MIN_VOX       = 200     # minimum voxels for an organ to be included
MAX_COORDS    = 5000    # max coords sampled per organ (speed vs accuracy)

# ---------------------------------------------------------------------------
# Geometry helpers  (match register_fixed.py exactly)
# ---------------------------------------------------------------------------

def _euler_to_R(rx, ry, rz):
    """R = Rz @ Ry @ Rx  (ZYX convention matching register_fixed)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], np.float32)
    return Rz @ Ry @ Rx


def _forward_transform(coords_xyz, R, t, center):
    """
    Move coords from moving→fixed space.
    SimpleITK inverse: x_m = R(x_f-c)+c+t  →  forward: x_f = R^T(x_m-c-t)+c
    """
    shifted = coords_xyz - center[None] - t[None]
    return (R.T @ shifted.T).T + center[None]


def _phys_to_vox_zyx(pts_xyz, origin, spacing, dir_inv):
    """Physical XYZ → integer voxel ZYX for numpy indexing."""
    shifted = pts_xyz - origin[None]
    vox_xyz = (dir_inv @ shifted.T).T / spacing[None]
    return np.round(vox_xyz[:, ::-1]).astype(np.int32)


# ---------------------------------------------------------------------------
# Precomputation  (called ONCE per study/phase before the optimizer loop)
# ---------------------------------------------------------------------------

def _erode_sitk(mask_np, spacing_xyz, erosion_mm):
    img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    img.SetSpacing(spacing_xyz)
    r = [max(1, int(round(erosion_mm / s))) for s in spacing_xyz]
    return sitk.GetArrayFromImage(sitk.BinaryErode(img, r)).astype(bool)


def _centroid_xyz(arr, label, sp, origin, direc):
    m = arr == label
    if m.sum() < MIN_VOX:
        return None
    cz, cy, cx = (np.argwhere(m).mean(axis=0))
    v = np.array([cx * sp[0], cy * sp[1], cz * sp[2]])
    return np.array(origin) + np.array(direc).reshape(3,3) @ v


def _sample_coords(eroded_mask, sp, origin, direc, label):
    """Extract physical XYZ coords of eroded voxels, subsampled to MAX_COORDS."""
    idx = np.argwhere(eroded_mask)          # (N,3) ZYX
    if len(idx) == 0:
        return np.zeros((0,3), np.float32)
    if len(idx) > MAX_COORDS:
        rng = np.random.default_rng(label)
        idx = idx[rng.choice(len(idx), MAX_COORDS, replace=False)]
    vox = np.column_stack([idx[:,2]*sp[0], idx[:,1]*sp[1], idx[:,0]*sp[2]])
    D = np.array(direc).reshape(3,3)
    return (np.array(origin) + (D @ vox.T).T).astype(np.float32)


class OrganData:
    """All static data for one organ, precomputed before the optimizer."""
    __slots__ = ("label","weight","fixed_centroid","fixed_eroded","fixed_count",
                 "moving_coords","moving_full_count","moving_centroid")

    def __init__(self, label, weight, fixed_centroid, fixed_eroded,
                 moving_coords, moving_full_count, moving_centroid):
        self.label = label; self.weight = weight
        self.fixed_centroid = fixed_centroid
        self.fixed_eroded   = fixed_eroded
        self.fixed_count    = int(fixed_eroded.sum())
        self.moving_coords  = moving_coords         # (N,3) physical XYZ
        self.moving_full_count = moving_full_count
        self.moving_centroid   = moving_centroid


class Precomp:
    """Batched precomputation for all organs, mirrors register_fixed.BatchPrecomp."""

    def __init__(self, fixed_seg: sitk.Image, moving_seg: sitk.Image,
                 fixed_vol: sitk.Image):
        fn = np.round(sitk.GetArrayFromImage(fixed_seg)).astype(np.int32)
        mn = np.round(sitk.GetArrayFromImage(moving_seg)).astype(np.int32)
        sp  = fixed_seg.GetSpacing()
        or_ = fixed_seg.GetOrigin()
        di  = fixed_seg.GetDirection()
        sp_m  = moving_seg.GetSpacing()
        or_m  = moving_seg.GetOrigin()
        di_m  = moving_seg.GetDirection()

        D_inv = np.linalg.inv(np.array(di).reshape(3,3)).astype(np.float32)
        sz = np.array(fixed_vol.GetSize(), float)
        self.center = (np.array(fixed_vol.GetOrigin())
                       + 0.5*(sz-1)*np.array(fixed_vol.GetSpacing())).astype(np.float32)
        self.origin  = np.array(or_,  np.float32)
        self.spacing = np.array(sp,   np.float32)
        self.dir_inv = D_inv
        self.shape_zyx = fn.shape     # (Z,Y,X)

        self.organs: list[OrganData] = []
        for label, weight in ORGAN_WEIGHTS.items():
            fm = fn == label
            mm = mn == label
            if fm.sum() < MIN_VOX or mm.sum() < MIN_VOX:
                continue
            fe = _erode_sitk(fm, sp, EROSION_MM)
            me = _erode_sitk(mm, sp_m, EROSION_MM)
            if fe.sum() == 0 or me.sum() == 0:
                continue
            fc = _centroid_xyz(fn, label, sp,   or_,  di)
            mc = _centroid_xyz(mn, label, sp_m, or_m, di_m)
            if fc is None or mc is None:
                continue
            coords = _sample_coords(me, sp_m, or_m, di_m, label)
            if len(coords) == 0:
                continue
            self.organs.append(OrganData(
                label, weight, fc.astype(np.float32), fe,
                coords, int(me.sum()), mc.astype(np.float32)
            ))

        if not self.organs:
            raise ValueError("No organs with sufficient voxels found")

        # stacked coords + organ index for one batched matmul
        self.all_coords = np.vstack([o.moving_coords for o in self.organs])
        self.organ_idx  = np.concatenate([
            np.full(len(o.moving_coords), i, np.int32)
            for i, o in enumerate(self.organs)
        ])
        print(f"    precomp: {len(self.organs)} organs, "
              f"{len(self.all_coords):,} coord points", flush=True)


# ---------------------------------------------------------------------------
# Fast loss functions  (no Resample in loop)
# ---------------------------------------------------------------------------

def _fast_organ_loss(params, batch: Precomp) -> float:
    rx, ry, rz, tx, ty, tz = params
    R = _euler_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], np.float32)
    Z, Y, X = batch.shape_zyx

    pts = _forward_transform(batch.all_coords, R, t, batch.center)
    vox = _phys_to_vox_zyx(pts, batch.origin, batch.spacing, batch.dir_inv)

    total = wt = 0.0
    for i, org in enumerate(batch.organs):
        # centroid (analytic)
        c_tf = _forward_transform(org.moving_centroid[None], R, t, batch.center)[0]
        cdist = float(np.linalg.norm(org.fixed_centroid - c_tf))

        # eroded Dice via coord lookup
        v = vox[batch.organ_idx == i]
        in_b = ((v[:,0]>=0)&(v[:,0]<Z)&(v[:,1]>=0)&(v[:,1]<Y)&(v[:,2]>=0)&(v[:,2]<X))
        v = v[in_b]
        if len(v) > 0:
            hits  = org.fixed_eroded[v[:,0], v[:,1], v[:,2]].sum()
            scale = org.moving_full_count / max(len(v), 1)
            inter = hits * scale
            denom = org.fixed_count + org.moving_full_count
            dice  = float(2.0 * inter / denom) if denom > 0 else 0.0
        else:
            dice = 0.0

        total += org.weight * (cdist + LAMBDA_DICE * (1.0 - dice))
        wt    += org.weight

    return total / max(wt, 1e-6)


def _fast_sobel_loss(params, batch: Precomp,
                     fixed_grad, moving_grad) -> float:
    """
    -NCC of gradient magnitudes sampled at the transformed moving organ coords.
    No full-volume Resample — sample fixed_grad and moving_grad at coord lookup.
    """
    rx, ry, rz, tx, ty, tz = params
    R = _euler_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], np.float32)
    Z, Y, X = batch.shape_zyx

    pts = _forward_transform(batch.all_coords, R, t, batch.center)
    vox = _phys_to_vox_zyx(pts, batch.origin, batch.spacing, batch.dir_inv)
    in_b = ((vox[:,0]>=0)&(vox[:,0]<Z)&(vox[:,1]>=0)&(vox[:,1]<Y)
            &(vox[:,2]>=0)&(vox[:,2]<X))

    if in_b.sum() < 10:
        return 1.0  # worst case

    vf = vox[in_b]
    gf = fixed_grad[vf[:,0], vf[:,1], vf[:,2]].astype(np.float64)

    # sample moving grad at ORIGINAL (pre-transform) coords
    # moving_coords are in moving space already
    orig_vox = _phys_to_vox_zyx(batch.all_coords[in_b],
                                  batch.origin, batch.spacing, batch.dir_inv)
    in_b2 = ((orig_vox[:,0]>=0)&(orig_vox[:,0]<Z)
             &(orig_vox[:,1]>=0)&(orig_vox[:,1]<Y)
             &(orig_vox[:,2]>=0)&(orig_vox[:,2]<X))

    if in_b2.sum() < 10:
        return 1.0

    gf2 = gf[in_b2]
    gm  = moving_grad[orig_vox[in_b2,0], orig_vox[in_b2,1], orig_vox[in_b2,2]].astype(np.float64)

    gf2 -= gf2.mean(); gm -= gm.mean()
    dn = np.linalg.norm(gf2) * np.linalg.norm(gm)
    ncc = float(np.dot(gf2, gm) / dn) if dn > 1e-6 else 0.0
    return -ncc


def _fast_full_loss(params, batch: Precomp,
                    fixed_grad, moving_grad) -> float:
    return (_fast_organ_loss(params, batch)
            + LAMBDA_SOBEL * _fast_sobel_loss(params, batch, fixed_grad, moving_grad))


# ---------------------------------------------------------------------------
# Centroid seed + explicit simplex  (identical to register_fixed Pass-1)
# ---------------------------------------------------------------------------

def _centroid_seed(batch: Precomp) -> np.ndarray:
    """Weighted mean (moving_centroid - fixed_centroid) → translation seed."""
    num = np.zeros(3, np.float64); den = 0.0
    for org in batch.organs:
        num += org.weight * (org.moving_centroid - org.fixed_centroid)
        den += org.weight
    return num / max(den, 1e-6)


def _init_simplex(x0: np.ndarray) -> np.ndarray:
    """8mm / 0.05rad explicit simplex — prevents the 0.00025 scipy default."""
    steps = np.array([0.05, 0.05, 0.05, 8.0, 8.0, 8.0])
    return np.vstack([x0] + [x0 + np.eye(6)[i]*steps[i] for i in range(6)])


def _params_to_tx(params, center) -> sitk.Euler3DTransform:
    tx = sitk.Euler3DTransform()
    tx.SetComputeZYX(True)
    tx.SetCenter(center.tolist())
    tx.SetRotation(float(params[0]), float(params[1]), float(params[2]))
    tx.SetTranslation([float(params[3]), float(params[4]), float(params[5])])
    return tx


# ---------------------------------------------------------------------------
# Per-metric register_* functions
# ---------------------------------------------------------------------------

def _make_batch(fixed_seg_path, moving_seg, fixed_vol):
    """Build Precomp with progress print."""
    print("    precomputing organ data...", flush=True)
    fixed_seg = sitk.ReadImage(fixed_seg_path)
    return Precomp(fixed_seg, moving_seg, fixed_vol)


def register_organ(fixed, moving, moving_seg, item) -> K.RegResult:
    batch = _make_batch(item["fixed_seg"], moving_seg, fixed)
    t0    = _centroid_seed(batch)
    x0    = np.array([0.0, 0.0, 0.0, t0[0], t0[1], t0[2]])
    print(f"    centroid seed: t={t0.round(1)}mm  loss0="
          f"{_fast_organ_loss(x0, batch):.2f}", flush=True)

    res = minimize(_fast_organ_loss, x0, args=(batch,), method="Nelder-Mead",
                   options={"maxiter": NM_MAXITER, "xatol": 0.1, "fatol": 0.001,
                            "disp": False, "initial_simplex": _init_simplex(x0)})
    print(f"    organ done: loss={res.fun:.4f} iters={res.nit}", flush=True)
    return K.RegResult(transform=_params_to_tx(res.x, batch.center))


def register_sobel(fixed, moving, moving_seg, item) -> K.RegResult:
    batch = _make_batch(item["fixed_seg"], moving_seg, fixed)
    print("    computing gradient magnitudes...", flush=True)
    fg = _grad_mag(sitk.GetArrayFromImage(fixed).astype(np.float32))
    mg = _grad_mag(sitk.GetArrayFromImage(moving).astype(np.float32))

    t0 = _centroid_seed(batch)
    x0 = np.array([0.0, 0.0, 0.0, t0[0], t0[1], t0[2]])
    print(f"    centroid seed: t={t0.round(1)}mm  loss0="
          f"{_fast_sobel_loss(x0, batch, fg, mg):.4f}", flush=True)

    res = minimize(_fast_sobel_loss, x0, args=(batch, fg, mg), method="Nelder-Mead",
                   options={"maxiter": NM_MAXITER, "xatol": 0.1, "fatol": 1e-4,
                            "disp": False, "initial_simplex": _init_simplex(x0)})
    print(f"    sobel done: loss={res.fun:.4f} iters={res.nit}", flush=True)
    return K.RegResult(transform=_params_to_tx(res.x, batch.center))


def register_full(fixed, moving, moving_seg, item) -> K.RegResult:
    batch = _make_batch(item["fixed_seg"], moving_seg, fixed)
    print("    computing gradient magnitudes...", flush=True)
    fg = _grad_mag(sitk.GetArrayFromImage(fixed).astype(np.float32))
    mg = _grad_mag(sitk.GetArrayFromImage(moving).astype(np.float32))

    t0 = _centroid_seed(batch)
    x0 = np.array([0.0, 0.0, 0.0, t0[0], t0[1], t0[2]])
    print(f"    centroid seed: t={t0.round(1)}mm  loss0="
          f"{_fast_full_loss(x0, batch, fg, mg):.4f}", flush=True)

    res = minimize(_fast_full_loss, x0, args=(batch, fg, mg), method="Nelder-Mead",
                   options={"maxiter": NM_MAXITER, "xatol": 0.1, "fatol": 0.001,
                            "disp": False, "initial_simplex": _init_simplex(x0)})
    print(f"    full done: loss={res.fun:.4f} iters={res.nit}", flush=True)
    return K.RegResult(transform=_params_to_tx(res.x, batch.center))


def register_mmi(fixed, moving, moving_seg, item) -> K.RegResult:
    """Mattes-MI via SimpleITK gradient-based optimizer (faster for smooth metric)."""
    print("    running MMI registration...", flush=True)
    f = sitk.Cast(fixed,  sitk.sitkFloat32)
    m = sitk.Cast(moving, sitk.sitkFloat32)

    init_tx = sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

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
    reg.SetInitialTransform(init_tx, inPlace=False)

    def cb():
        pass
    reg.AddCommand(sitk.sitkIterationEvent,
                   lambda: print(".", end="", flush=True))
    tx = reg.Execute(f, m)
    print(f"\n    mmi done: metric={reg.GetMetricValue():.4f}", flush=True)
    return K.RegResult(transform=tx)


def _grad_mag(vol_np: np.ndarray) -> np.ndarray:
    v = vol_np.astype(np.float64)
    return np.sqrt(sum(nd_sobel(v, axis=i)**2 for i in range(3))).astype(np.float32)


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
    K.make_cli(algo_tag, lambda _args: register_fn)
