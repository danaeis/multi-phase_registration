"""
MIND-NCC patch for run_rigid.py
================================
Paste the contents of this file into run_rigid.py in two places:

  1. Add the three functions (_mind_map, _fast_mind_loss, register_mind)
     AFTER the existing `_grad_mag` function (around line 406).

  2. In the `_METRICS` dict (bottom of file), add:
         "mind": ("R_mind", register_mind),

That's all. compare_config.py changes are in compare_config_mind_patch.py.
"""

# ---------------------------------------------------------------------------
# MIND descriptor  (Self-Similarity Context, Heinrich et al. 2012)
# ---------------------------------------------------------------------------
from scipy.ndimage import uniform_filter as _uf   # already imported via scipy


def _mind_map(vol_np: np.ndarray, d: int = 2, patch: int = 7) -> np.ndarray:
    """
    Compute a 6-channel MIND descriptor volume.

    For each voxel p the descriptor is:
        MIND(p, delta) = exp( -SSD_patch(p, p+delta) / H(p) )
    over 6 axis-aligned displacements delta = ±d voxels.
    H(p) = local variance (size=patch) + epsilon — acts as contrast normaliser.

    Args:
        vol_np : float32 numpy array, shape (Z, Y, X)
        d      : neighbour displacement in voxels (default 2 → 3 mm at 1.5 mm)
        patch  : size of the local averaging filter for SSD and H (default 7)

    Returns:
        mind : float32 array, shape (6, Z, Y, X)
    """
    vol = vol_np.astype(np.float32)

    # Local variance estimate — contrast normaliser
    local_mean = _uf(vol,     size=patch).astype(np.float32)
    local_sq   = _uf(vol**2,  size=patch).astype(np.float32)
    H = local_sq - local_mean**2 + 1e-5          # variance + epsilon

    channels = []
    for axis in range(3):
        for sign in (+1, -1):
            shifted = np.roll(vol, sign * d, axis=axis)
            ssd = _uf((vol - shifted) ** 2, size=patch).astype(np.float32)
            channels.append(np.exp(-ssd / H))

    return np.stack(channels, axis=0)             # (6, Z, Y, X)


# ---------------------------------------------------------------------------
# MIND-NCC loss  (drop-in for _fast_sobel_loss)
# ---------------------------------------------------------------------------

def _fast_mind_loss(params, batch: Precomp,
                    fixed_mind: np.ndarray, moving_mind: np.ndarray) -> float:
    """
    -mean_channel NCC of MIND descriptors sampled within the organ-mask region.

    fixed_mind  : (6, Z, Y, X) MIND map of the fixed volume
    moving_mind : (6, Z, Y, X) MIND map of the moving volume

    The moving organ coordinates are transformed into fixed space; MIND values
    are looked up at integer voxel positions (nearest-neighbour — consistent
    with the batched coord-lookup approach used by _fast_sobel_loss).
    """
    rx, ry, rz, tx, ty, tz = params
    R = _euler_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], np.float32)
    Z, Y, X = batch.shape_zyx

    # Transform moving organ coords → fixed space
    pts = _forward_transform(batch.all_coords, R, t, batch.center)
    vox = _phys_to_vox_zyx(pts, batch.origin, batch.spacing, batch.dir_inv)
    in_b = ((vox[:,0] >= 0) & (vox[:,0] < Z) &
            (vox[:,1] >= 0) & (vox[:,1] < Y) &
            (vox[:,2] >= 0) & (vox[:,2] < X))

    if in_b.sum() < 10:
        return 1.0

    vf = vox[in_b]                                # (N, 3) fixed voxel indices

    # Sample MIND at transformed fixed coords  →  shape (6, N)
    fm = fixed_mind[:, vf[:,0], vf[:,1], vf[:,2]].astype(np.float64)

    # Sample MIND at original (untransformed) moving coords  →  shape (6, N)
    orig_vox = _phys_to_vox_zyx(
        batch.all_coords[in_b],
        batch.origin, batch.spacing, batch.dir_inv)
    in_b2 = ((orig_vox[:,0] >= 0) & (orig_vox[:,0] < Z) &
             (orig_vox[:,1] >= 0) & (orig_vox[:,1] < Y) &
             (orig_vox[:,2] >= 0) & (orig_vox[:,2] < X))

    if in_b2.sum() < 10:
        return 1.0

    fm = fm[:, in_b2]                             # (6, M)
    mm = moving_mind[:, orig_vox[in_b2, 0],
                        orig_vox[in_b2, 1],
                        orig_vox[in_b2, 2]].astype(np.float64)  # (6, M)

    # Per-channel NCC, then average
    ncc_sum = 0.0
    valid   = 0
    for c in range(fm.shape[0]):
        f_c = fm[c] - fm[c].mean()
        m_c = mm[c] - mm[c].mean()
        dn  = np.linalg.norm(f_c) * np.linalg.norm(m_c)
        if dn > 1e-6:
            ncc_sum += float(np.dot(f_c, m_c) / dn)
            valid   += 1

    return -(ncc_sum / valid) if valid > 0 else 1.0


# ---------------------------------------------------------------------------
# register_mind  (mirrors register_sobel / register_full)
# ---------------------------------------------------------------------------

def register_mind(fixed: "sitk.Image", moving: "sitk.Image",
                  moving_seg: "sitk.Image", item: dict) -> "K.RegResult":
    """
    Rigid registration using MIND-NCC within the organ-mask region.

    Contrast-invariant: MIND normalises by local variance, so HU shifts
    between NC and CECT phases are suppressed — same motivation as Sobel-NCC
    but with richer local structure captured by the descriptor.
    """
    batch = _make_batch(item["fixed_seg"], moving_seg, fixed)

    print("    computing MIND descriptors...", flush=True)
    fvol = sitk.GetArrayFromImage(fixed).astype(np.float32)
    mvol = sitk.GetArrayFromImage(moving).astype(np.float32)
    fm   = _mind_map(fvol)
    mm   = _mind_map(mvol)
    print(f"    MIND shape: {fm.shape}  dtype={fm.dtype}", flush=True)

    t0 = _centroid_seed(batch)
    x0 = np.array([0.0, 0.0, 0.0, t0[0], t0[1], t0[2]])
    print(f"    centroid seed: t={t0.round(1)}mm  "
          f"loss0={_fast_mind_loss(x0, batch, fm, mm):.4f}", flush=True)

    res = minimize(
        _fast_mind_loss, x0, args=(batch, fm, mm),
        method="Nelder-Mead",
        options={
            "maxiter": NM_MAXITER,
            "xatol":   0.1,
            "fatol":   1e-4,
            "disp":    False,
            "initial_simplex": _init_simplex(x0),
        },
    )
    print(f"    mind done: loss={res.fun:.4f} iters={res.nit}", flush=True)
    return K.RegResult(transform=_params_to_tx(res.x, batch.center))


# ---------------------------------------------------------------------------
# _METRICS update — replace the existing dict in the CLI section with:
# ---------------------------------------------------------------------------
# _METRICS = {
#     "organ": ("R_organ", register_organ),
#     "sobel": ("R_sobel", register_sobel),
#     "full":  ("R_full",  register_full),
#     "mmi":   ("R_mmi",   register_mmi),
#     "mind":  ("R_mind",  register_mind),   # ← new
# }