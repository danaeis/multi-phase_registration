from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_logger import get_stage_logger

# ---------------------------------------------------------------------------
# Organ label weights
# ---------------------------------------------------------------------------
ORGAN_WEIGHTS: Dict[int, float] = {
    1: 3.0,   # liver
    2: 2.0,   # spleen
    3: 2.0,   # kidney_left
    4: 2.0,   # kidney_right
    5: 0.5,   # pancreas
    6: 0.5,   # gallbladder
    13: 2.0,  # aorta
    14: 1.5,  # inferior_vena_cava
    40: 1.5,  # spinal_cord
    21: 1.5,  # vertebrae_L1
    22: 1.5,  # vertebrae_L2
    23: 1.5,  # vertebrae_L3
}

# Organs used for Z-offset estimation — must be large + contrast-stable
ANCHOR_LABELS = {1, 2, 3, 4}   # liver, spleen, kidneys

# Minimum voxel count for an organ to be considered present
MIN_ORGAN_VOXELS = 200

# Minimum confidence to trust the 3D centroid offset (0–1).
# Below this threshold, fall back to baseline center crop for that series.
MIN_CONFIDENCE = 0.40

# Maximum plausible Z-offset in mm. Larger offsets are likely errors.
MAX_OFFSET_MM = 80.0

# ===========================================================================
# EXTENDED ROI: HEART → BLADDER
# ===========================================================================

# Superior boundary: topmost structures present in seg_reg (native TotalSegmentator labels).
# Target: liver dome / diaphragm level → pad up to cardiac base.
# Heart (119) and lungs (114-118) are tier-3 (NOT in seg_reg).
# Best proxies: upper ribs, sternum, upper thoracic vertebrae — all in seg_reg.
SUPERIOR_LABELS = {
    1, 2,           # liver, spleen  — top face sits at diaphragm
    45,             # sternum        — at cardiac level
    46,             # costal_cartilages
    60, 72,         # rib_left_1, rib_right_1
    61, 73,         # rib_left_2, rib_right_2
    62, 74,         # rib_left_3, rib_right_3
    29, 30, 31,     # vertebrae_T1, T2, T3
    40,             # spinal_cord
}

# Inferior boundary: lowest pelvic structures present in seg_reg.
# Bladder (102) and prostate (103) are tier-3 (NOT in seg_reg).
# Best proxies: femurs, hips, sacrum — extend furthest caudally in seg_reg.
INFERIOR_LABELS = {
    38, 39,         # sacrum, vertebrae_S1
    41, 42,         # hip_left, hip_right
    43, 44,         # femur_left, femur_right
    9, 10,          # iliopsoas_left, iliopsoas_right
    16, 17,         # iliac_artery_left, iliac_artery_right
    18, 19,         # iliac_vena_left, iliac_vena_right
}

# Safety margins (in slices at 1.5 mm/slice):
#   SUPERIOR_PADDING = 40 → ~60 mm above liver dome / upper rib ≈ reaches cardiac base
#   INFERIOR_PADDING = 50 → ~75 mm below femur/sacrum tip  ≈ covers bladder floor
SUPERIOR_PADDING = 40
INFERIOR_PADDING = 50

# ===========================================================================
# EXISTING HELPERS (unchanged from v2)
# ===========================================================================

def _present_labels(sl: np.ndarray, min_vox: int = 20) -> set:
    labels, counts = np.unique(sl, return_counts=True)
    return {int(l) for l, c in zip(labels, counts) if l > 0 and c >= min_vox}


def _compute_new_origin(sitk_img: sitk.Image, crop_start: int) -> List[float]:
    origin    = np.array(sitk_img.GetOrigin())
    spacing   = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    z_vec     = direction[:, 2]
    return (origin + crop_start * spacing[2] * z_vec).tolist()


def _force_origin(img: sitk.Image, origin) -> sitk.Image:
    out = sitk.Image(img)
    out.SetOrigin(origin)
    return out


def _resample_inplane(moving: sitk.Image, reference: sitk.Image,
                      interpolator=sitk.sitkLinear,
                      default_val: float = -1024.0) -> sitk.Image:
    ref_sz  = reference.GetSize()
    ref_sp  = reference.GetSpacing()
    new_sz  = (ref_sz[0], ref_sz[1], moving.GetSize()[2])
    new_sp  = (ref_sp[0], ref_sp[1], moving.GetSpacing()[2])

    # Use reference XY origin + direction so the resampled moving data
    # covers the SAME physical XY region as the reference.
    # Keep moving's Z origin to preserve its Z positioning before crop.
    ref_origin = reference.GetOrigin()
    mov_origin = moving.GetOrigin()
    output_origin = (ref_origin[0], ref_origin[1], mov_origin[2])

    r = sitk.ResampleImageFilter()
    r.SetSize(new_sz)
    r.SetOutputSpacing(new_sp)
    r.SetOutputOrigin(output_origin)               # FIXED: ref XY, moving Z
    r.SetOutputDirection(reference.GetDirection()) # FIXED: match reference direction
    r.SetInterpolator(interpolator)
    r.SetDefaultPixelValue(default_val)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving)


def _crop_sitk(sitk_img: sitk.Image, z_start: int, z_end: int) -> sitk.Image:
    """Crop Z axis, update origin correctly."""
    arr = sitk.GetArrayFromImage(sitk_img)[z_start:z_end]
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(sitk_img.GetSpacing())
    out.SetDirection(sitk_img.GetDirection())
    out.SetOrigin(_compute_new_origin(sitk_img, z_start))
    return out


def _organ_z_center(seg_reg_sitk: sitk.Image,
                    anchor_labels: set = ANCHOR_LABELS) -> Optional[int]:
    """Weighted Z-CoM of anchor organs. Falls back to volume midpoint."""
    arr  = np.round(sitk.GetArrayFromImage(seg_reg_sitk)).astype(np.int32)
    mask = np.zeros(arr.shape[0], dtype=np.float32)
    for z in range(arr.shape[0]):
        sl = arr[z]
        mask[z] = sum(
            ORGAN_WEIGHTS.get(lbl, 1.0) * float((sl == lbl).sum())
            for lbl in anchor_labels
            if (sl == lbl).sum() >= 50
        )
    if mask.sum() == 0:
        return arr.shape[0] // 2
    z_indices = np.arange(arr.shape[0], dtype=np.float32)
    return int(round(float(np.sum(z_indices * mask) / mask.sum())))

def _find_superior_extent(seg_np: np.ndarray, labels: set, min_vox: int = 15) -> int:
    """Find the smallest Z (highest in body) where any superior structure appears."""
    for z in range(seg_np.shape[0]):
        sl = seg_np[z]
        for lbl in labels:
            if np.sum(sl == lbl) >= min_vox:
                return z
    # Fallback: return a reasonable top if nothing found
    return max(0, seg_np.shape[0] // 6)


def _find_inferior_extent(seg_np: np.ndarray, labels: set, min_vox: int = 15) -> int:
    """Find the largest Z (lowest in body) where any inferior structure appears."""
    for z in range(seg_np.shape[0] - 1, -1, -1):
        sl = seg_np[z]
        for lbl in labels:
            if np.sum(sl == lbl) >= min_vox:
                return z
    # Fallback: return a reasonable bottom
    return min(seg_np.shape[0] - 1, int(seg_np.shape[0] * 0.85))
# ===========================================================================
# NEW — 3D centroid-based Z-offset computation
# ===========================================================================

def _organ_centroid_z_mm(seg_np: np.ndarray,
                         spacing: Tuple[float, float, float],
                         origin: Tuple[float, float, float],
                         direction: np.ndarray,
                         label: int) -> Optional[float]:
    """
    Compute the physical Z-coordinate (mm) of the centroid of a single
    organ label in a seg_reg volume.

    Uses the full 3D voxel mask, not a single slice.

    Args:
        seg_np    : (Z, Y, X) integer array from seg_reg
        spacing   : (sx, sy, sz) in mm — SimpleITK order (XYZ)
        origin    : physical origin in mm — SimpleITK order (XYZ)
        direction : 3×3 direction matrix (row-major from SimpleITK)
        label     : integer organ label to measure

    Returns:
        Physical Z-coordinate of the centroid in mm, or None if organ
        has fewer than MIN_ORGAN_VOXELS voxels.
    """
    mask = seg_np == label
    n    = int(mask.sum())
    if n < MIN_ORGAN_VOXELS:
        return None

    # scipy gives (Z, Y, X) order for center_of_mass
    cz_vox, cy_vox, cx_vox = ndi.center_of_mass(mask)

    # Convert voxel centroid → physical mm
    # SimpleITK spacing is (sx, sy, sz), origin is (ox, oy, oz)
    # direction is stored row-major: [d00,d01,d02, d10,d11,d12, d20,d21,d22]
    # Physical = origin + direction @ (voxel * spacing)
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dir3       = direction.reshape(3, 3)

    vox_scaled = np.array([cx_vox * sx, cy_vox * sy, cz_vox * sz])
    physical   = np.array([ox, oy, oz]) + dir3 @ vox_scaled

    # Return Z component (index 2 in physical XYZ)
    return float(physical[2])


def _voxel_z_to_physical_z(img_sitk: sitk.Image, z_vox: int) -> float:
    """Voxel Z-index → physical S-coordinate (mm) in LPS."""
    origin    = np.array(img_sitk.GetOrigin())
    spacing   = np.array(img_sitk.GetSpacing())
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    return float((origin + z_vox * spacing[2] * direction[:, 2])[2])


def _physical_z_to_voxel(img_sitk: sitk.Image, z_mm: float) -> int:
    """Physical S-coordinate (mm) → voxel Z-index."""
    origin    = np.array(img_sitk.GetOrigin())
    spacing   = np.array(img_sitk.GetSpacing())
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    s_dir     = float(direction[2, 2])          # typically -1.0 for head-first CT
    return int(round((z_mm - origin[2]) / (spacing[2] * s_dir)))


def _organ_centroid_xyz_mm(seg_np, spacing, origin, dir3, label):
    """Return physical XYZ centroid of label as np.ndarray (3,), or None."""
    mask = seg_np == label
    if int(mask.sum()) < MIN_ORGAN_VOXELS:
        return None
    cz, cy, cx = ndi.center_of_mass(mask)
    sx, sy, sz = spacing
    vox_scaled = np.array([cx * sx, cy * sy, cz * sz])
    return np.array(origin) + dir3 @ vox_scaled   # shape (3,)


def _inplane_grid_matches(moving: sitk.Image, reference: sitk.Image,
                           tol: float = 0.5) -> bool:
    """True only if moving and reference share the same physical XY grid."""
    ref_o = np.array(reference.GetOrigin())
    mov_o = np.array(moving.GetOrigin())
    ref_d = np.array(reference.GetDirection()).reshape(3, 3)
    mov_d = np.array(moving.GetDirection()).reshape(3, 3)
    return (
        moving.GetSize()[:2]    == reference.GetSize()[:2]          and
        np.allclose(ref_o[:2],    mov_o[:2], atol=tol)              and
        np.allclose(reference.GetSpacing()[:2],
                    moving.GetSpacing()[:2],    atol=tol)            and
        np.allclose(ref_d[:2, :2], mov_d[:2, :2], atol=1e-4)
    )


def compute_3d_centroid_offset(
    ref_seg_np, ref_spacing, ref_origin, ref_dir,
    mov_seg_np, mov_spacing, mov_origin, mov_dir,
    phase="?",
    ) -> Tuple[np.ndarray, float, dict]:
    ref_dir3 = np.array(ref_dir).reshape(3, 3)
    mov_dir3 = np.array(mov_dir).reshape(3, 3)
 
    xyz_offsets_mm: List[np.ndarray] = []
    xyz_weights:    List[float]      = []
    debug:          dict             = {}
    organ_names = {1:'liver', 2:'spleen', 3:'kidney_left', 4:'kidney_right'}
 
    max_possible_weight = sum(
        ORGAN_WEIGHTS[l] for l in ANCHOR_LABELS if l in ORGAN_WEIGHTS
    )
 
    for label in sorted(ANCHOR_LABELS):
        ref_c = _organ_centroid_xyz_mm(ref_seg_np, ref_spacing, ref_origin, ref_dir3, label)
        mov_c = _organ_centroid_xyz_mm(mov_seg_np, mov_spacing, mov_origin, mov_dir3, label)
        name  = organ_names.get(label, str(label))
 
        if ref_c is None or mov_c is None:
            reason = "missing in ref" if ref_c is None else "missing in mov"
            debug[name] = {"status": "skipped", "reason": reason}
            continue
 
        diff_xyz = ref_c - mov_c          # (3,) XYZ mm, ref - mov
        w        = ORGAN_WEIGHTS.get(label, 1.0)
        debug[name] = {
            "status":      "ok",
            "ref_xyz_mm":  ref_c.round(2).tolist(),
            "mov_xyz_mm":  mov_c.round(2).tolist(),
            "diff_xyz_mm": diff_xyz.round(2).tolist(),
            "diff_z_mm":   round(float(diff_xyz[2]), 2),
            "weight":      w,
        }
        xyz_offsets_mm.append(diff_xyz)
        xyz_weights.append(w)
 
    if not xyz_offsets_mm:
        print(f"      [{phase}] WARNING: no anchor organs found — offset=0, conf=0")
        return np.zeros(3), 0.0, debug
 
    wt  = np.array(xyz_weights)
    pts = np.array(xyz_offsets_mm)
    weighted_offset_xyz = (wt[:, None] * pts).sum(0) / wt.sum()
 
    # Single MAD filter on 3D distance from weighted mean
    if len(pts) > 2:
        norms = np.linalg.norm(pts - weighted_offset_xyz, axis=1)
        mad   = float(np.median(np.abs(norms - np.median(norms))))
        threshold = max(5.0, 2.0 * 1.4826 * mad)
        keep  = norms <= threshold
        if keep.sum() > 0 and keep.sum() < len(pts):
            weighted_offset_xyz = (wt[keep, None] * pts[keep]).sum(0) / wt[keep].sum()
            for name, keep_flag in zip(
                [k for k in debug if debug[k].get("status") == "ok"], keep
            ):
                if not keep_flag:
                    debug[name]["status"] = "outlier_removed"
 
    # Reject if magnitude implausibly large
    offset_mag = float(np.linalg.norm(weighted_offset_xyz))
    if offset_mag > MAX_OFFSET_MM:
        print(f"      [{phase}] WARNING: |offset|={offset_mag:.1f}mm > {MAX_OFFSET_MM}mm")
        return np.zeros(3), 0.0, debug
 
    # BUG 6b FIX: measure organ agreement as mean deviation from consensus,
    # not raw std of absolute offsets (which is large even when organs agree).
    if len(pts) > 1:
        deviations = np.linalg.norm(pts - weighted_offset_xyz, axis=1)
        consensus_std = float(deviations.mean())
        if consensus_std > 12.0:
            print(f"      [{phase}] WARNING: organ consensus_std={consensus_std:.1f}mm > 12mm")
            return np.zeros(3), 0.0, debug
 
    confidence = min(1.0, float(wt.sum()) / max_possible_weight)
    debug['_summary'] = {
        'offset_xyz_mm': weighted_offset_xyz.round(2).tolist(),
        'offset_mag_mm': round(offset_mag, 2),
        'confidence':    round(confidence, 3),
        'n_organs':      len(xyz_offsets_mm),
    }
    return weighted_offset_xyz, confidence, debug
# ===========================================================================
# UPDATED align_study() — uses compute_z_offset_3d() with fallback
# ===========================================================================

def align_study(
        study_id: str,
        series_info: Dict[str, dict],
        reference_series_id: str,
        output_dir: str,
        search_region: str = 'upper',          # kept for API compatibility, unused
        min_overlap_slices: int = 30,
        force_recompute: bool = False,
) -> dict:
    """
    Align all CT phases to the NC reference using 3D organ centroid matching.

    For each moving phase:
      1. compute_z_offset_3d() estimates the Z-offset in physical mm,
         aggregated over all available anchor organs (liver, spleen, kidneys).
      2. If confidence >= MIN_CONFIDENCE and |offset| <= MAX_OFFSET_MM,
         use the 3D centroid offset.
      3. Otherwise fall back to per-series organ Z-centre crop (baseline
         strategy) for that series only.
    """
    # FIX 2: normalise to absolute path — ITK's C-level NIfTI writer does not
    # resolve "../" relative paths, causing silent write failures on macOS/Linux.
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, f"{study_id}_alignment_metadata.json")

    if not force_recompute and os.path.exists(meta_path):
        print(f"   Already aligned — skipping ({study_id[:40]}...)")
        with open(meta_path) as f:
            return {"status": "skipped", "metadata": json.load(f)}
    log = get_stage_logger("alignment", study_id=study_id)

    print(f"\n{'='*80}")
    print(f"Aligning study (v3 — 3D centroid): {study_id}")
    print(f"Reference: {reference_series_id[:50]}...")
    print(f"{'='*80}")

    # ── Load reference ───────────────────────────────────────────────────
    ref          = series_info[reference_series_id]
    ref_img      = sitk.ReadImage(ref["image"])
    ref_img_np   = sitk.GetArrayFromImage(ref_img)
    ref_seg_full = sitk.ReadImage(ref["seg_full"]) if ref["seg_full"] else None

    if ref["seg_reg"] is None:
        raise ValueError(f"NC series has no seg_reg. Series: {reference_series_id}")

    ref_seg_reg_sitk = sitk.ReadImage(ref["seg_reg"])
    ref_seg_np       = np.round(
        sitk.GetArrayFromImage(ref_seg_reg_sitk)
    ).astype(np.int32)

    ref_spacing  = ref_img.GetSpacing()           # (sx, sy, sz)
    ref_origin   = ref_img.GetOrigin()            # (ox, oy, oz)
    ref_dir      = ref_img.GetDirection()         # flat 9-element tuple

    # ── Step 1: compute 3D centroid offsets for each moving phase ────────
    print("\n[Step 1] Computing 3D organ centroid Z-offsets...")

    alignment: Dict[str, dict] = {}

    # Reference entry (offset = 0 by definition)
    alignment[reference_series_id] = {
        "phase":          ref["phase"],
        "z_offset":       0,
        "confidence":     1.0,
        "method":         "reference",
        "original_size":  ref_img_np.shape,
        "img_sitk":       ref_img,
        "seg_full_sitk":  ref_seg_full,
        "seg_reg_sitk":   ref_seg_reg_sitk,
    }

    for sid, info in series_info.items():
        if sid == reference_series_id:
            continue

        phase = info["phase"]
        print(f"\n   [{phase}] {sid[:50]}...")

        mov_img    = sitk.ReadImage(info["image"])
        mov_img_np = sitk.GetArrayFromImage(mov_img)
        mov_seg_full = sitk.ReadImage(info["seg_full"]) if info["seg_full"] else None

        if info["seg_reg"] is None:
            raise ValueError(f"Moving series {sid} has no seg_reg.")

        mov_seg_reg_sitk = sitk.ReadImage(info["seg_reg"])

        # In-plane resample if needed (same as v2)
        # if mov_img_np.shape[1:] != ref_img_np.shape[1:]:
        if not _inplane_grid_matches(mov_img, ref_img):
            print(f"      In-plane mismatch — resampling...")
            mov_img          = _resample_inplane(mov_img, ref_img)
            mov_img_np       = sitk.GetArrayFromImage(mov_img)
            mov_seg_reg_sitk = _resample_inplane(
                mov_seg_reg_sitk, ref_img,
                interpolator=sitk.sitkNearestNeighbor, default_val=0
            )
            if mov_seg_full is not None:
                mov_seg_full = _resample_inplane(
                    mov_seg_full, ref_img,
                    interpolator=sitk.sitkNearestNeighbor, default_val=0
                )

        mov_seg_np   = np.round(
            sitk.GetArrayFromImage(mov_seg_reg_sitk)
        ).astype(np.int32)
        mov_spacing  = mov_img.GetSpacing()
        mov_origin   = mov_img.GetOrigin()
        mov_dir      = mov_img.GetDirection()
        mov_z_sp     = float(mov_spacing[2])   # Z spacing in mm

        # ── 3D centroid offset ───────────────────────────────────────────
        offset_xyz_mm, confidence, organ_debug = compute_3d_centroid_offset(
            ref_seg_np,  ref_spacing, ref_origin, np.array(ref_dir),
            mov_seg_np,  mov_spacing, mov_origin, np.array(mov_dir),
            phase=phase,
        )

        MIN_Z_OFFSET_MM = 7.5    # below this, treat as same-session noise
        MIN_XY_OFFSET_MM = 5.0  # below this, XY shift is within registration capture range

        # FIX 1: initialise all output variables before the confidence branch so
        # they are always defined when the alignment dict and log.metric are written,
        # regardless of which branch executes.
        method       = "no_offset"
        apply_xy     = False
        apply_z      = False
        z_offset_vox = 0

        if confidence >= MIN_CONFIDENCE:
            # Apply XY translation if it exceeds the noise threshold
            xy_offset_mm   = offset_xyz_mm[:2]   # [x, y]
            z_offset_mm    = float(offset_xyz_mm[2])
            apply_xy = np.linalg.norm(xy_offset_mm) >= MIN_XY_OFFSET_MM
            apply_z  = abs(z_offset_mm) >= MIN_Z_OFFSET_MM

            if apply_xy:
                # Shift the moving image origin in XY to align centroids
                # This is a pure translation: new_origin = old_origin + xy_offset
                # The resample_inplane function already handles XY resampling;
                # we need to adjust the reference origin it uses.
                new_xy_origin = (
                    float(mov_origin[0]) + float(xy_offset_mm[0]),
                    float(mov_origin[1]) + float(xy_offset_mm[1]),
                    float(mov_origin[2]),
                )
                _xy_t = sitk.TranslationTransform(3, [
                    float(xy_offset_mm[0]),   # ← positive, not negative
                    float(xy_offset_mm[1]),
                    0.0,
                ])
                xy_shifted_img = sitk.Resample(
                    mov_img, mov_img, _xy_t,
                    sitk.sitkLinear, -1024.0, mov_img.GetPixelID(),
                )
                xy_shifted_seg_reg = sitk.Resample(
                    mov_seg_reg_sitk, mov_seg_reg_sitk, _xy_t,
                    sitk.sitkNearestNeighbor, 0, mov_seg_reg_sitk.GetPixelID(),
                )
                if mov_seg_full is not None:
                    mov_seg_full = sitk.Resample(
                        mov_seg_full, mov_seg_full, _xy_t,
                        sitk.sitkNearestNeighbor, 0, mov_seg_full.GetPixelID(),
                    )
                mov_img          = xy_shifted_img
                mov_seg_reg_sitk = xy_shifted_seg_reg
                mov_img_np       = sitk.GetArrayFromImage(mov_img)
                mov_seg_np       = np.round(sitk.GetArrayFromImage(mov_seg_reg_sitk)).astype(np.int32)
                print(f"      [{phase}] Applied XY translation: "
                      f"({xy_offset_mm[0]:+.1f}, {xy_offset_mm[1]:+.1f}) mm")

            # # Convert Z offset from mm to voxels for the crop step
            # mov_dir3   = np.array(mov_img.GetDirection()).reshape(3, 3)
            # mov_z_sp   = float(mov_img.GetSpacing()[2])
            # z_sign     = float(mov_dir3[2, 2])
            # z_offset_vox = int(round(z_offset_mm / mov_z_sp)) if apply_z else 0
            # method = "3d_centroid_xyz"
            # print(f"      [{phase}] Z-offset: {z_offset_mm:+.1f}mm = {z_offset_vox} voxels "
            #       f"{'(applied)' if apply_z else '(skipped — same-session noise)'}")
            if apply_z:
                # Apply Z translation physically — consistent with XY shift above.
                # TranslationTransform(3, [tx, ty, tz]): output[p] = input[p - t]
                # To shift content up by z_offset_mm: t_z = -z_offset_mm
                _z_t = sitk.TranslationTransform(3, [0.0, 0.0, -z_offset_mm])
                mov_img = sitk.Resample(
                    mov_img, mov_img, _z_t,
                    sitk.sitkLinear, -1024.0, mov_img.GetPixelID(),
                )
                mov_seg_reg_sitk = sitk.Resample(
                    mov_seg_reg_sitk, mov_seg_reg_sitk, _z_t,
                    sitk.sitkNearestNeighbor, 0, mov_seg_reg_sitk.GetPixelID(),
                )
                if mov_seg_full is not None:
                    mov_seg_full = sitk.Resample(
                        mov_seg_full, mov_seg_full, _z_t,
                        sitk.sitkNearestNeighbor, 0, mov_seg_full.GetPixelID(),
                    )
                mov_img_np = sitk.GetArrayFromImage(mov_img)

                # FIX 5: also update every image's Z-origin by z_offset_mm so that
                # Step 2's series_z_mm computation sees the corrected physical extent.
                # Without this the origin stays at the old Z position, Step 2 still
                # uses unshifted extents, and the crop intersection is wrong → gap.
                for _im in [mov_img, mov_seg_reg_sitk] + (
                        [mov_seg_full] if mov_seg_full is not None else []):
                    _o      = list(_im.GetOrigin())
                    _o[2]  += z_offset_mm   # advance Z origin by the applied offset
                    _im.SetOrigin(_o)
                # Re-read numpy array from the origin-updated image
                mov_img_np = sitk.GetArrayFromImage(mov_img)
                mov_seg_np = np.round(
                    sitk.GetArrayFromImage(mov_seg_reg_sitk)
                ).astype(np.int32)
                print(f"      [{phase}] Applied Z shift: content + origin "
                      f"{z_offset_mm:+.1f}mm")

            z_offset_vox = 0   # ← always 0 now; physical resample handles it
            method = "3d_centroid_xyz"

        else:
            z_offset_vox = 0
            apply_xy     = False
            method       = "fallback_crop"
            print(f"      [{phase}] LOW CONFIDENCE ({confidence:.2f}) — no offset applied")

        log.metric("centroid_offset", {
            "series_id":     sid,
            "phase":         phase,
            "offset_xyz_mm": offset_xyz_mm.tolist() if hasattr(offset_xyz_mm, "tolist")
                             else list(offset_xyz_mm),
            "offset_mag_mm": round(float(np.linalg.norm(offset_xyz_mm)), 2),
            "confidence":    round(float(confidence), 3),
            "method":        method,
            "apply_xy":      apply_xy,
            "apply_z":       apply_z,
            "z_offset_vox":  z_offset_vox,
        })
        if confidence < MIN_CONFIDENCE:
            log.warning(
                f"[{phase}] Low confidence ({confidence:.2f}) — no offset applied. "
                f"Check seg_reg organ count for series {sid[:40]}."
            )

        alignment[sid] = {
            "phase":          phase,
            "z_offset":       z_offset_vox,
            "offset_xyz_mm":  offset_xyz_mm.tolist(),
            "confidence":     confidence,
            "method":         method,
            "organ_debug":    organ_debug,
            "original_size":  mov_img_np.shape,
            "img_sitk":       mov_img,
            "seg_full_sitk":  mov_seg_full,
            "seg_reg_sitk":   mov_seg_reg_sitk,
        }

    for sid, d in alignment.items():
        img = d["img_sitk"]
        top_mm = _voxel_z_to_physical_z(img, 0)
        bot_mm = _voxel_z_to_physical_z(img, d["original_size"][0] - 1)
        print(f"   {d['phase']:15s}: Z range [{top_mm:.1f} → {bot_mm:.1f}mm]  "
            f"depth={d['original_size'][0]} slices")


    # ── Step 2: common Z-overlap in PHYSICAL MM space ────────────────────
    # Abandon voxel-offset arithmetic entirely. Convert each series' full
    # extent to physical mm, take the intersection, then map back to voxels.
    # This is robust to volumes of different depths starting at different Z.
    print("\n[Step 2] Computing common Z-overlap (physical mm)...")

    # Collect physical Z extent of each series (after any in-plane resampling)
    series_z_mm: Dict[str, Tuple[float, float]] = {}
    for sid, d in alignment.items():
        img   = d["img_sitk"]
        D     = d["original_size"][0]
        # In head-first CT: voxel 0 = most superior (least negative S)
        #                   voxel D-1 = most inferior (most negative S)
        top_mm = _voxel_z_to_physical_z(img, 0)
        bot_mm = _voxel_z_to_physical_z(img, D - 1)
        # Ensure top > bot in physical S (superior is less negative = larger value)
        sup_mm = max(top_mm, bot_mm)   # superior boundary (less negative)
        inf_mm = min(top_mm, bot_mm)   # inferior boundary (more negative)
        series_z_mm[sid] = (sup_mm, inf_mm)
        print(f"   {d['phase']:15s}: Z [{inf_mm:.1f} → {sup_mm:.1f}mm]  "
            f"depth={D}")

    # Physical intersection: most inferior superior-boundary, most superior inferior-boundary
    common_sup_mm = min(v[0] for v in series_z_mm.values())  # least superior top
    common_inf_mm = max(v[1] for v in series_z_mm.values())  # least inferior bottom
    print(f"\n   Physical overlap: {common_inf_mm:.1f}mm → {common_sup_mm:.1f}mm  "
        f"= {(common_sup_mm - common_inf_mm):.1f}mm")

    if common_sup_mm <= common_inf_mm:
        raise ValueError(
            f"No physical Z overlap across series: "
            f"sup={common_sup_mm:.1f}mm, inf={common_inf_mm:.1f}mm"
        )

    # ── Step 3: convert shared mm window → per-series voxel crop indices ─
    print("\n[Step 3] Cropping all series to common physical window...")

    # Reference crop — sets the final_origin
    ref_img   = alignment[reference_series_id]["img_sitk"]
    ref_cs    = _physical_z_to_voxel(ref_img, common_sup_mm)
    ref_ce    = _physical_z_to_voxel(ref_img, common_inf_mm)
    if ref_cs > ref_ce:
        ref_cs, ref_ce = ref_ce, ref_cs
    ref_cs    = max(0, min(ref_cs, alignment[reference_series_id]["original_size"][0]))
    ref_ce    = max(0, min(ref_ce, alignment[reference_series_id]["original_size"][0]))

    ref_sitk_cropped = _crop_sitk(ref_img, ref_cs, ref_ce)
    final_origin     = ref_sitk_cropped.GetOrigin()
    overlap_depth    = ref_ce - ref_cs

    print(f"   Reference NC: crop [{ref_cs}:{ref_ce}] = {overlap_depth} slices")

    if overlap_depth < min_overlap_slices:
        raise ValueError(
            f"Insufficient overlap: {overlap_depth} < {min_overlap_slices} slices."
        )

    out_volumes:  Dict[str, sitk.Image] = {}
    out_seg_full: Dict[str, Optional[sitk.Image]] = {}
    out_seg_reg:  Dict[str, Optional[sitk.Image]] = {}

    for sid, d in alignment.items():
        img = d["img_sitk"]
        D   = d["original_size"][0]

        if sid == reference_series_id:
            cs, ce = ref_cs, ref_ce
        else:
            cs = _physical_z_to_voxel(img, common_sup_mm)
            ce = _physical_z_to_voxel(img, common_inf_mm)
            if cs > ce:
                cs, ce = ce, cs
            cs = max(0, min(cs, D))
            ce = max(0, min(ce, D))
            if (ce - cs) != overlap_depth:
                diff = (ce - cs) - overlap_depth
                ce  -= diff
            cs = max(0, min(cs, D))
            ce = max(0, min(ce, D))

        # Sanity: verify physical Z of crop start matches reference crop start
        crop_top_mm = _voxel_z_to_physical_z(img, cs)
        gap         = abs(crop_top_mm - _voxel_z_to_physical_z(ref_img, ref_cs))
        if gap > 10.0:
            log.warning(
                f"[{d['phase']}] crop boundary gap={gap:.1f}mm > 10mm — "
                f"Z-origin mismatch between phases."
            )
        status = "OK" if gap <= 10.0 else f"WARNING {gap:.1f}mm gap"
        print(f"   {d['phase']:15s}: crop [{cs}:{ce}] ...")

        vol_cropped      = _force_origin(_crop_sitk(img, cs, ce), final_origin)
        out_volumes[sid] = vol_cropped

        sf = d["seg_full_sitk"]
        out_seg_full[sid] = (
            _force_origin(_crop_sitk(sf, cs, ce), final_origin)
            if sf is not None else None
        )
        out_seg_reg[sid] = _force_origin(
            _crop_sitk(d["seg_reg_sitk"], cs, ce), final_origin
        )


    # ── Step 4: save ──────────────────────────────────────────────────────
    print("\n[Step 4] Saving aligned volumes...")

    saved_files: Dict[str, dict] = {}

    for sid, d in alignment.items():
        phase  = d["phase"]
        prefix = os.path.join(output_dir, f"{study_id}_{sid}")

        vol_path = f"{prefix}_aligned.nii.gz"
        sitk.WriteImage(out_volumes[sid], vol_path)
        saved: dict = {"volume": vol_path}

        sf = out_seg_full.get(sid)
        if sf is not None:
            sf_path = f"{prefix}_aligned_seg_full.nii.gz"
            sitk.WriteImage(sf, sf_path)
            saved["seg_full"] = sf_path

        sr = out_seg_reg.get(sid)
        if sr is not None:
            sr_path = f"{prefix}_aligned_seg_reg.nii.gz"
            sitk.WriteImage(sr, sr_path)
            saved["seg_reg"] = sr_path

        saved_files[sid] = saved
        print(f"   [{phase}] → {os.path.basename(vol_path)}")

    # ── Step 5: metadata ──────────────────────────────────────────────────
    metadata = {
        "study_id":         study_id,
        "reference_series": reference_series_id,
        "method":           "3d_centroid_v3",
        "min_confidence":   MIN_CONFIDENCE,
        "max_offset_mm":    MAX_OFFSET_MM,
        "final_origin":     list(final_origin),
        "final_spacing":    list(out_volumes[reference_series_id].GetSpacing()),
        "final_size":       list(out_volumes[reference_series_id].GetSize()),
        "common_overlap": {
            "sup_mm":    round(common_sup_mm, 2),
            "inf_mm":    round(common_inf_mm, 2),
            "ref_start": int(ref_cs),
            "ref_end":   int(ref_ce),
            "depth":     int(overlap_depth),
        },
        "series": {
            sid: {
                "phase":          d["phase"],
                "confidence":     float(d["confidence"]),
                "method":         d["method"],
                "original_shape": [int(x) for x in d["original_size"]],
                "z_range_mm":     [round(series_z_mm[sid][1], 2),
                                round(series_z_mm[sid][0], 2)],
                "organ_debug":    d.get("organ_debug", {}),
            } for sid, d in alignment.items()
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n   Metadata → {os.path.basename(meta_path)}")

    print(f"\n{'='*80}")
    print(f"Alignment complete (v3): {overlap_depth} slice overlap, "
          f"{len(out_volumes)} series")
    print(f"{'='*80}")

    for sid, d in alignment.items():
            log.stage_summary({
                "series_id":     sid,
                "phase":         d["phase"],
                "status":        "ok",
                "confidence":    float(d["confidence"]),
                "method":        d["method"],
                "z_offset_vox":  int(d["z_offset"]),
                "offset_xyz_mm": d.get("offset_xyz_mm", [0, 0, 0]),
                "overlap_depth": int(overlap_depth),
            })

    return {
        "status":           "complete",
        "metadata":         metadata,
        "aligned_volumes":  out_volumes,
        "aligned_seg_full": out_seg_full,
        "aligned_seg_reg":  out_seg_reg,
    }


# ===========================================================================
# BASELINE — unchanged from v2
# ===========================================================================
def baseline_crop_study(
        study_id: str,
        series_info: Dict[str, dict],
        reference_series_id: str,
        output_dir: str,
        min_overlap_slices: int = 30,
        force_recompute: bool = False,
) -> dict:
    """
    Baseline crop: each series is cropped to the physical mm intersection of
    all series' organ extents (superior/inferior).

    No Z-alignment is applied. The crop window is determined by the
    most-limited series — if Arterial has no pelvis, nobody gets pelvis.

    Superior boundary: first slice containing any SUPERIOR_LABELS organ
    Inferior boundary: last slice containing any INFERIOR_LABELS organ
    Both are converted to physical S-mm, intersection taken, then each
    series is cropped back to those physical mm boundaries.
    """
    # FIX 2b: normalise to absolute path for the same ITK NIfTI write reason.
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, f"{study_id}_baseline_metadata.json")

    # FIX 4: instantiate the logger so --baseline runs create the
    # pipeline_logs/alignment/ folder and write structured records,
    # identical to align_study behaviour.
    log = get_stage_logger("alignment", study_id=study_id)

    if not force_recompute and os.path.exists(meta_path):
        print(f"   Already done (baseline) — skipping...")
        log.info(f"Skipping baseline — already done: {study_id[:50]}")
        with open(meta_path) as f:
            return {"status": "skipped", "metadata": json.load(f)}

    print(f"\n{'='*80}")
    print(f"BASELINE CROP (organ intersection): {study_id}")
    print(f"{'='*80}")

    ref     = series_info[reference_series_id]
    ref_img = sitk.ReadImage(ref["image"])

    series_data: Dict[str, dict] = {}

    # ── Step 1: per-series organ extents in physical mm ──────────────────
    print("\n[Step 1] Computing per-series organ extents (physical mm)...")

    for sid, info in series_info.items():
        img          = sitk.ReadImage(info["image"])
        seg_reg_sitk = sitk.ReadImage(info["seg_reg"])
        seg_full     = sitk.ReadImage(info["seg_full"]) if info["seg_full"] else None

        if not _inplane_grid_matches(img, ref_img):
            print(f"   [{info['phase']}] In-plane resampling...")
            img          = _resample_inplane(img, ref_img)
            seg_reg_sitk = _resample_inplane(
                seg_reg_sitk, ref_img, sitk.sitkNearestNeighbor, 0
            )
            if seg_full is not None:
                seg_full = _resample_inplane(
                    seg_full, ref_img, sitk.sitkNearestNeighbor, 0
                )

        img_np  = sitk.GetArrayFromImage(img)
        seg_np  = np.round(sitk.GetArrayFromImage(seg_reg_sitk)).astype(np.int32)
        depth   = img_np.shape[0]

        # Voxel extents of superior/inferior landmarks
        sup_vox = _find_superior_extent(seg_np, SUPERIOR_LABELS)
        inf_vox = _find_inferior_extent(seg_np, INFERIOR_LABELS)

        # Convert to physical S-coordinate (mm)
        sup_mm  = _voxel_z_to_physical_z(img, sup_vox)
        inf_mm  = _voxel_z_to_physical_z(img, inf_vox)

        # In LPS head-first CT: superior = more negative S.
        # Sanity: if no inferior landmark found, use the actual volume bottom.
        vol_top_mm    = _voxel_z_to_physical_z(img, 0)
        vol_bottom_mm = _voxel_z_to_physical_z(img, depth - 1)

        print(f"   [{info['phase']:15s}]  "
              f"sup_vox={sup_vox:4d} ({sup_mm:.1f}mm)  "
              f"inf_vox={inf_vox:4d} ({inf_mm:.1f}mm)  "
              f"depth={depth}  "
              f"vol=[{vol_top_mm:.1f}:{vol_bottom_mm:.1f}mm]")

        series_data[sid] = {
            "phase":         info["phase"],
            "img_sitk":      img,
            "seg_reg_sitk":  seg_reg_sitk,
            "seg_full_sitk": seg_full,
            "depth":         depth,
            "sup_vox":       sup_vox,
            "inf_vox":       inf_vox,
            "sup_mm":        sup_mm,
            "inf_mm":        inf_mm,
        }

    # ── Step 2: physical mm Z-intersection — same logic as align_study ──────
    # FIX CROP: use the raw physical Z extents of each series (volume start/end),
    # not organ landmarks. This makes baseline produce the SAME crop window as
    # aligned, differing only in that no centroid-offset correction is applied.
    # That is the correct definition of "baseline" for an ablation: same crop,
    # no alignment.
    print("\n[Step 2] Computing common Z-overlap (physical mm intersection)...")

    series_z_extents: Dict[str, Tuple[float, float]] = {}
    for sid, d in series_data.items():
        img   = d["img_sitk"]
        D     = d["depth"]
        top_mm = _voxel_z_to_physical_z(img, 0)
        bot_mm = _voxel_z_to_physical_z(img, D - 1)
        sup_mm_ext = max(top_mm, bot_mm)   # physically superior (less negative)
        inf_mm_ext = min(top_mm, bot_mm)   # physically inferior (more negative)
        series_z_extents[sid] = (sup_mm_ext, inf_mm_ext)
        print(f"   [{d['phase']:15s}]  Z extent [{inf_mm_ext:.1f} → {sup_mm_ext:.1f}mm]")

    # Intersection: most-constrained superior top, most-constrained inferior bottom
    common_top_mm    = min(v[0] for v in series_z_extents.values())  # least-superior top
    common_bottom_mm = max(v[1] for v in series_z_extents.values())  # least-inferior bottom
    print(f"\n   Physical overlap: {common_bottom_mm:.1f}mm → {common_top_mm:.1f}mm  "
          f"= {common_top_mm - common_bottom_mm:.1f}mm")

    if common_top_mm <= common_bottom_mm:
        raise ValueError(
            f"No physical Z overlap across series. "
            f"sup={common_top_mm:.1f}mm, inf={common_bottom_mm:.1f}mm"
        )

    # ── Step 3: convert shared mm window to per-series voxel crop indices ─
    # Same logic as align_study Step 3: map physical mm boundaries back to
    # per-series voxel indices. No padding, no organ-landmark offsets.
    print("\n[Step 3] Converting mm window to per-series voxel crops...")

    for sid, d in series_data.items():
        img = d["img_sitk"]
        # common_top_mm is the superior (less negative) boundary — maps to lower voxel index
        # common_bottom_mm is the inferior (more negative) boundary — maps to higher voxel index
        cs_raw = _physical_z_to_voxel(img, common_top_mm)
        ce_raw = _physical_z_to_voxel(img, common_bottom_mm)

        if cs_raw > ce_raw:
            cs_raw, ce_raw = ce_raw, cs_raw

        cs = max(0, min(cs_raw, d["depth"]))
        ce = max(0, min(ce_raw, d["depth"]))

        d["crop_start"] = cs
        d["crop_end"]   = ce
        print(f"   [{d['phase']:15s}]  crop [{cs}:{ce}] = {ce-cs} slices")

    depths = {sid: d["crop_end"] - d["crop_start"] for sid, d in series_data.items()}
    # Use NC reference depth as the canonical window depth (same as align_study)
    window_depth = depths[reference_series_id]
    all_depths   = list(depths.values())
    if max(all_depths) - min(all_depths) > 2:
        print(f"   ⚠  crop depths not uniform {all_depths} — rounding from different spacings")

    if window_depth < min_overlap_slices:
        raise ValueError(
            f"Common window too narrow: {window_depth} < {min_overlap_slices} slices."
        )

    # ── Step 4: crop each series and set its own correct origin ──────────
    # FIX 3: do NOT force all series to the NC reference's origin.
    # All series are cropped to the same physical mm window, so each crop-start
    # voxel already corresponds to the same physical boundary — but _physical_z_to_voxel
    # rounds differently per series, causing 1-2 voxel shifts. Stamping every header
    # with the NC origin hides this: the pixel content starts at a different physical
    # slice than the header claims, making ITK-SNAP overlay them with a Z offset.
    # Fix: compute each series' origin from ITS OWN crop start, so header = content.
    print("\n[Step 4] Cropping all series to common window...")

    out_volumes  = {}
    out_seg_full = {}
    out_seg_reg  = {}

    for sid, d in series_data.items():
        cs, ce = d["crop_start"], d["crop_end"]

        # Per-series correct origin: physical position of THIS series' crop-start voxel
        series_origin = _compute_new_origin(d["img_sitk"], cs)

        out_volumes[sid] = _force_origin(
            _crop_sitk(d["img_sitk"], cs, ce), series_origin
        )
        out_seg_reg[sid] = _force_origin(
            _crop_sitk(d["seg_reg_sitk"], cs, ce), series_origin
        )
        if d["seg_full_sitk"] is not None:
            out_seg_full[sid] = _force_origin(
                _crop_sitk(d["seg_full_sitk"], cs, ce), series_origin
            )
        print(f"   [{d['phase']:15s}] cropped [{cs}:{ce}] ({ce-cs} slices)  "
              f"origin_z={series_origin[2]:.1f}mm")

    # ── Step 5: save ──────────────────────────────────────────────────────
    print("\n[Step 5] Saving baseline volumes...")
    saved_files: Dict[str, dict] = {}

    for sid, d in series_data.items():
        prefix   = os.path.join(output_dir, f"{study_id}_{sid}")
        vol_path = f"{prefix}_baseline.nii.gz"
        sitk.WriteImage(out_volumes[sid], vol_path)
        saved: dict = {"volume": vol_path}

        sf = out_seg_full.get(sid)
        if sf is not None:
            sf_path = f"{prefix}_baseline_seg_full.nii.gz"
            sitk.WriteImage(sf, sf_path)
            saved["seg_full"] = sf_path

        sr = out_seg_reg.get(sid)
        if sr is not None:
            sr_path = f"{prefix}_baseline_seg_reg.nii.gz"
            sitk.WriteImage(sr, sr_path)
            saved["seg_reg"] = sr_path

        saved_files[sid] = saved
        print(f"   [{d['phase']}] → {os.path.basename(vol_path)}")

    metadata = {
        "study_id":           study_id,
        "reference_series":   reference_series_id,
        "method":             "baseline_organ_intersection_v4",
        "common_top_mm":      round(common_top_mm, 2),
        "common_bottom_mm":   round(common_bottom_mm, 2),
        "window_depth":       int(window_depth),
        "superior_padding":   SUPERIOR_PADDING,
        "inferior_padding":   INFERIOR_PADDING,
        "series": {sid: {
            "phase":       d["phase"],
            "sup_vox":     int(d["sup_vox"]),
            "inf_vox":     int(d["inf_vox"]),
            "sup_mm":      round(d["sup_mm"], 2),
            "inf_mm":      round(d["inf_mm"], 2),
            "crop_start":  int(d["crop_start"]),
            "crop_end":    int(d["crop_end"]),
        } for sid, d in series_data.items()},
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # FIX 4b: emit structured log records so baseline runs populate
    # pipeline_logs/alignment/ identically to align_study.
    log.metric("baseline_crop", {
        "n_series":         len(series_data),
        "window_depth":     int(window_depth),
        "common_top_mm":    round(common_top_mm, 2),
        "common_bottom_mm": round(common_bottom_mm, 2),
    })
    for sid, d in series_data.items():
        log.stage_summary({
            "series_id":    sid,
            "phase":        d["phase"],
            "status":       "ok",
            "method":       "baseline_crop",
            "crop_start":   int(d["crop_start"]),
            "crop_end":     int(d["crop_end"]),
            "window_depth": int(window_depth),
            "origin_z_mm":  round(_compute_new_origin(
                                d["img_sitk"], d["crop_start"])[2], 2),
        })

    print(f"\n{'='*80}")
    print(f"Baseline crop complete: {window_depth} slices  "
          f"({common_top_mm:.1f}mm → {common_bottom_mm:.1f}mm)")
    print(f"{'='*80}")

    return {
        "status":      "complete",
        "metadata":    metadata,
        "out_volumes": out_volumes,
        "out_seg_full": out_seg_full,
        "out_seg_reg":  out_seg_reg,
    }
# ===========================================================================
# VERIFICATION UTILITY (unchanged from v2)
# ===========================================================================

def verify_study_alignment(study_dir: str) -> bool:
    """Check all _aligned.nii.gz in a folder share the same grid."""
    import glob
    files = [f for f in glob.glob(os.path.join(study_dir, "*_aligned.nii.gz"))
             if "seg" not in f]
    if not files:
        print(f"No aligned volumes in {study_dir}")
        return False
    ref      = sitk.ReadImage(files[0])
    ref_size = ref.GetSize()
    ok       = True
    for f in files:
        img   = sitk.ReadImage(f)
        match = img.GetSize() == ref_size
        tag   = "✓" if match else "✗"
        print(f"  {tag} {os.path.basename(f)}: {img.GetSize()}")
        if not match:
            ok = False
    return ok