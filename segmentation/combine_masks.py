import os
import sys
import json
import numpy as np
import nibabel as nib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_logger import get_stage_logger

log = get_stage_logger("segmentation")

# ---------------------------------------------------------------------------
# LABEL MAP
# ---------------------------------------------------------------------------
# Format:  label_int: (stem, tier, in_reg_mask)
#
#   tier 1 = bone        -> _seg_full + _seg_reg  (highest registration weight)
#   tier 2 = stable soft -> _seg_full + _seg_reg
#   tier 3 = other soft  -> _seg_full only         (evaluation, not metric mask)
#
#   in_reg_mask must be True for tier 1 and tier 2, False for tier 3.
#
# Gut (colon, small_bowel, duodenum, rectum) is not listed — excluded entirely.
#
# TotalSegmentator does NOT output lung_left / lung_right as merged files;
# it outputs per-lobe files which are listed under tier 3 below.
# ---------------------------------------------------------------------------

LABEL_MAP = {

    # =========================================================================
    # TIER 2 — stable soft organs  (labels 1–20)
    # These labels are fixed to match align_data_v3.py's ORGAN_WEIGHTS.
    # =========================================================================

    1:  ("liver",                        2, True),
    2:  ("spleen",                       2, True),
    3:  ("kidney_left",                  2, True),
    4:  ("kidney_right",                 2, True),
    5:  ("pancreas",                     2, True),
    6:  ("gallbladder",                  2, True),

    # Core muscles (paraspinal + iliopsoas — contrast-independent, stable)
    7:  ("autochthon_left",              2, True),
    8:  ("autochthon_right",             2, True),
    9:  ("iliopsoas_left",               2, True),
    10: ("iliopsoas_right",              2, True),

    # Major vessels
    13: ("aorta",                        2, True),
    14: ("inferior_vena_cava",           2, True),
    15: ("portal_vein_and_splenic_vein", 2, True),
    16: ("iliac_artery_left",            2, True),
    17: ("iliac_artery_right",           2, True),
    18: ("iliac_vena_left",              2, True),
    19: ("iliac_vena_right",             2, True),

    # =========================================================================
    # TIER 1 — bones  (labels 21–99)
    # =========================================================================

    # Lumbar spine — labels 21–25 fixed to match align_data_v3.py (21=L1,22=L2,23=L3)
    21: ("vertebrae_L1",  1, True),
    22: ("vertebrae_L2",  1, True),
    23: ("vertebrae_L3",  1, True),
    24: ("vertebrae_L4",  1, True),
    25: ("vertebrae_L5",  1, True),

    # Lower thoracic (inside abdominal crop)
    26: ("vertebrae_T10", 1, True),
    27: ("vertebrae_T11", 1, True),
    28: ("vertebrae_T12", 1, True),

    # Upper thoracic (often outside abdominal crop — kept for completeness)
    29: ("vertebrae_T1",  1, True),
    30: ("vertebrae_T2",  1, True),
    31: ("vertebrae_T3",  1, True),
    32: ("vertebrae_T4",  1, True),
    33: ("vertebrae_T5",  1, True),
    34: ("vertebrae_T6",  1, True),
    35: ("vertebrae_T7",  1, True),
    36: ("vertebrae_T8",  1, True),
    37: ("vertebrae_T9",  1, True),

    # Sacrum / S1
    38: ("sacrum",        1, True),
    39: ("vertebrae_S1",  1, True),

    # Spinal cord (not bone, but rigid and contrast-independent — treat as tier 1)
    40: ("spinal_cord",   1, True),

    # Pelvis bones
    41: ("hip_left",      1, True),
    42: ("hip_right",     1, True),
    43: ("femur_left",    1, True),
    44: ("femur_right",   1, True),

    # Sternum + costal cartilage
    45: ("sternum",           1, True),
    46: ("costal_cartilages", 1, True),

    # Shoulder girdle (usually outside abdominal crop)
    47: ("scapula_left",    1, True),
    48: ("scapula_right",   1, True),
    49: ("clavicula_left",  1, True),
    50: ("clavicula_right", 1, True),

    # Ribs left — all 12; .nii.gz extension added automatically
    60: ("rib_left_1",  1, True),
    61: ("rib_left_2",  1, True),
    62: ("rib_left_3",  1, True),
    63: ("rib_left_4",  1, True),
    64: ("rib_left_5",  1, True),
    65: ("rib_left_6",  1, True),
    66: ("rib_left_7",  1, True),
    67: ("rib_left_8",  1, True),
    68: ("rib_left_9",  1, True),
    69: ("rib_left_10", 1, True),
    70: ("rib_left_11", 1, True),
    71: ("rib_left_12", 1, True),

    # Ribs right — all 12
    72: ("rib_right_1",  1, True),
    73: ("rib_right_2",  1, True),
    74: ("rib_right_3",  1, True),
    75: ("rib_right_4",  1, True),
    76: ("rib_right_5",  1, True),
    77: ("rib_right_6",  1, True),
    78: ("rib_right_7",  1, True),
    79: ("rib_right_8",  1, True),
    80: ("rib_right_9",  1, True),
    81: ("rib_right_10", 1, True),
    82: ("rib_right_11", 1, True),
    83: ("rib_right_12", 1, True),

    # =========================================================================
    # TIER 3 — other soft organs  (labels 100–149)
    # Included in _seg_full (widens crop box + evaluation),
    # NOT included in _seg_reg (excluded from registration metric).
    # =========================================================================

    100: ("stomach",             3, False),
    101: ("esophagus",           3, False),
    102: ("urinary_bladder",     3, False),
    103: ("prostate",            3, False),
    104: ("adrenal_gland_left",  3, False),
    105: ("adrenal_gland_right", 3, False),
    106: ("kidney_cyst_left",    3, False),
    107: ("kidney_cyst_right",   3, False),

    # Gluteal muscles (large volume, useful for XY bounding box only)
    108: ("gluteus_maximus_left",  3, False),
    109: ("gluteus_maximus_right", 3, False),
    110: ("gluteus_medius_left",   3, False),
    111: ("gluteus_medius_right",  3, False),
    112: ("gluteus_minimus_left",  3, False),
    113: ("gluteus_minimus_right", 3, False),

    # Lung lobes (TotalSegmentator per-lobe output — merged lung files don't exist)
    114: ("lung_lower_lobe_left",   3, False),
    115: ("lung_lower_lobe_right",  3, False),
    116: ("lung_middle_lobe_right", 3, False),
    117: ("lung_upper_lobe_left",   3, False),
    118: ("lung_upper_lobe_right",  3, False),

    # Heart
    119: ("heart", 3, False),

    # EXCLUDED (no label assigned, not in any output mask):
    #   colon, small_bowel, duodenum, rectum
}

# Convenience sets for downstream consumers
BONE_LABELS        = {lbl for lbl, (_, tier, _ir) in LABEL_MAP.items() if tier == 1}
STABLE_SOFT_LABELS = {lbl for lbl, (_, tier, _ir) in LABEL_MAP.items() if tier == 2}
OTHER_SOFT_LABELS  = {lbl for lbl, (_, tier, _ir) in LABEL_MAP.items() if tier == 3}
REG_LABELS         = {lbl for lbl, (_, tier, in_reg) in LABEL_MAP.items() if in_reg}

TIER_WEIGHTS = {1: 3.0, 2: 1.5, 3: 0.0}   # registration metric weights per tier

# ── ADD after the LABEL_MAP block, before build_masks ────────────────────

# Minimum voxels required per organ for a "present" call
MIN_ORGAN_VOXELS = 500

# Organs that MUST be present for a usable seg_reg.
# Key = label int, value = human name for the error message.
# If fewer than MIN_REQUIRED_PRESENT of these pass the voxel threshold,
# the output is flagged as unusable and the script exits non-zero.
from typing import Dict
REQUIRED_ANCHOR_LABELS: Dict[int, str] = {
    1: "liver",
    2: "spleen",
    3: "kidney_left",
    4: "kidney_right",
}
MIN_REQUIRED_PRESENT = 2   # need at least 2 of the 4 anchors


def check_seg_sanity(seg_reg: np.ndarray, found: dict) -> dict:
    """
    Validate that the seg_reg has enough content for alignment and registration.

    Returns a dict with keys:
        ok           : bool  — True if safe to proceed
        anchor_found : int   — how many anchor organs passed the voxel threshold
        bone_found   : int   — how many bone labels are present
        warnings     : list[str]
        errors       : list[str]
    """
    import numpy as np
    warnings_out = []
    errors_out   = []

    arr = seg_reg  # (Z, Y, X) uint16

    # ── Check 1: anchor organ presence ───────────────────────────────────
    anchor_found = 0
    for lbl, name in REQUIRED_ANCHOR_LABELS.items():
        count = int((arr == lbl).sum())
        if count >= MIN_ORGAN_VOXELS:
            anchor_found += 1
        else:
            warnings_out.append(
                f"Anchor organ missing or too small: {name} (lbl {lbl}), "
                f"voxels={count} < {MIN_ORGAN_VOXELS}"
            )

    if anchor_found < MIN_REQUIRED_PRESENT:
        errors_out.append(
            f"Only {anchor_found}/{len(REQUIRED_ANCHOR_LABELS)} anchor organs "
            f"present — seg_reg is too sparse for reliable alignment. "
            f"Likely cause: wrong HU range (re-run dicom_processor) or OOM during segmentation."
        )

    # ── Check 2: bone presence (spine) ───────────────────────────────────
    SPINE_LABELS = {21, 22, 23, 24, 25, 40}
    bone_found = sum(
        1 for lbl in SPINE_LABELS if (arr == lbl).sum() >= MIN_ORGAN_VOXELS
    )
    if bone_found == 0:
        errors_out.append(
            "No spine labels found in seg_reg (L1-L5, spinal_cord). "
            "Kabsch SVD will be ill-conditioned. "
            "Likely cause: HU values out of range — verify bone HU > 200."
        )
    elif bone_found < 3:
        warnings_out.append(
            f"Only {bone_found}/6 spine labels present — Kabsch XY may be underdetermined."
        )

    # ── Check 3: HU sanity on the raw seg count (indirect proxy) ─────────
    # If seg_reg is suspiciously empty overall (<5% of expected body volume)
    total_seg_vox = int((arr > 0).sum())
    total_vox     = arr.size
    seg_fraction  = total_seg_vox / max(total_vox, 1)
    if seg_fraction < 0.01:
        errors_out.append(
            f"seg_reg covers only {seg_fraction*100:.2f}% of volume — nearly empty. "
            f"TotalSegmentator likely failed on this volume."
        )

    ok = len(errors_out) == 0
    return {
        "ok":            ok,
        "anchor_found":  anchor_found,
        "bone_found":    bone_found,
        "seg_fraction":  round(seg_fraction, 4),
        "warnings":      warnings_out,
        "errors":        errors_out,
    }
# ---------------------------------------------------------------------------
# MASK BUILDER
# ---------------------------------------------------------------------------

def build_masks(seg_dir: str):
    """
    Load all per-organ masks and build seg_full + seg_reg arrays.

    Overlap policy: lower label number wins (first-come in label order).
    This means tier-2 soft organs (labels 1–20) win over bones (21–99)
    on overlap, which is intentional — soft organ boundaries are typically
    more accurate than bone margins in abdominal CT.

    Returns
    -------
    seg_full : np.ndarray uint16 or None
    seg_reg  : np.ndarray uint16 or None
    affine   : np.ndarray (4,4) or None
    header   : nibabel header or None
    found    : dict  {label: bool}
    """
    seg_full = None
    seg_reg  = None
    affine   = None
    header   = None
    found    = {}

    for label in sorted(LABEL_MAP.keys()):
        stem, tier, in_reg = LABEL_MAP[label]

        # Build file path — stem should NOT include .nii.gz
        fpath = os.path.join(seg_dir, stem + ".nii.gz")
        if not os.path.exists(fpath):
            found[label] = False
            continue

        img = nib.load(fpath)
        arr = (img.get_fdata() > 0.5).astype(bool)

        if seg_full is None:
            seg_full = np.zeros(arr.shape, dtype=np.uint16)
            seg_reg  = np.zeros(arr.shape, dtype=np.uint16)
            affine   = img.affine.copy()
            header   = img.header

        if arr.shape != seg_full.shape:
            print(f"  ⚠  Shape mismatch [{label}] {stem}: {arr.shape} vs "
                  f"{seg_full.shape} — skip")
            found[label] = False
            continue

        n_vox     = int(arr.sum())
        empty_f   = seg_full == 0
        n_new_f   = int((arr & empty_f).sum())
        n_overlap = int((arr & ~empty_f).sum())

        seg_full[arr & empty_f] = label

        if in_reg:
            empty_r = seg_reg == 0
            seg_reg[arr & empty_r] = label

        found[label] = True
        reg_tag = "reg" if in_reg else "   "
        print(
            f"  ✓ [{label:3d}|t{tier}|{reg_tag}] {stem:<40} "
            f"vox={n_vox:<8,} new={n_new_f:<8,} overlap={n_overlap:,}"
        )

    return seg_full, seg_reg, affine, header, found


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python combine_masks.py <seg_dir> <output_prefix>")
        print("")
        print("  <seg_dir>       : TotalSegmentator output directory")
        print("  <output_prefix> : path prefix (no extension).")
        print("                    Old-style '_seg.nii.gz' suffix is stripped")
        print("                    automatically for backward compatibility.")
        print("")
        print("  Writes:")
        print("    <output_prefix>_seg_full.nii.gz  (all organs, for crop bbox)")
        print("    <output_prefix>_seg_reg.nii.gz   (bones + stable soft, for reg)")
        sys.exit(1)

    seg_dir       = sys.argv[1]
    output_prefix = sys.argv[2]

    # Backward-compat: strip old single-file suffix
    for suffix in ("_seg.nii.gz", ".nii.gz"):
        if output_prefix.endswith(suffix):
            output_prefix = output_prefix[: -len(suffix)]
            break

    out_full = output_prefix + "_seg_full.nii.gz"
    out_reg  = output_prefix + "_seg_reg.nii.gz"

    # Use the series prefix to tag logs
    _stem     = os.path.basename(output_prefix)
    _parts    = _stem.split("_")
    _study_id  = _parts[0] if len(_parts) > 0 else "unknown"
    _series_id = _parts[1] if len(_parts) > 1 else "unknown"
    log = get_stage_logger("segmentation",
                            study_id=_study_id, series_id=_series_id)

    print(f"\n{'='*70}")
    print(f"combine_masks.py")
    print(f"  seg_dir : {seg_dir}")
    print(f"  out     : {out_full}")
    print(f"          : {out_reg}")
    print(f"{'='*70}\n")
    seg_full, seg_reg, affine, header, found = build_masks(seg_dir)
 
    # MOVE None check FIRST — must be before sanity check
    if seg_full is None:
        msg = f"No organ files found in seg_dir: {seg_dir}"
        print(f"\nERROR: {msg}")
        log.error(msg)
        log.stage_summary({"status": "failed", "error": "no_organ_files"})
        sys.exit(1)
 
    # COMPUTE n_found here, before any reference to it
    n_found   = sum(found.values())
    n_total   = len(found)
    n_full_v  = int((seg_full > 0).sum())
    n_reg_v   = int((seg_reg  > 0).sum())
    missing   = [LABEL_MAP[lbl][0] for lbl, ok in found.items() if not ok]
 
    # Now safe to run sanity check (seg_reg is guaranteed non-None)
    sanity = check_seg_sanity(seg_reg, found)
    status_str = "PASS" if sanity["ok"] else "FAIL"
    print(f"\n  Sanity check: {status_str}")
    print(f"    anchor organs : {sanity['anchor_found']} / {len(REQUIRED_ANCHOR_LABELS)}")
    print(f"    spine labels  : {sanity['bone_found']} / 6")
    print(f"    seg coverage  : {sanity['seg_fraction']*100:.1f}%")
 
    log.metric("seg_sanity", {
        "anchor_found":   sanity["anchor_found"],
        "bone_found":     sanity["bone_found"],
        "seg_fraction":   sanity["seg_fraction"],
        "n_organs_found": n_found,          # ← now defined
        "ok":             sanity["ok"],
    })
 
    for w in sanity["warnings"]:
        print(f"    ⚠  {w}")
        log.warning(w)
 
    for e in sanity["errors"]:
        print(f"    ✗  {e}")
        log.error(e)
 
    log.stage_summary({
        "status":         "ok" if sanity["ok"] else "seg_failed",
        "anchor_found":   sanity["anchor_found"],
        "bone_found":     sanity["bone_found"],
        "seg_fraction":   sanity["seg_fraction"],
        "n_organs_found": n_found,
    })
 
    if not sanity["ok"]:
        flag_path = output_prefix + "_SEG_FAILED.txt"
        with open(flag_path, "w") as fh:
            fh.write("\n".join(sanity["errors"]) + "\n")
        print(f"\n  Flagged as unusable → {flag_path}")
        sys.exit(2)
 
    # Summary (n_found already computed above — remove the duplicate block below)
    print(f"\nSummary")
    print(f"  Organ files found  : {n_found} / {n_total}")
    print(f"  seg_full voxels    : {n_full_v:,}")
    print(f"  seg_reg  voxels    : {n_reg_v:,}")
    print(f"  seg_full labels    : {np.unique(seg_full).tolist()}")
    print(f"  seg_reg  labels    : {np.unique(seg_reg).tolist()}")
    if missing:
        print(f"  Missing organs     : {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))

    out_dir = os.path.dirname(out_full)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(seg_full, affine, header), out_full)
    nib.save(nib.Nifti1Image(seg_reg,  affine, header), out_reg)

    # Write a small JSON sidecar for debugging / downstream consumers
    sidecar = output_prefix + "_mask_info.json"
    info = {
        "seg_dir": seg_dir,
        "n_organs_found": n_found,
        "n_organs_total": n_total,
        "seg_full_voxels": n_full_v,
        "seg_reg_voxels":  n_reg_v,
        "label_map": {
            str(lbl): {
                "stem": stem,
                "tier": tier,
                "in_reg": in_reg,
                "found": found.get(lbl, False),
            }
            for lbl, (stem, tier, in_reg) in sorted(LABEL_MAP.items())
        },
    }
    with open(sidecar, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"\nSaved {out_full}")
    print(f"Saved {out_reg}")
    print(f"Saved {sidecar}")


if __name__ == "__main__":
    main()
