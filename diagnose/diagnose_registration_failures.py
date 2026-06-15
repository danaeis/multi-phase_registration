"""
diagnose_registration_failures.py
==================================
Diagnoses catastrophic organ centroid misalignment (80–160 mm) in the
multi-phase CT registration pipeline.

Four root causes are checked in order:

  1. SEG_REG QUALITY — are there enough anchor organs for Kabsch?
     → <3 common organs → Kabsch raises ValueError → identity fallback → huge error.

  2. BODY ORIENTATION / FLIP — did DICOMOrient produce a reflected body?
     → det(direction) sign mismatch between phases → Kabsch finds the "mirror" 
       solution even though it passes numerically → 80–160 mm displacement.

  3. Z-OFFSET SIGN / MAGNITUDE — did align_data compute a bad crop?
     → Organ centroids in aligned seg_reg are outside the expected Z range 
       relative to NC → wrong crop window passed wrong anatomy to registration.

  4. IN-PLANE RESIDUAL (XY) — large lateral offsets before Kabsch?
     → Systematic X or Y shift >40 mm that Nelder-Mead cannot escape from.

Usage
-----
# Single study — supply aligned seg_reg directory and labels CSV:
python diagnose_registration_failures.py --study_dir /data/aligned_volumes/STUDY_ID \\
    --labels_csv /data/labels.csv

# Batch mode:
python diagnose_registration_failures.py --all \\
    --base_dir /data/aligned_volumes \\
    --labels_csv /data/labels.csv \\
    --out_csv /data/diagnosis_report.csv

# You can also point it at rigid0 outputs (to check if Kabsch fixed things):
python diagnose_registration_failures.py --all \\
    --base_dir /data/rigid_registered \\
    --seg_postfix _rigid0_seg_reg.nii.gz \\
    --labels_csv /data/labels.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Anchor organs used in Z-alignment (same as align_data_v3)
ANCHOR_LABELS: Dict[int, str] = {
    1: "liver",
    2: "spleen",
    3: "kidney_L",
    4: "kidney_R",
}

# Stable organs used for Kabsch (same as register_v2 ALL_STABLE_LABELS)
KABSCH_LABELS: Dict[int, str] = {
    1:  "liver",
    2:  "spleen",
    3:  "kidney_L",
    4:  "kidney_R",
    5:  "pancreas",
    6:  "gallbladder",
    13: "aorta",
    14: "IVC",
    21: "L1",
    22: "L2",
    23: "L3",
    40: "spinal_cord",
}

# Bone labels used in Kabsch Pass 0a (from register_v2 BONE_LABELS)
BONE_LABELS_FOR_KABSCH: Dict[int, str] = {
    21: "L1",
    22: "L2",
    23: "L3",
    24: "L4",
    25: "L5",
    40: "spinal_cord",
}

MIN_ORGAN_VOXELS = 200
LARGE_INPLANE_THRESHOLD_MM = 40.0   # XY shift beyond this → Nelder-Mead failure risk
LARGE_Z_THRESHOLD_MM = 80.0         # Z shift beyond this → crop failure
CENTROID_ALARM_MM = 15.0            # flag organs beyond this in the report


# ===========================================================================
# UTILITIES
# ===========================================================================

def get_organ_centroid_phys(seg_sitk: sitk.Image, label: int) -> Optional[np.ndarray]:
    """Return physical mm centroid of a label, or None if too few voxels."""
    arr = sitk.GetArrayFromImage(seg_sitk).astype(np.int32)
    mask = arr == label
    if mask.sum() < MIN_ORGAN_VOXELS:
        return None
    zz, yy, xx = np.where(mask)
    cz, cy, cx = zz.mean(), yy.mean(), xx.mean()
    sp = np.array(seg_sitk.GetSpacing())   # (sx, sy, sz)
    or_ = np.array(seg_sitk.GetOrigin())   # (ox, oy, oz)
    di = np.array(seg_sitk.GetDirection()).reshape(3, 3)
    vox_scaled = np.array([cx * sp[0], cy * sp[1], cz * sp[2]])
    return or_ + di @ vox_scaled


def direction_det(seg_sitk: sitk.Image) -> float:
    return float(np.linalg.det(
        np.array(seg_sitk.GetDirection()).reshape(3, 3)
    ))


def count_present_labels(seg_sitk: sitk.Image,
                          label_dict: Dict[int, str]) -> Tuple[int, List[int]]:
    arr = sitk.GetArrayFromImage(seg_sitk).astype(np.int32)
    present = [lbl for lbl in label_dict if (arr == lbl).sum() >= MIN_ORGAN_VOXELS]
    return len(present), present


# ===========================================================================
# PER-STUDY DIAGNOSIS
# ===========================================================================

def diagnose_study(
        study_id: str,
        study_dir: str,
        labels_df,
        seg_postfix: str = "_aligned_seg_reg.nii.gz",
        ref_phase: str = "Non-contrast",
        verbose: bool = True,
) -> List[dict]:
    """
    Run all four checks for a single study.

    Returns a list of per-series result dicts.
    """
    from scipy.spatial.distance import cdist

    rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    nc_rows = rows[rows["Label"] == ref_phase]
    if nc_rows.empty:
        if verbose:
            print(f"  [SKIP] No {ref_phase} series found for {study_id[:40]}")
        return []

    nc_sid = nc_rows.iloc[0]["SeriesInstanceUID"]

    # Find NC seg_reg
    nc_seg_path = os.path.join(study_dir, f"{study_id}_{nc_sid}{seg_postfix}")
    if not os.path.exists(nc_seg_path):
        if verbose:
            print(f"  [SKIP] NC seg_reg not found: {nc_seg_path}")
        return []

    nc_seg = sitk.ReadImage(nc_seg_path)
    nc_det = direction_det(nc_seg)
    nc_n_kabsch, nc_present = count_present_labels(nc_seg, KABSCH_LABELS)
    nc_n_bone,   nc_bone    = count_present_labels(nc_seg, BONE_LABELS_FOR_KABSCH)
    nc_n_anchor, nc_anchor  = count_present_labels(nc_seg, ANCHOR_LABELS)

    nc_centroids = {}
    for lbl, name in KABSCH_LABELS.items():
        c = get_organ_centroid_phys(nc_seg, lbl)
        if c is not None:
            nc_centroids[lbl] = c

    if verbose:
        print(f"\n{'='*70}")
        print(f"Study: {study_id[:60]}")
        print(f"  NC seg: {len(nc_centroids)} organs  |  det={nc_det:+.3f}  "
              f"|  bone organs={nc_n_bone}  |  anchor organs={nc_n_anchor}")

    results = []

    for _, row in rows.iterrows():
        sid   = row["SeriesInstanceUID"]
        phase = row["Label"]
        if sid == nc_sid:
            continue

        seg_path = os.path.join(study_dir, f"{study_id}_{sid}{seg_postfix}")
        if not os.path.exists(seg_path):
            results.append({
                "study_id": study_id, "series_id": sid, "phase": phase,
                "status": "file_missing", "seg_path": seg_path,
            })
            if verbose:
                print(f"  [{phase}] ⚠️  File not found: {seg_path}")
            continue

        mov_seg = sitk.ReadImage(seg_path)
        mov_det = direction_det(mov_seg)
        mov_n_kabsch, mov_present = count_present_labels(mov_seg, KABSCH_LABELS)
        mov_n_bone,   mov_bone    = count_present_labels(mov_seg, BONE_LABELS_FOR_KABSCH)
        mov_n_anchor, mov_anchor  = count_present_labels(mov_seg, ANCHOR_LABELS)

        mov_centroids = {}
        for lbl, name in KABSCH_LABELS.items():
            c = get_organ_centroid_phys(mov_seg, lbl)
            if c is not None:
                mov_centroids[lbl] = c

        common_kabsch = sorted(set(nc_centroids) & set(mov_centroids))
        common_bone   = [l for l in common_kabsch if l in BONE_LABELS_FOR_KABSCH]
        common_anchor = [l for l in common_kabsch if l in ANCHOR_LABELS]

        # ── Check 1: Kabsch organ count ──────────────────────────────────
        kabsch_bone_ok   = len(common_bone) >= 3
        kabsch_organs_ok = len(common_kabsch) >= 3
        kabsch_fail      = not kabsch_organs_ok

        # ── Check 2: Direction determinant sign consistency ───────────────
        det_sign_match = np.sign(nc_det) == np.sign(mov_det)
        # Large det mismatch can also indicate resampling flipped an axis
        det_diff = abs(nc_det - mov_det)

        # ── Check 3: Z-range of anchor organ centroids ────────────────────
        z_deltas_mm: Dict[int, float] = {}
        for lbl in common_anchor:
            nc_z = nc_centroids[lbl][2]
            mv_z = mov_centroids[lbl][2]
            z_deltas_mm[lbl] = mv_z - nc_z

        # Z offset = mean of anchor-organ Z deltas (in physical mm)
        z_offset_mean_mm = float(np.mean(list(z_deltas_mm.values()))) if z_deltas_mm else float("nan")
        z_offset_std_mm  = float(np.std(list(z_deltas_mm.values())))  if z_deltas_mm else float("nan")
        bad_z_offset     = abs(z_offset_mean_mm) > LARGE_Z_THRESHOLD_MM if z_deltas_mm else True

        # ── Check 4: In-plane (XY) residual of anchor organs ─────────────
        xy_deltas_mm: Dict[int, float] = {}
        for lbl in common_anchor:
            nc_xy = nc_centroids[lbl][:2]
            mv_xy = mov_centroids[lbl][:2]
            xy_deltas_mm[lbl] = float(np.linalg.norm(mv_xy - nc_xy))

        xy_mean_mm      = float(np.mean(list(xy_deltas_mm.values()))) if xy_deltas_mm else float("nan")
        xy_max_mm       = float(np.max(list(xy_deltas_mm.values())))  if xy_deltas_mm else float("nan")
        large_inplane   = xy_mean_mm > LARGE_INPLANE_THRESHOLD_MM     if xy_deltas_mm else False

        # ── 3D centroid distances for all common organs ───────────────────
        centroid_3d: Dict[str, float] = {}
        for lbl in common_kabsch:
            d = float(np.linalg.norm(mov_centroids[lbl] - nc_centroids[lbl]))
            centroid_3d[KABSCH_LABELS[lbl]] = d

        # ── Summary flags ─────────────────────────────────────────────────
        flags = []
        if kabsch_fail:
            flags.append("KABSCH_FAIL_TOO_FEW_ORGANS")
        if not kabsch_bone_ok:
            flags.append("KABSCH_BONE_TOO_FEW")
        if not det_sign_match:
            flags.append("DIRECTION_DET_SIGN_MISMATCH")
        if det_diff > 0.5:
            flags.append(f"DIRECTION_DET_DIFF={det_diff:.2f}")
        if bad_z_offset:
            flags.append(f"BAD_Z_OFFSET={z_offset_mean_mm:.1f}mm")
        if large_inplane:
            flags.append(f"LARGE_XY_RESIDUAL={xy_mean_mm:.1f}mm")

        # Flag any organ with huge 3D centroid distance
        alarm_organs = [f"{org}={d:.1f}mm"
                        for org, d in centroid_3d.items()
                        if d > CENTROID_ALARM_MM]
        if alarm_organs:
            flags.append("LARGE_CENTROID_DIST: " + " | ".join(alarm_organs))

        status = "BAD" if flags else "OK"

        r = {
            "study_id":           study_id,
            "series_id":          sid,
            "phase":              phase,
            "status":             status,
            "flags":              flags,
            # organ counts
            "nc_n_kabsch_organs": nc_n_kabsch,
            "mov_n_kabsch_organs": mov_n_kabsch,
            "common_kabsch_organs": len(common_kabsch),
            "common_bone_organs": len(common_bone),
            "common_anchor_organs": len(common_anchor),
            # direction
            "nc_det":             nc_det,
            "mov_det":            mov_det,
            "det_sign_match":     det_sign_match,
            # Z offset (physical mm)
            "z_offset_mean_mm":   round(z_offset_mean_mm, 1),
            "z_offset_std_mm":    round(z_offset_std_mm, 1),
            # XY residual
            "xy_mean_mm":         round(xy_mean_mm, 1),
            "xy_max_mm":          round(xy_max_mm, 1),
            # 3D centroid distances per organ
            "centroid_3d":        {k: round(v, 1) for k, v in centroid_3d.items()},
            "mean_centroid_3d":   round(float(np.mean(list(centroid_3d.values()))), 1)
                                  if centroid_3d else float("nan"),
        }
        results.append(r)

        if verbose:
            flag_str = " | ".join(flags) if flags else "—"
            print(f"\n  [{phase}]  status={status}")
            print(f"    Common organs (Kabsch): {len(common_kabsch)}  "
                  f"(bone: {len(common_bone)})  (anchor: {len(common_anchor)})")
            print(f"    det NC={nc_det:+.3f}  det mov={mov_det:+.3f}  "
                  f"sign_match={det_sign_match}")
            print(f"    Z-offset mean={z_offset_mean_mm:+.1f}mm  "
                  f"std={z_offset_std_mm:.1f}mm")
            print(f"    XY-residual  mean={xy_mean_mm:.1f}mm  "
                  f"max={xy_max_mm:.1f}mm")
            if centroid_3d:
                print(f"    3D centroid distances:")
                for org, d in sorted(centroid_3d.items(), key=lambda x: -x[1]):
                    flag = "  ⚠️" if d > CENTROID_ALARM_MM else ""
                    print(f"      {org:15s}: {d:.1f} mm{flag}")
            print(f"    FLAGS: {flag_str}")

    return results


# ===========================================================================
# BATCH + REPORT
# ===========================================================================

def run_batch(
        base_dir: str,
        labels_df,
        seg_postfix: str = "_aligned_seg_reg.nii.gz",
        ref_phase: str = "Non-contrast",
        out_csv: Optional[str] = None,
        verbose: bool = True,
) -> "pd.DataFrame":
    import pandas as pd

    all_results = []
    study_dirs = sorted([
        d for d in Path(base_dir).iterdir() if d.is_dir()
    ])

    print(f"\nBatch diagnosis: {len(study_dirs)} study directories")
    print(f"  seg_postfix : {seg_postfix}")
    print(f"  ref_phase   : {ref_phase}")
    print(f"  base_dir    : {base_dir}\n")

    for sd in study_dirs:
        study_id = sd.name
        res = diagnose_study(
            study_id=study_id,
            study_dir=str(sd),
            labels_df=labels_df,
            seg_postfix=seg_postfix,
            ref_phase=ref_phase,
            verbose=verbose,
        )
        all_results.extend(res)

    if not all_results:
        print("No results collected — check paths and postfixes.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # === FIXED: Robust flags handling ===
    if "flags" in df.columns:
        # Ensure every row has a list
        df["flags"] = df["flags"].apply(lambda x: x if isinstance(x, list) else [])
        
        df["flags_str"] = df["flags"].apply(lambda x: " | ".join(x) if x else "")
        df = df.drop(columns=["centroid_3d", "flags"], errors="ignore")
        df = df.rename(columns={"flags_str": "flags"})

    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print(f"{'='*70}")
    n_total = len(df)
    n_bad   = (df["status"] == "BAD").sum()
    n_miss  = (df["status"] == "file_missing").sum()
    print(f"  Total series  : {n_total}")
    print(f"  OK            : {(df['status']=='OK').sum()}")
    print(f"  BAD           : {n_bad}  ({100*n_bad/max(n_total,1):.1f}%)")
    print(f"  File missing  : {n_miss}")

    if n_bad > 0:
        bad_df = df[df["status"] == "BAD"].copy()
        print(f"\nBAD cases breakdown:")
        for _, row in bad_df.iterrows():
            print(f"  {row['study_id'][:40]}  [{row['phase']}]")
            print(f"    {row.get('flags', '')}")
            print(f"    mean_centroid_3d={row.get('mean_centroid_3d', np.nan):.1f}mm  "
                  f"common_organs={row.get('common_kabsch_organs', 0)}  "
                  f"xy_mean={row.get('xy_mean_mm', np.nan):.1f}mm  "
                  f"z_offset={row.get('z_offset_mean_mm', np.nan):+.1f}mm")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\n✓ Results saved → {out_csv}")

    return df

# ===========================================================================
# QUICK STANDALONE CHECK (no labels CSV needed)
# ===========================================================================

def quick_check_pair(nc_seg_path: str, mov_seg_path: str,
                     phase: str = "moving") -> dict:
    """
    Run diagnosis on a single NC + moving pair without needing a labels CSV.

    Example:
        from diagnose_registration_failures import quick_check_pair
        r = quick_check_pair("NC_aligned_seg_reg.nii.gz", "ART_aligned_seg_reg.nii.gz")
        print(r)
    """
    nc_seg  = sitk.ReadImage(nc_seg_path)
    mov_seg = sitk.ReadImage(mov_seg_path)

    nc_det  = direction_det(nc_seg)
    mov_det = direction_det(mov_seg)

    nc_c  = {lbl: get_organ_centroid_phys(nc_seg,  lbl) for lbl in KABSCH_LABELS}
    mov_c = {lbl: get_organ_centroid_phys(mov_seg, lbl) for lbl in KABSCH_LABELS}
    nc_c  = {k: v for k, v in nc_c.items()  if v is not None}
    mov_c = {k: v for k, v in mov_c.items() if v is not None}

    common = sorted(set(nc_c) & set(mov_c))
    centroid_3d = {KABSCH_LABELS[l]: round(float(np.linalg.norm(mov_c[l] - nc_c[l])), 1)
                   for l in common}

    common_bone   = [l for l in common if l in BONE_LABELS_FOR_KABSCH]
    common_anchor = [l for l in common if l in ANCHOR_LABELS]

    z_deltas = [float(mov_c[l][2] - nc_c[l][2]) for l in common_anchor]
    xy_dists = [float(np.linalg.norm(mov_c[l][:2] - nc_c[l][:2])) for l in common_anchor]

    flags = []
    if len(common) < 3:           flags.append(f"KABSCH_FAIL:{len(common)}_organs")
    if len(common_bone) < 3:      flags.append(f"KABSCH_BONE_FAIL:{len(common_bone)}_organs")
    if np.sign(nc_det) != np.sign(mov_det): flags.append("DET_SIGN_MISMATCH")
    if z_deltas and abs(np.mean(z_deltas)) > LARGE_Z_THRESHOLD_MM:
        flags.append(f"BAD_Z_OFFSET:{np.mean(z_deltas):+.1f}mm")
    if xy_dists and np.mean(xy_dists) > LARGE_INPLANE_THRESHOLD_MM:
        flags.append(f"LARGE_XY:{np.mean(xy_dists):.1f}mm")

    result = {
        "phase": phase,
        "status": "BAD" if flags else "OK",
        "flags": flags,
        "nc_det": nc_det, "mov_det": mov_det,
        "nc_organs": len(nc_c), "mov_organs": len(mov_c),
        "common_organs": len(common), "common_bone_organs": len(common_bone),
        "centroid_3d_mm": centroid_3d,
        "mean_centroid_3d_mm": round(float(np.mean(list(centroid_3d.values()))), 1)
                               if centroid_3d else float("nan"),
        "z_offset_mean_mm": round(float(np.mean(z_deltas)), 1) if z_deltas else float("nan"),
        "xy_mean_mm": round(float(np.mean(xy_dists)), 1) if xy_dists else float("nan"),
    }

    print(f"\n[quick_check_pair] phase={phase}  status={result['status']}")
    print(f"  NC  organs={len(nc_c)}  det={nc_det:+.4f}")
    print(f"  MOV organs={len(mov_c)}  det={mov_det:+.4f}")
    print(f"  Common Kabsch organs : {len(common)}  (bone: {len(common_bone)})")
    print(f"  Z-offset mean        : {result['z_offset_mean_mm']:+.1f} mm")
    print(f"  XY-residual mean     : {result['xy_mean_mm']:.1f} mm")
    print(f"  3D centroid distances:")
    for org, d in sorted(centroid_3d.items(), key=lambda x: -x[1]):
        flag = "  ⚠️" if d > CENTROID_ALARM_MM else ""
        print(f"    {org:15s}: {d:.1f} mm{flag}")
    print(f"  FLAGS: {' | '.join(flags) if flags else 'none'}")
    return result


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Diagnose registration pipeline failures.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--all", action="store_true",
                   help="Batch over all study subdirs in --base_dir")
    p.add_argument("--base_dir",  default=None)
    p.add_argument("--study_dir", default=None)
    p.add_argument("--study_id",  default=None)
    p.add_argument("--labels_csv", default="../ncct_cect/vindr_ds/labels.csv")
    p.add_argument("--seg_postfix", default="_aligned_seg_reg.nii.gz",
                   help="Postfix for seg_reg files, e.g.:\n"
                        "  aligned stage  : _aligned_seg_reg.nii.gz\n"
                        "  after rigid0   : _rigid0_seg_reg.nii.gz\n"
                        "  after rigid1   : _rigid1_seg_reg.nii.gz")
    p.add_argument("--ref_phase", default="Non-contrast")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    import pandas as pd
    labels_df = pd.read_csv(args.labels_csv)

    if args.all:
        if not args.base_dir:
            print("ERROR: --base_dir required for batch mode"); sys.exit(1)
        run_batch(
            base_dir=args.base_dir,
            labels_df=labels_df,
            seg_postfix=args.seg_postfix,
            ref_phase=args.ref_phase,
            out_csv=args.out_csv,
            verbose=not args.quiet,
        )
    else:
        if not args.study_dir or not args.study_id:
            print("ERROR: --study_dir and --study_id required for single-study mode")
            sys.exit(1)
        results = diagnose_study(
            study_id=args.study_id,
            study_dir=args.study_dir,
            labels_df=labels_df,
            seg_postfix=args.seg_postfix,
            ref_phase=args.ref_phase,
            verbose=True,
        )
        for r in results:
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
