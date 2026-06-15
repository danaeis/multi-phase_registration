"""
align_and_crop.py
=================
Batch runner for multi-phase CT Z-alignment.

Supports TWO input layouts:
1. Legacy (subfolder per study):
   {INPUT_DIR}/{study_id}/{study_id}_{series_id}_crop.nii.gz
   {INPUT_DIR}/{study_id}/{study_id}_{series_id}_seg_reg.nii.gz
   {INPUT_DIR}/{study_id}/{study_id}_{series_id}_seg_full.nii.gz

2. New flat layout:
   nifti_unprocessed/{study_id}_{series_id}*standardized.nii.gz
   ts_segmentation/{study_id}*{series_id}*seg_reg.nii.gz
   ts_segmentation/{study_id}*{series_id}*seg_full.nii.gz

Usage:
    python align_all_data.py                         # legacy example
    python align_all_data.py --new-layout            # new flat layout
    python align_all_data.py <study_id> --new-layout
    python align_all_data.py --all --new-layout
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional
import glob

import pandas as pd

from configs import MAIN_PATH
from align_data_v3 import align_study, baseline_crop_study, verify_study_alignment
from pipeline_logger import get_stage_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGACY_INPUT_DIR = Path(MAIN_PATH + "cropped_volumes")
NEW_IMAGE_DIR    = Path(MAIN_PATH + "nifti_unprocessed_volumes")
NEW_SEG_DIR      = Path(MAIN_PATH + "ts_segmentations")

OUTPUT_DIR   = Path(MAIN_PATH + "aligned_volumes")
BASELINE_DIR = Path(MAIN_PATH + "baseline_volumes")

LABELS_CSV   = str(Path(MAIN_PATH) / "labels.csv")

EXAMPLE_STUDY_ID = "1.2.840.113619.2.278.3.717616.260.1578649418.868"


# ===========================================================================
# FILE DISCOVERY - SUPPORTS BOTH LAYOUTS
# ===========================================================================

def get_input_mode(new_layout: bool = False):
    """Return directories based on layout mode."""
    if new_layout:
        print(f"🔄 Using **NEW FLAT** layout")
        print(f"   Images dir : {NEW_IMAGE_DIR}")
        print(f"   Segs dir   : {NEW_SEG_DIR}")
        return {
            "image_dir": NEW_IMAGE_DIR,
            "seg_dir": NEW_SEG_DIR,
            "is_flat": True
        }
    else:
        print(f"🔄 Using **LEGACY** layout: {LEGACY_INPUT_DIR}")
        return {
            "image_dir": LEGACY_INPUT_DIR,
            "seg_dir": LEGACY_INPUT_DIR,
            "is_flat": False
        }

def find_image_file(study_id: str, series_id: str, mode: dict) -> Optional[Path]:
    if not mode["is_flat"]:
        p = mode["image_dir"] / study_id / f"{study_id}_{series_id}_crop.nii.gz"
        return p if p.exists() else None
    else:
        # More flexible pattern
        pattern = str(mode["image_dir"] / f"*{study_id}*{series_id}*standardized.nii.gz")
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
        return None


def find_seg_reg_file(study_id: str, series_id: str, mode: dict) -> Optional[Path]:
    if not mode["is_flat"]:
        p = mode["seg_dir"] / study_id / f"{study_id}_{series_id}_seg_reg.nii.gz"
        return p if p.exists() else None
    else:
        pattern = str(mode["seg_dir"] / f"*{study_id}*{series_id}*seg_reg.nii.gz")
        matches = glob.glob(pattern)
        return Path(matches[0]) if matches else None


def find_seg_full_file(study_id: str, series_id: str, mode: dict) -> Optional[Path]:
    if not mode["is_flat"]:
        p = mode["seg_dir"] / study_id / f"{study_id}_{series_id}_seg_full.nii.gz"
        return p if p.exists() else None
    else:
        pattern = str(mode["seg_dir"] / f"*{study_id}*{series_id}*seg_full.nii.gz")
        matches = glob.glob(pattern)
        return Path(matches[0]) if matches else None


def find_series_files(study_id: str, series_id: str, mode: dict) -> dict:
    image    = find_image_file(study_id, series_id, mode)
    seg_reg  = find_seg_reg_file(study_id, series_id, mode)
    seg_full = find_seg_full_file(study_id, series_id, mode)

    return {
        "image":    str(image)    if image    else None,
        "seg_reg":  str(seg_reg)  if seg_reg  else None,
        "seg_full": str(seg_full) if seg_full else None,
    }


def find_noncontrast_series(study_rows: pd.DataFrame, study_id: str, mode: dict) -> str:
    nc_rows = study_rows[study_rows["Label"] == "Non-contrast"]
    if nc_rows.empty:
        raise ValueError(
            f"No 'Non-contrast' label in CSV for {study_id}. "
            f"Labels present: {study_rows['Label'].tolist()}"
        )

    print(f"  🔍 Looking for Non-contrast series for study {study_id}...")

    missing = []
    for _, row in nc_rows.iterrows():
        sid = row["SeriesInstanceUID"]
        img_path = find_image_file(study_id, sid, mode)
        if img_path:
            print(f"  ✅ Found Non-contrast image: {sid}")
            return sid
        missing.append(sid)

    raise FileNotFoundError(
        f"{len(nc_rows)} Non-contrast entries in CSV for {study_id} "
        f"but none found on disk.\nMissing series: {missing}"
    )

# ===========================================================================
# BUILD series_info DICT
# ===========================================================================

def build_series_info(study_id: str, study_rows: pd.DataFrame, mode: dict) -> Dict[str, dict]:
    series_info: Dict[str, dict] = {}

    for _, row in study_rows.iterrows():
        sid   = row["SeriesInstanceUID"]
        phase = row["Label"]
        files = find_series_files(study_id, sid, mode)

        if files["image"] is None:
            print(f"  ⚠  [{phase:15s}] missing image — skipped")
            continue

        if files["seg_reg"] is None:
            print(f"  ⚠  [{phase:15s}] missing seg_reg — skipped")
            continue

        series_info[sid] = {
            "image":    files["image"],
            "seg_reg":  files["seg_reg"],
            "seg_full": files["seg_full"],
            "phase":    phase,
        }

        seg_full_tag = "✓" if files["seg_full"] else "⊘"
        print(f"  ✓ [{phase:15s}]  seg_reg=✓  seg_full={seg_full_tag}  image=✓")

    return series_info

# ===========================================================================
# RUN ONE STUDY (updated to accept mode)
# ===========================================================================

def run_study(study_id: str,
              labels_df: pd.DataFrame,
              new_layout: bool = False,
              force_recompute: bool = False) -> dict:
    """
    Align all phases in one study.
    """
    mode = get_input_mode(new_layout)
    
    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    if study_rows.empty:
        return {"status": "not_in_csv"}

    # Identify Non-contrast reference
    try:
        ref_sid = find_noncontrast_series(study_rows, study_id, mode)
    except (ValueError, FileNotFoundError) as e:
        print(f"  ⊘ {str(e).splitlines()[0]}")
        return {"status": "no_reference", "reason": str(e)}

    print(f"  Ref (NC): {ref_sid[:50]}...")

    # Build series info
    series_info = build_series_info(study_id, study_rows, mode)

    if len(series_info) < 2:
        return {
            "status": "too_few_series",
            "reason": f"only {len(series_info)} usable series on disk",
        }

    if ref_sid not in series_info:
        return {
            "status": "no_reference",
            "reason": "Non-contrast series missing seg_reg — cannot align",
        }

    # Run alignment
    study_output_dir = (OUTPUT_DIR / study_id).resolve()
    study_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = align_study(
            study_id=study_id,
            series_info=series_info,
            reference_series_id=ref_sid,
            output_dir=str(study_output_dir),
            search_region="upper",
            min_overlap_slices=30,
            force_recompute=force_recompute,
        )
    except Exception as e:
        log = get_stage_logger("alignment", study_id=study_id)
        log.error(f"Alignment failed: {e}", exc_info=True)
        log.stage_summary({"status": "failed", "error": str(e)})
        raise
    
    return result


# ===========================================================================
# PRINT RESULTS (safe for both modes)
# ===========================================================================

def print_results(result, is_baseline=False):
    if result.get("status") != "complete":
        return

    meta = result.get("metadata", {})
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    
    if is_baseline:
        depth = meta.get("window_depth", "N/A")
        print(f"  Baseline crop depth : {depth} slices")
    else:
        overlap = meta.get("common_overlap", {})
        depth = overlap.get("depth", "N/A")
        print(f"  Overlap window : {depth} slices")

    print(f"  Series aligned : {len(result.get('aligned_volumes', []))}")
    
    print(f"\n  Per-series:")
    for sid, info in meta.get("series", {}).items():
        print(f"    [{info.get('phase', 'Unknown'):15s}]  "
              f"z_offset={info.get('z_offset', 0):+d}  "
              f"score={info.get('similarity', 0):.4f}")

# ===========================================================================
# SINGLE-STUDY ENTRY POINT
# ===========================================================================

def process_single_study(study_id: str, new_layout: bool = False, force_recompute: bool = False):
    print(f"\n{'='*80}")
    print(f"SINGLE STUDY: {study_id}  (layout: {'NEW FLAT' if new_layout else 'LEGACY'})")
    print(f"{'='*80}")

    labels_df = pd.read_csv(LABELS_CSV)
    result    = run_study(study_id, labels_df, new_layout, force_recompute)
    status    = result.get("status", "unknown")

    if status in ("not_in_csv", "no_reference", "too_few_series", "error"):
        print(f"\n❌ {status}: {result.get('reason', '')}")
        return

    if status == "skipped":
        print(f"\n⊘ Already aligned (pass --force-recompute to redo)")
        return

    print_results(result)
    verify_study_alignment(str(OUTPUT_DIR / study_id))


# ===========================================================================
# BATCH ENTRY POINT
# ===========================================================================

def process_all_studies(new_layout: bool = False, force_recompute: bool = False):
    labels_df = pd.read_csv(LABELS_CSV)
    study_ids = labels_df["StudyInstanceUID"].unique()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_name = "NEW FLAT" if new_layout else "LEGACY"
    print(f"\n{'='*80}")
    print(f"BATCH ALIGNMENT — {len(study_ids)} studies  "
          f"layout={mode_name}  force={force_recompute}")
    print(f"{'='*80}")

    counts  = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list = []

    for idx, study_id in enumerate(study_ids, 1):
        print(f"\n[{idx}/{len(study_ids)}] {study_id[:55]}...")

        result = run_study(study_id, labels_df, new_layout, force_recompute)
        status = result.get("status", "unknown")

        if status == "complete":
            depth = result["metadata"]["common_overlap"]["depth"]
            print(f"  ✓ OK — {depth} slices overlap")
            counts["ok"] += 1
        elif status == "skipped":
            print(f"  ⊘ Already done")
            counts["skipped"] += 1
        else:
            reason = result.get("reason", status)[:70]
            print(f"  ✗ {status}: {reason}")
            counts["failed"] += 1
            failures.append((study_id, reason))

    print(f"\n{'='*80}")
    print(f"DONE  ✓{counts['ok']}  ⊘{counts['skipped']}  "
          f"✗{counts['failed']}  / {len(study_ids)}")
    if failures:
        print(f"\nFailed studies ({len(failures)}):")
        for sid, reason in failures[:20]:
            print(f"  {sid[:50]}... → {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(f"{'='*80}")


# ===========================================================================
# BASELINE RUNNERS (similarly updated)
# ===========================================================================

def run_study_baseline(study_id: str,
                       labels_df: pd.DataFrame,
                       new_layout: bool = False,
                       force_recompute: bool = False) -> dict:
    mode = get_input_mode(new_layout)
    
    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    if study_rows.empty:
        return {"status": "not_in_csv"}

    try:
        ref_sid = find_noncontrast_series(study_rows, study_id, mode)
    except (ValueError, FileNotFoundError) as e:
        print(f"  ⊘ {str(e).splitlines()[0]}")
        return {"status": "no_reference", "reason": str(e)}

    print(f"  Ref (NC): {ref_sid[:50]}...")

    series_info = build_series_info(study_id, study_rows, mode)

    if len(series_info) < 2:
        return {"status": "too_few_series",
                "reason": f"only {len(series_info)} usable series on disk"}

    if ref_sid not in series_info:
        return {"status": "no_reference",
                "reason": "Non-contrast series missing seg_reg"}

    study_output_dir = BASELINE_DIR / study_id
    study_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = baseline_crop_study(
            study_id=study_id,
            series_info=series_info,
            reference_series_id=ref_sid,
            output_dir=str(study_output_dir),
            min_overlap_slices=30,
            force_recompute=force_recompute,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}

    return result


def process_single_baseline(study_id: str, new_layout: bool = False, force_recompute: bool = False):
    print(f"\n{'='*80}")
    print(f"BASELINE (single study): {study_id}  (layout: {'NEW' if new_layout else 'LEGACY'})")
    print(f"{'='*80}")

    labels_df = pd.read_csv(LABELS_CSV)
    result    = run_study_baseline(study_id, labels_df, new_layout, force_recompute)
    # ... (rest similar to process_single_study, omitted for brevity)

    status    = result.get("status", "unknown")

    if status in ("not_in_csv", "no_reference", "too_few_series", "error"):
        print(f"\n❌ {status}: {result.get('reason', '')}")
        return

    if status == "skipped":
        print(f"\n⊘ Already aligned (pass --force-recompute to redo)")
        return

    meta = result["metadata"]
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"  Series aligned : {len(result.get('out_volumes', {}))}")
    print(f"  Crop window    : {meta['window_depth']} slices  "
        f"({meta['common_top_mm']:.1f}mm → {meta['common_bottom_mm']:.1f}mm)")
    print(f"\n  Per-series:")
    for sid, info in meta["series"].items():
        print(f"    [{info['phase']:15s}]  "
            f"sup={info['sup_vox']:4d} ({info['sup_mm']:.1f}mm)  "
            f"inf={info['inf_vox']:4d} ({info['inf_mm']:.1f}mm)  "
            f"crop=[{info['crop_start']}:{info['crop_end']}]")
    verify_study_alignment(str(BASELINE_DIR / study_id))


def process_all_baseline(new_layout: bool = False, force_recompute: bool = False):
    labels_df = pd.read_csv(LABELS_CSV)
    study_ids = labels_df["StudyInstanceUID"].unique()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    mode_name = "NEW FLAT" if new_layout else "LEGACY"
    print(f"\n{'='*80}")
    print(f"BATCH BASELINE — {len(study_ids)} studies  "
          f"layout={mode_name}  force={force_recompute}")
    print(f"{'='*80}")

    counts  = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list = []

    for idx, study_id in enumerate(study_ids, 1):
        print(f"\n[{idx}/{len(study_ids)}] {study_id[:55]}...")

        result = run_study_baseline(study_id, labels_df, new_layout, force_recompute)
        status = result.get("status", "unknown")

        if status == "complete":
            depth = result["metadata"]["window_depth"]
            print(f"  ✓ OK — {depth} slices overlap")
            counts["ok"] += 1
        elif status == "skipped":
            print(f"  ⊘ Already done")
            counts["skipped"] += 1
        else:
            reason = result.get("reason", status)[:70]
            print(f"  ✗ {status}: {reason}")
            counts["failed"] += 1
            failures.append((study_id, reason))

    print(f"\n{'='*80}")
    print(f"DONE  ✓{counts['ok']}  ⊘{counts['skipped']}  "
          f"✗{counts['failed']}  / {len(study_ids)}")
    if failures:
        print(f"\nFailed studies ({len(failures)}):")
        for sid, reason in failures[:20]:
            print(f"  {sid[:50]}... → {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(f"{'='*80}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    args = sys.argv[1:]
    new_layout = "--new-layout" in args or "--flat" in args
    baseline = "--baseline" in args
    force = "--force-recompute" in args

    print(f"🚀 Starting with new_layout={new_layout}, baseline={baseline}")
    
    if "--all" in args:
        if baseline:
            process_all_baseline(new_layout=new_layout, force_recompute=force)
        else:
            process_all_studies(new_layout=new_layout, force_recompute=force)
    else:
        sid = next((a for a in args if not a.startswith("--")), EXAMPLE_STUDY_ID)
        if baseline:
            process_single_baseline(sid, new_layout=new_layout, force_recompute=force)
        else:
            process_single_study(sid, new_layout=new_layout, force_recompute=force)
