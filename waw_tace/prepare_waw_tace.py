"""
prepare_waw_tace.py
===================
Adapter: turn the WAW-TACE release layout into a VinDr-shaped dataset root, so
every downstream stage of this pipeline runs on it unchanged.

WAW-TACE ships (Zenodo 12741586, CC BY 4.0):
    source/{case}/{case}_{phase}_scan.nii.gz
with phase codes  0=native  1=late arterial  2=portal venous  3=delayed.

This script writes, under --out:
    nifti_unprocessed_volumes/{study}_{series}_standardized.nii.gz
    nifti_unprocessed_volumes/{study}_{series}_standardized_metadata.txt
    labels.csv       StudyInstanceUID,SeriesInstanceUID,Label
    excluded.csv     one row per dropped case + reason  (goes in the paper)
    cohort.csv       per-series native geometry, for slice-thickness strata

ID scheme: study = "waw%04d", series = "p{0..3}". NOT the raw "24"/"0", because
  * pandas types a bare numeric ID column as int64, and every
    labels_df[... == study] comparison against an os.listdir string then
    silently matches nothing;
  * alignment/align_and_crop.py globs *{study}*{series}*standardized.nii.gz,
    which with study="2", series="0" matches most of the cohort;
  * fixed-width padding stops "waw002" matching inside "waw0024".
No underscores inside either token — segmentation/run_segmentation.sh recovers
study and series by splitting the filename on "_".

Standardisation is NOT reimplemented here. It calls
StandardizedDICOMProcessor.{reorient_to_lps,resample_to_target_spacing,
clip_hu_range} so the -1024 resample fill and the [-1024, 3000] clip stay
single-sourced with the DICOM path that produced the VinDr cohort. Output is
float32 at 1.5mm isotropic in LPS, byte-for-byte comparable to VinDr.

Usage
-----
  python prepare_waw_tace.py --source /data/waw_tace/source --out /data/waw_tace/derived
  python prepare_waw_tace.py ... --phases 0 1 2 3        # include delayed
  python prepare_waw_tace.py ... --cases 10 23 24        # subset
  python prepare_waw_tace.py ... --workers 8
  python prepare_waw_tace.py ... --report-only           # cohort tables, no volumes
  python prepare_waw_tace.py ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "dcm2nii"))
sys.path.insert(0, str(_HERE.parent))

from dicom_processor import StandardizedDICOMProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# WAW-TACE phase codes -> this pipeline's Label vocabulary.
#
# Keep these strings EXACTLY as VinDr uses them. "Non-contrast" is hardcoded as
# the fixed reference in register_fixed.py, deformable_registration_v2.py and
# align_and_crop.find_noncontrast_series; matching the vocabulary is what makes
# this cohort run with zero changes to those files.
# ---------------------------------------------------------------------------
PHASE_LABELS: Dict[int, str] = {
    0: "Non-contrast",
    1: "Arterial",
    2: "Venous",
    3: "Delayed",
}

REF_PHASE_CODE = 0
# A study is usable iff it has the reference phase AND at least one of these.
CONTRAST_PHASE_CODES = (1, 2)

TARGET_SPACING = (1.5, 1.5, 1.5)

# Slice-thickness strata for reporting (mm). WAW-TACE ranges 1.25-5.0 and varies
# within a patient; resampling 5.0 -> 1.5 is a 3.3x upsample and must be a
# reported variable, not a hidden one.
THIN_MAX_MM = 2.5

SCAN_RE = re.compile(r"^(?P<case>\d+)_(?P<phase>\d+)_scan\.nii(?:\.gz)?$")


def study_id(case: int) -> str:
    return f"waw{case:04d}"


def series_id(phase: int) -> str:
    return f"p{phase}"


# ===========================================================================
# DISCOVERY
# ===========================================================================

def scan_source(source: Path,
                phases: Tuple[int, ...],
                cases: Optional[List[int]] = None,
                ) -> Tuple[Dict[int, Dict[int, Path]], List[dict]]:
    """
    Walk source/ and return ({case: {phase: path}}, [excluded rows]).

    Only phases in `phases` are collected. Folders that are empty, hold no
    recognisable scan, or lack the reference phase are reported as excluded
    rather than silently dropped.
    """
    found: Dict[int, Dict[int, Path]] = {}
    excluded: List[dict] = []

    if not source.is_dir():
        raise SystemExit(f"--source is not a directory: {source}")

    case_dirs = sorted(
        (d for d in source.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if cases:
        want = set(cases)
        case_dirs = [d for d in case_dirs if int(d.name) in want]

    for d in case_dirs:
        case = int(d.name)
        per_phase: Dict[int, Path] = {}
        unparsed: List[str] = []
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            m = SCAN_RE.match(f.name)
            if not m:
                unparsed.append(f.name)
                continue
            ph = int(m.group("phase"))
            if ph in phases:
                per_phase[ph] = f

        if not per_phase:
            excluded.append({
                "case": case, "study_id": study_id(case),
                "phases_present": "",
                "reason": "no_scan_files" if not unparsed else "unrecognised_filenames",
                "detail": ";".join(unparsed[:5]),
            })
            continue

        present = sorted(per_phase)
        if REF_PHASE_CODE not in per_phase:
            excluded.append({
                "case": case, "study_id": study_id(case),
                "phases_present": ",".join(map(str, present)),
                "reason": "no_reference_phase",
                "detail": f"phase {REF_PHASE_CODE} ({PHASE_LABELS[REF_PHASE_CODE]}) absent",
            })
            continue

        if not any(p in per_phase for p in CONTRAST_PHASE_CODES):
            excluded.append({
                "case": case, "study_id": study_id(case),
                "phases_present": ",".join(map(str, present)),
                "reason": "no_contrast_phase",
                "detail": f"none of {CONTRAST_PHASE_CODES} present",
            })
            continue

        found[case] = per_phase

    return found, excluded


# ===========================================================================
# CONVERSION (one series)
# ===========================================================================

def convert_series(src: Path,
                   out_vol: Path,
                   overwrite: bool = False) -> dict:
    """
    Reorient -> resample to 1.5mm iso -> clip HU -> write.

    Returns a cohort row with the NATIVE geometry recorded, so results can be
    stratified by slice thickness afterwards.
    """
    row: dict = {"src": str(src), "out": str(out_vol), "status": "ok"}

    if out_vol.exists() and not overwrite:
        img = sitk.ReadImage(str(out_vol))
        row.update(status="skipped",
                   final_size="x".join(map(str, img.GetSize())),
                   final_spacing_mm="x".join(f"{s:.3f}" for s in img.GetSpacing()))
        return row

    proc = StandardizedDICOMProcessor(target_spacing=TARGET_SPACING)

    img = sitk.ReadImage(str(src))
    native_spacing = img.GetSpacing()
    native_size = img.GetSize()
    native_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection())

    row.update(
        native_size="x".join(map(str, native_size)),
        native_spacing_mm="x".join(f"{s:.3f}" for s in native_spacing),
        native_slice_thickness_mm=round(float(native_spacing[2]), 3),
        native_inplane_mm=round(float(native_spacing[0]), 3),
        native_orientation=native_orient,
        thickness_stratum="thin" if native_spacing[2] <= THIN_MAX_MM else "thick",
    )

    # WAW-TACE ships LAS; the pipeline standardises to LPS.
    img = proc.reorient_to_lps(img, native_orient)
    img = proc.resample_to_target_spacing(img)
    img = proc.clip_hu_range(img)          # -> float32, [-1024, 3000]

    out_vol.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_vol))

    arr = sitk.GetArrayViewFromImage(img)
    row.update(
        final_size="x".join(map(str, img.GetSize())),
        final_spacing_mm="x".join(f"{s:.3f}" for s in img.GetSpacing()),
        hu_min=round(float(arr.min()), 1),
        hu_max=round(float(arr.max()), 1),
    )

    # Same sidecar format as the DICOM path, plus the native geometry the
    # thickness stratification needs.
    meta = str(out_vol).replace(".nii.gz", "_metadata.txt")
    with open(meta, "w") as f:
        f.write(f"Original NIfTI path: {src}\n")
        f.write(f"Source dataset: WAW-TACE (Zenodo 12741586, CC BY 4.0)\n")
        f.write(f"Native spacing: {native_spacing}\n")
        f.write(f"Native size: {native_size}\n")
        f.write(f"Native orientation: {native_orient}\n")
        f.write(f"Target spacing: {TARGET_SPACING}\n")
        f.write(f"Final spacing: {img.GetSpacing()}\n")
        f.write(f"Final size: {img.GetSize()}\n")
        f.write(f"Target orientation: {proc.target_orientation}\n")
        f.write(f"Final direction: {img.GetDirection()}\n")
        f.write(f"HU clip range: [{proc._CLIP_LOW_HU:.0f}, {proc._CLIP_HIGH_HU:.0f}]\n")
        f.write(f"Final HU min: {arr.min():.1f}\n")
        f.write(f"Final HU max: {arr.max():.1f}\n")

    return row


def _worker(task: Tuple[int, int, str, str, bool]) -> dict:
    case, phase, src, out_vol, overwrite = task
    base = {"case": case, "study_id": study_id(case),
            "phase_code": phase, "series_id": series_id(phase),
            "label": PHASE_LABELS[phase]}
    try:
        base.update(convert_series(Path(src), Path(out_vol), overwrite))
    except Exception as e:                                  # noqa: BLE001
        base.update(status="failed", error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=3))
    return base


# ===========================================================================
# TABLES
# ===========================================================================

LABELS_HEADER = ["StudyInstanceUID", "SeriesInstanceUID", "Label"]
EXCLUDED_HEADER = ["case", "study_id", "phases_present", "reason", "detail"]


def write_labels(out_root: Path,
                 found: Dict[int, Dict[int, Path]]) -> int:
    path = out_root / "labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABELS_HEADER)
        for case in sorted(found):
            for phase in sorted(found[case]):
                w.writerow([study_id(case), series_id(phase), PHASE_LABELS[phase]])
                n += 1
    print(f"  labels.csv        {n} series / {len(found)} studies  -> {path}")
    return n


def write_excluded(out_root: Path, excluded: List[dict]) -> None:
    path = out_root / "excluded.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXCLUDED_HEADER)
        w.writeheader()
        for r in sorted(excluded, key=lambda x: x["case"]):
            w.writerow({k: r.get(k, "") for k in EXCLUDED_HEADER})
    print(f"  excluded.csv      {len(excluded)} cases dropped        -> {path}")
    from collections import Counter
    for reason, k in Counter(r["reason"] for r in excluded).most_common():
        print(f"      {reason:24s} {k}")


def write_cohort(out_root: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path = out_root / "cohort.csv"
    cols = ["case", "study_id", "series_id", "phase_code", "label", "status",
            "native_size", "native_spacing_mm", "native_inplane_mm",
            "native_slice_thickness_mm", "native_orientation",
            "thickness_stratum", "final_size", "final_spacing_mm",
            "hu_min", "hu_max", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["case"], x["phase_code"])):
            w.writerow(r)
    print(f"  cohort.csv        {len(rows)} series                  -> {path}")

    thin = sum(1 for r in rows if r.get("thickness_stratum") == "thin")
    thick = sum(1 for r in rows if r.get("thickness_stratum") == "thick")
    if thin or thick:
        print(f"      slice thickness: thin(<={THIN_MAX_MM}mm)={thin}  "
              f"thick(>{THIN_MAX_MM}mm)={thick}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert WAW-TACE into a VinDr-shaped pipeline root.")
    p.add_argument("--source", required=True,
                   help="WAW-TACE source/ directory (folders named by case id).")
    p.add_argument("--out", required=True,
                   help="Output dataset root — this becomes MAIN_PATH.")
    p.add_argument("--phases", nargs="*", type=int, default=[0, 1, 2],
                   choices=[0, 1, 2, 3],
                   help="Phase codes to include. Default 0 1 2 (VinDr-comparable); "
                        "add 3 for the delayed-phase robustness run.")
    p.add_argument("--cases", nargs="*", type=int, default=None,
                   help="Optional subset of case ids.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel conversion processes.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reconvert series whose output already exists.")
    p.add_argument("--report-only", action="store_true",
                   help="Write labels/excluded only; convert nothing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen; write nothing.")
    args = p.parse_args()

    source = Path(args.source).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    vol_dir = out_root / "nifti_unprocessed_volumes"
    phases = tuple(sorted(set(args.phases)))

    print("=" * 74)
    print("prepare_waw_tace.py")
    print(f"  source  : {source}")
    print(f"  out     : {out_root}")
    print(f"  phases  : {phases}  -> {[PHASE_LABELS[p_] for p_ in phases]}")
    print(f"  workers : {args.workers}")
    print("=" * 74)

    if REF_PHASE_CODE not in phases:
        print(f"ERROR: phase {REF_PHASE_CODE} ({PHASE_LABELS[REF_PHASE_CODE]}) is the "
              f"fixed reference and must be included.")
        return 2

    found, excluded = scan_source(source, phases, args.cases)
    print(f"\nusable studies: {len(found)}   excluded cases: {len(excluded)}")
    for case in sorted(found):
        print(f"  {study_id(case):10s} phases {sorted(found[case])}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        for r in sorted(excluded, key=lambda x: x["case"]):
            print(f"  DROP {r['study_id']:10s} [{r['phases_present'] or '-'}] "
                  f"{r['reason']}")
        return 0

    if not found:
        print("\nNothing usable found — check --source layout.")
        write_excluded(out_root, excluded)
        return 1

    print("\ntables:")
    write_labels(out_root, found)
    write_excluded(out_root, excluded)

    if args.report_only:
        print("\n--report-only: no volumes converted.")
        return 0

    tasks = [(case, phase, str(src), str(vol_dir / f"{study_id(case)}_{series_id(phase)}_standardized.nii.gz"), args.overwrite)
             for case in sorted(found) for phase, src in sorted(found[case].items())]

    print(f"\nconverting {len(tasks)} series ...")
    rows: List[dict] = []
    n_ok = n_skip = n_fail = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                n_ok += r["status"] == "ok"
                n_skip += r["status"] == "skipped"
                n_fail += r["status"] == "failed"
                _progress(i, len(tasks), r)
    else:
        for i, t in enumerate(tasks, 1):
            r = _worker(t)
            rows.append(r)
            n_ok += r["status"] == "ok"
            n_skip += r["status"] == "skipped"
            n_fail += r["status"] == "failed"
            _progress(i, len(tasks), r)

    print()
    write_cohort(out_root, rows)

    print(f"\nDONE  ok={n_ok}  skipped={n_skip}  failed={n_fail}  / {len(tasks)}")
    if n_fail:
        print("\nfailures:")
        for r in rows:
            if r["status"] == "failed":
                print(f"  {r['study_id']}_{r['series_id']}: {r.get('error')}")
    print(f"\nNext:  export MAIN_PATH={out_root}/")
    return 1 if n_fail else 0


def _progress(i: int, n: int, r: dict) -> None:
    flag = {"ok": "+", "skipped": ".", "failed": "!"}[r["status"]]
    print(f"  [{i:4d}/{n}] {flag} {r['study_id']}_{r['series_id']}"
          f" {r.get('native_spacing_mm', '')}"
          f" -> {r.get('final_size', '')}"
          + (f"  {r.get('error')}" if r["status"] == "failed" else ""))


if __name__ == "__main__":
    sys.exit(main())
