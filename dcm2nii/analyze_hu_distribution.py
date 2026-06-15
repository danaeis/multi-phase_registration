"""
analyze_hu_distribution.py
===========================
Scan all NIfTI volumes in a directory tree and report HU value
distributions so you can pick a safe clipping range.

Reports per-volume and aggregate statistics:
  - true min / max (raw stored values)
  - percentiles: 0.1, 1, 5, 50, 95, 99, 99.9
  - fraction of voxels below common sentinel thresholds
    (-2048, -1500, -1024, -1000)
  - fraction of voxels above +1000, +1500, +2000

Run:
    python analyze_hu_distribution.py
    python analyze_hu_distribution.py --dir /path/to/nifti_unprocessed_volumes
    python analyze_hu_distribution.py --dir /path/to/volumes --out hu_report.csv
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from configs import MAIN_PATH

# ---------------------------------------------------------------------------
# Default path — point at the raw NIfTI output from dcm2nifti
# ---------------------------------------------------------------------------
DEFAULT_DIR = MAIN_PATH + "nifti_unprocessed_volumes"


# ===========================================================================
# CORE
# ===========================================================================

SENTINEL_THRESHOLDS_LOW  = [-4096, -2048, -1500, -1024, -1000]
SENTINEL_THRESHOLDS_HIGH = [1000, 1500, 2000, 3000]
PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def analyze_volume(path: str) -> Optional[dict]:
    """
    Load one NIfTI volume and compute HU statistics.

    We only read the body-tissue region for percentiles — voxels below -1500
    are almost certainly air/padding and would dominate the percentile
    calculation, masking the tissue distribution.

    Returns None if the file cannot be read.
    """
    try:
        img   = sitk.ReadImage(path)
        arr   = sitk.GetArrayFromImage(img).astype(np.float32).ravel()
        total = arr.size

        row: dict = {
            "path":       path,
            "filename":   os.path.basename(path),
            "n_voxels":   total,
            "true_min":   float(arr.min()),
            "true_max":   float(arr.max()),
        }

        # Fraction of voxels at/below each low sentinel
        for thr in SENTINEL_THRESHOLDS_LOW:
            frac = float((arr <= thr).sum()) / total
            row[f"frac_le_{abs(thr)}"] = round(frac, 6)

        # Fraction of voxels at/above each high sentinel
        for thr in SENTINEL_THRESHOLDS_HIGH:
            frac = float((arr >= thr).sum()) / total
            row[f"frac_ge_{thr}"] = round(frac, 6)

        # Percentiles on ALL voxels (to see the true distribution)
        pcts_all = np.percentile(arr, PERCENTILES)
        for p, v in zip(PERCENTILES, pcts_all):
            row[f"pct_{p}_all"] = round(float(v), 1)

        # Percentiles on body-tissue only (exclude extreme background)
        body = arr[arr > -1000]
        if body.size > 1000:
            pcts_body = np.percentile(body, PERCENTILES)
            for p, v in zip(PERCENTILES, pcts_body):
                row[f"pct_{p}_body"] = round(float(v), 1)
        else:
            for p in PERCENTILES:
                row[f"pct_{p}_body"] = None

        return row

    except Exception as e:
        print(f"  Could not read {path}: {e}")
        return None


def find_nifti_files(root_dir: str) -> List[str]:
    """Recursively find all .nii.gz and .nii files."""
    files = (
        glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True) +
        glob.glob(os.path.join(root_dir, "**", "*.nii"),    recursive=True)
    )
    # Exclude segmentation files
    files = [f for f in files if "seg" not in os.path.basename(f).lower()]
    return sorted(files)


# ===========================================================================
# MAIN
# ===========================================================================

def run(input_dir: str, out_csv: Optional[str] = None, max_files: int = 0):
    files = find_nifti_files(input_dir)
    if max_files > 0:
        files = files[:max_files]

    print(f"\nFound {len(files)} CT volumes in {input_dir}")
    print("Analysing HU distributions...\n")

    rows = []
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {os.path.basename(f)}", end="  ", flush=True)
        row = analyze_volume(f)
        if row:
            rows.append(row)
            print(f"min={row['true_min']:.0f}  max={row['true_max']:.0f}  "
                  f"frac<=-1024={row['frac_le_1024']:.3f}")
        else:
            print("FAILED")

    if not rows:
        print("No volumes could be analysed.")
        return

    df = pd.DataFrame(rows)

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")
    print(f"  Volumes analysed : {len(df)}")

    print(f"\n  True value range across all volumes:")
    print(f"    Minimum of minimums : {df['true_min'].min():.0f}")
    print(f"    Maximum of maximums : {df['true_max'].max():.0f}")
    print(f"    Median minimum      : {df['true_min'].median():.0f}")
    print(f"    Median maximum      : {df['true_max'].median():.0f}")

    print(f"\n  Fraction of voxels below sentinel thresholds (median across volumes):")
    for thr in SENTINEL_THRESHOLDS_LOW:
        col  = f"frac_le_{abs(thr)}"
        med  = df[col].median()
        p95  = df[col].quantile(0.95)
        print(f"    <= {thr:6d} HU : median={med:.4f}  95th-pct={p95:.4f}")

    print(f"\n  Fraction of voxels above sentinel thresholds (median across volumes):")
    for thr in SENTINEL_THRESHOLDS_HIGH:
        col  = f"frac_ge_{thr}"
        med  = df[col].median()
        p95  = df[col].quantile(0.95)
        print(f"    >= +{thr:5d} HU : median={med:.4f}  95th-pct={p95:.4f}")

    print(f"\n  Percentiles on ALL voxels (median across volumes):")
    for p in PERCENTILES:
        col = f"pct_{p}_all"
        print(f"    {p:5.1f}th  : {df[col].median():.0f} HU")

    print(f"\n  Percentiles on body-tissue voxels only (>-1000 HU, median across volumes):")
    for p in PERCENTILES:
        col = f"pct_{p}_body"
        valid = df[col].dropna()
        if len(valid):
            print(f"    {p:5.1f}th  : {valid.median():.0f} HU")

    # ── Sentinel detection ─────────────────────────────────────────────────
    print(f"\n  Sentinel value detection:")
    for thr in [-4096, -2048]:
        col       = f"frac_le_{abs(thr)}"
        n_affected = (df[col] > 0.01).sum()   # >1% of voxels at this level
        print(f"    Volumes with >1% voxels <= {thr}: "
              f"{n_affected} / {len(df)} "
              f"({100*n_affected/len(df):.1f}%)")

    # ── Clipping recommendation ────────────────────────────────────────────
    print(f"\n  Clipping range recommendation:")
    low_candidates  = [-1500, -1024, -1000]
    high_candidates = [1000, 1500, 2000]

    for lo in low_candidates:
        col      = f"frac_le_{abs(lo)}"
        if col in df.columns:
            lost = df[col].median() * 100
            print(f"    Lower clip at {lo:6d}: loses median {lost:.2f}% of voxels "
                  f"(background/padding)")

    for hi in high_candidates:
        col = f"frac_ge_{hi}"
        if col in df.columns:
            lost = df[col].median() * 100
            print(f"    Upper clip at +{hi:5d}: loses median {lost:.2f}% of voxels "
                  f"(bone/metal)")

    # ── Save results ───────────────────────────────────────────────────────
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\n  Raw per-volume data saved → {out_csv}")

        summary_path = out_csv.replace(".csv", "_summary.csv")
        summary_cols = (
            ["filename", "true_min", "true_max"]
            + [f"frac_le_{abs(t)}" for t in SENTINEL_THRESHOLDS_LOW]
            + [f"frac_ge_{t}"      for t in SENTINEL_THRESHOLDS_HIGH]
            + [f"pct_{p}_all"      for p in [1, 5, 50, 95, 99]]
            + [f"pct_{p}_body"     for p in [1, 5, 50, 95, 99]]
        )
        summary_cols = [c for c in summary_cols if c in df.columns]
        df[summary_cols].to_csv(summary_path, index=False)
        print(f"  Summary CSV saved           → {summary_path}")

    print(f"\n{'='*80}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse HU distributions across NIfTI CT volumes."
    )
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Root directory containing NIfTI volumes")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (optional)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max number of files to process (0 = all)")
    args = parser.parse_args()

    run(input_dir=args.dir, out_csv=args.out, max_files=args.max)