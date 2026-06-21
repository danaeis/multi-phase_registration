"""
run_deformable_ablation.py
==========================
Orchestrate the full deformable B-spline metric ablation:

  For each metric (or a user-specified subset):
    1. Run deformable_registration_v2.py --all --skip --metric <m>
    2. Run evaluate_registration.py --all on the output directory
    3. Save per-metric CSV to evaluation/

  After all metrics: load every CSV and print a side-by-side comparison table
  (median ± IQR per metric and per organ), saved to evaluation/.

Metrics available
-----------------
  grad_ncc     Central-diff |∇HU| + ANTS NCC          [headline]
  sobel_ncc    Sobel |∇HU| (σ=0.75mm) + ANTS NCC
  mmi          Mattes MI on raw HU
  seg_dist     Organ signed-dist maps + MSE
  mind_ncc     MIND 6-offset descriptor + ANTS NCC
  sobel_binary Per-organ Otsu binary edges + MSE
  mind_sobel   MIND × (1 + sobel_binary) + ANTS NCC

Usage
-----
  # Full ablation (all 7 metrics, skip already-done series):
  python run_deformable_ablation.py

  # Subset:
  python run_deformable_ablation.py --metrics grad_ncc mind_ncc mind_sobel

  # Re-run evaluation only (registration already done):
  python run_deformable_ablation.py --skip-reg

  # Registration only (evaluate later):
  python run_deformable_ablation.py --skip-eval

  # Dry run — show commands without executing:
  python run_deformable_ablation.py --dry-run

Outputs
-------
  evaluation/eval_deformable_{metric}.csv        per-(study,phase,organ) rows
  evaluation/eval_deformable_{metric}_summary.csv  per-(organ,phase) median±IQR
  evaluation/deformable_comparison.csv           one row per metric (ablation table)
  evaluation/deformable_comparison_perorgan.csv  one row per (metric,organ)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths — mirrors deformable_registration_v2.py's own path logic
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_HERE_PAR = _HERE.parent
sys.path.insert(0, str(_HERE / "registration"))

try:
    from configs import MAIN_PATH  # type: ignore
except ImportError:
    MAIN_PATH = os.environ.get("MAIN_PATH", "../../ncct_cect/vindr_ds/")

if not MAIN_PATH.endswith("/"):
    MAIN_PATH += "/"

MAIN = Path(MAIN_PATH).resolve()

# Mirrors OUTPUT_DIR_BASE in deformable_registration_v2.py
_DEFORMABLE_BASE = MAIN / "deformable_registered"
LABELS_CSV       = str(MAIN / "labels.csv")

# Evaluation output goes here (already exists from prior runs)
EVAL_DIR = _HERE / "evaluation"

# Postfixes written by deformable_registration_v2.py
VOL_POSTFIX = "_deformable.nii.gz"
SEG_POSTFIX = "_deformable_seg_reg.nii.gz"

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
ALL_METRICS = (
    "grad_ncc",
    "sobel_ncc",
    "mmi",
    "seg_dist",
    "mind_ncc",
    "sobel_binary",
    "mind_sobel",
)

METRIC_LABEL: Dict[str, str] = {
    "grad_ncc":     "grad-NCC  (central-diff ∇HU)",
    "sobel_ncc":    "Sobel-NCC (σ=0.75 mm ∇HU)",
    "mmi":          "Mattes MI (raw HU)",
    "seg_dist":     "Organ dist-MSE",
    "mind_ncc":     "MIND-NCC  (6-offset)",
    "sobel_binary": "Binary-Sobel MSE (per-organ Otsu)",
    "mind_sobel":   "MIND×(1+sobel) NCC",
}

# Eval columns in the order they appear in the printed table
EVAL_COLS = ["eroded_dice", "hd95_mm", "sobel_ncc", "centroid_mm"]
COL_FMT   = {"eroded_dice": ".3f", "hd95_mm": ".2f",
              "sobel_ncc": ".3f",   "centroid_mm": ".2f"}
COL_HEADER = {"eroded_dice": "Dice↑", "hd95_mm": "HD95↓(mm)",
              "sobel_ncc": "∇NCC↑", "centroid_mm": "Cent↓(mm)"}


# ===========================================================================
# STEP 1 — Registration
# ===========================================================================

def output_dir_for(metric: str, baseline: bool = False) -> Path:
    """Mirrors the out_dir computation in deformable_registration_v2.py __main__."""
    d = _DEFORMABLE_BASE.parent / (f"deformable_registered_{metric}")
    if baseline:
        d = Path(str(d) + "_baseline")
    return d


def run_registration(
        metric: str,
        grid: float,
        iters: int,
        sampling: float,
        skip_existing: bool,
        no_field: bool,
        baseline: bool,
        dry_run: bool,
) -> bool:
    """
    Run deformable_registration_v2.py for one metric.

    Returns True on success, False on subprocess failure.
    """
    reg_script = str(_HERE / "deformable_registration_v2.py")
    cmd = [
        sys.executable, reg_script,
        "--metric",   metric,
        "--all",
        "--grid",     str(grid),
        "--iters",    str(iters),
        "--sampling", str(sampling),
    ]
    if skip_existing:
        cmd.append("--skip")
    if no_field:
        cmd.append("--no-field")
    if baseline:
        cmd.append("--baseline")

    print(f"\n  CMD: {' '.join(cmd)}")
    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_HERE))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✓" if ok else "✗"
    print(f"\n  {status} Registration [{metric}] finished in {timedelta(seconds=int(elapsed))}")
    return ok


# ===========================================================================
# STEP 2 — Evaluation
# ===========================================================================

def eval_csv_path(metric: str) -> Path:
    return EVAL_DIR / f"eval_deformable_{metric}.csv"


def run_evaluation(
        metric: str,
        out_dir: Path,
        dry_run: bool,
) -> bool:
    """
    Run evaluate_registration.py --all on one metric's registration output.

    Returns True on success (or dry_run), False on failure or missing dir.
    """
    if not out_dir.exists():
        print(f"  ✗ Output dir missing, skipping eval: {out_dir}")
        return False

    eval_script = str(_HERE_PAR / "evaluate_pipeline" / "evaluate_registration.py")
    csv_out     = str(eval_csv_path(metric))
    cmd = [
        sys.executable, eval_script,
        "--all",
        "--base_dir",   str(out_dir),
        "--labels_csv", LABELS_CSV,
        "--seg_postfix", SEG_POSTFIX,
        "--vol_postfix", VOL_POSTFIX,
        "--out_csv",    csv_out,
    ]

    print(f"\n  CMD: {' '.join(cmd)}")
    if dry_run:
        return True

    EVAL_DIR.mkdir(exist_ok=True)
    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(_HERE))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✓" if ok else "✗"
    print(f"\n  {status} Evaluation [{metric}] finished in {timedelta(seconds=int(elapsed))}")
    return ok


# ===========================================================================
# STEP 3 — Comparison
# ===========================================================================

def _iqr(series: pd.Series) -> float:
    a = series.dropna().values
    if len(a) == 0:
        return float("nan")
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def _med(series: pd.Series) -> float:
    a = series.dropna().values
    return float(np.median(a)) if len(a) > 0 else float("nan")


def load_all_results(metrics: List[str]) -> Dict[str, pd.DataFrame]:
    """Load per-metric CSVs that exist on disk. Warn about missing ones."""
    dfs: Dict[str, pd.DataFrame] = {}
    for m in metrics:
        p = eval_csv_path(m)
        if p.exists():
            dfs[m] = pd.read_csv(p)
        else:
            print(f"  ⚠  No eval CSV for '{m}': {p}")
    return dfs


def build_overall_comparison(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One row per metric.
    Columns: metric_label, then for each eval col: median and IQR.
    """
    rows = []
    for metric, df in dfs.items():
        row: dict = {"metric": METRIC_LABEL.get(metric, metric)}
        n_pairs = len(df)
        row["n_pairs"] = n_pairs
        for col in EVAL_COLS:
            if col in df.columns:
                row[f"{col}_med"] = _med(df[col])
                row[f"{col}_iqr"] = _iqr(df[col])
            else:
                row[f"{col}_med"] = float("nan")
                row[f"{col}_iqr"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_organ_comparison(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One row per (metric, organ).
    """
    rows = []
    for metric, df in dfs.items():
        if "organ" not in df.columns:
            continue
        for organ, grp in df.groupby("organ"):
            row: dict = {
                "metric": METRIC_LABEL.get(metric, metric),
                "metric_key": metric,
                "organ": organ,
                "n_pairs": len(grp),
            }
            for col in EVAL_COLS:
                if col in grp.columns:
                    row[f"{col}_med"] = _med(grp[col])
                    row[f"{col}_iqr"] = _iqr(grp[col])
                else:
                    row[f"{col}_med"] = float("nan")
                    row[f"{col}_iqr"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def print_overall_table(comp: pd.DataFrame) -> None:
    """Pretty-print the overall comparison table."""
    if comp.empty:
        print("  (no results to compare)")
        return

    # Header
    col_w = 32
    num_w = 12
    header_parts = [f"{'Metric':<{col_w}}"]
    for col in EVAL_COLS:
        header_parts.append(f"{COL_HEADER[col]:>{num_w}}")
    print("\n" + "=" * (col_w + num_w * len(EVAL_COLS) + 2))
    print("DEFORMABLE METRIC COMPARISON  (median ± IQR, all organs × phases)")
    print("=" * (col_w + num_w * len(EVAL_COLS) + 2))
    print("".join(header_parts))
    print("-" * (col_w + num_w * len(EVAL_COLS) + 2))

    for _, row in comp.iterrows():
        parts = [f"{row['metric']:<{col_w}}"]
        for col in EVAL_COLS:
            med = row.get(f"{col}_med", float("nan"))
            iqr = row.get(f"{col}_iqr", float("nan"))
            fmt = COL_FMT[col]
            if np.isnan(med):
                cell = f"{'—':>{num_w}}"
            else:
                cell_str = f"{med:{fmt}}±{iqr:{fmt}}"
                cell = f"{cell_str:>{num_w}}"
            parts.append(cell)
        print("".join(parts))

    print("=" * (col_w + num_w * len(EVAL_COLS) + 2))


def print_per_organ_table(per_organ: pd.DataFrame, metric_key: str, col: str = "eroded_dice") -> None:
    """Print one eval column across organs for a given metric."""
    sub = per_organ[per_organ["metric_key"] == metric_key]
    if sub.empty:
        return
    print(f"\n  [{METRIC_LABEL.get(metric_key, metric_key)}]  {COL_HEADER.get(col, col)}  per organ:")
    for _, row in sub.sort_values("organ").iterrows():
        med = row.get(f"{col}_med", float("nan"))
        iqr_val = row.get(f"{col}_iqr", float("nan"))
        fmt = COL_FMT.get(col, ".3f")
        print(f"    {row['organ']:<22}  {med:{fmt}} ± {iqr_val:{fmt}}")


def print_organ_ranking(per_organ: pd.DataFrame, col: str = "eroded_dice") -> None:
    """
    For each organ, rank metrics by median of col.
    Useful for spotting which metric wins on which organ.
    """
    if per_organ.empty:
        return

    ascending = col in ("hd95_mm", "centroid_mm")  # lower is better
    med_col   = f"{col}_med"
    hdr       = COL_HEADER.get(col, col)

    print(f"\n{'='*60}")
    print(f"Per-organ ranking by {hdr}  ({'↑ higher' if not ascending else '↓ lower'} = better)")
    print(f"{'='*60}")

    for organ, grp in per_organ.groupby("organ"):
        ranked = grp.sort_values(med_col, ascending=ascending)
        print(f"\n  {organ}")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            med = row[med_col]
            fmt = COL_FMT.get(col, ".3f")
            if np.isnan(med):
                continue
            marker = "★ " if rank == 1 else f"{rank}. "
            print(f"    {marker}{row['metric_key']:<15}  {med:{fmt}}")


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full deformable metric ablation (registration + evaluation + comparison).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--metrics", nargs="+", choices=ALL_METRICS, default=list(ALL_METRICS),
        help="Metrics to run (default: all 7).",
    )
    ap.add_argument(
        "--skip-reg",  action="store_true",
        help="Skip registration; only run evaluation on existing output dirs.",
    )
    ap.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation; only run registration.",
    )
    ap.add_argument(
        "--skip", action="store_true", default=True,
        help="Pass --skip to registration (skip already-processed series). Default: on.",
    )
    ap.add_argument(
        "--no-skip", dest="skip", action="store_false",
        help="Force re-registration of all series (overrides --skip).",
    )
    ap.add_argument(
        "--grid", type=float, default=25.0,
        help="B-spline grid spacing in mm (default=25.0).",
    )
    ap.add_argument(
        "--iters", type=int, default=150,
        help="LBFGSB max iterations per pyramid level (default=150).",
    )
    ap.add_argument(
        "--sampling", type=float, default=0.15,
        help="Random metric sampling fraction (default=0.15).",
    )
    ap.add_argument(
        "--no-field", action="store_true",
        help="Do not save displacement fields (faster, less disk).",
    )
    ap.add_argument(
        "--baseline", action="store_true",
        help="Use fixed_baseline_rigid_registered input instead of aligned.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print commands but do not execute them.",
    )
    args = ap.parse_args()

    metrics = args.metrics
    t_start = time.time()

    print(f"\n{'='*70}")
    print(f"DEFORMABLE METRIC ABLATION")
    print(f"  metrics  : {metrics}")
    print(f"  data     : {MAIN}")
    print(f"  grid     : {args.grid}mm   iters={args.iters}   sampling={args.sampling:.0%}")
    print(f"  skip-reg : {args.skip_reg}   skip-eval: {args.skip_eval}")
    print(f"  dry-run  : {args.dry_run}")
    print(f"{'='*70}")

    reg_status:  Dict[str, bool] = {}
    eval_status: Dict[str, bool] = {}

    for idx, metric in enumerate(metrics, 1):
        out_dir = output_dir_for(metric, baseline=args.baseline)

        print(f"\n{'━'*70}")
        print(f"[{idx}/{len(metrics)}]  {METRIC_LABEL.get(metric, metric)}")
        print(f"  output dir : {out_dir}")
        print(f"{'━'*70}")

        # ── Registration ─────────────────────────────────────────────────────
        if not args.skip_reg:
            ok = run_registration(
                metric=metric,
                grid=args.grid,
                iters=args.iters,
                sampling=args.sampling,
                skip_existing=args.skip,
                no_field=args.no_field,
                baseline=args.baseline,
                dry_run=args.dry_run,
            )
            reg_status[metric] = ok
            if not ok:
                print(f"  ✗ Registration failed for {metric} — skipping evaluation.")
                eval_status[metric] = False
                continue
        else:
            print(f"  ⏭  Registration skipped (--skip-reg).")
            reg_status[metric] = True

        # ── Evaluation ───────────────────────────────────────────────────────
        if not args.skip_eval:
            ok = run_evaluation(
                metric=metric,
                out_dir=out_dir,
                dry_run=args.dry_run,
            )
            eval_status[metric] = ok
        else:
            print(f"  ⏭  Evaluation skipped (--skip-eval).")
            eval_status[metric] = True

    # ── Comparison table ─────────────────────────────────────────────────────
    if not args.skip_eval and not args.dry_run:
        print(f"\n{'='*70}")
        print("BUILDING COMPARISON TABLE")
        print(f"{'='*70}")

        dfs       = load_all_results(metrics)
        overall   = build_overall_comparison(dfs)
        per_organ = build_per_organ_comparison(dfs)

        # Print overall table
        print_overall_table(overall)

        # Print per-organ dice ranking
        print_organ_ranking(per_organ, col="eroded_dice")
        print_organ_ranking(per_organ, col="hd95_mm")

        # Save CSVs
        EVAL_DIR.mkdir(exist_ok=True)
        comp_path     = EVAL_DIR / "deformable_comparison.csv"
        organ_path    = EVAL_DIR / "deformable_comparison_perorgan.csv"
        overall.to_csv(comp_path,  index=False)
        per_organ.to_csv(organ_path, index=False)
        print(f"\n  ✓ Comparison table  → {comp_path}")
        print(f"  ✓ Per-organ table   → {organ_path}")

    # ── Run summary ──────────────────────────────────────────────────────────
    elapsed = timedelta(seconds=int(time.time() - t_start))
    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE  (total time: {elapsed})")
    print(f"{'='*70}")
    for m in metrics:
        r = "✓" if reg_status.get(m, True) else "✗"
        e = "✓" if eval_status.get(m, True) else "✗"
        skip_marker = "(skipped)" if args.skip_reg else ""
        print(f"  {m:<16}  reg={r}{skip_marker}  eval={e}")


if __name__ == "__main__":
    main()
