"""
aggregate_profiles.py — Aggregate per-registration profiling JSONL files into
a single summary CSV and a per-algorithm LaTeX table snippet.

Run after all algorithms have been executed:
    python aggregate_profiles.py

Outputs:
    {RESULTS_DIR}/profiling/profiling_detail.csv    — one row per registration
    {RESULTS_DIR}/profiling/profiling_summary.csv   — mean ± std per algo×input
    {RESULTS_DIR}/profiling/profiling_table.tex     — LaTeX table for the paper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare_config as C

PROFILING_DIR = C.RESULTS_DIR / "profiling"

# Pretty names for the comparison table
ALGO_NAMES = {
    "R_organ":        "Rigid (organ)",
    "R_sobel":        "Rigid (Sobel-NCC)",
    "R_full":         "Rigid (proposed)",
    "R_mmi":          "Rigid (MMI)",
    "B2_deeds":       "DEEDS",
    "B3_ants":        "ANTs-SyN",
    "B4_vxm":         "VoxelMorph-NMI",
    "B4_vxm_training":"VoxelMorph (train)",
    "A3_pass0":       "Kabsch (SVD)",
    "A4_pass01":      "Rigid P0+1",
    "A5_pass012":     "Rigid P0+1+2",
    "A6_bspline":     "B-spline deformable",
}


def load_all_records() -> pd.DataFrame:
    records = []
    for jsonl in sorted(PROFILING_DIR.glob("profiling_*.jsonl")):
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if not records:
        print(f"No profiling records found in {PROFILING_DIR}")
        print("Run the baseline algorithms first, then re-run this script.")
        return pd.DataFrame()
    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    # Inference rows (have a 'phase' field)
    infer = df[df.get("mode", pd.Series(["infer"] * len(df))) != "train"].copy() \
            if "mode" in df.columns else df.copy()
    infer = infer[infer["algo"].isin(ALGO_NAMES)]

    group_cols = ["algo", "input"]
    agg = (
        infer
        .groupby(group_cols)[["elapsed_sec", "peak_ram_mb", "peak_gpu_mb"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["algo_name"] = agg["algo"].map(ALGO_NAMES).fillna(agg["algo"])
    return agg


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns:
        return pd.DataFrame()
    return df[df["mode"] == "train"].copy()


def write_latex(summary: pd.DataFrame, train_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference time, peak RAM, and peak GPU memory per registration "
        r"(mean\,$\pm$\,std over the test set). "
        r"Training time is reported separately for learning-based methods.}",
        r"\label{tab:profiling}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & Input & Time (s) & RAM (MB) & GPU (MB) \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        t_mean = f"{row['elapsed_sec_mean']:.1f}"
        t_std  = f"{row['elapsed_sec_std']:.1f}" if not pd.isna(row['elapsed_sec_std']) else "—"
        r_mean = f"{row['peak_ram_mb_mean']:.0f}" if not pd.isna(row.get('peak_ram_mb_mean')) else "—"
        r_std  = f"{row['peak_ram_mb_std']:.0f}"  if not pd.isna(row.get('peak_ram_mb_std'))  else ""
        g_mean = f"{row['peak_gpu_mb_mean']:.0f}" if not pd.isna(row.get('peak_gpu_mb_mean')) else "—"
        g_std  = f"{row['peak_gpu_mb_std']:.0f}"  if not pd.isna(row.get('peak_gpu_mb_std'))  else ""

        ram_cell = f"{r_mean}$\\pm${r_std}" if r_std else r_mean
        gpu_cell = f"{g_mean}$\\pm${g_std}" if g_std else g_mean
        lines.append(
            f"  {row['algo_name']} & {row['input']} & "
            f"${t_mean}\\pm{t_std}$ & {ram_cell} & {gpu_cell} \\\\"
        )

    if not train_df.empty:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{Training (one-time cost)}} \\")
        for _, row in train_df.iterrows():
            algo = row.get("algo", "")
            loss = row.get("loss_type", "")
            total_h = row.get("total_elapsed_sec", 0) / 3600
            per_ep  = row.get("time_per_epoch_sec", 0)
            gpu_mb  = row.get("peak_gpu_mb")
            gpu_str = f"{gpu_mb:.0f}" if gpu_mb is not None else "—"
            lines.append(
                f"  {ALGO_NAMES.get(algo, algo)} ({loss}) & — & "
                f"{total_h:.1f}\\,h ({per_ep:.0f}s/ep) & — & {gpu_str} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table written to {out_path}")


def main() -> None:
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_records()
    if df.empty:
        return

    # Detail CSV
    detail_path = PROFILING_DIR / "profiling_detail.csv"
    df.to_csv(detail_path, index=False)
    print(f"Detail CSV: {detail_path}  ({len(df)} rows)")

    summary = summarise(df)
    train_df = training_rows(df)

    # Summary CSV
    summary_path = PROFILING_DIR / "profiling_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV: {summary_path}  ({len(summary)} rows)")

    # Print to terminal
    print("\n── Inference profiling summary ──────────────────────────────")
    cols = ["algo_name", "input", "elapsed_sec_mean", "elapsed_sec_std",
            "peak_ram_mb_mean", "peak_gpu_mb_mean", "elapsed_sec_count"]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].to_string(index=False))

    if not train_df.empty:
        print("\n── Training profiling ───────────────────────────────────────")
        tcols = ["algo", "loss_type", "total_elapsed_sec",
                 "time_per_epoch_sec", "peak_gpu_mb", "params_M", "gflops"]
        tavail = [c for c in tcols if c in train_df.columns]
        print(train_df[tavail].to_string(index=False))

    # LaTeX snippet
    tex_path = PROFILING_DIR / "profiling_table.tex"
    write_latex(summary, train_df, tex_path)


if __name__ == "__main__":
    main()
