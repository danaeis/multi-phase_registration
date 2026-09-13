"""
generate_final_report.py
========================
Produce a combined evaluation + profiling report for ALL conditions on the
test split.

Steps it runs automatically:
  1. eval_compare_conditions.py --all --split test   (metrics)
  2. aggregate_profiles.py                           (time + resource)
  3. Merge both into one wide table
  4. Write:
       {RESULTS_DIR}/report/final_report.csv
       {RESULTS_DIR}/report/final_report.tex
       {RESULTS_DIR}/report/final_report_profiling.csv  (raw profiling join)

Usage:
    python generate_final_report.py [--split {all,train,test}] [--workers N]
    python generate_final_report.py --only-report   # skip re-evaluation, just merge

Requires:
    eval_compare_conditions.py and aggregate_profiles.py on sys.path (same dir).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import compare_config as C
from aggregate_profiles import load_all_records, summarise

REPORT_DIR = C.RESULTS_DIR / "report"

# ---------------------------------------------------------------------------
# Step 1 – run eval_compare_conditions if needed
# ---------------------------------------------------------------------------

def run_evaluation(split: str, workers: int) -> None:
    cmd = [
        sys.executable, str(_HERE / "eval_compare_conditions.py"),
        "--all", "--split", split, "--workers", str(workers),
    ]
    print(f"\n{'='*60}")
    print("Running evaluation …")
    print(" ".join(cmd))
    print("="*60)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 2 – load metrics table
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    """Load comparison_table.csv written by eval_compare_conditions."""
    p = C.RESULTS_DIR / "comparison_table.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"comparison_table.csv not found at {p}.\n"
            "Run eval_compare_conditions.py first (or omit --only-report)."
        )
    df = pd.read_csv(p)
    # The CSV has Method as first col (set_index in eval script)
    if "Method" in df.columns:
        df = df.set_index("Method")
    return df


# ---------------------------------------------------------------------------
# Step 3 – load profiling summary
# ---------------------------------------------------------------------------

def load_profiling() -> pd.DataFrame:
    """Aggregate JSONL profiling records → per-condition mean ± std."""
    raw = load_all_records()
    if raw.empty:
        print("  ⚠ No profiling records found — time/RAM columns will be empty.")
        return pd.DataFrame(columns=["condition_tag", "time_mean_s", "time_std_s",
                                     "ram_mean_mb", "gpu_mean_mb", "n_registrations"])

    # Build condition_tag from (algo, input) to match compare_config keys
    raw["condition_tag"] = raw.apply(
        lambda r: f"{r['algo']}__{r['input']}" if "input" in raw.columns
                  and pd.notna(r.get("input")) and r.get("input") != ""
                  else r["algo"],
        axis=1,
    )

    # Filter inference rows only
    if "mode" in raw.columns:
        raw = raw[raw["mode"] != "train"]

    agg = (
        raw.groupby("condition_tag")
        .agg(
            time_mean_s  =("elapsed_sec",  "mean"),
            time_std_s   =("elapsed_sec",  "std"),
            ram_mean_mb  =("peak_ram_mb",  "mean"),
            gpu_mean_mb  =("peak_gpu_mb",  "mean"),
            n_registrations=("elapsed_sec", "count"),
        )
        .round(2)
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Step 4 – merge and format
# ---------------------------------------------------------------------------

_ORDERED_CONDITIONS = [
    # A-series (ablation)
    "A1_crop_only", "A2_zalign", "A3_pass0", "A4_pass01", "A5_pass012",
    "A3b_pass0_noalign", "A4b_pass01_noalign", "A5b_pass012_noalign",
    "A6_bspline", "A6_bspline_baseline",
    # R-series (rigid objectives, on aligned input)
    "R_organ__aligned",  "R_organ__baseline",
    "R_sobel__aligned",  "R_sobel__baseline",
    "R_full__aligned",   "R_full__baseline",
    "R_mmi__aligned",    "R_mmi__baseline",
    "R_mind__aligned",   "R_mind__baseline",
    # B-series (external baselines)
    "B2_deeds__aligned", "B2_deeds__baseline", "B2_deeds__raw",
    "B3_ants__aligned",  "B3_ants__baseline",  "B3_ants__raw",
    "B4_vxm__aligned",   "B4_vxm__baseline",   "B4_vxm__raw",
    "B5_unigradicon__aligned", "B5_unigradicon__baseline",
    "B5_unigradicon_io__aligned",
]


def _tag_to_method(tag: str) -> str:
    """Map condition tag → pretty row label (fallback to tag itself)."""
    return C.ROW_LABELS.get(tag, tag)


def merge_tables(metrics_df: pd.DataFrame, prof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join metrics (indexed by Method label) with profiling (indexed by condition_tag).
    Returns a DataFrame indexed by condition_tag with all columns.
    """
    # Re-index metrics by condition tag
    tag_to_label = {tag: _tag_to_method(tag) for tag in C.CONDITIONS}
    label_to_tag = {v: k for k, v in tag_to_label.items()}

    metrics_reset = metrics_df.copy()
    metrics_reset.index.name = "Method"
    metrics_reset = metrics_reset.reset_index()
    metrics_reset["condition_tag"] = metrics_reset["Method"].map(label_to_tag)

    merged = metrics_reset.merge(prof_df, on="condition_tag", how="outer")

    # Reorder rows
    order = {t: i for i, t in enumerate(_ORDERED_CONDITIONS)}
    merged["_sort"] = merged["condition_tag"].map(order).fillna(999)
    merged = merged.sort_values("_sort").drop(columns=["_sort"])
    merged = merged.reset_index(drop=True)

    return merged


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex(df: pd.DataFrame, out_path: Path, split: str) -> None:
    metric_cols = [
        ("Eroded Dice ↑",   "Eroded Dice",    ""),
        ("HD95 ↓ (mm)",     "HD95",           " (mm)"),
        ("Centroid ↓ (mm)", "Centroid",       " (mm)"),
        ("|∇HU|-NCC ↑",    "$|{\\nabla}$HU|-NCC", ""),
        ("MIND-NCC ↑",      "MIND-NCC",       ""),
        ("%|J|<0 ↓",        "$\\%|J|{<}0$",   ""),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Registration results on the " + split + r" split "
        r"(median\,$\pm$\,IQR). "
        r"Time: mean\,$\pm$\,std per moving volume. "
        r"RAM: peak RSS. "
        r"$\dagger$~deformable method.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l" + "r" * (len(metric_cols) + 3) + r"}",
        r"\toprule",
    ]

    # Header
    hdr_names = " & ".join(
        [r"\textbf{Method}"]
        + [f"\\textbf{{{n[1]}}}" + n[2] for n in metric_cols]
        + [r"\textbf{Time (s)}", r"\textbf{RAM (MB)}", r"\textbf{GPU (MB)}"]
    )
    lines.append(hdr_names + r" \\")
    lines.append(r"\midrule")

    sections = [
        ("Ablation",          [t for t in _ORDERED_CONDITIONS if t.startswith("A")]),
        ("Rigid objectives",  [t for t in _ORDERED_CONDITIONS if t.startswith("R")]),
        ("External baselines",[t for t in _ORDERED_CONDITIONS if t.startswith("B")]),
    ]

    for sec_label, tags in sections:
        lines.append(r"\multicolumn{" + str(len(metric_cols)+4) +
                     r"}{l}{\textit{" + sec_label + r"}} \\")
        for tag in tags:
            row = df[df["condition_tag"] == tag]
            if row.empty:
                continue
            row = row.iloc[0]
            method = _tag_to_method(tag)
            # shorten for table width
            method = method.replace("Rigid · ", "").replace("[aligned]", "[al]") \
                           .replace("[baseline]", "[bl]").replace("[raw]", "[rw]")

            cells = [method]
            for col, _, _ in metric_cols:
                v = row.get(col, "—")
                cells.append("—" if pd.isna(v) or str(v).strip() == "" else str(v))

            # time
            tm = row.get("time_mean_s")
            ts = row.get("time_std_s")
            if pd.notna(tm):
                t_str = f"${tm:.1f}\\pm{ts:.1f}$" if pd.notna(ts) else f"{tm:.1f}"
            else:
                t_str = "—"
            cells.append(t_str)

            ram = row.get("ram_mean_mb")
            cells.append(f"{ram:.0f}" if pd.notna(ram) else "—")
            gpu = row.get("gpu_mean_mb")
            cells.append(f"{gpu:.0f}" if pd.notna(gpu) else "—")

            lines.append("  " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"   # replace last \midrule
    lines += [
        r"\end{tabular}",
        r"\end{table*}",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"✓ LaTeX → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate final combined report.")
    p.add_argument("--split",       choices=["all","train","test"], default="test")
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--only-report", action="store_true",
                   help="Skip re-evaluation; just merge existing CSVs.")
    args = p.parse_args()

    if not args.only_report:
        run_evaluation(args.split, args.workers)

    print("\n── Loading metrics …")
    metrics = load_metrics()
    print(f"   {len(metrics)} conditions in comparison_table.csv")

    print("── Loading profiling …")
    prof = load_profiling()
    print(f"   {len(prof)} condition-level profiling rows")

    merged = merge_tables(metrics, prof)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_p = REPORT_DIR / "final_report.csv"
    merged.to_csv(csv_p, index=False)
    print(f"✓ CSV  → {csv_p}")

    # Raw profiling join
    prof_p = REPORT_DIR / "final_report_profiling.csv"
    prof.to_csv(prof_p, index=False)
    print(f"✓ profiling CSV → {prof_p}")

    # LaTeX
    tex_p = REPORT_DIR / "final_report.tex"
    write_latex(merged, tex_p, args.split)

    # Terminal preview
    preview_cols = ["condition_tag", "Eroded Dice ↑", "HD95 ↓ (mm)",
                    "Centroid ↓ (mm)", "time_mean_s", "ram_mean_mb"]
    available = [c for c in preview_cols if c in merged.columns]
    print(f"\n{'='*80}")
    print(f"FINAL REPORT  (split={args.split})")
    print("="*80)
    print(merged[available].to_string(index=False))


if __name__ == "__main__":
    main()