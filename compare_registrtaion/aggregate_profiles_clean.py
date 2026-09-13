"""
aggregate_profiles_clean.py — Trustworthy aggregation of the profiling JSONL logs.

Differs from aggregate_profiles.py in four ways, each of which matters:

1. Drops records with elapsed_sec < MIN_SEC. profile_stage() writes a record even
   when register_fn raises, so a crashed DEEDS call lands as elapsed_sec=0.0.
   Those rows are 94% of profiling_B2_deeds_baseline.jsonl and would drag the
   DEEDS mean from ~282 s down to ~14 s — a fabricated 20x speed advantage.
2. De-duplicates on (algo, input, study, phase). The JSONL files are append-only,
   so every re-run appends another record for the same pair. Keeps the last.
3. Does NOT filter algos against a hard-coded name map. aggregate_profiles.py's
   ALGO_NAMES lacks R_mind, B5_unigradicon and B5_unigradicon_io, so it silently
   drops the uniGradICON baseline entirely.
4. Reports median + IQR alongside mean +/- std. Runtimes here are right-skewed
   and the mean is not the number to defend.

peak_gpu_mb is null in every existing record: _common.py calls profile_stage()
without use_gpu=True, so the CUDA branch never runs. The GPU column is emitted
but will be empty until that call site is fixed and the GPU methods re-run.

peak_ram_mb is the parent process RSS. Valid for the in-process methods
(R_*, B3_ants via SimpleITK/ANTsPy, B5_unigradicon), NOT for B2_deeds, which
shells out to the deedsBCV binary. DEEDS RAM is marked invalid in the output.

Usage:
    python aggregate_profiles_clean.py                 # all studies
    python aggregate_profiles_clean.py --split test    # test split only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare_config as C

PROFILING_DIR = C.RESULTS_DIR / "profiling"

# Methods whose real work happens in a child process -> parent RSS is meaningless.
SUBPROCESS_ALGOS = {"B2_deeds"}

DISPLAY_NAMES = {
    "R_organ":           "Rigid, centroid+eDice",
    "R_sobel":           "Rigid, Sobel-NCC",
    "R_full":            "Rigid, proposed",
    "R_mind":            "Rigid, MIND-NCC",
    "R_mmi":             "Rigid, MMI",
    "B2_deeds":          "DEEDS",
    "B3_ants":           "ANTs-SyN",
    "B4_vxm":            "VoxelMorph",
    "B5_unigradicon":    "uniGradICON",
    "B5_unigradicon_io": "uniGradICON + IO",
    "A3_pass0":          "Kabsch (SVD)",
    "A4_pass01":         "Rigid P0+1",
    "A5_pass012":        "Rigid P0+1+2",
    "A6_bspline":        "B-spline deformable",
}


def load_records() -> pd.DataFrame:
    records = []
    for jsonl in sorted(PROFILING_DIR.glob("profiling_*.jsonl")):
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        raise SystemExit(f"No profiling records found in {PROFILING_DIR}")
    return pd.DataFrame(records)


def clean(df: pd.DataFrame, min_sec: float, split: str | None) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["elapsed_sec"] >= min_sec].copy()
    n_crash = n0 - len(df)

    df = df.drop_duplicates(subset=["algo", "input", "study", "phase"], keep="last")
    n_dup = n0 - n_crash - len(df)

    n_split = 0
    if split:
        from generate_split import load_split
        keep = load_split(split)
        before = len(df)
        df = df[df["study"].isin(keep)]
        n_split = before - len(df)

    print(f"  loaded {n0} records")
    print(f"  dropped {n_crash} with elapsed_sec < {min_sec}s (failed registrations)")
    print(f"  dropped {n_dup} duplicate (algo,input,study,phase) re-runs")
    if split:
        print(f"  dropped {n_split} outside the '{split}' split")
    print(f"  -> {len(df)} usable records")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (algo, ikey), grp in df.groupby(["algo", "input"]):
        t = grp["elapsed_sec"]
        ram = grp["peak_ram_mb"].dropna()
        gpu = grp["peak_gpu_mb"].dropna()
        ram_valid = algo not in SUBPROCESS_ALGOS
        out.append({
            "algo":           algo,
            "algo_name":      DISPLAY_NAMES.get(algo, algo),
            "input":          ikey,
            "n":              len(grp),
            "n_studies":      grp["study"].nunique(),
            "time_median_s":  round(float(t.median()), 2),
            "time_q1_s":      round(float(t.quantile(0.25)), 2),
            "time_q3_s":      round(float(t.quantile(0.75)), 2),
            "time_mean_s":    round(float(t.mean()), 2),
            "time_std_s":     round(float(t.std()), 2),
            "ram_median_mb":  round(float(ram.median()), 1) if len(ram) and ram_valid else np.nan,
            "ram_mean_mb":    round(float(ram.mean()), 1) if len(ram) and ram_valid else np.nan,
            "ram_valid":      ram_valid,
            "gpu_median_mb":  round(float(gpu.median()), 1) if len(gpu) else np.nan,
        })
    return pd.DataFrame(out).sort_values(["input", "time_median_s"])


def write_latex(summary: pd.DataFrame, out_path: Path, split: str | None) -> None:
    cap = ("Computational cost per registration pair (median, inter-quartile range). "
           "RAM is peak resident set size of the registration process; it is not "
           "measurable for DEEDS, which runs in a child process. "
           "GPU memory was not recorded in this run.")
    if split:
        cap += f" Restricted to the {split} split."
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + cap + "}", r"\label{tab:profiling}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Method & Input & $n$ & Time (s) & RAM (MB) \\", r"\midrule",
    ]
    for _, r in summary.iterrows():
        t = f"{r['time_median_s']:.1f} [{r['time_q1_s']:.1f}--{r['time_q3_s']:.1f}]"
        ram = f"{r['ram_median_mb']:.0f}" if pd.notna(r["ram_median_mb"]) else r"n/a$^{\dagger}$"
        lines.append(f"  {r['algo_name']} & {r['input']} & {int(r['n'])} & {t} & {ram} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\\[2pt] \footnotesize $^{\dagger}$ external binary; parent-process RSS "
        r"does not capture it.",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  LaTeX table -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sec", type=float, default=1.0,
                    help="Records faster than this are treated as failed calls.")
    ap.add_argument("--split", choices=["train", "test"], default=None,
                    help="Restrict to one split (default: all studies).")
    args = ap.parse_args()

    df = clean(load_records(), args.min_sec, args.split)
    summary = summarise(df)

    suffix = f"_{args.split}" if args.split else ""
    detail_p = PROFILING_DIR / f"profiling_clean_detail{suffix}.csv"
    summ_p = PROFILING_DIR / f"profiling_clean_summary{suffix}.csv"
    tex_p = PROFILING_DIR / f"profiling_clean_table{suffix}.tex"

    df.to_csv(detail_p, index=False)
    summary.to_csv(summ_p, index=False)
    print(f"  detail  -> {detail_p}")
    print(f"  summary -> {summ_p}")
    write_latex(summary, tex_p, args.split)

    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
