"""
evaluate_all.py
===============
Run evaluate_registration over every condition in config.CONDITIONS, attach the
deformable-only folding metric (%|J|<0), and emit a single comparison table
(median ± IQR per organ x phase) plus a LaTeX version for the paper.

Usage
-----
# All conditions that have output on disk:
python evaluate_all.py

# Only some conditions:
python evaluate_all.py --conditions A1_crop_only A2_zalign A5_pass012

# Custom labels / reference phase / erosion:
python evaluate_all.py --labels_csv /data/labels.csv --ref_phase Non-contrast --erosion_mm 6.0

Outputs
-------
results/{tag}/eval_detail.csv     per (study, phase, organ) rows for one condition
results/all_conditions.csv        every condition concatenated (long format)
results/comparison_table.csv      one row per condition (the experiment table)
results/comparison_table.tex      same, as LaTeX
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C
import evaluate_registration as E


# ---------------------------------------------------------------------------
# Folding metric (%|J|<0) for deformable conditions
# ---------------------------------------------------------------------------

def folding_for_condition(cond: C.Condition,
                          labels_df: pd.DataFrame,
                          ref_phase: str) -> pd.DataFrame:
    """
    Compute %|J|<0 per (study, phase) for one deformable condition.

    Returns a DataFrame with columns [study_id, phase, neg_jac_pct].
    Missing / unreadable DVFs are skipped (logged), not fatal.
    """
    rows: List[dict] = []
    if cond.dvf_postfix is None or not cond.base_dir.exists():
        return pd.DataFrame(rows)

    study_dirs = sorted(d for d in os.listdir(cond.base_dir)
                        if (cond.base_dir / d).is_dir())

    for study_id in study_dirs:
        study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
        for _, r in study_rows.iterrows():
            sid, phase = r["SeriesInstanceUID"], r["Label"]
            if phase == ref_phase:
                continue
            dvf_path = (cond.base_dir / study_id /
                        f"{study_id}_{sid}{cond.dvf_postfix}")
            if not dvf_path.exists():
                continue
            try:
                pct = E.neg_jacobian_pct(str(dvf_path), displacement_in_mm=True)
            except Exception as e:                       # pragma: no cover
                print(f"    ⚠ folding calc failed [{study_id[:20]}…/{phase}]: {e}")
                pct = None
            if pct is not None:
                rows.append({"study_id": study_id, "phase": phase,
                             "neg_jac_pct": pct})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def run_condition(cond: C.Condition,
                  labels_df: pd.DataFrame,
                  labels_csv: str,
                  ref_phase: str,
                  erosion_mm: float,
                  workers: int = 1) -> Optional[pd.DataFrame]:
    """Evaluate one condition. Returns the per-organ detail df (with condition tag)."""
    print(f"\n{'#'*80}\n# CONDITION: {cond.tag}\n{'#'*80}")

    if not cond.base_dir.exists():
        print(f"  base_dir not found ({cond.base_dir}) — condition not run yet, skipping.")
        return None

    df = E.evaluate_all(
        base_dir=str(cond.base_dir),
        labels_csv=labels_csv,
        seg_postfix=cond.seg_postfix,
        vol_postfix=cond.vol_postfix,
        ref_phase=ref_phase,
        erosion_mm=erosion_mm,
        out_csv=None,                      # we save ourselves below
        workers=workers,
    )
    if df is None or df.empty:
        print(f"  no organ results for {cond.tag} — check postfixes.")
        return None

    df.insert(0, "condition", cond.tag)

    # Attach folding metric (deformable only); merge by (study, phase)
    if cond.is_deformable:
        fold = folding_for_condition(cond, labels_df, ref_phase)
        if not fold.empty:
            df = df.merge(fold, on=["study_id", "phase"], how="left")
            print(f"  folding (%|J|<0) attached for "
                  f"{fold['study_id'].nunique()} studies")
        else:
            df["neg_jac_pct"] = np.nan
            print("  ⚠ deformable condition but no DVF files found — %|J|<0 = n/a")
    else:
        df["neg_jac_pct"] = np.nan          # rigid: analytically 0 -> report "—"

    cond.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cond.detail_csv, index=False)
    print(f"  ✓ detail → {cond.detail_csv}")
    return df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _fmt_med_iqr(series: pd.Series, fmt: str) -> str:
    s = series.dropna()
    if s.empty:
        return "n/a"
    return f"{s.median():{fmt}} ± {E.iqr(s):{fmt}}"


def build_comparison_table(detail_by_cond: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per condition: median ± IQR per metric, plus %|J|<0."""
    rows = []
    for tag in C.CONDITIONS:                      # registry order
        df = detail_by_cond.get(tag)
        cond = C.CONDITIONS[tag]
        label = C.ROW_LABELS.get(tag, tag)

        if df is None or df.empty:
            rows.append({"Method": label, "Eroded Dice ↑": "—", "HD95 ↓ (mm)": "—",
                         "Centroid ↓ (mm)": "—", "|∇HU|-NCC ↑": "—", "%|J|<0 ↓": "—"})
            continue

        if cond.is_deformable and df["neg_jac_pct"].notna().any():
            # one value per (study,phase); average those, not the organ-duplicated rows
            per_case = df.drop_duplicates(["study_id", "phase"])["neg_jac_pct"]
            jac = f"{per_case.mean():.2f}"
        else:
            jac = "—"                              # rigid: analytically zero

        rows.append({
            "Method":          label,
            "Eroded Dice ↑":   _fmt_med_iqr(df["eroded_dice"], ".3f"),
            "HD95 ↓ (mm)":     _fmt_med_iqr(df["hd95_mm"],     ".1f"),
            "Centroid ↓ (mm)": _fmt_med_iqr(df["centroid_mm"], ".1f"),
            "|∇HU|-NCC ↑":     _fmt_med_iqr(df["sobel_ncc"],   ".3f"),
            "%|J|<0 ↓":        jac,
        })
    return pd.DataFrame(rows).set_index("Method")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate all registration conditions.")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Subset of condition tags (default: all in registry).")
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--ref_phase",  default=C.REF_PHASE)
    p.add_argument("--erosion_mm", type=float, default=C.EROSION_MM)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes per condition (default: 1)")
    args = p.parse_args()

    tags = args.conditions or list(C.CONDITIONS)
    unknown = [t for t in tags if t not in C.CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown condition(s): {unknown}\n"
                         f"Valid: {list(C.CONDITIONS)}")

    labels_df = pd.read_csv(args.labels_csv)

    detail_by_cond: Dict[str, pd.DataFrame] = {}
    for tag in tags:
        df = run_condition(C.CONDITIONS[tag], labels_df,
                           args.labels_csv, args.ref_phase, args.erosion_mm,
                           workers=args.workers)
        if df is not None:
            detail_by_cond[tag] = df

    if not detail_by_cond:
        raise SystemExit("\nNo conditions produced results. Nothing to aggregate.")

    # Concatenate long-format detail across conditions
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(detail_by_cond.values(), ignore_index=True)
    all_path = C.RESULTS_DIR / "all_conditions.csv"
    all_df.to_csv(all_path, index=False)

    # Comparison table
    table = build_comparison_table(detail_by_cond)

    print(f"\n{'='*80}\nCOMPARISON TABLE  (median ± IQR over organs × phases)\n{'='*80}")
    print(table.to_string())

    csv_path = C.RESULTS_DIR / "comparison_table.csv"
    tex_path = C.RESULTS_DIR / "comparison_table.tex"
    table.to_csv(csv_path)
    with open(tex_path, "w") as fh:
        fh.write(table.to_latex(escape=False))

    print(f"\n✓ long detail        → {all_path}")
    print(f"✓ comparison (CSV)   → {csv_path}")
    print(f"✓ comparison (LaTeX) → {tex_path}")


if __name__ == "__main__":
    main()