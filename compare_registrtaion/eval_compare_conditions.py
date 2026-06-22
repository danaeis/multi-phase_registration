"""
eval_compare_conditions.py
==========================
Evaluate conditions defined in compare_config.py and write per-condition
detail CSVs plus an aggregated comparison table.

Mirrors evaluate_pipeline/evaluate_all.py but imports compare_config
(A-series + B-series × 3 inputs) instead of evaluate_pipeline/config.py.

Usage
-----
# All conditions with output on disk:
python eval_compare_conditions.py --all

# One algorithm across all inputs:
python eval_compare_conditions.py --prefix B2_deeds

# Hand-picked set:
python eval_compare_conditions.py --conditions B2_deeds__aligned B3_ants__raw

# Parallelise per-condition metric computation:
python eval_compare_conditions.py --all --workers 4

Outputs
-------
all_baseline_algorithms/{tag}/eval_detail.csv   per (study, phase, organ)
all_baseline_algorithms/all_conditions.csv      long format, all conditions
all_baseline_algorithms/comparison_table.csv    one row per condition
all_baseline_algorithms/comparison_table.tex    same, LaTeX
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── path setup ───────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_EVAL_PKG = _HERE.parent / "evaluate_pipeline"
sys.path.insert(0, str(_HERE))       # compare_config.py lives here
sys.path.insert(0, str(_EVAL_PKG))   # evaluate_registration.py lives here

import compare_config as C
import evaluate_registration as E


# ---------------------------------------------------------------------------
# Folding metric (%|J|<0) — identical logic to evaluate_all.py
# ---------------------------------------------------------------------------

def folding_for_condition(
    cond: C.Condition,
    labels_df: pd.DataFrame,
    ref_phase: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    if cond.dvf_postfix is None or not cond.base_dir.exists():
        return pd.DataFrame(rows)

    study_dirs = sorted(
        d for d in os.listdir(cond.base_dir) if (cond.base_dir / d).is_dir()
    )
    for study_id in study_dirs:
        study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
        for _, r in study_rows.iterrows():
            sid, phase = r["SeriesInstanceUID"], r["Label"]
            if phase == ref_phase:
                continue
            dvf_path = cond.base_dir / study_id / f"{study_id}_{sid}{cond.dvf_postfix}"
            if not dvf_path.exists():
                continue
            try:
                pct = E.neg_jacobian_pct(str(dvf_path), displacement_in_mm=True)
            except Exception as exc:
                print(f"    ⚠ folding failed [{study_id[:20]}/{phase}]: {exc}")
                pct = None
            if pct is not None:
                rows.append({"study_id": study_id, "phase": phase, "neg_jac_pct": pct})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def run_condition(
    cond: C.Condition,
    labels_df: pd.DataFrame,
    labels_csv: str,
    ref_phase: str,
    erosion_mm: float,
    workers: int = 1,
) -> Optional[pd.DataFrame]:
    print(f"\n{'#'*70}\n# CONDITION: {cond.tag}\n{'#'*70}")

    if not cond.base_dir.exists():
        print(f"  base_dir not found: {cond.base_dir} — skipping.")
        return None

    df = E.evaluate_all(
        base_dir=str(cond.base_dir),
        labels_csv=labels_csv,
        seg_postfix=cond.seg_postfix,
        vol_postfix=cond.vol_postfix,
        ref_phase=ref_phase,
        erosion_mm=erosion_mm,
        out_csv=None,
        workers=workers,
    )
    if df is None or df.empty:
        print(f"  no organ results for {cond.tag} — check postfixes / base_dir.")
        return None

    df.insert(0, "condition", cond.tag)

    if cond.is_deformable:
        fold = folding_for_condition(cond, labels_df, ref_phase)
        if not fold.empty:
            df = df.merge(fold, on=["study_id", "phase"], how="left")
            print(f"  folding attached for {fold['study_id'].nunique()} studies")
        else:
            df["neg_jac_pct"] = np.nan
            print("  ⚠ deformable but no DVF files found — %|J|<0 = n/a")
    else:
        df["neg_jac_pct"] = np.nan   # rigid: analytically zero → report "—"

    cond.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cond.detail_csv, index=False)
    print(f"  ✓ detail → {cond.detail_csv}")
    return df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _fmt(series: pd.Series, fmt: str) -> str:
    s = series.dropna()
    return "n/a" if s.empty else f"{s.median():{fmt}} ± {E.iqr(s):{fmt}}"


def build_comparison_table(detail_by_cond: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tag, cond in C.CONDITIONS.items():
        df    = detail_by_cond.get(tag)
        label = C.ROW_LABELS.get(tag, tag)
        if df is None or df.empty:
            rows.append({
                "Method": label,
                "Eroded Dice ↑": "—", "HD95 ↓ (mm)": "—",
                "Centroid ↓ (mm)": "—", "|∇HU|-NCC ↑": "—", "%|J|<0 ↓": "—",
            })
            continue

        if cond.is_deformable and df["neg_jac_pct"].notna().any():
            per_case = df.drop_duplicates(["study_id", "phase"])["neg_jac_pct"]
            jac = f"{per_case.mean():.2f}"
        else:
            jac = "—"

        rows.append({
            "Method":          label,
            "Eroded Dice ↑":   _fmt(df["eroded_dice"], ".3f"),
            "HD95 ↓ (mm)":     _fmt(df["hd95_mm"],     ".1f"),
            "Centroid ↓ (mm)": _fmt(df["centroid_mm"], ".1f"),
            "|∇HU|-NCC ↑":     _fmt(df["sobel_ncc"],   ".3f"),
            "%|J|<0 ↓":        jac,
        })

    return pd.DataFrame(rows).set_index("Method")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate compare_config.py conditions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--conditions", nargs="*", default=None,
        help="Explicit condition tags to evaluate.",
    )
    p.add_argument(
        "--prefix", default=None,
        help="Evaluate all conditions whose tag starts with this (e.g. B2_deeds).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Evaluate all conditions in compare_config (ignores --conditions/--prefix).",
    )
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--ref_phase",  default=C.REF_PHASE)
    p.add_argument("--erosion_mm", type=float, default=C.EROSION_MM)
    p.add_argument("--workers",    type=int,   default=1,
                   help="Parallel worker processes per condition (default: 1).")
    args = p.parse_args()

    if args.all or (args.conditions is None and args.prefix is None):
        tags = list(C.CONDITIONS)
    elif args.prefix:
        tags = [t for t in C.CONDITIONS if t.startswith(args.prefix)]
        if not tags:
            raise SystemExit(
                f"No conditions with prefix '{args.prefix}'.\n"
                f"Available: {list(C.CONDITIONS)}"
            )
    else:
        tags = args.conditions or []

    unknown = [t for t in tags if t not in C.CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown condition(s): {unknown}\nValid: {list(C.CONDITIONS)}")

    labels_df = pd.read_csv(args.labels_csv)

    detail_by_cond: Dict[str, pd.DataFrame] = {}
    for tag in tags:
        df = run_condition(
            C.CONDITIONS[tag], labels_df,
            args.labels_csv, args.ref_phase, args.erosion_mm,
            workers=args.workers,
        )
        if df is not None:
            detail_by_cond[tag] = df

    if not detail_by_cond:
        print("\nNo conditions produced results. Nothing to aggregate.")
        return

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_df   = pd.concat(detail_by_cond.values(), ignore_index=True)
    all_path = C.RESULTS_DIR / "all_conditions.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\n✓ long detail → {all_path}")

    if len(detail_by_cond) > 1 or args.all:
        table = build_comparison_table(detail_by_cond)
        print(f"\n{'='*60}\nCOMPARISON TABLE  (median ± IQR)\n{'='*60}")
        print(table.to_string())

        csv_p = C.RESULTS_DIR / "comparison_table.csv"
        tex_p = C.RESULTS_DIR / "comparison_table.tex"
        table.to_csv(csv_p)
        with open(tex_p, "w") as fh:
            fh.write(table.to_latex(escape=False))

        print(f"✓ comparison CSV   → {csv_p}")
        print(f"✓ comparison LaTeX → {tex_p}")


if __name__ == "__main__":
    main()
