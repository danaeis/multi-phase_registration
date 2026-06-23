"""
gather_results.py
=================
Collect all CSV files produced under RESULTS_DIR condition subdirectories
and write one merged file per CSV type (eval_detail, resource_stats, …).

Each merged file gets a leading `condition` column populated from the
subdirectory name when not already present in the source CSV.

Outputs written to RESULTS_DIR/:
    all_eval_detail.csv        — merged from {condition}/eval_detail.csv
    all_resource_stats.csv     — merged from {condition}/resource_stats.csv
    all_<name>.csv             — for any other CSV type found

Usage
-----
python gather_results.py                        # all CSVs in RESULTS_DIR
python gather_results.py --results_dir /path    # custom directory
python gather_results.py --split test           # filter eval rows to test split
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare_config as C


def gather(
    results_dir: Path,
    study_ids: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Walk each subdirectory of results_dir, load every CSV, tag it with the
    directory name, and return a dict keyed by CSV filename (e.g. 'eval_detail.csv').
    """
    by_type: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    subdirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not subdirs:
        print(f"No subdirectories found under {results_dir}")
        return {}

    for subdir in subdirs:
        condition = subdir.name
        csvs = sorted(subdir.glob("*.csv"))
        if not csvs:
            continue

        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"  ⚠  [{condition}] {csv_path.name}: read error — {exc}")
                continue

            if df.empty:
                continue

            # Ensure a leading 'condition' column (eval_detail already has one;
            # resource_stats has it too, but this is safe to call either way)
            if "condition" not in df.columns:
                df.insert(0, "condition", condition)
            else:
                # Standardise: make condition the first column
                cols = ["condition"] + [c for c in df.columns if c != "condition"]
                df = df[cols]

            # Optionally filter eval rows to a study split
            if study_ids is not None and "study_id" in df.columns:
                df = df[df["study_id"].isin(study_ids)]
                if df.empty:
                    continue

            by_type[csv_path.name].append(df)

            if verbose:
                n = len(df)
                extra = (f"  {df['study_id'].nunique()} studies"
                         if "study_id" in df.columns else "")
                print(f"  [{condition}] {csv_path.name}: {n} rows{extra}")

    merged: Dict[str, pd.DataFrame] = {}
    for csv_name, frames in by_type.items():
        merged[csv_name] = pd.concat(frames, ignore_index=True)

    return merged


def write_merged(results_dir: Path, merged: Dict[str, pd.DataFrame]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    print()
    for csv_name, df in merged.items():
        out_path = results_dir / f"all_{csv_name}"
        df.to_csv(out_path, index=False)

        n_cond = df["condition"].nunique()
        n_rows = len(df)
        extra = (f", {df['study_id'].nunique()} studies"
                 if "study_id" in df.columns else "")
        print(f"✓  {out_path.name}  ({n_rows} rows, {n_cond} conditions{extra})")


def print_summary(merged: Dict[str, pd.DataFrame]) -> None:
    if "eval_detail.csv" not in merged:
        return

    df = merged["eval_detail.csv"]
    print(f"\n{'='*60}")
    print("COVERAGE SUMMARY  (conditions × studies in eval_detail)")
    print(f"{'='*60}")
    counts = (
        df.groupby("condition")["study_id"]
        .nunique()
        .rename("n_studies")
        .reset_index()
    )
    # Align to the canonical condition order from compare_config
    order = {tag: i for i, tag in enumerate(C.CONDITIONS)}
    counts["_ord"] = counts["condition"].map(lambda t: order.get(t, 9999))
    counts = counts.sort_values("_ord").drop(columns="_ord")
    for _, row in counts.iterrows():
        bar = "█" * min(int(row["n_studies"]), 60)
        print(f"  {row['condition']:<35}  {row['n_studies']:>3}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge all per-condition CSVs into one file per type.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--results_dir", default=str(C.RESULTS_DIR),
        help=f"Root directory containing condition subdirs (default: {C.RESULTS_DIR}).",
    )
    p.add_argument(
        "--split", choices=["all", "train", "test"], default="all",
        help="Filter eval rows to a train/test split (default: all).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file progress lines.",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    study_ids = None
    if args.split != "all":
        from generate_split import load_split as _load_split
        study_ids = _load_split(args.split)
        print(f"Split filter: '{args.split}' → {len(study_ids)} studies\n")

    print(f"Scanning: {results_dir}\n")
    merged = gather(results_dir, study_ids=study_ids, verbose=not args.quiet)

    if not merged:
        print("No CSV data found.")
        return

    write_merged(results_dir, merged)
    print_summary(merged)


if __name__ == "__main__":
    main()
