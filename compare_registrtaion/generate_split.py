"""
generate_split.py — Create a reproducible train/test split at the study level.

Usage:
    python generate_split.py                         # default 80/20, seed 42
    python generate_split.py --test_ratio 0.25 --seed 0

    # Evaluation sample: mark exactly N randomly chosen studies as "test" and
    # everything else "train". Every condition then runs on the SAME N studies,
    # which is what makes a cross-algorithm table comparable — and it is the
    # only honest way to cut compute, since the alternative (each algorithm on
    # whatever it got through) compares methods on different data.
    python generate_split.py --sample 40 --sample_seed 7

Outputs:
    {MAIN_PATH}/splits/train_test_split.csv   columns: study_id, split
    {MAIN_PATH}/splits/eval_sample.csv        the sampled studies + provenance
                                              (written only with --sample)

All learning-based runners (VoxelMorph, LapIRN, …) should call load_split()
to filter their training pairs. Evaluation is ALWAYS run on test studies only
to avoid data leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare_config as C


SPLIT_CSV = C.MAIN / "splits" / "train_test_split.csv"


def generate_split(
    labels_csv: Path = C.LABELS_CSV,
    output_csv: Path = SPLIT_CSV,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    col = "StudyInstanceUID" if "StudyInstanceUID" in df.columns else "study_id"
    studies = df[col].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(studies)

    n_test = max(1, round(len(shuffled) * test_ratio))
    test_set  = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    split_df = pd.DataFrame({
        "study_id": list(shuffled),
        "split":    ["test" if s in test_set else "train" for s in shuffled],
    })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    print(f"Split written to {output_csv}")
    print(f"  total: {len(studies)} studies")
    print(f"  train: {len(train_set)}  ({len(train_set)/len(studies)*100:.0f}%)")
    print(f"  test:  {len(test_set)}   ({len(test_set)/len(studies)*100:.0f}%)")
    print(f"  seed:  {seed}")
    return split_df


def load_split(split: str = "train", split_csv: Path = SPLIT_CSV) -> set[str]:
    """Return the set of study_ids for the requested split ('train' or 'test')."""
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}\n"
            "Run: python generate_split.py"
        )
    df = pd.read_csv(split_csv)
    return set(df.loc[df["split"] == split, "study_id"].tolist())


SAMPLE_CSV = C.MAIN / "splits" / "eval_sample.csv"


def generate_sample(
    n: int,
    labels_csv: Path = C.LABELS_CSV,
    output_csv: Path = SPLIT_CSV,
    sample_csv: Path = SAMPLE_CSV,
    seed: int = 42,
    require_ref_phase: bool = True,
) -> pd.DataFrame:
    """
    Mark exactly `n` randomly chosen studies as "test", the rest "train".

    Used to cut the cost of the full condition grid. Every runner and the
    evaluator already filter on --split test, so after this every condition is
    measured on the identical study set — the comparison stays paired, which is
    what lets a with/without-front-end difference be tested per study rather
    than between two differently-sized groups.

    Studies without the reference phase are excluded from the draw: no fixed
    image means no registration, so sampling one would silently shrink n.

    Writes the provenance (seed, n, the drawn ids) to `sample_csv` so the
    sample can be reproduced and reported.
    """
    df = pd.read_csv(labels_csv, dtype=str)
    col = "StudyInstanceUID" if "StudyInstanceUID" in df.columns else "study_id"

    if require_ref_phase:
        eligible = sorted(df.loc[df["Label"] == C.REF_PHASE, col].unique())
        n_dropped = df[col].nunique() - len(eligible)
        if n_dropped:
            print(f"  excluded {n_dropped} studies with no '{C.REF_PHASE}' phase")
    else:
        eligible = sorted(df[col].unique())

    if n > len(eligible):
        raise SystemExit(
            f"--sample {n} exceeds the {len(eligible)} eligible studies. "
            f"Use --sample {len(eligible)} or fewer."
        )

    rng = np.random.default_rng(seed)
    drawn = set(rng.choice(np.array(eligible, dtype=object), size=n, replace=False))

    all_studies = sorted(df[col].unique())
    split_df = pd.DataFrame({
        "study_id": all_studies,
        "split":    ["test" if s in drawn else "train" for s in all_studies],
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    # Provenance: how many moving pairs this sample actually buys, per phase.
    sample_rows = df[df[col].isin(drawn)]
    moving = sample_rows[sample_rows["Label"] != C.REF_PHASE]
    sample_df = pd.DataFrame({
        "study_id": sorted(drawn),
        "n_phases": [int((sample_rows[col] == s).sum()) for s in sorted(drawn)],
    })
    sample_df["sample_seed"] = seed
    sample_df["sample_n"] = n
    sample_df["eligible_pool"] = len(eligible)
    sample_df.to_csv(sample_csv, index=False)

    print(f"Sample written to {output_csv}")
    print(f"  eligible pool : {len(eligible)} studies (have '{C.REF_PHASE}')")
    print(f"  sampled       : {n} studies  -> split=test")
    print(f"  held out      : {len(all_studies) - n} studies -> split=train (unused)")
    print(f"  moving pairs  : {len(moving)}")
    print(moving["Label"].value_counts().to_string())
    print(f"  seed          : {seed}")
    print(f"  provenance    : {sample_csv}")
    return split_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate train/test split")
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--output",     default=str(SPLIT_CSV))
    p.add_argument("--sample",     type=int,   default=None,
                   help="Draw exactly N studies as the test set instead of a "
                        "ratio split. Use this to cut the cost of the full "
                        "condition grid while keeping every condition on the "
                        "same studies.")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Seed for --sample (report this in the paper).")
    p.add_argument("--sample_csv", default=str(SAMPLE_CSV))
    args = p.parse_args()

    if args.sample is not None:
        generate_sample(
            n=args.sample,
            labels_csv=Path(args.labels_csv),
            output_csv=Path(args.output),
            sample_csv=Path(args.sample_csv),
            seed=args.sample_seed,
        )
    else:
        generate_split(
            labels_csv=Path(args.labels_csv),
            output_csv=Path(args.output),
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
