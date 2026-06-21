"""
generate_split.py — Create a reproducible train/test split at the study level.

Usage:
    python generate_split.py                         # default 80/20, seed 42
    python generate_split.py --test_ratio 0.25 --seed 0

Outputs:
    {MAIN_PATH}/splits/train_test_split.csv   columns: study_id, split

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
    studies = df["study_id"].unique()
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate train/test split")
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--output",     default=str(SPLIT_CSV))
    args = p.parse_args()
    generate_split(
        labels_csv=Path(args.labels_csv),
        output_csv=Path(args.output),
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
