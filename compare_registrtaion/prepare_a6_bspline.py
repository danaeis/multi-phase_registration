"""
prepare_a6_bspline.py
=====================
Populate the compare_config A6_bspline condition directory from an existing
deformable_registration_v2.py output directory.

deformable_registration_v2.py writes:
    MAIN/deformable_registered_{metric}/{study}/{study}_{series}_deformable.nii.gz
    MAIN/deformable_registered_{metric}/{study}/{study}_{series}_deformable_seg_reg.nii.gz
    MAIN/deformable_registered_{metric}/{study}/{study}_{series}_deformable_dvf.nii.gz

compare_config.py A6_bspline expects:
    MAIN/all_baseline_algorithms/A6_bspline/{study}/{study}_{series}_bspline.nii.gz
    MAIN/all_baseline_algorithms/A6_bspline/{study}/{study}_{series}_bspline_seg_reg.nii.gz
    MAIN/all_baseline_algorithms/A6_bspline/{study}/{study}_{series}_bspline_dvf.nii.gz

This script creates hardlinks (zero extra disk space; falls back to copy on
cross-device mount) so evaluate_compare_conditions.py can evaluate A6_bspline
without re-running registration.

Usage
-----
# Use sobel_ncc (default — best metric per deformable ablation):
python prepare_a6_bspline.py

# Use a different metric:
python prepare_a6_bspline.py --metric grad_ncc

# Overwrite any existing targets:
python prepare_a6_bspline.py --force

# Dry run — show what would be linked without touching the filesystem:
python prepare_a6_bspline.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import compare_config as C

# deformable_registration_v2.py postfixes
SRC_POSTFIXES = {
    "_deformable.nii.gz":         "_bspline.nii.gz",
    "_deformable_seg_reg.nii.gz": "_bspline_seg_reg.nii.gz",
    "_deformable_dvf.nii.gz":     "_bspline_dvf.nii.gz",
}

DEFAULT_METRIC = "sobel_ncc"


def link_or_copy(src: Path, dst: Path, dry_run: bool) -> str:
    if dry_run:
        return "dry"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Link/copy deformable_registered_{metric} → A6_bspline location."
    )
    p.add_argument("--metric", default=DEFAULT_METRIC,
                   help=f"Deformable metric to use (default: {DEFAULT_METRIC}).")
    p.add_argument("--force",   action="store_true",
                   help="Overwrite existing targets.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without touching disk.")
    args = p.parse_args()

    src_dir  = C.MAIN / f"deformable_registered_{args.metric}"
    dst_cond = C.CONDITIONS.get("A6_bspline")
    if dst_cond is None:
        raise SystemExit("A6_bspline not found in compare_config.CONDITIONS")
    dst_dir = dst_cond.base_dir

    if not src_dir.exists():
        raise SystemExit(
            f"Source directory not found: {src_dir}\n"
            f"Run first:\n"
            f"  python ../registration/deformable_registration_v2.py "
            f"--metric {args.metric} --all --skip"
        )

    print(f"Source : {src_dir}")
    print(f"Dest   : {dst_dir}")
    print(f"Metric : {args.metric}  |  dry_run={args.dry_run}  force={args.force}")
    print()

    n_ok = n_skip = n_missing = 0

    for study_dir in sorted(src_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        study = study_dir.name
        out_study = dst_dir / study

        if not args.dry_run:
            out_study.mkdir(parents=True, exist_ok=True)

        for src_suf, dst_suf in SRC_POSTFIXES.items():
            for src_file in sorted(study_dir.glob(f"*{src_suf}")):
                stem = src_file.name[: -len(src_suf)]
                dst_file = out_study / f"{stem}{dst_suf}"

                if dst_file.exists() and not args.force:
                    n_skip += 1
                    continue

                if not src_file.exists():
                    n_missing += 1
                    continue

                if dst_file.exists() and args.force and not args.dry_run:
                    dst_file.unlink()

                action = link_or_copy(src_file, dst_file, args.dry_run)
                n_ok += 1
                if args.dry_run:
                    print(f"  [dry] {study}/{stem}{dst_suf}")

    print(f"\nA6_bspline prepare done:")
    print(f"  linked/copied : {n_ok}")
    print(f"  skipped (exist): {n_skip}")
    print(f"  source missing: {n_missing}")
    if args.dry_run:
        print("  (dry run — nothing written)")


if __name__ == "__main__":
    main()
