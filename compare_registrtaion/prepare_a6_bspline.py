"""
prepare_a6_bspline.py
=====================
Populate the compare_config A6_bspline (or A6_bspline_baseline) condition
directory from an existing deformable_registration_v2.py output directory,
using hardlinks (zero extra disk space; falls back to copy on cross-device).

deformable_registration_v2.py writes:
    MAIN/deformable_registered_{metric}/           (aligned, default)
    MAIN/deformable_registered_{metric}_baseline/  (with --baseline)
  with postfixes: _deformable.nii.gz  _deformable_seg_reg.nii.gz  _deformable_dvf.nii.gz

compare_config.py expects:
    A6_bspline          → all_baseline_algorithms/A6_bspline/
    A6_bspline_baseline → all_baseline_algorithms/A6_bspline_baseline/
  with postfixes: _bspline.nii.gz  _bspline_seg_reg.nii.gz  _bspline_dvf.nii.gz

Usage
-----
python prepare_a6_bspline.py                   # aligned, sobel_ncc
python prepare_a6_bspline.py --baseline        # baseline (no z-align), sobel_ncc
python prepare_a6_bspline.py --metric grad_ncc
python prepare_a6_bspline.py --force           # overwrite existing targets
python prepare_a6_bspline.py --dry-run         # show what would be linked
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

# deformable_registration_v2.py output postfixes → compare_config expected postfixes
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


def prepare(metric: str, baseline: bool, force: bool, dry_run: bool) -> None:
    # Source: where deformable_registration_v2.py wrote its output
    suffix = f"_{metric}_baseline" if baseline else f"_{metric}"
    src_dir = C.MAIN / f"deformable_registered{suffix}"

    # Destination: what compare_config CONDITIONS expect
    cond_tag = "A6_bspline_baseline" if baseline else "A6_bspline"
    dst_cond = C.CONDITIONS.get(cond_tag)
    if dst_cond is None:
        raise SystemExit(f"{cond_tag} not found in compare_config.CONDITIONS")
    dst_dir = dst_cond.base_dir

    if not src_dir.exists():
        kind = "baseline" if baseline else "aligned"
        raise SystemExit(
            f"Source not found: {src_dir}\n"
            f"Run first:\n"
            f"  python ../registration/deformable_registration_v2.py "
            f"--metric {metric} --all --skip"
            + (" --baseline" if baseline else "")
        )

    print(f"Source ({cond_tag}): {src_dir}")
    print(f"Dest              : {dst_dir}")
    print(f"dry_run={dry_run}  force={force}")

    n_ok = n_skip = n_missing = 0

    for study_dir in sorted(src_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        study = study_dir.name
        out_study = dst_dir / study
        if not dry_run:
            out_study.mkdir(parents=True, exist_ok=True)

        for src_suf, dst_suf in SRC_POSTFIXES.items():
            for src_file in sorted(study_dir.glob(f"*{src_suf}")):
                stem = src_file.name[: -len(src_suf)]
                dst_file = out_study / f"{stem}{dst_suf}"

                if dst_file.exists() and not force:
                    n_skip += 1
                    continue
                if not src_file.exists():
                    n_missing += 1
                    continue
                if dst_file.exists() and force and not dry_run:
                    dst_file.unlink()

                link_or_copy(src_file, dst_file, dry_run)
                n_ok += 1
                if dry_run:
                    print(f"  [dry] {study}/{stem}{dst_suf}")

    print(f"\n{cond_tag} prepare done:")
    print(f"  linked/copied  : {n_ok}")
    print(f"  skipped (exist): {n_skip}")
    print(f"  source missing : {n_missing}")
    if dry_run:
        print("  (dry run — nothing written)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Link deformable_registered_{metric}[_baseline] → A6_bspline[_baseline]."
    )
    p.add_argument("--metric",   default=DEFAULT_METRIC,
                   help=f"Deformable metric (default: {DEFAULT_METRIC}).")
    p.add_argument("--baseline", action="store_true",
                   help="Prepare A6_bspline_baseline from the _baseline output dir.")
    p.add_argument("--force",    action="store_true",
                   help="Overwrite existing target files.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Show what would be linked without writing anything.")
    args = p.parse_args()
    prepare(args.metric, args.baseline, args.force, args.dry_run)


if __name__ == "__main__":
    main()
