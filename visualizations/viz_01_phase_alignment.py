"""
viz_01_phase_alignment.py
=========================
Visual 1: Phase alignment before / after comparison.

Produces a 2-row × 3-column figure:
  Row 1 (Before): NC / Arterial / Venous at their ORIGINAL Z-positions
  Row 2 (After) : NC / Arterial / Venous after your pipeline (aligned)

Each column shows the same anatomical level so misalignment is immediately
visible in row 1 and corrected in row 2.

Usage
-----
python viz_01_phase_alignment.py \
    --before_dir  /path/to/aligned_volumes/STUDY_ID \
    --after_dir   /path/to/rigid_registered/STUDY_ID \
    --study_id    1.2.840.113619... \
    --labels_csv  /path/to/labels.csv \
    --out         figures/fig1_alignment.png

The script uses the NC volume as the spatial reference and samples the
axial slice closest to the liver centroid (from seg_reg if available,
else the mid-slice of the volume).

Dependencies
------------
pip install nibabel matplotlib numpy SimpleITK pandas
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import SimpleITK as sitk
    def load_vol(path):
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)   # (Z, Y, X)
        return arr, img.GetSpacing()
except ImportError:
    import nibabel as nib
    def load_vol(path):
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)       # (X, Y, Z) → transpose
        arr = arr.transpose(2, 1, 0)        # → (Z, Y, X)
        return arr, img.header.get_zooms()


# ── colour / style constants ────────────────────────────────────────────────
PHASE_COLORS = {
    "Non-contrast": "#3A6EA8",
    "Arterial":     "#C0392B",
    "Venous":       "#7D3C98",
}
PHASE_ORDER   = ["Non-contrast", "Arterial", "Venous"]
HU_WINDOW     = (-200, 300)   # soft-tissue window
FIG_DPI       = 200


def find_vol(directory: str, study_id: str, series_id: str,
             suffix: str) -> Path | None:
    candidates = [
        Path(directory) / f"{study_id}_{series_id}{suffix}",
        Path(directory) / f"{series_id}{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def liver_slice(vol_arr: np.ndarray, seg_path=None) -> int:
    """Return Z index of liver centroid, or mid-slice as fallback."""
    if seg_path is not None and Path(seg_path).exists():
        try:
            seg, _ = load_vol(seg_path)
            liver = seg == 1
            if liver.sum() > 200:
                zz = np.where(liver.any(axis=(1, 2)))[0]
                return int(zz.mean())
        except Exception:
            pass
    return vol_arr.shape[0] // 2


def window(arr: np.ndarray, wmin=-200, wmax=300) -> np.ndarray:
    return np.clip((arr - wmin) / (wmax - wmin), 0, 1)


def make_figure(before_slices, after_slices, phase_names, out_path):
    n = len(phase_names)
    fig = plt.figure(figsize=(5 * n, 8), facecolor="black")
    gs  = gridspec.GridSpec(2, n, figure=fig,
                            hspace=0.06, wspace=0.04,
                            left=0.04, right=0.96, top=0.88, bottom=0.06)

    row_labels = ["Before registration_1", "After registration_1"]
    row_colors = ["#C0392B", "#1A7A4A"]

    for row, (slices, rlabel, rcol) in enumerate(
            zip([before_slices, after_slices], row_labels, row_colors)):
        for col, (sl, phase) in enumerate(zip(slices, phase_names)):
            ax = fig.add_subplot(gs[row, col])
            if sl is not None:
                ax.imshow(window(sl, *HU_WINDOW), cmap="gray",
                          aspect="equal", interpolation="bilinear")
            else:
                ax.set_facecolor("#111")
                ax.text(0.5, 0.5, "Not found", color="white",
                        ha="center", va="center", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # Phase label (top row only)
            if row == 0:
                pc = PHASE_COLORS.get(phase, "#AAAAAA")
                ax.set_title(phase, color=pc, fontsize=13,
                             fontweight="bold", pad=6)

            # Row label (left column only)
            if col == 0:
                ax.set_ylabel(rlabel, color=rcol, fontsize=11,
                              fontweight="bold", labelpad=8)

            # Thin coloured border
            for spine in ax.spines.values():
                spine.set_edgecolor(rcol)
                spine.set_linewidth(1.5)

    fig.suptitle("Multi-Phase CT — Axial Slice at Liver Level",
                 color="white", fontsize=14, fontweight="bold", y=0.96)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Figure 1: phase alignment before/after")
    ap.add_argument("--before_dir", required=True,
                    help="Directory with UNALIGNED volumes (original NIfTIs or "
                         "aligned_volumes/STUDY_ID for the 'before' row)")
    ap.add_argument("--after_dir",  required=True,
                    help="Directory with REGISTERED volumes (rigid_registered/"
                         "STUDY_ID or deformable_registered/STUDY_ID)")
    ap.add_argument("--study_id",   required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--before_vol_suffix", default="_aligned.nii.gz",
                    help="Volume filename suffix for 'before' directory")
    ap.add_argument("--after_vol_suffix",  default="_rigid2.nii.gz",
                    help="Volume filename suffix for 'after' directory")
    ap.add_argument("--before_seg_suffix", default="_aligned_seg_reg.nii.gz")
    ap.add_argument("--after_seg_suffix",  default="_rigid2_seg_reg.nii.gz")
    ap.add_argument("--out", default="figures/fig1_phase_alignment.png")
    args = ap.parse_args()

    labels_df  = pd.read_csv(args.labels_csv)
    study_rows = labels_df[labels_df["StudyInstanceUID"] == args.study_id]

    if study_rows.empty:
        sys.exit(f"Study {args.study_id} not found in labels CSV.")

    # Sort phases in canonical order
    phase_map = {}
    for _, row in study_rows.iterrows():
        phase_map[row["Label"]] = row["SeriesInstanceUID"]

    phases = [p for p in PHASE_ORDER if p in phase_map]
    if not phases:
        sys.exit("No recognised phases found.")

    # Find NC reference slice index (from 'after' NC volume)
    nc_sid = phase_map.get("Non-contrast")
    ref_z  = None
    if nc_sid:
        vol_path = find_vol(args.after_dir, args.study_id, nc_sid,
                            args.after_vol_suffix)
        seg_path = find_vol(args.after_dir, args.study_id, nc_sid,
                            args.after_seg_suffix)
        if vol_path:
            arr, _ = load_vol(vol_path)
            ref_z  = liver_slice(arr, seg_path)
            print(f"Reference slice: Z={ref_z} (liver centroid in NC)")

    before_slices, after_slices = [], []

    for phase in phases:
        sid = phase_map[phase]

        # ── BEFORE ──────────────────────────────────────────────────────
        vp_b = find_vol(args.before_dir, args.study_id, sid,
                        args.before_vol_suffix)
        if vp_b:
            arr_b, _ = load_vol(vp_b)
            # For 'before': use the same absolute Z so misalignment is visible
            z_b = ref_z if ref_z is not None else arr_b.shape[0] // 2
            z_b = min(z_b, arr_b.shape[0] - 1)
            before_slices.append(arr_b[z_b])
        else:
            print(f"  [before] {phase}: volume not found at {vp_b}")
            before_slices.append(None)

        # ── AFTER ───────────────────────────────────────────────────────
        vp_a = find_vol(args.after_dir, args.study_id, sid,
                        args.after_vol_suffix)
        if vp_a:
            arr_a, _ = load_vol(vp_a)
            z_a = ref_z if ref_z is not None else arr_a.shape[0] // 2
            z_a = min(z_a, arr_a.shape[0] - 1)
            after_slices.append(arr_a[z_a])
        else:
            print(f"  [after]  {phase}: volume not found at {vp_a}")
            after_slices.append(None)

    make_figure(before_slices, after_slices, phases, args.out)


if __name__ == "__main__":
    main()
