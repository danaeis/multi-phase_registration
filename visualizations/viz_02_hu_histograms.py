"""
viz_02_hu_histograms.py
=======================
Visual 2: Per-organ HU intensity histograms across CT phases.

For each organ in the seg_reg mask, overlays the HU distribution from
NC, Arterial, and Venous phases on the same axes.  Visually demonstrates
WHY MMI and NCC are invalid across contrast phases.

The aorta subplot is the most dramatic: ~40 HU (NC) → ~300 HU (arterial).

Output: one figure with one subplot per organ (3×4 grid or auto-sized).

Usage
-----
python viz_02_hu_histograms.py \
    --vol_dir    /path/to/aligned_volumes/STUDY_ID \
    --study_id   1.2.840.113619... \
    --labels_csv /path/to/labels.csv \
    --vol_suffix  _aligned.nii.gz \
    --seg_suffix  _aligned_seg_reg.nii.gz \
    --out         figures/fig2_hu_histograms.png

Dependencies
------------
pip install nibabel matplotlib numpy SimpleITK pandas scipy
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
    def load_arr(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
except ImportError:
    import nibabel as nib
    def load_arr(path):
        img = nib.load(str(path))
        return np.asarray(img.dataobj).transpose(2, 1, 0)


# ── Config ──────────────────────────────────────────────────────────────────
ORGAN_LABELS = {
    1:  "Liver",
    2:  "Spleen",
    3:  "Kidney (L)",
    4:  "Kidney (R)",
    13: "Aorta",
    14: "IVC",
    40: "Spinal Cord",
    21: "Vertebra L1",
    7:  "Autochthon (L)",
    8:  "Autochthon (R)",
}

PHASE_COLORS = {
    "Non-contrast": "#3A6EA8",
    "Arterial":     "#C0392B",
    "Venous":       "#7D3C98",
}
PHASE_ORDER = ["Non-contrast", "Arterial", "Venous"]
PHASE_SHORT = {"Non-contrast": "NC", "Arterial": "Art", "Venous": "Ven"}

HU_RANGE    = (-200, 500)
N_BINS      = 80
MIN_VOXELS  = 200
FIG_DPI     = 200
NCOLS       = 4
BG          = "#F9F9F9"   # matches your DML slide background


def find_file(directory, study_id, series_id, suffix):
    for p in [
        Path(directory) / f"{study_id}_{series_id}{suffix}",
        Path(directory) / f"{series_id}{suffix}",
    ]:
        print(f"  Checking for {p}...")
        if p.exists():
            return p
    return None


def organ_hu(vol: np.ndarray, seg: np.ndarray, label: int):
    mask = seg == label
    if mask.sum() < MIN_VOXELS:
        return None
    return vol[mask].ravel()


def main():
    ap = argparse.ArgumentParser(
        description="Figure 2: per-organ HU histograms across phases")
    ap.add_argument("--vol_dir",    required=True)
    ap.add_argument("--study_id",   required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--vol_suffix", default="_aligned.nii.gz")
    ap.add_argument("--seg_suffix", default="_aligned_seg_reg.nii.gz")
    ap.add_argument("--out", default="figures/fig2_hu_histograms.png")
    ap.add_argument("--organs", nargs="+", type=int, default=None,
                    help="Organ label ints to include (default: all defined)")
    args = ap.parse_args()

    labels_df  = pd.read_csv(args.labels_csv)
    study_rows = labels_df[labels_df["StudyInstanceUID"] == args.study_id]
    if study_rows.empty:
        sys.exit(f"Study {args.study_id} not found in labels CSV.")

    phase_map = {row["Label"]: row["SeriesInstanceUID"]
                 for _, row in study_rows.iterrows()}
    phases = [p for p in PHASE_ORDER if p in phase_map]

    # Load volumes and segs
    data = {}   # phase → (vol_arr, seg_arr)
    for phase in phases:
        sid  = phase_map[phase]
        print(f"Loading [{phase}] data...")
        print(f"  Searching for volume with suffix '{args.vol_suffix}'...")
        print(f"  Searching for seg with suffix '{args.seg_suffix}'...")
        print(f"  Study ID: {args.study_id}, Series ID: {sid}")
        vp   = find_file(args.vol_dir, args.study_id, sid, args.vol_suffix)
        print(f"Loading [{phase}] volume from {vp}")
        sp   = find_file(args.vol_dir, args.study_id, sid, args.seg_suffix)
        print(f"Loading [{phase}] seg from {sp}")
        if vp is None or sp is None:
            print(f"  [{phase}] vol or seg missing — skip")
            continue
        data[phase] = (load_arr(vp).astype(np.float32),
                       np.round(load_arr(sp)).astype(np.int32))
        print(f"  Loaded [{phase}]: vol={data[phase][0].shape}")

    if not data:
        sys.exit("No phase data loaded.")

    organ_ids = args.organs if args.organs else list(ORGAN_LABELS.keys())
    # Filter to organs present in at least one phase
    present = [lbl for lbl in organ_ids
               if any((seg == lbl).sum() >= MIN_VOXELS
                      for _, seg in data.values())]
    if not present:
        sys.exit("No organs found in any phase.")

    nrows = int(np.ceil(len(present) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(5 * NCOLS, 3.5 * nrows),
                             facecolor=BG)
    axes = np.array(axes).reshape(-1)

    for ax_idx, label in enumerate(present):
        ax = axes[ax_idx]
        ax.set_facecolor("white")
        organ_name = ORGAN_LABELS.get(label, f"Label {label}")
        found_any  = False

        for phase in phases:
            if phase not in data:
                continue
            vol, seg = data[phase]
            hu = organ_hu(vol, seg, label)
            if hu is None:
                continue
            hu = np.clip(hu, *HU_RANGE)
            ax.hist(hu, bins=N_BINS, range=HU_RANGE,
                    color=PHASE_COLORS.get(phase, "#999"),
                    alpha=0.55, density=True,
                    label=PHASE_SHORT.get(phase, phase))
            # Vertical median line
            ax.axvline(float(np.median(hu)),
                       color=PHASE_COLORS.get(phase, "#999"),
                       linewidth=1.5, linestyle="--", alpha=0.9)
            found_any = True

        ax.set_title(organ_name, fontsize=11, fontweight="bold",
                     color="#22373A")
        ax.set_xlabel("HU", fontsize=9, color="#444")
        ax.set_ylabel("Density", fontsize=9, color="#444")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=9, framealpha=0.7)

    # Hide unused axes
    for ax in axes[len(present):]:
        ax.set_visible(False)

    # Shared legend at top
    handles = [
        plt.Line2D([0], [0], color=PHASE_COLORS[p], linewidth=3, label=p)
        for p in phases if p in data
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        "HU Intensity Distributions per Organ across CT Phases\n"
        "(dashed lines = median — demonstrates why MMI/NCC are invalid across phases)",
        fontsize=12, fontweight="bold", color="#22373A", y=1.04)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
