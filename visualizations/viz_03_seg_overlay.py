"""
viz_03_segmentation_overlay.py
==============================
Visual 3: Segmentation mask overlay on CT axial slices.

Produces a 1×3 figure (NC / Arterial / Venous) showing an axial CT slice
with the seg_reg mask overlaid in translucent per-organ colours.  Makes the
dual-mask strategy concrete and shows TotalSegmentator quality.

Output: one figure with three annotated CT slices + a shared colour legend.

Usage
-----
python viz_03_segmentation_overlay.py \
    --vol_dir    /path/to/aligned_volumes/STUDY_ID \
    --study_id   1.2.840.113619... \
    --labels_csv /path/to/labels.csv \
    --vol_suffix  _aligned.nii.gz \
    --seg_suffix  _aligned_seg_reg.nii.gz \
    --out         figures/fig3_seg_overlay.png

Options
-------
--z_offset   INT   manual Z-slice offset from liver centroid (default 0)
--alpha      FLOAT overlay transparency 0–1 (default 0.45)

Dependencies
------------
pip install nibabel matplotlib numpy SimpleITK pandas
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

try:
    import SimpleITK as sitk
    def load_arr(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
except ImportError:
    import nibabel as nib
    def load_arr(path):
        img = nib.load(str(path))
        return np.asarray(img.dataobj).transpose(2, 1, 0)


# ── Organ colours (RGBA tuples 0-1) ─────────────────────────────────────────
ORGAN_PALETTE = {
    1:  ("#4FC3F7", "Liver"),           # light blue
    2:  ("#EF5350", "Spleen"),          # red
    3:  ("#AB47BC", "Kidney L"),        # purple
    4:  ("#CE93D8", "Kidney R"),        # light purple
    5:  ("#FFB74D", "Pancreas"),        # orange
    6:  ("#A5D6A7", "Gallbladder"),     # light green
    7:  ("#80CBC4", "Autochthon L"),    # teal
    8:  ("#4DB6AC", "Autochthon R"),    # teal dark
    9:  ("#F48FB1", "Iliopsoas L"),     # pink
    10: ("#F06292", "Iliopsoas R"),     # dark pink
    13: ("#FF8A65", "Aorta"),           # salmon
    14: ("#FFCC02", "IVC"),             # yellow
    21: ("#B0BEC5", "Vertebra L1"),     # grey-blue
    22: ("#90A4AE", "Vertebra L2"),
    23: ("#78909C", "Vertebra L3"),
    40: ("#80DEEA", "Spinal Cord"),     # cyan
    38: ("#CFD8DC", "Sacrum"),
    41: ("#ECEFF1", "Hip L"),
    42: ("#F5F5F5", "Hip R"),
}

PHASE_COLORS = {
    "Non-contrast": "#3A6EA8",
    "Arterial":     "#C0392B",
    "Venous":       "#7D3C98",
}
PHASE_ORDER = ["Non-contrast", "Arterial", "Venous"]

HU_WINDOW = (-200, 300)
FIG_DPI   = 200
BG        = "#F9F9F9"


def find_file(directory, study_id, series_id, suffix):
    for p in [
        Path(directory) / f"{study_id}_{series_id}{suffix}",
        Path(directory) / f"{series_id}{suffix}",
    ]:
        if p.exists():
            return p
    return None


def liver_z(seg: np.ndarray) -> int:
    mask = seg == 1
    if mask.sum() > 200:
        zz = np.where(mask.any(axis=(1, 2)))[0]
        return int(zz.mean())
    return seg.shape[0] // 2


def window_hu(arr, wmin=-200, wmax=300):
    return np.clip((arr - wmin) / (wmax - wmin), 0, 1)


def overlay_seg(vol_slice: np.ndarray, seg_slice: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """Blend seg colours onto grayscale CT slice. Returns (H,W,4) RGBA."""
    gray = window_hu(vol_slice)
    rgb  = np.stack([gray, gray, gray], axis=-1)   # (H,W,3)

    out  = np.concatenate([rgb, np.ones((*gray.shape, 1))], axis=-1)

    for label, (hex_col, _) in ORGAN_PALETTE.items():
        mask = seg_slice == label
        if not mask.any():
            continue
        c = matplotlib.colors.to_rgb(hex_col)
        for ch, cv in enumerate(c):
            out[mask, ch] = (1 - alpha) * out[mask, ch] + alpha * cv
        out[mask, 3] = 1.0

    return out


def make_figure(data, ref_z, out_path, z_offset=0, alpha=0.45):
    phases = [p for p in PHASE_ORDER if p in data]
    n      = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), facecolor=BG)
    if n == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        vol, seg = data[phase]
        z = min(max(ref_z + z_offset, 0), vol.shape[0] - 1)
        composite = overlay_seg(vol[z], seg[z], alpha=alpha)

        ax.imshow(composite, aspect="equal", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])

        pc = PHASE_COLORS.get(phase, "#555")
        ax.set_title(phase, color=pc, fontsize=13, fontweight="bold", pad=6)

        for spine in ax.spines.values():
            spine.set_edgecolor(pc)
            spine.set_linewidth(2)

        ax.text(0.02, 0.02, f"Z = {z}", color="white", fontsize=8,
                transform=ax.transAxes, va="bottom",
                bbox=dict(facecolor="black", alpha=0.5, pad=2))

    # Shared organ legend (only organs visible in at least one slice)
    visible_labels = set()
    for vol, seg in data.values():
        z = min(max(ref_z + z_offset, 0), vol.shape[0] - 1)
        for lbl in np.unique(seg[z]):
            if lbl > 0 and lbl in ORGAN_PALETTE:
                visible_labels.add(lbl)

    legend_patches = [
        mpatches.Patch(color=ORGAN_PALETTE[lbl][0],
                       label=ORGAN_PALETTE[lbl][1])
        for lbl in sorted(visible_labels)
        if lbl in ORGAN_PALETTE
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=min(len(legend_patches), 8),
               fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Segmentation Overlay — Aligned Axial Slices (seg_reg mask)",
        fontsize=13, fontweight="bold", color="#22373A", y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Figure 3: segmentation overlay on CT slices")
    ap.add_argument("--vol_dir",    required=True)
    ap.add_argument("--study_id",   required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--vol_suffix", default="_aligned.nii.gz")
    ap.add_argument("--seg_suffix", default="_aligned_seg_reg.nii.gz")
    ap.add_argument("--out",        default="figures/fig3_seg_overlay.png")
    ap.add_argument("--z_offset",   type=int,   default=0,
                    help="Z-slice offset from liver centroid")
    ap.add_argument("--alpha",      type=float, default=0.45,
                    help="Overlay alpha (0=transparent, 1=opaque)")
    args = ap.parse_args()

    labels_df  = pd.read_csv(args.labels_csv)
    study_rows = labels_df[labels_df["StudyInstanceUID"] == args.study_id]
    if study_rows.empty:
        sys.exit(f"Study {args.study_id} not found in labels CSV.")

    phase_map = {row["Label"]: row["SeriesInstanceUID"]
                 for _, row in study_rows.iterrows()}

    data = {}
    for phase in PHASE_ORDER:
        if phase not in phase_map:
            continue
        sid = phase_map[phase]
        vp  = find_file(args.vol_dir, args.study_id, sid, args.vol_suffix)
        sp  = find_file(args.vol_dir, args.study_id, sid, args.seg_suffix)
        if vp is None or sp is None:
            print(f"  [{phase}] vol or seg missing — skip")
            continue
        data[phase] = (load_arr(vp).astype(np.float32),
                       np.round(load_arr(sp)).astype(np.int32))
        print(f"  Loaded [{phase}]: {data[phase][0].shape}")

    if not data:
        sys.exit("No phase data loaded.")

    # Determine reference Z from NC (or first available phase)
    ref_phase = "Non-contrast" if "Non-contrast" in data else list(data)[0]
    _, ref_seg = data[ref_phase]
    ref_z = liver_z(ref_seg)
    print(f"Reference Z slice: {ref_z} (liver centroid)")

    make_figure(data, ref_z, args.out, args.z_offset, args.alpha)


if __name__ == "__main__":
    main()
