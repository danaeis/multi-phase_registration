"""
viz_05_centroid_distribution.py
================================
Visual 5: Centroid displacement distribution across pipeline stages.

Produces two complementary plots:

  (a) Violin + strip plot — per-study centroid displacement distribution,
      one violin per pipeline stage, one panel per organ.  Shows spread
      and outliers.

  (b) Improvement waterfall — paired per-study centroid displacement at
      two stages (e.g. aligned vs deformable), sorted by 'before' value.
      Arrows show the per-study improvement.  Immediately reveals which
      studies benefited most and which regressed.

CSV columns expected (from evaluate_registration.py):
    study_id, phase, organ, eroded_dice, centroid_mm

Usage
-----
python viz_05_centroid_distribution.py \
    --csvs \
        aligned:eval_aligned.csv \
        rigid:eval_rigid2.csv \
        deformable:eval_deformable.csv \
    --out_dir figures/

Options
-------
--phases     STR [STR ...]   phases to include (default: Arterial Venous)
--organs     STR [STR ...]   organs for violin plot (default: top 6)
--waterfall_stages  STR STR  two stage names for the waterfall (default: first and last)
--waterfall_organ   STR      organ for the waterfall (default: liver)

Dependencies
------------
pip install pandas matplotlib numpy scipy
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

BG          = "#F9F9F9"
FIG_DPI     = 200

STAGE_COLORS = [
    "#1B786F",
    "#EB801A",
    "#3A6EA8",
    "#7D3C98",
    "#C0392B",
]

ORGAN_SHORT = {
    "liver":              "Liver",
    "spleen":             "Spleen",
    "kidney_left":        "Kidney L",
    "kidney_right":       "Kidney R",
    "aorta":              "Aorta",
    "inferior_vena_cava": "IVC",
    "pancreas":           "Pancreas",
    "gallbladder":        "Gallbladder",
    "vertebrae_L1":       "L1",
    "vertebrae_L2":       "L2",
    "vertebrae_L3":       "L3",
}

DEFAULT_ORGANS = [
    "liver", "spleen", "kidney_left", "kidney_right",
    "aorta", "vertebrae_L1",
]


def load_csvs(csv_specs):
    stages = {}
    for spec in csv_specs:
        if ":" not in spec:
            sys.exit(f"Bad --csvs format: '{spec}'. Use LABEL:PATH.")
        label, path = spec.split(":", 1)
        if not Path(path).exists():
            print(f"  WARNING: {path} not found — skipping '{label}'")
            continue
        df = pd.read_csv(path)
        df["_stage"] = label
        stages[label] = df
        print(f"  Loaded '{label}': {len(df)} rows")
    return stages


def filter_phase(df, phases):
    return df[df["phase"].isin(phases)] if phases else df


# ── (a) Violin + strip chart ─────────────────────────────────────────────────
def violin_plot(stages, organ_list, phases, out_path):
    stage_names = list(stages.keys())
    n_organs    = len(organ_list)

    fig, axes = plt.subplots(
        1, n_organs,
        figsize=(3.5 * n_organs, 5),
        facecolor=BG, sharey=False
    )
    if n_organs == 1:
        axes = [axes]

    for ax, organ in zip(axes, organ_list):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plot_data = []
        positions = []
        colors    = []

        for i, (stage_name, df) in enumerate(stages.items()):
            sub = filter_phase(df, phases)
            vals = sub[sub["organ"] == organ]["centroid_mm"].dropna().values
            if len(vals) < 3:
                plot_data.append([np.nan])
            else:
                plot_data.append(vals)
            positions.append(i + 1)
            colors.append(STAGE_COLORS[i % len(STAGE_COLORS)])

        # Violin
        parts = ax.violinplot(
            [v for v in plot_data if not (len(v) == 1 and np.isnan(v[0]))],
            positions=[p for p, v in zip(positions, plot_data)
                       if not (len(v) == 1 and np.isnan(v[0]))],
            showmedians=True, showextrema=False, widths=0.7
        )
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.55)
            pc.set_edgecolor(colors[j])
        parts["cmedians"].set_color("#22373A")
        parts["cmedians"].set_linewidth(2)

        # Strip (jitter)
        rng = np.random.default_rng(42)
        for i, (vals, pos, col) in enumerate(
                zip(plot_data, positions, colors)):
            if len(vals) == 1 and np.isnan(vals[0]):
                continue
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color=col, alpha=0.4, s=18, zorder=3)

        # Clinical thresholds
        ax.axhline(5,  color="#1A7A4A", linewidth=1, linestyle="--",
                   alpha=0.7, label="5mm (excellent)")
        ax.axhline(10, color="#EB801A", linewidth=1, linestyle="--",
                   alpha=0.7, label="10mm (acceptable)")
        ax.axhline(15, color="#C0392B", linewidth=1, linestyle="--",
                   alpha=0.7, label="15mm (failure)")

        ax.set_xticks(range(1, len(stage_names) + 1))
        ax.set_xticklabels(stage_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(ORGAN_SHORT.get(organ, organ),
                     fontsize=11, fontweight="bold", color="#22373A")
        ax.set_ylabel("Centroid displacement (mm)" if ax == axes[0] else "",
                       fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=STAGE_COLORS[i], label=s, alpha=0.7)
        for i, s in enumerate(stage_names)
    ] + [
        plt.Line2D([0], [0], color="#1A7A4A", linestyle="--", label="5mm"),
        plt.Line2D([0], [0], color="#EB801A", linestyle="--", label="10mm"),
        plt.Line2D([0], [0], color="#C0392B", linestyle="--", label="15mm"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    phase_str = " + ".join(phases) if phases else "all phases"
    fig.suptitle(
        f"Centroid Displacement Distribution per Stage  (phases: {phase_str})",
        fontsize=13, fontweight="bold", color="#22373A", y=1.01)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── (b) Improvement waterfall ────────────────────────────────────────────────
def waterfall_plot(stages, stage_a, stage_b, organ, phases, out_path):
    if stage_a not in stages or stage_b not in stages:
        print(f"  Waterfall: stages '{stage_a}' or '{stage_b}' not found — skip")
        return

    def get_per_study(stage_name):
        df  = stages[stage_name]
        sub = filter_phase(df, phases)
        sub = sub[sub["organ"] == organ][["study_id", "centroid_mm"]]
        return sub.groupby("study_id")["centroid_mm"].mean()

    vals_a = get_per_study(stage_a)
    vals_b = get_per_study(stage_b)
    common = vals_a.index.intersection(vals_b.index)

    if len(common) == 0:
        print(f"  Waterfall: no common studies for '{organ}' — skip")
        return

    va = vals_a.loc[common].values
    vb = vals_b.loc[common].values
    delta = va - vb   # positive = improved

    sort_idx = np.argsort(va)[::-1]   # sort by 'before' descending
    va, vb, delta = va[sort_idx], vb[sort_idx], delta[sort_idx]

    x = np.arange(len(common))

    fig, ax = plt.subplots(figsize=(max(12, len(common) * 0.4 + 2), 5),
                           facecolor=BG)
    ax.set_facecolor("white")

    # Before bars (light)
    ax.bar(x, va, width=0.6, color=STAGE_COLORS[0], alpha=0.35,
           label=f"Before ({stage_a})", zorder=2)
    # After bars (solid)
    ax.bar(x, vb, width=0.6, color=STAGE_COLORS[1], alpha=0.85,
           label=f"After ({stage_b})", zorder=3)

    # Arrow: improvement direction
    for xi, (a, b) in enumerate(zip(va, vb)):
        col = "#1A7A4A" if b < a else "#C0392B"
        ax.annotate("", xy=(xi, b + 0.2), xytext=(xi, a + 0.2),
                    arrowprops=dict(arrowstyle="->", color=col,
                                   lw=1.2, mutation_scale=10))

    # Thresholds
    ax.axhline(5,  color="#1A7A4A", linewidth=1.2, linestyle="--", alpha=0.8,
               label="5mm excellent")
    ax.axhline(10, color="#EB801A", linewidth=1.2, linestyle="--", alpha=0.8,
               label="10mm acceptable")
    ax.axhline(15, color="#C0392B", linewidth=1.2, linestyle="--", alpha=0.8,
               label="15mm failure")

    improved = (delta > 0).sum()
    regressed = (delta < 0).sum()
    ax.set_title(
        f"Per-Study Centroid Displacement: {stage_a} → {stage_b}  |  "
        f"Organ: {ORGAN_SHORT.get(organ, organ)}\n"
        f"Improved: {improved}/{len(common)} studies   "
        f"Regressed: {regressed}/{len(common)} studies   "
        f"Mean Δ: {delta.mean():+.1f} mm",
        fontsize=11, fontweight="bold", color="#22373A", pad=8)

    ax.set_xlabel("Study (sorted by 'before' displacement)", fontsize=10)
    ax.set_ylabel("Centroid displacement (mm)", fontsize=10)
    ax.set_xticks([])
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── (c) Overall improvement heatmap ─────────────────────────────────────────
def improvement_heatmap(stages, baseline_stage, phases, out_path):
    """Heatmap: rows=organs, cols=stages, value=mean centroid_mm."""
    stage_names = list(stages.keys())
    organs = DEFAULT_ORGANS

    mat = np.full((len(organs), len(stage_names)), np.nan)
    for j, (stage_name, df) in enumerate(stages.items()):
        sub = filter_phase(df, phases)
        for i, organ in enumerate(organs):
            vals = sub[sub["organ"] == organ]["centroid_mm"].dropna().values
            if len(vals) > 0:
                mat[i, j] = vals.mean()

    fig, ax = plt.subplots(
        figsize=(max(5, len(stage_names) * 1.5 + 1.5), max(4, len(organs) * 0.55 + 1)),
        facecolor=BG)
    ax.set_facecolor("white")

    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=np.nanpercentile(mat, 95))

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(organs)))
    ax.set_yticklabels([ORGAN_SHORT.get(o, o) for o in organs], fontsize=10)

    # Annotate cells
    for i in range(len(organs)):
        for j in range(len(stage_names)):
            v = mat[i, j]
            txt = f"{v:.1f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if (not np.isnan(v) and v > np.nanpercentile(mat, 60)) else "#22373A")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean centroid displacement (mm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Mean Centroid Displacement (mm) — Organ × Stage\n"
        f"(Green = better alignment, Red = worse | phases: "
        f"{', '.join(phases) if phases else 'all'})",
        fontsize=11, fontweight="bold", color="#22373A", pad=10)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Figure 5: centroid displacement distributions")
    ap.add_argument("--csvs", nargs="+", required=True,
                    metavar="LABEL:PATH")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--phases", nargs="+",
                    default=["Arterial", "Venous"])
    ap.add_argument("--organs", nargs="*", default=None)
    ap.add_argument("--waterfall_stages", nargs=2, default=None,
                    metavar=("BEFORE", "AFTER"),
                    help="Two stage names for the waterfall plot")
    ap.add_argument("--waterfall_organ", default="liver")
    args = ap.parse_args()

    stages = load_csvs(args.csvs)
    if not stages:
        sys.exit("No valid CSV files loaded.")

    stage_names = list(stages.keys())
    phases      = args.phases
    organ_list  = args.organs if args.organs else DEFAULT_ORGANS
    out_dir     = Path(args.out_dir)

    # (a) Violin
    violin_plot(
        stages, organ_list, phases,
        out_path=out_dir / "fig5a_centroid_violin.png"
    )

    # (b) Waterfall
    wa_stages = args.waterfall_stages or (
        [stage_names[0], stage_names[-1]] if len(stage_names) >= 2 else None
    )
    if wa_stages:
        waterfall_plot(
            stages,
            stage_a=wa_stages[0],
            stage_b=wa_stages[1],
            organ=args.waterfall_organ,
            phases=phases,
            out_path=out_dir / "fig5b_centroid_waterfall.png"
        )

    # (c) Heatmap
    improvement_heatmap(
        stages,
        baseline_stage=stage_names[0],
        phases=phases,
        out_path=out_dir / "fig5c_centroid_heatmap.png"
    )


if __name__ == "__main__":
    main()
