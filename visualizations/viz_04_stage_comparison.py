"""
viz_04_stage_comparison.py
==========================
Visual 4: Evaluation metric comparison across pipeline stages.

Reads multiple evaluation CSV files (one per pipeline stage, produced by
evaluate_registration.py) and generates:

  (a) A grouped bar chart  — Eroded Dice per organ, grouped by stage
  (b) A grouped bar chart  — Centroid displacement (mm) per organ, by stage
  (c) A summary table PNG  — mean ± std for each metric × stage

CSV columns expected (from evaluate_registration.py):
    study_id, phase, seg_postfix, label, organ,
    eroded_dice, ncc, centroid_mm

Usage
-----
python viz_04_stage_comparison.py \
    --csvs \
        aligned:eval_aligned.csv \
        rigid:eval_rigid2.csv \
        deformable:eval_deformable.csv \
    --out_dir figures/

# The --csvs argument takes "LABEL:PATH" pairs (colon-separated).
# LABEL becomes the stage name in the legend.

Options
-------
--phases    STR [STR ...]  phases to include (default: Arterial Venous)
--organs    STR [STR ...]  organs to include (default: all)
--metric_dice     eroded_dice   (column name)
--metric_centroid centroid_mm   (column name)

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
import matplotlib.ticker as ticker

BG          = "#F9F9F9"
FIG_DPI     = 200
STAGE_COLORS = [
    "#1B786F",   # teal (matches DML slides)
    "#EB801A",   # orange
    "#3A6EA8",   # blue
    "#7D3C98",   # purple
    "#C0392B",   # red
    "#2E7D32",   # dark green
]

ORGAN_ORDER = [
    "liver", "spleen", "kidney_left", "kidney_right",
    "aorta", "inferior_vena_cava",
    "pancreas", "gallbladder",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
]
ORGAN_SHORT = {
    "liver":              "Liver",
    "spleen":             "Spleen",
    "kidney_left":        "Kid. L",
    "kidney_right":       "Kid. R",
    "aorta":              "Aorta",
    "inferior_vena_cava": "IVC",
    "pancreas":           "Pancr.",
    "gallbladder":        "G.Bladd",
    "vertebrae_L1":       "L1",
    "vertebrae_L2":       "L2",
    "vertebrae_L3":       "L3",
}


def load_csvs(csv_specs):
    """Parse 'LABEL:PATH' pairs and return dict {label: DataFrame}."""
    stages = {}
    for spec in csv_specs:
        if ":" not in spec:
            sys.exit(f"Bad --csvs format: '{spec}'. Use LABEL:PATH.")
        label, path = spec.split(":", 1)
        if not Path(path).exists():
            print(f"  WARNING: {path} not found — skipping stage '{label}'")
            continue
        df = pd.read_csv(path)
        df["_stage"] = label
        stages[label] = df
        print(f"  Loaded '{label}': {len(df)} rows from {path}")
    return stages


def agg(df, metric, phases):
    """Group by organ, return mean and sem."""
    sub = df[df["phase"].isin(phases)] if phases else df
    g   = sub.groupby("organ")[metric].agg(["mean", "sem"]).reset_index()
    return g


def pick_organs(stages, phases):
    """Return sorted organ list present in all stages."""
    sets = []
    for df in stages.values():
        sub = df[df["phase"].isin(phases)] if phases else df
        sets.append(set(sub["organ"].dropna().unique()))
    common = set.intersection(*sets) if sets else set()
    return [o for o in ORGAN_ORDER if o in common] + \
           sorted(common - set(ORGAN_ORDER))


def bar_chart(stages, organ_list, metric, ylabel, title, out_path,
              phases, ylim=None):
    stage_names = list(stages.keys())
    x  = np.arange(len(organ_list))
    w  = 0.8 / len(stage_names)

    fig, ax = plt.subplots(figsize=(max(10, len(organ_list) * 0.9 + 2), 5),
                           facecolor=BG)
    ax.set_facecolor("white")

    for i, (stage_name, df) in enumerate(stages.items()):
        g      = agg(df, metric, phases)
        g_map  = g.set_index("organ")
        means  = [g_map.loc[o, "mean"] if o in g_map.index else np.nan
                  for o in organ_list]
        sems   = [g_map.loc[o, "sem"]  if o in g_map.index else 0.0
                  for o in organ_list]
        offset = (i - len(stage_names) / 2 + 0.5) * w
        bars = ax.bar(x + offset, means, width=w * 0.92,
                      color=STAGE_COLORS[i % len(STAGE_COLORS)],
                      label=stage_name, alpha=0.88, zorder=3)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="black", capsize=3,
                    linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([ORGAN_SHORT.get(o, o) for o in organ_list],
                       rotation=35, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color="#22373A", pad=10)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    if ylim:
        ax.set_ylim(ylim)

    phase_str = " + ".join(phases) if phases else "all phases"
    ax.text(0.99, 0.98, f"Phases: {phase_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#666")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def summary_table(stages, organ_list, metrics, phases, out_path):
    """Render a clean summary table as PNG."""
    rows = []
    for metric_col, metric_label in metrics:
        for stage_name, df in stages.items():
            sub = df[df["phase"].isin(phases)] if phases else df
            for organ in organ_list:
                org_df = sub[sub["organ"] == organ][metric_col].dropna()
                if len(org_df) == 0:
                    continue
                rows.append({
                    "Metric":  metric_label,
                    "Stage":   stage_name,
                    "Organ":   ORGAN_SHORT.get(organ, organ),
                    "Mean":    f"{org_df.mean():.3f}",
                    "Std":     f"{org_df.std():.3f}",
                    "N":       len(org_df),
                })
    if not rows:
        print("  No data for summary table — skipping")
        return

    tdf = pd.DataFrame(rows)
    # Pivot: rows = Organ × Metric, cols = Stage
    pivot = tdf.pivot_table(
        index=["Metric", "Organ"], columns="Stage",
        values="Mean", aggfunc="first"
    ).reset_index()

    n_rows, n_cols = pivot.shape
    fig_h = max(4, 0.35 * n_rows + 1.5)
    fig_w = max(8, 1.8 * n_cols + 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.axis("off")

    col_labels  = list(pivot.columns)
    cell_text   = pivot.values.tolist()
    cell_colors = []

    # Colour gradient per data cell: green=good, red=bad (per metric)
    for r, row_vals in enumerate(cell_text):
        metric_name = str(row_vals[0])
        is_dice     = "Dice" in metric_name or "dice" in metric_name
        row_colors  = ["#EEEEEE", "#EEEEEE"]   # first two cols (Metric, Organ)
        for v in row_vals[2:]:
            try:
                fv = float(v)
                # dice higher = greener; centroid lower = greener
                ratio = fv if is_dice else (1 - min(fv / 20.0, 1.0))
                g = int(180 + ratio * 60)
                r2 = int(255 - ratio * 80)
                b = int(180 - ratio * 40)
                row_colors.append(f"#{r2:02X}{g:02X}{b:02X}")
            except (ValueError, TypeError):
                row_colors.append("#FFFFFF")
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#22373A")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Registration Quality Summary — Mean Metrics per Organ × Stage\n"
        f"(Phases: {', '.join(phases) if phases else 'all'})",
        fontsize=11, fontweight="bold", color="#22373A", pad=12)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Figure 4: pipeline stage comparison bar charts + table")
    ap.add_argument("--csvs", nargs="+", required=True,
                    metavar="LABEL:PATH",
                    help="e.g. aligned:eval_aligned.csv rigid:eval_rigid.csv")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--phases", nargs="+",
                    default=["Arterial", "Venous"],
                    help="Phases to include in aggregation")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Organ names to include (default: auto from data)")
    args = ap.parse_args()

    stages = load_csvs(args.csvs)
    if not stages:
        sys.exit("No valid CSV files loaded.")

    phases     = args.phases
    organ_list = args.organs if args.organs else pick_organs(stages, phases)
    print(f"Organs: {organ_list}")
    print(f"Phases: {phases}")

    out_dir = Path(args.out_dir)

    # ── (a) Eroded Dice bar chart ──────────────────────────────────────────
    bar_chart(
        stages, organ_list,
        metric="eroded_dice",
        ylabel="Eroded Dice (higher = better)",
        title="Eroded Dice Coefficient per Organ × Pipeline Stage",
        out_path=out_dir / "fig4a_dice_by_stage.png",
        phases=phases,
        ylim=(0, 1.05),
    )

    # ── (b) Centroid displacement bar chart ────────────────────────────────
    bar_chart(
        stages, organ_list,
        metric="centroid_mm",
        ylabel="Centroid Displacement (mm) — lower = better",
        title="Centroid Displacement per Organ × Pipeline Stage",
        out_path=out_dir / "fig4b_centroid_by_stage.png",
        phases=phases,
    )

    # ── (c) Summary table ──────────────────────────────────────────────────
    summary_table(
        stages, organ_list,
        metrics=[("eroded_dice", "Eroded Dice"),
                 ("centroid_mm", "Centroid (mm)")],
        phases=phases,
        out_path=out_dir / "fig4c_summary_table.png",
    )


if __name__ == "__main__":
    main()
