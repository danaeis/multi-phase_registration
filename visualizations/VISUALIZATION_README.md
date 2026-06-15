# Registration Pipeline — Visualization Scripts

Five scripts that produce all evaluation figures for the presentation.
All outputs use the DML lab colour scheme (F9F9F9 background, 22373A text, EB801A accent).

---

## Setup

```bash
pip install SimpleITK nibabel matplotlib numpy pandas scipy
```

Set your paths once at the top, then copy-paste the commands below:

```bash
# ── Edit these ────────────────────────────────────────────────────────────
STUDY_ID="1.2.840.113619.2.359.3.2831208971.108.1589585466.773"  # one example study
LABELS_CSV="/path/to/labels.csv"

ALIGNED_DIR="/path/to/aligned_volumes/${STUDY_ID}"
RIGID_DIR="/path/to/rigid_registered/${STUDY_ID}"
DEFORMABLE_DIR="/path/to/deformable_registered/${STUDY_ID}"

EVAL_ALIGNED="/path/to/eval_aligned.csv"
EVAL_RIGID="/path/to/eval_rigid2.csv"
EVAL_DEFORMABLE="/path/to/eval_deformable.csv"

FIGURES_DIR="./figures"
mkdir -p "$FIGURES_DIR"
```

---

## Figure 1 — Phase Alignment Before / After

**What it shows:** 2 rows × 3 columns.
Row 1 = same absolute Z-slice from NC / Arterial / Venous *before* registration
(you can see different anatomy per panel — the problem).
Row 2 = same Z-slice *after* registration (anatomy matches — the solution).

```bash
python viz_01_phase_alignment.py \
    --before_dir      "$ALIGNED_DIR" \
    --after_dir       "$RIGID_DIR" \
    --study_id        "$STUDY_ID" \
    --labels_csv      "$LABELS_CSV" \
    --before_vol_suffix "_aligned.nii.gz" \
    --after_vol_suffix  "_rigid2.nii.gz" \
    --before_seg_suffix "_aligned_seg_reg.nii.gz" \
    --after_seg_suffix  "_rigid2_seg_reg.nii.gz" \
    --out             "${FIGURES_DIR}/fig1_phase_alignment.png"
```

**Tips:**
- Pick a study where the misalignment is visually obvious (large Z-offset).
- The script auto-selects the liver centroid slice. Add `--z_offset 10` to
  move to a different anatomical level (e.g., kidney level).
- For the "before" row you can also point `--before_dir` at the raw
  standardized NIfTIs (before any alignment) for maximum effect.

**Output:** `figures/fig1_phase_alignment.png` — drop straight into slide 4.

---

## Figure 2 — HU Intensity Histograms

**What it shows:** One subplot per organ. NC / Arterial / Venous HU
distributions overlaid with dashed median lines. Visually proves why MMI
and NCC are invalid across phases (aorta: ~40 HU NC → ~300 HU arterial).

```bash
python viz_02_hu_histograms.py \
    --vol_dir    "$ALIGNED_DIR" \
    --study_id   "$STUDY_ID" \
    --labels_csv "$LABELS_CSV" \
    --vol_suffix "_aligned.nii.gz" \
    --seg_suffix "_aligned_seg_reg.nii.gz" \
    --out        "${FIGURES_DIR}/fig2_hu_histograms.png"
```

**Tips:**
- Default shows 10 organs. Override with `--organs 1 2 3 4 13 14` (label ints)
  to show only the most dramatic ones for the slide (liver, kidneys, aorta, IVC).
- Works on *aligned* volumes — no registration needed.

**Output:** `figures/fig2_hu_histograms.png` — use on the "Why MMI/NCC fail" slide.

---

## Figure 3 — Segmentation Overlay

**What it shows:** Three axial CT slices (NC / Arterial / Venous) with the
seg_reg mask overlaid in per-organ colours. Makes the dual-mask strategy
concrete and demonstrates TotalSegmentator quality.

```bash
python viz_03_seg_overlay.py \
    --vol_dir    "$ALIGNED_DIR" \
    --study_id   "$STUDY_ID" \
    --labels_csv "$LABELS_CSV" \
    --vol_suffix "_aligned.nii.gz" \
    --seg_suffix "_aligned_seg_reg.nii.gz" \
    --out        "${FIGURES_DIR}/fig3_seg_overlay.png" \
    --alpha      0.45
```

**Tips:**
- `--alpha 0.3` for more CT visibility; `--alpha 0.6` for bolder masks.
- `--z_offset 15` shifts to a more superior slice (shows vertebrae + aorta
  better); `--z_offset -10` to go inferior (shows iliopsoas + hips).
- Try on the NC phase first — clearest anatomy for the background CT.

**Output:** `figures/fig3_seg_overlay.png` — use on the "Segmentation Architecture" slide.

---

## Figure 4 — Pipeline Stage Comparison (requires eval CSVs)

**What it shows:**
- `fig4a_dice_by_stage.png`    — grouped bar chart: Eroded Dice per organ, by stage
- `fig4b_centroid_by_stage.png` — grouped bar chart: centroid mm per organ, by stage
- `fig4c_summary_table.png`   — colour-coded mean ± std summary table

```bash
python viz_04_stage_comparison.py \
    --csvs \
        "Z-Aligned:${EVAL_ALIGNED}" \
        "Rigid:${EVAL_RIGID}" \
        "Deformable:${EVAL_DEFORMABLE}" \
    --out_dir  "$FIGURES_DIR" \
    --phases   "Arterial" "Venous"
```

**Tips:**
- The `LABEL:PATH` pairs become the legend labels — choose clear names.
- `--phases Arterial` to show only arterial phase if venous data is incomplete.
- `--organs liver spleen kidney_left kidney_right aorta vertebrae_L1` to
  restrict to the 6 most important organs for a cleaner slide figure.
- Add more stages by adding more `LABEL:PATH` pairs to `--csvs`.

**Output:** three PNGs in `figures/` — use fig4a + fig4c on the evaluation slide.

---

## Figure 5 — Centroid Displacement Distribution (requires eval CSVs)

**What it shows:**
- `fig5a_centroid_violin.png`    — violin + strip plot per organ, one violin per stage.
  Clinical threshold lines at 5mm / 10mm / 15mm.
- `fig5b_centroid_waterfall.png` — per-study before→after arrows sorted by 'before'
  displacement. Shows which studies improved and which regressed.
- `fig5c_centroid_heatmap.png`   — organ × stage heatmap of mean centroid displacement.
  Green = better, Red = worse.

```bash
python viz_05_centroid_distribution.py \
    --csvs \
        "Z-Aligned:${EVAL_ALIGNED}" \
        "Rigid:${EVAL_RIGID}" \
        "Deformable:${EVAL_DEFORMABLE}" \
    --out_dir          "$FIGURES_DIR" \
    --phases           "Arterial" "Venous" \
    --waterfall_stages "Z-Aligned" "Deformable" \
    --waterfall_organ  "liver"
```

**Tips:**
- Change `--waterfall_organ` to `aorta` or `spleen` for different stories.
- The waterfall is the most convincing single-slide figure — shows N studies
  as individual data points, not just aggregate means.
- The heatmap (fig5c) is ideal as a compact slide summary table replacement.

**Output:** three PNGs in `figures/` — use fig5a + fig5b on the evaluation slide.

---

## Batch run (all 5)

```bash
python viz_01_phase_alignment.py \
    --before_dir "$ALIGNED_DIR" --after_dir "$RIGID_DIR" \
    --study_id "$STUDY_ID" --labels_csv "$LABELS_CSV" \
    --out "${FIGURES_DIR}/fig1_phase_alignment.png"

python viz_02_hu_histograms.py \
    --vol_dir "$ALIGNED_DIR" --study_id "$STUDY_ID" \
    --labels_csv "$LABELS_CSV" \
    --out "${FIGURES_DIR}/fig2_hu_histograms.png"

python viz_03_seg_overlay.py \
    --vol_dir "$ALIGNED_DIR" --study_id "$STUDY_ID" \
    --labels_csv "$LABELS_CSV" \
    --out "${FIGURES_DIR}/fig3_seg_overlay.png"

python viz_04_stage_comparison.py \
    --csvs "Z-Aligned:${EVAL_ALIGNED}" "Rigid:${EVAL_RIGID}" "Deformable:${EVAL_DEFORMABLE}" \
    --out_dir "$FIGURES_DIR"

python viz_05_centroid_distribution.py \
    --csvs "Z-Aligned:${EVAL_ALIGNED}" "Rigid:${EVAL_RIGID}" "Deformable:${EVAL_DEFORMABLE}" \
    --out_dir "$FIGURES_DIR" \
    --waterfall_stages "Z-Aligned" "Deformable"
```

---

## Outputs summary

| File | Slide to use on |
|------|----------------|
| `fig1_phase_alignment.png` | Motivation / Problem statement |
| `fig2_hu_histograms.png` | Why segmentation-based loss |
| `fig3_seg_overlay.png` | Segmentation mask architecture |
| `fig4a_dice_by_stage.png` | Evaluation metrics (Dice) |
| `fig4b_centroid_by_stage.png` | Evaluation metrics (centroid) |
| `fig4c_summary_table.png` | Summary / results table |
| `fig5a_centroid_violin.png` | Evaluation — distribution |
| `fig5b_centroid_waterfall.png` | Evaluation — per-study improvement |
| `fig5c_centroid_heatmap.png` | Compact summary heatmap |
