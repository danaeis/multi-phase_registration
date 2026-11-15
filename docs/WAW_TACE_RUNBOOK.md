# WAW-TACE server runbook

Copy-paste commands for the remote server. The reasoning behind the choices is in
[WAW_TACE_PLAN.md](WAW_TACE_PLAN.md); this file is just the sequence.

Written 2026-11-15. Nothing here has been executed — the laptop has 23 GB free and
the download is 48 GB, so every step below is server-side.

**Paths I could not verify.** `$REPO` is guessed from the `export DEEDS_BIN=...` comment
at the bottom of `compare_registrtaion/_common.py`; `$DATA` is a placeholder. Fix both in
§1 before running anything. Likewise I could not check which Python packages or GPUs the
server has — §2 verifies that.

---

## 0. What ships with this runbook

Four new/changed things. They are written and syntax-checked but **not yet
committed or pushed** — do that from the laptop before §2's `git pull`, or `scp`
them across:

| file | what it is |
|---|---|
| `waw_tace/prepare_waw_tace.py` | **new** — converts WAW-TACE into a VinDr-shaped root |
| `waw_tace/check_tumor_masks.py` | **new** — the 20 MB gate check (§3) |
| `compare_registrtaion/run_waw_tace.sh` | **new** — registration grid, cut condition list |
| `{dcm2nii,alignment,registration,compare_registrtaion}/configs.py` | **changed** — `MAIN_PATH` now `os.environ.get("MAIN_PATH", <old default>)` |
| `compare_registrtaion/check_progress.sh` | **changed** — honours `STATE_FILE` / `LOG_DIR` |

The `configs.py` change is the whole mechanism: one env var selects the cohort and every
stage runs unmodified. The VinDr defaults are unchanged, so existing runs behave exactly
as before (§9 verifies that).

---

## 1. Set these once per shell

```bash
export REPO=/media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration
export DATA=/media/external20/saeedeh_danaei/waw_tace          # <-- pick a disk with ~1 TB
export MAIN_PATH=$DATA/derived/
export PYTHON=python3
```

`MAIN_PATH` **must be absolute and end in a slash.** Stages run from different working
directories (`registration/`, `compare_registrtaion/`, …) and the old default was a
relative `../../ncct_cect/vindr_ds/`; a relative override would resolve to a different
place in each stage. `run_waw_tace.sh` refuses to start on a relative `MAIN_PATH`.

Put those four lines in `~/.wawtace_env` and `source` it in every tmux pane.

## 2. Get the code and check the environment

```bash
cd $REPO && git pull
```

```bash
cd $REPO && $PYTHON -c "
import SimpleITK, pandas, numpy, scipy, pydicom, nibabel
print('SimpleITK', SimpleITK.__version__, '| pandas', pandas.__version__)
import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
import shutil; print('TotalSegmentator:', shutil.which('TotalSegmentator'))
print('ANTs antsRegistration:', shutil.which('antsRegistration'))
"
```

`prepare_waw_tace.py` imports `StandardizedDICOMProcessor` from the DICOM stage, which
imports `pydicom` at module level — so pydicom must be present even though no DICOM is
read. That import is deliberate: it keeps the `-1024` resample fill and the
`[-1024, 3000]` clip single-sourced with the code that produced the VinDr cohort.

```bash
ls -l $REPO/compare_registrtaion/deedsBCV/{deedsBCV,linearBCV,applyBCVfloat}
df -h $DATA
```

Need ~1 TB free at `$DATA` for the full cohort (§8 has the breakdown). If the disk is
smaller, decide the cohort cut **now**, at §5, not halfway through §7.

## 3. Gate check first — 20 MB, before the 48 GB

This decides whether the independent-label (tumour Dice) table is viable, which is the
part that answers "your Dice is circular — the objective optimises the same masks you
evaluate". Do not skip it to save an afternoon.

```bash
mkdir -p $DATA/raw && cd $DATA/raw
curl -L -O "https://zenodo.org/records/12741586/files/ct_hcc_metadata_v2.csv?download=1"
curl -L -O "https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1"
```

```bash
cd $DATA/raw && mv 'ct_hcc_metadata_v2.csv?download=1' ct_hcc_metadata_v2.csv 2>/dev/null; mv 'tumor_masks_wawtace_v1_08_05_2024.zip?download=1' tumor_masks_wawtace_v1_08_05_2024.zip 2>/dev/null; ls -lh
```

```bash
mkdir -p $DATA/tumor_masks && unzip -q -d $DATA/tumor_masks $DATA/raw/tumor_masks_wawtace_v1_08_05_2024.zip && find $DATA/tumor_masks -type f | head
```

```bash
cd $REPO && $PYTHON waw_tace/check_tumor_masks.py \
    --masks    $DATA/tumor_masks \
    --metadata $DATA/raw/ct_hcc_metadata_v2.csv \
    --phases 0 1 2 \
    --out      $DATA/tumor_eval_candidates.csv
```

Read the VERDICT line. It counts patients with a tumour mask in the **reference** phase
*and* in at least one moving phase — the only ones from which a registration Dice is
computable, because the masks were drawn on whichever phase showed the lesion best.

- **≥ ~25 patients** → viable. Build `evaluate_pipeline/evaluate_tumor.py` (§7 of the
  plan) and keep it in the paper as the independent ruler.
- **< 25** → demote to a qualitative figure and say so explicitly; the paper then rests
  on the front-end transfer table alone.
- **0** → drop that deliverable. Still worth knowing before downloading 48 GB.

Also keep the printed metadata column list — it is the authoritative phase-availability
source and should be cross-checked against what §5 discovers on disk.

## 4. Download and extract the scans

```bash
cd $DATA/raw && for i in 1 2 3 4; do
  curl -L -C - -O "https://zenodo.org/records/12741586/files/ct_scans_${i}_4_wawtace_09_05_24.zip?download=1"
done
```

`-C -` resumes a partial transfer; re-run the loop if it drops. 48 GB total.

```bash
cd $DATA/raw && for f in ct_scans_*.zip*; do mv "$f" "${f%%\?*}"; done; ls -lh
```

Extract each part into **one shared `source/`** — the parts split patients, not
directory levels:

```bash
mkdir -p $DATA/source && for i in 1 2 3 4; do
  unzip -q -n -d $DATA/source $DATA/raw/ct_scans_${i}_4_wawtace_09_05_24.zip
done
```

```bash
ls $DATA/source | head; echo "---"; ls $DATA/source | wc -l; du -sh $DATA/source
```

Expect ~233 case folders. If the zips nest an extra level (e.g. `$DATA/source/ct_scans_1_4/24/`),
flatten it so folders named by case id sit directly under `source/`, or point `--source`
at the real parent in §5. Check with `find $DATA/source -name '*_scan.nii.gz' | head -3`.

Once §5 succeeds, delete the zips to reclaim 48 GB:

```bash
rm -i $DATA/raw/ct_scans_*.zip
```

## 5. Convert to a VinDr-shaped root

Dry run first — it writes nothing and prints the keep/drop decision per case:

```bash
cd $REPO && $PYTHON waw_tace/prepare_waw_tace.py \
    --source $DATA/source --out $DATA/derived \
    --phases 0 1 2 --dry-run
```

Expect roughly 200 usable studies of 233. A study is kept only if it has phase 0
(the fixed `Non-contrast` reference) **and** at least one of phase 1/2. On the 13-case
local sample this keeps 7 and drops 6 — including case 25, which has phases 1,2,3 but no
non-contrast, and three empty folders.

Then convert (`--workers` to taste; each worker holds one volume in RAM, ~1 GB):

```bash
cd $REPO && $PYTHON waw_tace/prepare_waw_tace.py \
    --source $DATA/source --out $DATA/derived \
    --phases 0 1 2 --workers 8 2>&1 | tee $DATA/prepare.log
```

Resumable: re-running skips series whose output exists (`--overwrite` to force).

Output under `$DATA/derived/`:

| file | purpose |
|---|---|
| `nifti_unprocessed_volumes/waw0024_p0_standardized.nii.gz` | float32, 1.5 mm iso, LPS, HU ∈ [−1024, 3000] — same spec as VinDr |
| `..._standardized_metadata.txt` | VinDr sidecar format **plus native spacing** |
| `labels.csv` | `waw0024,p0,Non-contrast` — the vocabulary the pipeline already hardcodes |
| `excluded.csv` | dropped cases with reasons → the paper's data section |
| `cohort.csv` | native geometry per series + `thickness_stratum` |

### Verify before spending GPU time

```bash
cd $DATA/derived && head -4 labels.csv && echo "studies: $(tail -n +2 labels.csv | cut -d, -f1 | sort -u | wc -l)" && echo "series: $(tail -n +2 labels.csv | wc -l)" && cut -d, -f3 labels.csv | sort | uniq -c
```

```bash
cd $REPO && $PYTHON -c "
import pandas as pd, glob, SimpleITK as sitk
df = pd.read_csv('$MAIN_PATH/labels.csv')
assert df.StudyInstanceUID.dtype == object, 'study IDs parsed as numeric — IDs must be waw####'
print('study id dtype OK:', df.StudyInstanceUID.dtype, '| example', df.StudyInstanceUID.iloc[0])
c = pd.read_csv('$MAIN_PATH/cohort.csv')
print(c.thickness_stratum.value_counts().to_string())
print('native slice thickness mm: min %.2f max %.2f' % (c.native_slice_thickness_mm.min(), c.native_slice_thickness_mm.max()))
f = sorted(glob.glob('$MAIN_PATH/nifti_unprocessed_volumes/*_standardized.nii.gz'))[0]
im = sitk.ReadImage(f)
print(f.split('/')[-1], im.GetSize(), im.GetSpacing(), im.GetPixelIDTypeAsString())
"
```

The `study id dtype` assertion is the important one. If a study-ID column ever parses as
int64, `labels_df[labels_df["StudyInstanceUID"] == study]` compares ints to `os.listdir`
strings, matches nothing, and the whole run completes "successfully" having processed zero
studies. The `waw%04d` scheme prevents it; the assertion proves it.

**Decide the cohort size now.** A6 B-spline measured ~27 min/phase on VinDr, so ~200
studies × ~2 moving phases ≈ 180 h single-threaded, and DEEDS/ANTs are the same order.
If that does not fit the timeline, cut the **cohort**, not the conditions — rerun §5 with
a stratified `--cases` subset of ~60 (sample within `cohort.csv`'s thin/thick strata and
phase-availability groups) and state the sampling rule in the paper. VinDr reached
p = 1.5e-6 on 53 test studies, so 60 has ample power for the front-end effect.

## 6. Segmentation, split, alignment

TotalSegmentator over every standardised volume, then `combine_masks.py`:

```bash
cd $REPO/segmentation && cat > run_segmentation_waw.sh <<'EOF'
#!/bin/bash
set -e
: "${MAIN_PATH:?set MAIN_PATH first}"
export LOG_ROOT="${PIPELINE_LOG_ROOT:-pipeline_logs_waw}"
mkdir -p "$LOG_ROOT"
DATA_DIR="${MAIN_PATH}nifti_unprocessed_volumes"
output_dir="${MAIN_PATH}ts_segmentations"
mkdir -p "$output_dir"
for vol in "$DATA_DIR"/*_standardized.nii.gz; do
    file_name=$(basename "$vol" .nii.gz)
    base_name="${file_name/_standardized/}"
    IFS='_' read -r study_name vol_name _ <<< "$base_name"
    prefix="${output_dir}/${study_name}_${vol_name}"
    if [ -f "${prefix}_seg_reg.nii.gz" ]; then echo "SKIP $base_name"; continue; fi
    seg_dir="${output_dir}/${study_name}_${vol_name}_segs"
    mkdir -p "$seg_dir"
    echo "── $base_name"
    TotalSegmentator -i "$vol" -o "$seg_dir"
    python3 combine_masks.py "$seg_dir" "$prefix" || echo "WARN combine_masks exit $? on $base_name"
done
EOF
chmod +x run_segmentation_waw.sh
```

This is `run_segmentation.sh` with three changes: paths come from `MAIN_PATH`, it skips
series already done (so it resumes), and it does not abort the loop when `combine_masks`
exits non-zero — exit 2 means the sanity check found `seg_reg` too sparse, which is a
per-series finding to count, not a reason to stop.

```bash
cd $REPO/segmentation && tmux new-session -d -s wawseg './run_segmentation_waw.sh 2>&1 | tee seg_waw.log'
tmux attach -t wawseg
```

~600 series. Budget by your per-volume TotalSegmentator time; this is the GPU-bound step.

Afterwards, count the sanity failures — they are a reportable number, and the
thick-slice series are where to expect them:

```bash
grep -c WARN $REPO/segmentation/seg_waw.log; ls $MAIN_PATH/ts_segmentations/*_seg_reg.nii.gz | wc -l
```

Split — WAW-TACE is **100% test**. Nothing is trained on it: VoxelMorph keeps its
VinDr weights, because zero-shot transfer is the claim. `--test_ratio 1.0` marks every
study `test` so the existing `--split test` plumbing works unchanged:

```bash
cd $REPO/compare_registrtaion && $PYTHON generate_split.py --test_ratio 1.0
```

Alignment — A2 (z-aligned) and A1 (crop-only baseline):

```bash
cd $REPO/alignment && $PYTHON align_and_crop.py --all --new-layout 2>&1 | tee align_waw.log
```

```bash
cd $REPO/alignment && $PYTHON align_and_crop.py --all --new-layout --baseline 2>&1 | tee align_waw_baseline.log
```

Before the full `--all`, sanity-check one study — the flat-layout file discovery globs
`*{study}*{series}*standardized.nii.gz`, and it must resolve to exactly one file:

```bash
cd $REPO/alignment && $PYTHON align_and_crop.py waw0010 --new-layout
```

```bash
ls $MAIN_PATH/aligned_volumes | wc -l; ls $MAIN_PATH/baseline_volumes | wc -l
```

## 7. Registration grid

```bash
cd $REPO/compare_registrtaion && tmux new-session -d -s wawtace \
  'bash run_waw_tace.sh 2>&1 | tee run_logs_waw/master.log'
tmux attach -t wawtace
```

The script prints its cohort size and refuses to start without an absolute `MAIN_PATH` or
a `labels.csv`. It runs the cut condition list from the plan (A1, A2, A3–A5, A3b–A5b, A6,
A6b, R_full, R_mmi, B2 DEEDS, B3 ANTs, B5 uGI+IO-50 — aligned and baseline inputs, no
`raw` arm) as seven parallel state-guarded jobs. Dropped versus the VinDr run: `R_organ`,
`R_sobel`, `B4_vxm`, `B5_unigradicon` without IO.

Monitor:

```bash
cd $REPO/compare_registrtaion && STATE_FILE=.run_state_waw LOG_DIR=run_logs_waw bash check_progress.sh
```

```bash
tail -f $REPO/compare_registrtaion/run_logs_waw/job_a_series_aligned.log
```

Resume after any interruption by re-running the same `run_waw_tace.sh` — completed steps
are skipped via `.run_state_waw`, and the watchers resume from the studies already in each
`eval_detail.csv`.

## 8. Evaluation

The watchers evaluate each condition as it finishes. The final table:

```bash
cd $REPO/compare_registrtaion && $PYTHON eval_compare_conditions.py --all --split test
```

```bash
ls $MAIN_PATH/all_baseline_algorithms/; column -s, -t < $MAIN_PATH/all_baseline_algorithms/comparison_table.csv 2>/dev/null | head -30
```

Then the two things that make this external validation rather than a second table:

**Slice-thickness stratification.** WAW-TACE z-spacing runs 1.25–5.0 mm and varies within
a patient; resampling 5.0 → 1.5 mm is a 3.3× upsample. Join `cohort.csv` onto the eval
rows and report the A1→A2 gain separately for `thin` (≤2.5 mm) and `thick` (>2.5 mm). If
the gain holds in both, that is stronger than not asking; if it does not, that is the
finding.

**Cohort table.** `excluded.csv` + the `combine_masks` sanity-failure count from §6 are
the data-section numbers. Report them; a reviewer who finds 233 patients in the citation
and ~200 in your table will ask.

If §3 cleared, build `evaluate_pipeline/evaluate_tumor.py` per plan §4.4 (tumour Dice +
centroid, one label, no erosion, reusing each condition's stored transform — no
re-registration) and run it over `$DATA/tumor_eval_candidates.csv`.

## 9. Regression check — VinDr must still work

The `configs.py` edits are defaulted, so nothing should change. Prove it rather than
assume it, in a shell **without** `MAIN_PATH` set:

```bash
cd $REPO/compare_registrtaion && env -u MAIN_PATH $PYTHON -c "
import compare_config as C
print('MAIN:', C.MAIN)
print('labels exists:', C.LABELS_CSV.exists())
print('conditions:', len(C.CONDITIONS))
"
```

It must print the VinDr root. Then confirm the frozen numbers are untouched:

```bash
cd $REPO/compare_registrtaion && env -u MAIN_PATH STATE_FILE=.run_state bash check_progress.sh | head -40
```

## 10. Order of work

1. §1–2 paths and environment.
2. §3 gate check — 20 MB. **Settle the tumour-mask decision before downloading 48 GB.**
3. §4 download and extract.
4. §5 convert, verify, **and decide the cohort size**.
5. §6 segmentation (the GPU-bound stretch), split, alignment.
6. §7 registration grid.
7. §8 evaluation, stratification, cohort table, tumour table.
8. §9 regression check — before you trust any VinDr number you re-quote.
