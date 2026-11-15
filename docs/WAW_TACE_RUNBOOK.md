# WAW-TACE server runbook

Copy-paste commands for the remote server. The reasoning behind the choices is in
[WAW_TACE_PLAN.md](WAW_TACE_PLAN.md); this file is just the sequence.

Written 2026-11-15, corrected after first contact with the server. Nothing in §3 onward
has been executed yet. The code is on the server as of `daebd95` (§2), and the four
WAW-TACE scan archives were **already downloaded** there (§4) — 44 GB, still zipped.

`$REPO` is confirmed as
`/media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration` (not the
older `data_reg_pipeline` clone beside it — see §2). `$DATA` is still yours to choose in
§1. Which Python packages and GPUs the server has is unverified — §2b checks that.

---

## 0. What ships with this runbook

These are committed and pushed to `danaeis/multi-phase_registration` `main`
(f2d99c3). The server cannot reach GitHub, so §2 moves them across as a bundle:

| file | what it is |
|---|---|
| `waw_tace/prepare_waw_tace.py` | **new** — converts WAW-TACE into a VinDr-shaped root |
| `waw_tace/check_tumor_masks.py` | **new** — the 20 MB gate check (§3) |
| `compare_registrtaion/run_waw_tace.sh` | **new** — registration grid, cut condition list |
| `compare_registrtaion/failure_rates.py` | **new** — failure rate per condition + paired with/without-front-end contrast |
| `compare_registrtaion/generate_split.py` | **changed** — `--sample N --sample_seed S` draws a fixed evaluation sample |
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

## 2. Get the code onto the server

The server reaches GitHub over **HTTPS through a proxy**, but **not over SSH** — ssh gets
`Could not resolve hostname github.com` because there is no direct DNS, while git's HTTPS
traffic resolves on the proxy side. So the clone's remote must be an HTTPS URL. Done once:

```bash
cd $REPO && git remote set-url origin https://github.com/danaeis/multi-phase_registration.git
```

```bash
cd $REPO && git status --short && git fetch origin && git pull --ff-only
```

Check `git status --short` first and deal with anything it lists, because two kinds of
local state will stop the merge:

- an **untracked** file that the incoming commits also add — git refuses rather than
  overwrite it. Move it aside, pull, then `diff` the two and delete the copy if identical.
  (`compare_registrtaion/measure_resources.py` hit exactly this: it had been created on
  the server and separately committed upstream in 84a6845. The copies were identical.)
- a **modified** tracked file. Print `git diff` for it and decide; do not discard it
  blind.

Then verify the pull took:

```bash
cd $REPO && ls waw_tace/ docs/ compare_registrtaion/failure_rates.py && grep -n "gradientConvergenceTolerance\|_deformable_dvf" registration/deformable_registration_v2.py
```

`gradientConvergenceTolerance=1e-12` and `_deformable_dvf.nii.gz` are the fixed values. If
you see `1e-5` or `_deformable_field`, the fast-forward did not take and A6 is back to
producing identity transforms.

**Not the `data_reg_pipeline` directory.** `/media/external20/saeedeh_danaei/data_reg_pipeline`
is a different, older lineage: on branch `master` (this repo has only `main` and
`server-thesis-results`), with `check_z.ipynb` tracked, and with no `compare_registrtaion/`
or `evaluate_pipeline/`. Pulling this repo into it would merge two unrelated histories.
The clone to use is `sample_data_reg/multi-phase_registration`.

Do **not** rsync the repo tree onto the server: the laptop's `deedsBCV` binaries are
macOS arm64 and would clobber the Linux builds, and `--delete` would wipe server-only
ignored files such as the uniGradICON weights.

## 2b. Check the environment

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

## 3. Gate check first — 20 MB

This decides whether the independent-label (tumour Dice) table is viable, which is the
part that answers "your Dice is circular — the objective optimises the same masks you
evaluate". The scans are already on the server (§4) but **the tumour masks and metadata
are not** — only the four `ct_scans_*.zip` were fetched. These two files are 20 MB.

The server reaches GitHub over its HTTPS proxy; whether it reaches Zenodo is a separate
question, so test first:

```bash
curl -sI -o /dev/null -w '%{http_code}\n' "https://zenodo.org/records/12741586/files/ct_hcc_metadata_v2.csv?download=1"
```

`200` (or `302`) means fetch them directly on the server:

```bash
mkdir -p $DATA/raw && cd $DATA/raw && curl -L -o ct_hcc_metadata_v2.csv "https://zenodo.org/records/12741586/files/ct_hcc_metadata_v2.csv?download=1" && curl -L -o tumor_masks.zip "https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1" && ls -lh
```

Anything else (`000`, a proxy error) means the proxy covers github only — download those
two on the laptop and move them across; they are small enough that any route works.

```bash
mkdir -p $DATA/tumor_masks && unzip -q -d $DATA/tumor_masks $DATA/raw/tumor_masks.zip && find $DATA/tumor_masks -type f | head
```

```bash
cd $REPO && $PYTHON waw_tace/check_tumor_masks.py --masks $DATA/tumor_masks --metadata $DATA/raw/ct_hcc_metadata_v2.csv --phases 0 1 2 --out $DATA/tumor_eval_candidates.csv
```

Read the VERDICT line. It counts patients with a tumour mask in the **reference** phase
*and* in at least one moving phase — the only ones from which a registration Dice is
computable, because the masks were drawn on whichever phase showed the lesion best.

- **≥ ~25 patients** → viable. Build `evaluate_pipeline/evaluate_tumor.py` (plan §4.4)
  and keep it in the paper as the independent ruler.
- **< 25** → demote to a qualitative figure and say so explicitly; the paper then rests
  on the front-end transfer table alone.
- **0** → drop that deliverable, and state that it was checked.

Keep the printed metadata column list — it is the authoritative phase-availability source
and should be cross-checked against what §5 finds on disk.

## 4. Extract the scans — already downloaded

The four archives are **already on the server**, 44 GB total:

```bash
ls -lh /media/external20/saeedeh_danaei/WAW-Tace/
```

So there is nothing to download here and no laptop disk problem. Set `$DATA` in §1 to
wherever you want the derived tree; the archives can stay where they are.

The zips hold `.nii.gz` files, which are already compressed, so extraction is roughly
1:1 — budget about another 44 GB. Check before starting:

```bash
df -h /media/external20/saeedeh_danaei/
```

Extract all four parts into **one shared `source/`** — the parts split patients, not
directory levels:

```bash
mkdir -p $DATA/source && for i in 1 2 3 4; do unzip -q -n -d $DATA/source /media/external20/saeedeh_danaei/WAW-Tace/ct_scans_${i}_4_wawtace_09_05_24.zip; done
```

```bash
ls $DATA/source | head && echo "---" && ls $DATA/source | wc -l && find $DATA/source -name "*_scan.nii.gz" | wc -l && du -sh $DATA/source
```

Expect ~233 case folders and roughly 650–800 `*_scan.nii.gz`. If the zips nest an extra
level (e.g. `$DATA/source/ct_scans_1_4/24/`), flatten it so the case-id folders sit
directly under `source/`, or point `--source` at the real parent in §5:

```bash
find $DATA/source -name "*_scan.nii.gz" | head -3
```

Keep the zips until §5 has run and its `cohort.csv` row count looks right — they are the
only copy, and re-downloading needs a network route that may not be there.

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
VinDr weights, because zero-shot transfer is the claim.

Draw the **evaluation sample** here, with a recorded seed. Every runner and the
evaluator filter on `--split test`, so marking exactly N studies as `test` is what makes
all algorithms and all states land on the identical studies:

```bash
cd $REPO/compare_registrtaion && $PYTHON generate_split.py --sample 40 --sample_seed 7
```

Pick N from the compute budget in §5 — 40 studies × ~2 moving phases × ~27 min ≈ 36 h for
A6, against ~180 h for the full ~200. It writes `splits/eval_sample.csv` recording the
seed, N, the eligible pool and the drawn study IDs, so the sample is reproducible and
reportable. Studies with no non-contrast phase are excluded from the draw, since without
a fixed image there is nothing to register and sampling one would quietly shrink N.

Sample **once, globally** — never per algorithm. Each condition reading the same split is
what keeps the with/without-front-end comparison paired per (study, phase, organ), which
is what `failure_rates.py` tests in §8. Per-algorithm samples would compare methods on
different data and the paired tests would be meaningless.

To use every eligible study instead, drop `--sample` and pass `--test_ratio 1.0`:

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
A6b, R_full, R_mmi, B2 DEEDS, B3 ANTs, B5 uGI+IO-50) as seven parallel state-guarded jobs.
The B-series runs all three input tiers — raw, baseline, aligned — because the crop step
is `raw` → `baseline` and the front-end claim is `raw` → `aligned`; §8's paired contrast
needs both. Dropped versus the VinDr run: `R_organ`, `R_sobel`, `B4_vxm`, and
`B5_unigradicon` without IO.

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

### Failure rates, and performance with vs without the front-end

```bash
cd $REPO/compare_registrtaion && $PYTHON failure_rates.py --split test
```

`run_waw_tace.sh` already runs this at the end; the standalone call is for re-running
after a threshold change. It answers both halves of the question directly:

**Failure rate per condition** (`failure_rates.csv` / `.tex`). A (study, phase, organ)
triple fails if its eroded Dice is below 0.5 — the threshold this project already reports
— **or** if the method produced no output for it at all. That second clause is the point:
the median ± IQR table silently rewards a registrar that crashes on hard cases, because
its median is then computed over the easy remainder. This project has already been bitten
by it (plain uniGradICON's crop-only run covers 176 of 578 samples, and those samples are
an easy subset: raw Dice 0.745 vs 0.590 for the rest). The table prints `failure_pct`
beside `failure_pct_of_evaluated`; a large gap between the two is the warning sign, and
`coverage_pct` names it outright.

**With vs without, paired** (`front_end_effect.csv`). Each algorithm ran on three input
tiers, so the front-end decomposes into the two steps and the whole:

| step | contrast |
|---|---|
| crop | `raw` → `baseline` |
| z-align | `baseline` → `aligned` |
| front-end | `raw` → `aligned` |

Each is paired over the triples the two tiers share: McNemar exact on the discordant
failures (did the step change failure?) and Wilcoxon signed-rank on Dice where both
survived (did it change accuracy?). Both are reported because on this data they disagree
informatively — the front-end is a failure-mode eliminator, so it moves the tail far more
than the median. The A-series rows cover the same contrast for the pipeline's own stages.

Verified against the VinDr results already on the laptop: `failure_pct_of_evaluated`
reproduces the thesis's published failure rates (DEEDS 3.8%, ANTs 9.9%, A4 13.5% vs the
13.4% on record), and the two conditions the project notes as unusable surface on their
own as low `coverage_pct` — `R_full__raw` at 3.1% and `B5_unigradicon__baseline` at 26.2%.

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

1. §1 paths; §2 bundle the code to the server (no `git pull` there); §2b environment.
2. §3 gate check — 20 MB (tumour masks + metadata; these were NOT in the earlier
   download). **Settle the tumour-mask decision before the segmentation run.**
3. §4 extract the scan archives already on the server.
4. §5 convert, verify, **and decide the cohort size**.
5. §6 segmentation (the GPU-bound stretch), split, alignment.
6. §7 registration grid.
7. §8 evaluation, stratification, cohort table, tumour table.
8. §9 regression check — before you trust any VinDr number you re-quote.
