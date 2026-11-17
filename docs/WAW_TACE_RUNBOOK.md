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

## 3. Gate check — DONE 2026-11-16, result negative

Ran, so do not repeat it. `check_tumor_masks.py` over all 377 masks (233 patients):

| phase | patients with a tumour mask |
|---|---|
| 0 non-contrast | 3 |
| 1 arterial | 126 |
| 2 venous | 74 |
| 3 delayed | 30 |

Those sum to 233 — the total number of patients with any mask — so **every patient has
masks in exactly one phase**. Qualifying patients (mask in the reference phase and in a
moving phase): **0**. No pairing rescues it; a registration Dice needs the lesion drawn
in both images being registered, and that never occurs in this dataset.

The tumour-Dice deliverable is dropped. Say so explicitly in the paper, with these
counts — a reader who knows WAW-TACE ships tumour masks will otherwise wonder why they
went unused. The masks still serve for qualitative overlay figures.

The circularity objection it was meant to answer is handled instead by evaluating
**tier-3 organs** (plan §3.1b): `seg_full` already carries 20 organs the registration
objective never sees, and `register_fixed.py` already warps it alongside `seg_reg`. That
is a change to `evaluate_registration.ORGAN_LABELS`, not new data.

Useful by-product — `ct_hcc_metadata_v2.csv` has 854 rows, one per scan, with columns
`patient_id, ct_phase, ct_file_name, tumor_count, slice_thickness`. That matches the 854
extracted scans exactly, and its `slice_thickness` is a cleaner source for the
thickness stratification than re-deriving it from headers.

## 4. Extract the scans — DONE 2026-11-16

The archives live read-only at `/media/external20/saeedeh_danaei/WAW-Tace/` (44 GB, four
parts). They were extracted into the working tree with:

```bash
mkdir -p $DATA/source && for i in 1 2 3 4; do unzip -q -n -d $DATA/source $SCANS/ct_scans_${i}_4_wawtace_09_05_24.zip; done
```

Result: **233 case folders, 854 `*_scan.nii.gz`** — matching `ct_hcc_metadata_v2.csv`
row for row. `unzip -d` only reads the archive, so the source directory being read-only
does not matter.

Paths in use on the server:

```
REPO  = /media/external20/saeedeh_danaei/sample_data_reg/multi-phase_registration
DATA  = /media/external20/saeedeh_danaei/sample_data_reg/ncct_cect/waw_tace
SCANS = /media/external20/saeedeh_danaei/WAW-Tace          (read-only archives)
MAIN_PATH = $DATA/derived/
```

`$DATA` sits beside `vindr_ds/`, which is the parallel layout the rest of this plan
assumes. Disk at the time: 3.0 TB free of 21 TB.

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

### What the conversion found (2026-11-16)

598 series converted, 0 failed. 200 studies usable of 233; 33 dropped, all but one for
`no_reference_phase` (arterial + venous present, no non-contrast) plus `waw0378` with no
scan files at all.

Two cohort properties that must be reported, both different from VinDr:

- **Within-study slice-thickness mismatch is the norm, not the exception**: 136 of 200
  studies have phases differing by ≥2×, and non-contrast runs up to 7.5 mm. VinDr came
  through the DICOM pipeline with consistent series; WAW-TACE is heterogeneous clinical
  data where a thin arterial against a thick venous phase is routine practice. That makes
  it a *harder* and more realistic test of the front-end — which strengthens the transfer
  claim if it holds, but only if stated. **Stratify by the per-study max/min thickness
  ratio**, not only the per-series thin/thick flag `cohort.csv` carries.
- **One non-axial series**: `waw0540_p1` is LSP (coronal, 98 native slices at 3 mm)
  against 597 LAS. Reorientation handled it correctly and it is in the right physical
  space, but a study whose phases disagree in acquisition plane is an avoidable confound.
  Excluded via `--exclude waw0540` in §6.

**Decide the cohort size now.** A6 B-spline measured ~27 min/phase on VinDr, so ~200
studies × ~2 moving phases ≈ 180 h single-threaded, and DEEDS/ANTs are the same order.
If that does not fit the timeline, cut the **cohort**, not the conditions — rerun §5 with
a stratified `--cases` subset of ~60 (sample within `cohort.csv`'s thin/thick strata and
phase-availability groups) and state the sampling rule in the paper. VinDr reached
p = 1.5e-6 on 53 test studies, so 60 has ample power for the front-end effect.

## 6. Segmentation, split, alignment

### Draw the evaluation sample FIRST

Order matters here. Segmentation is the long pole, and everything downstream filters on
`--split test`, so segmenting studies outside the sample buys nothing until the sample is
widened. Drawing it first turns ~598 TotalSegmentator runs into ~120.

```bash
cd $REPO && export MAIN_PATH=$DATA/derived/ && python3 compare_registrtaion/generate_split.py --sample 40 --sample_seed 7
```

Done on 2026-11-16 with seed 7: eligible pool 200 studies, 40 sampled, **79 moving pairs**
(40 arterial, 39 venous). Comparable to VinDr's 53 test studies. Record the seed in the
paper.

### Segment the sampled studies

```bash
cd $REPO/segmentation && export MAIN_PATH=$DATA/derived/ && bash run_segmentation_split.sh --split test --exclude waw0540 --dry-run
```

```bash
cd $REPO/segmentation && nohup bash -c 'source /media/external20/saeedeh_danaei/ncct_env/bin/activate && export MAIN_PATH=$DATA/derived/ && bash run_segmentation_split.sh --split test --exclude waw0540' > $DATA/seg_waw.log 2>&1 & echo "started pid $!"
```

The environment is a **venv**, and it must be activated *inside* the backgrounded
command — a `nohup` shell does not inherit an interactive activation, so a prior
`source .../activate` in your terminal does not reach it. Getting this wrong is how the
first attempt failed: `TotalSegmentator` was absent, every series failed in turn, and the
run ended seconds later with the whole work list in the crash list. The script now
refuses to start in that state rather than proving it 120 times.

`run_segmentation_split.sh` takes its paths from `MAIN_PATH`, resumes (a series with a
`_seg_reg.nii.gz` is skipped), and does **not** abort the loop when `combine_masks.py`
exits 2 — that means the sanity check found `seg_reg` too sparse, which is a per-series
finding to count, not a reason to stop the other 119. It prints the sparse and crashed
lists at the end; both are reportable numbers, the sparse ones especially, since
thick-slice series are where TotalSegmentator is expected to struggle.

`--exclude waw0540` drops the one study with a non-axial series (§5).

```bash
tail -f $DATA/seg_waw.log
```

### Split-aware alignment

```bash
cd $REPO/alignment && export MAIN_PATH=$DATA/derived/ && python3 align_and_crop.py waw0002 --new-layout
```

Run that single study first: the flat-layout discovery globs
`*{study}*{series}*standardized.nii.gz` and must resolve to exactly one file per call.
The `waw%04d` / `p{n}` ID scheme is what makes that safe, and this is the cheapest place
to confirm it.

Then A2 (z-aligned) and A1 (crop-only):

```bash
cd $REPO/alignment && export MAIN_PATH=$DATA/derived/ && python3 align_and_crop.py --all --new-layout 2>&1 | tee $DATA/align_waw.log
```

```bash
cd $REPO/alignment && export MAIN_PATH=$DATA/derived/ && python3 align_and_crop.py --all --new-layout --baseline 2>&1 | tee $DATA/align_waw_baseline.log
```

```bash
ls $DATA/derived/aligned_volumes | wc -l && ls $DATA/derived/baseline_volumes | wc -l
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
2. ~~§3 gate check~~ — done 2026-11-16, negative; the tumour-Dice deliverable is dropped
   and replaced by the tier-3 organ evaluation (plan §3.1b).
3. ~~§4 extract~~ — done 2026-11-16: 233 cases, 854 scans.
4. §5 convert, verify, **and decide the cohort size**.
5. §6 segmentation (the GPU-bound stretch), split, alignment.
6. §7 registration grid.
7. §8 evaluation, stratification, cohort table, tumour table.
8. §9 regression check — before you trust any VinDr number you re-quote.
