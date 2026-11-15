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

**The server has no internet, so `git pull` from GitHub will not work.** Code reaches it
only through a bundle you transfer. On the laptop (canonical clone):

```bash
cd /Users/macbook/develope/thesis/multi-phase_registration && git pull --ff-only
```

```bash
mkdir -p /Users/macbook/develope/thesis/transfer && cd /Users/macbook/develope/thesis/multi-phase_registration && git bundle create /Users/macbook/develope/thesis/transfer/multi-phase_registration.bundle main server-thesis-results
```

Move that bundle to the server by whatever route you normally use, then on the server —
clearing the uncommitted B-spline edit first, which would otherwise block the
fast-forward:

```bash
cd $REPO && git status --short
```

```bash
cd $REPO && git checkout -- registration/deformable_registration_v2.py
```

```bash
cd $REPO && git fetch /path/to/multi-phase_registration.bundle main:refs/remotes/origin/main && git switch main && git merge --ff-only refs/remotes/origin/main
```

Confirm the WAW-TACE files arrived and that the B-spline fixes are the ones on disk:

```bash
cd $REPO && ls waw_tace/ compare_registrtaion/run_waw_tace.sh compare_registrtaion/failure_rates.py && grep -n "gradientConvergenceTolerance\|_deformable_dvf" registration/deformable_registration_v2.py
```

`gradientConvergenceTolerance=1e-12` and `_deformable_dvf.nii.gz` are the fixed values; if
you see `1e-5` or `_deformable_field`, the fast-forward did not take.

Do **not** rsync the repo tree onto the server: the laptop's `deedsBCV` binaries are macOS
arm64 and would clobber the Linux builds, and `--delete` would wipe server-only ignored
files such as the uniGradICON weights.

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

## 3. Gate check first — 20 MB, before the 48 GB

This decides whether the independent-label (tumour Dice) table is viable, which is the
part that answers "your Dice is circular — the objective optimises the same masks you
evaluate". Do not skip it to save an afternoon.

**Download where there is internet, not on the server** — it has none (§2). These two
files are 20 MB, so run the `curl` on the laptop and move them across:

```bash
mkdir -p ~/Downloads/wawtace && cd ~/Downloads/wawtace
curl -L -O "https://zenodo.org/records/12741586/files/ct_hcc_metadata_v2.csv?download=1"
curl -L -O "https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1"
```

The gate check itself is pure Python over filenames, so it can run on the laptop too —
that is the cheapest place to settle this decision. Substitute your own paths for
`$DATA/raw` and `$DATA/tumor_masks` below if you do.

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

**This is the step with a real logistical problem, and it needs your decision.** The four
scan archives are 48 GB; the server has no internet, and the laptop has ~23 GB free. So
neither machine can do this alone on its own disk. Pick a route before starting:

- an external drive on the laptop — download there, then transfer to the server;
- a third machine with internet and disk that can reach the server;
- a proxy or mirror on the server's network, if one exists for data (GitHub is blocked,
  but your network may differ for other hosts — worth checking before assuming);
- one archive at a time through the laptop: download, transfer, delete, repeat. Each part
  is 10.4–13.5 GB, which fits in 23 GB one at a time.

The last option needs no new hardware and is the fallback if nothing else is available.

Wherever it runs, the download itself is:

```bash
cd $DATA/raw && for i in 1 2 3 4; do
  curl -L -C - -O "https://zenodo.org/records/12741586/files/ct_scans_${i}_4_wawtace_09_05_24.zip?download=1"
done
```

`-C -` resumes a partial transfer; re-run the loop if it drops. 48 GB total. Everything
from §5 onward runs on the server, where the extracted `source/` lives.

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
2. §3 gate check — 20 MB, runnable on the laptop. **Settle the tumour-mask decision
   before moving 48 GB anywhere.**
3. §4 decide the download route (the server has no internet and the laptop has ~23 GB
   free), then download and extract.
4. §5 convert, verify, **and decide the cohort size**.
5. §6 segmentation (the GPU-bound stretch), split, alignment.
6. §7 registration grid.
7. §8 evaluation, stratification, cohort table, tumour table.
8. §9 regression check — before you trust any VinDr number you re-quote.
