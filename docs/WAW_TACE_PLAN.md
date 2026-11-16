# WAW-TACE external validation — execution plan

Written 2026-11-15. Nothing has been executed; the full dataset lives on the remote
server. **The copy-paste command sequence is in [WAW_TACE_RUNBOOK.md](WAW_TACE_RUNBOOK.md)** —
this file is the reasoning behind it.

The code the runbook calls is written: `waw_tace/prepare_waw_tace.py`,
`waw_tace/check_tumor_masks.py`, `compare_registrtaion/run_waw_tace.sh`,
`compare_registrtaion/failure_rates.py`, the `--sample` flag on
`compare_registrtaion/generate_split.py`, and the `MAIN_PATH` env override in the four
`configs.py` files (§4.1). `evaluate_pipeline/evaluate_tumor.py` is **not** being written:
the §3.1 gate check came back negative. The tier-3 organ evaluation in §3.1b replaces it
and is a change to `evaluate_registration.ORGAN_LABELS`, not a new script.

## 0. Why this run exists

The defensible contribution of this work is the **front-end transfer result**: crop +
z-align (A1→A2) improves every downstream registrar, including off-the-shelf SOTA, and
the deformable stage (A6) only ties ANTs / uGI+IO. A single-cohort (VinDr) result cannot
support that claim for publication, because the front-end is tuned on the same intensity
and geometry distribution it is evaluated on.

WAW-TACE is the right second cohort:

- independent institution (Warsaw), independent scanners, different acquisition protocol;
- same task shape — multiphase abdominal CT, non-contrast reference + contrast phases;
- it ships **hand-drawn HCC tumour masks**, which are labels the registration objective
  never sees. VinDr evaluation uses TotalSegmentator masks, which the organ-centroid /
  eDice objective partly optimises — a reviewer will call that circular. Tumour Dice on
  WAW-TACE closes that hole.

So the run has two deliverables, in priority order:

1. **Transfer table** — A1/A2/A5/A6 + the strongest baselines, re-measured on WAW-TACE,
   showing the A1→A2 front-end gain reproduces out-of-distribution.
2. **Independent-label table** — tumour Dice / tumour centroid error on the subset with
   tumour masks in two or more phases, as an objective-independent ruler.

**Deliverable 2 was checked on 2026-11-16 and is not computable** — every WAW-TACE
patient has tumour masks in exactly one phase, so no registration Dice can be formed from
them (§3.1). Its purpose, answering the circularity objection, is served instead by
evaluating tier-3 organs, which `seg_full` already carries and the objective never sees
(§3.1b).

## 1. What the sample data actually is

Local sample: `/Users/macbook/develope/thesis/ncct_cect/waw_tace/source/` — 13 case
folders, 27 scans, named `{case}/{case}_{phase}_scan.nii.gz`.

Phase codes (confirmed against the Zenodo record, not inferred):

| code | phase | pipeline `Label` to use |
|---|---|---|
| 0 | native / precontrast | `Non-contrast` |
| 1 | late arterial | `Arterial` |
| 2 | portal venous | `Venous` |
| 3 | delayed | `Delayed` |

Keep those exact strings. `Non-contrast` is hardcoded as the fixed reference in
`registration/register_fixed.py:901`, `registration/deformable_registration_v2.py:883`
and `alignment/align_and_crop.py` (`find_noncontrast_series`). Matching VinDr's
vocabulary means zero code change there. VinDr already carries a 4th label (`Other`,
102 rows), so `Delayed` flows through the label-agnostic loops without special casing.

### Geometry — this is the part that needs work

Measured across all 27 sample scans:

- Orientation is **LAS**, not the LPS the pipeline standardises to. Reorientation required.
- In-plane spacing 0.613–0.922 mm; all 512×512.
- **Slice thickness 1.25–5.0 mm**, varying *within* a patient (case 10: 1.25 / 2.5 / 2.5).
- HU floor is **−3024** on 24 of 27 scans — the same scanner sentinel the DICOM stage
  already clips. Ceiling up to 3071.
- Some scans are stored `float64` (2.5× the disk of `int16` for no information).

The existing standardisation (`dcm2nii/dicom_processor.py`: LPS reorient → 1.5³ resample
with −1024 fill → clip to [−1024, 3000]) is exactly right for this data, but it is wired
to a DICOM reader. It needs a NIfTI-input sibling (§4).

Resampling 5.0 mm → 1.5 mm z is a 3.3× upsample and is not free. Record native z-spacing
per series in the metadata sidecar and **stratify the final results by slice thickness**
(≤2.5 mm vs >2.5 mm). If the front-end gain holds in both strata that is a stronger
result than hiding the variable; if it does not, that is a finding, not a failure.

### Phase availability — the real cohort size

Not every patient has every phase, and the gaps are not at the end. Sample:

| case | phases present | usable? |
|---|---|---|
| 10, 23 | 0,1,2,3 | yes |
| 2, 29, 60 | 0,2,3 | yes |
| 21 | 0,1,2 | yes |
| 24 | 0,1 | yes |
| 25 | 1,2,3 | **no — no NC reference** |
| 20 | 2 | no |
| 28 | 1 | no |
| 7, 26, 27 | *(empty folders)* | no |

7 of 13 usable in the sample. Full dataset: 233 patients, 200 with a precontrast phase,
so expect **~200 usable studies** — against VinDr's 183 NC studies / 53 test studies.
Comparable cohort, slightly larger.

Usability rule for `labels.csv`: keep a study iff it has phase 0 **and** at least one of
phase 1 / 2. Everything else is dropped with a logged reason.

## 2. Design decision: one pipeline, two dataset roots

Do **not** fork the pipeline for WAW-TACE. The publication claim is that the front-end
transfers; that claim is only credible if the same code runs on both cohorts. So:

- WAW-TACE becomes a second `MAIN_PATH` root, laid out identically to `vindr_ds/`:
  `nifti_unprocessed_volumes/`, `ts_segmentations/`, `labels.csv`, `splits/`, …
- One new adapter script converts `source/{case}/{case}_{p}_scan.nii.gz` into exactly the
  artefacts the DICOM stage would have produced.
- Every stage from segmentation onward runs unchanged, selected by `MAIN_PATH`.

### Study / series ID scheme — `waw0024` / `p0`

Not the raw `24` / `0`. Three reasons, all of them bugs if ignored:

1. `pd.read_csv` types a column of `24`, `25`, `60` as **int64**. `labels_df[labels_df["StudyInstanceUID"] == study]` compares against `os.listdir` strings and silently
   matches nothing — every study skipped, no error.
2. `alignment/align_and_crop.py:82` globs `*{study_id}*{series_id}*standardized.nii.gz`.
   With `study=2`, `series=0` that matches most of the cohort.
3. Zero-padding to a fixed width stops `waw002` matching inside `waw0024`.

So: `study_id = waw%04d`, `series_id = p{0..3}`. No underscores inside either token —
`segmentation/run_segmentation.sh` splits the filename on `_` to recover study and series.

## 3. Blocking decisions — settle these before any compute

1. **~~Does the tumour-mask evaluation have enough cases?~~ SETTLED 2026-11-16: NO.**
   Ran `check_tumor_masks.py` on all 377 masks (233 patients). Masks per phase:
   non-contrast 3, arterial 126, venous 74, delayed 30 — summing to 233, which is the
   number of patients with any mask at all. **Every patient has masks in exactly one
   phase**, so no pair of phases co-occurs for anyone. Qualifying patients: 0.

   This is structural, not a sample-size problem: there is no pairing that rescues it,
   not NC→arterial and not arterial→venous, because a registration Dice needs the lesion
   outlined in both images being registered and that never happens here. The masks were
   drawn once, on whichever phase showed the lesion best. Deliverable 2 is dropped —
   state in the paper that it was checked and why it is not computable, rather than
   leaving the reader to wonder why a dataset with tumour masks was not used for them.

   The masks remain usable for qualitative figures (overlay a warped arterial tumour on
   the NC phase) and the per-phase counts are worth reporting as cohort description.

1b. **The circularity problem that deliverable 2 was for — solve it with tier-3 organs
    instead.** The real objection is that the evaluation grades registration with the
    same TotalSegmentator masks the rigid objective optimises. All 11 organs in
    `evaluate_registration.ORGAN_LABELS` are in `seg_reg`, i.e. the objective sees every
    one of them.

    But `combine_masks.py` already splits the segmentation in two: `seg_reg` (71 organs,
    tier 1–2, fed to the objective) and `seg_full` (adds 20 tier-3 organs — stomach,
    esophagus, adrenal glands, urinary bladder, gluteal muscles, lung lobes — explicitly
    marked "evaluation, not metric mask"). `seg_full` is written for every series and
    **already warped alongside `seg_reg` by `register_fixed.py`**, so the independent
    labels exist on every run already and are simply never scored.

    Evaluating on tier-3 organs gives labels the registration objective never saw, on
    both cohorts, with no new data. Best candidates for abdominal registration: stomach
    (deformable, contrast-variable — a hard, fair test), esophagus, and the adrenal
    glands (small but positionally stable). Skip lung lobes and gluteal muscles, which
    sit at the crop boundary.

    One honest caveat: `seg_full` also widens the crop bounding box, so tier-3 organs
    are *not* independent of the **crop** front-end — only of the registration
    objective. That is fine for the circularity charge, which is aimed at the organ
    overlap term; and the crop claim is evidenced by failure-rate reduction in DEEDS /
    ANTs / uniGradICON, whose objectives are intensity-based and see no mask at all.

2. **Organ masks: re-run TotalSegmentator, or use the shipped ones?** The record ships
   1.0 GB of per-phase TotalSegmentator organ masks. Recommendation: **re-run TS on the
   standardised volumes**, as VinDr did. The shipped masks are in native anisotropic
   geometry (needing nearest-neighbour resampling that will not agree with the CT
   interpolation), and `segmentation/combine_masks.py` expects per-organ files from a
   specific TS version — a version mismatch in organ naming (`vertebrae_L1`, per-lobe
   lungs) would silently empty parts of `seg_reg`. Use the shipped masks only as a
   sanity cross-check on a handful of cases. If GPU time turns out to be the binding
   constraint, revisit.
3. **Delayed phase in or out?** Recommendation: **primary run on phases 0/1/2 only**, for
   a direct column-for-column comparison against the VinDr table; `Delayed` carried in
   `labels.csv` but excluded from the headline numbers, then reported as a secondary
   4-phase robustness row. Mixing a phase VinDr never had into the headline table
   invites "the cohorts aren't comparable".
4. **No training on WAW-TACE.** VoxelMorph is the only trained component. Evaluate the
   **VinDr-trained** weights zero-shot. That is the transfer claim; retraining on
   WAW-TACE answers a different and weaker question. So WAW-TACE is 100% test: generate
   the split with every study marked `test` so the `--split test` plumbing works unchanged.
5. **Where does this run?** Not locally — see §6.

## 4. Code changes required

Deliberately small. Anything larger than this list means the "same pipeline" claim is
getting shaky.

### 4.1 Make `MAIN_PATH` overridable (4 one-line edits)

`dcm2nii/configs.py`, `alignment/configs.py`, `registration/configs.py`,
`compare_registrtaion/configs.py` all hardcode the VinDr root. Change each to:

```python
import os
MAIN_PATH = os.environ.get("MAIN_PATH", '../../ncct_cect/vindr_ds/')
```

`compare_registrtaion/compare_config.py` and `evaluate_pipeline/config.py` already prefer
`configs.MAIN_PATH` with an env fallback, so they inherit this for free. After the edit,
`MAIN_PATH=../../ncct_cect/waw_tace/derived/` selects the cohort for every stage.

### 4.2 New: `waw_tace/prepare_waw_tace.py`

The only genuinely new code. Inputs `source/`, outputs a VinDr-shaped root.

Per `source/{case}/{case}_{p}_scan.nii.gz`:

1. Read with SimpleITK; reorient LAS → LPS (`StandardizedDICOMProcessor.reorient_to_lps`).
2. Resample to 1.5³ mm, linear, fill `-1024` (`resample_to_target_spacing`). Do not
   reimplement the fill value — that `-1024` choice is load-bearing per its own comment.
3. Clip to [−1024, 3000] (`clip_hu_range`, which also casts to **float32** — matching the
   VinDr volumes on disk, which are float32, not int16).
4. Write `nifti_unprocessed_volumes/waw{case:04d}_p{p}_standardized.nii.gz`.
5. Write `..._standardized_metadata.txt` in the existing format, **plus native in-plane
   and z spacing** so the thickness stratification in §1 is possible later.

All three geometry methods are called on `StandardizedDICOMProcessor` rather than copied,
so the cohorts cannot drift apart. That class imports `pydicom` at module level, so
pydicom must be installed even though no DICOM is read.

Then emit:

- `labels.csv` with `StudyInstanceUID,SeriesInstanceUID,Label` = `waw0024,p0,Non-contrast`,
  only for studies passing the §1 usability rule;
- `excluded.csv` — one row per dropped case with the reason (no NC, single phase, empty
  folder, unreadable). This table goes in the paper's data section.

Flags: `--source`, `--out`, `--cases`, `--phases 0 1 2`, `--workers`, `--overwrite`,
`--report-only`, `--dry-run`.

### 4.3 New: `waw_tace/prepare_tumor_masks.py` (gated on decision 3.1)

Resample each `{case}_{p}_{i}_tumor_seg` into the standardised grid with
**nearest-neighbour**, merge per-case tumour indices into one label volume, write to
`tumor_masks/waw{case:04d}_p{p}_tumor.nii.gz`. Keep it out of `seg_reg` — the whole point
is that no registrar ever sees it.

### 4.4 New: `evaluate_pipeline/evaluate_tumor.py` (gated on decision 3.1)

Mirrors `evaluate_registration.py`'s structure but with one label and no erosion:
tumour Dice, tumour centroid displacement (mm), one row per (study, phase, condition).
It applies the same stored transform each condition already wrote, so it needs no
re-registration.

### 4.5 Minor: visualisation phase order

`visualizations/viz_01_phase_alignment.py:64` and `viz_02_hu_histograms.py:71` hardcode
`PHASE_ORDER = ["Non-contrast", "Arterial", "Venous"]`. Append `"Delayed"` with a colour
when the secondary 4-phase run is figured. Not needed for the primary run.

## 5. Execution order

### Stage 0 — local smoke test, on the 13-case sample (do this first)

The point is to shake out the adapter and the ID scheme on data that fits on the laptop,
not to produce numbers.

```bash
export MAIN_PATH=../../ncct_cect/waw_tace/derived/
python waw_tace/prepare_waw_tace.py --source ../ncct_cect/waw_tace/source --phases 0 1 2
```

Check, before going further:

- `labels.csv` has 7 studies (10, 2, 21, 23, 24, 29, 60), `excluded.csv` has 6 with reasons;
- every standardised volume is 1.5³, LPS, `int16`, HU in [−1024, 3000];
- study IDs survive a `pd.read_csv` round-trip as `str`, not `int64`.

Then one study end to end — TotalSegmentator → `combine_masks` → align → rigid:

```bash
cd segmentation && bash run_segmentation.sh          # ~10 min/volume on CPU
cd ../alignment  && python align_and_crop.py waw0010 --new-layout
cd ../registration && python register_fixed.py waw0010 --skip
```

Watch specifically for: the `*{study}*{series}*` glob in `align_and_crop.py` resolving to
exactly one file per call, and `combine_masks.py` sanity-passing (it exits 2 on a sparse
`seg_reg`, which is also the honest answer if TS struggles on 5 mm slices).

### Stage 1 — full data, on the server

```
download (48 GB, CC BY 4.0) → unzip 4 scan parts → prepare_waw_tace.py --all
  → run_segmentation.sh  (~200 studies × 3 phases ≈ 600 TS runs)
  → generate_split.py --test_ratio 1.0        # everything is test
  → align_and_crop.py --all --new-layout      # A2 aligned
  → align_and_crop.py --all --new-layout --baseline   # A1 crop-only
```

### Stage 2 — registration grid

Reuse `compare_registrtaion/run_all_resume.sh` as-is; it is already resumable and
state-guarded. **But cut the condition list.** VinDr ran ~30 conditions; WAW-TACE needs
only the ones the external-validity argument rests on:

| keep | why |
|---|---|
| `A1_crop_only`, `A2_zalign` | the front-end gain itself — the headline |
| `A5_pass012` | best rigid, cheap |
| `A6_bspline` | the full proposed method |
| `A3b/A4b/A5b_noalign` | isolates z-align out of distribution |
| `R_full`, `R_mmi` | proposed rigid objective vs the standard one |
| `B2_deeds`, `B3_ants`, `B5_unigradicon_io` | the three that tied or won on VinDr |

Drop `R_organ`, `R_sobel`, `B4_vxm` and `B5_unigradicon` (no IO). **Keep the `--input raw`
arm** for the B-series: the crop contrast is `raw` → `baseline` and the headline transfer
claim is `raw` → `aligned`, so without a raw tier neither can be re-tested here (see §5b).
The saving comes from the cohort size instead.

### Stage 3 — evaluation

```bash
cd compare_registrtaion && python eval_compare_conditions.py --all --split test
python ../evaluate_pipeline/evaluate_tumor.py --all        # if 3.1 cleared
```

Then the two tables from §0, plus the slice-thickness stratification and the
`excluded.csv` cohort table.

## 5b. Failure rate, and the evaluation sample

Two reporting requirements that shape how the grid is run.

### Failure rate, not just median Dice

The headline claim is that the front-end is a **failure-mode eliminator** — on VinDr it
moved rigid MIND-NCC from 57% to 10% failure while barely touching the median, and the
gain per engine tracks that engine's own raw failure rate (MIND-NCC 60.6% fail → +0.369;
DEEDS 5.9% → +0.020; ANTs → n.s.). A median ± IQR table cannot show that, so
`compare_registrtaion/failure_rates.py` reports it directly.

A (study, phase, organ) triple fails if its eroded Dice is below 0.5 — reusing the
threshold already in the manuscript rather than inventing a second one — **or** if the
condition produced no output for it. The second clause matters more than it looks: a
median computed over whatever a registrar survived flatters the registrar. This project
has already hit that exact trap, with plain uniGradICON's crop-only run covering 176 of
578 samples that turned out to be an easy subset (raw Dice 0.745 vs 0.590 for the rest).

Denominator is the union of triples **any** condition evaluated, not the label-derived
product of pairs × organs: TotalSegmentator legitimately finds no gallbladder in a patient
who has had one removed, and charging every method for that is noise. But a triple some
method produced is one this method could have, so a gap there is a real failure.

Validated against the VinDr results already on the laptop: the per-evaluated failure rate
reproduces the published numbers (DEEDS 3.8%, ANTs 9.9%, A4 13.5% against 13.4% on
record), and both conditions the project records as unusable surface unprompted as low
coverage — `R_full__raw` 3.1%, `B5_unigradicon__baseline` 26.2%.

### With vs without the front-end, decomposed and paired

Each algorithm runs on three input tiers, so the front-end splits into its two steps:

| step | contrast | what VinDr found |
|---|---|---|
| crop | `raw` → `baseline` | carries the entire effect |
| z-align | `baseline` → `aligned` | contributes nothing (−0.007 to +0.002), hurts standalone |
| front-end | `raw` → `aligned` | the headline transfer result |

Each is paired over shared triples: McNemar exact on discordant failures, Wilcoxon
signed-rank on Dice where both tiers succeeded. Reporting both is deliberate — a front-end
can leave the median untouched while removing the catastrophic tail, and the tail is what
decides whether a pipeline is usable.

Note this means the **`raw` input arm must be kept** for the B-series on WAW-TACE, which
reverses the cut in §Stage 2 of the plan: without `raw` there is no crop contrast and the
headline claim cannot be re-tested on the second cohort. Keep `--input all` for DEEDS,
ANTs and uGI+IO; the saving has to come from the cohort instead.

### One sample, drawn once, for everything

To keep the grid affordable, `generate_split.py --sample N --sample_seed S` marks exactly
N randomly chosen studies as `test`. Every runner and the evaluator already filter on
`--split test`, so all algorithms and all states land on the identical studies.

That is not merely convenient — it is required. The paired tests above compare conditions
per (study, phase, organ); sampling per algorithm would compare methods on different data
and the pairing would be void. Studies without the reference phase are excluded from the
draw, since they have no fixed image. `splits/eval_sample.csv` records the seed, N, the
eligible pool and the drawn IDs, so the sample is reproducible and reportable — state the
seed in the paper.

Power: VinDr reached p = 1.5e-6 for A6 vs A4 on 53 test studies, and the front-end effects
are an order of magnitude larger (failure rate deltas of 20–47 points). N = 40 is ample;
N = 40 also cuts A6 from ~180 h to ~36 h.

## 6. Compute and disk — the binding constraint

**This cannot run on the laptop.** 23 GB free on `/System/Volumes/Data`; the download
alone is 48 GB. Local scope is Stage 0 only (sample ≈ 1.2 GB standardised + segs).

Server budget, ~200 studies × 3 phases = 600 series:

| item | estimate |
|---|---|
| raw download + unzip | ~100 GB (delete zips after) |
| standardised volumes | 600 × ~44 MB ≈ 26 GB |
| `seg_full` + `seg_reg` | ≈ 50 GB |
| A1 + A2 crops | ≈ 50 GB |
| 11 conditions × warped vol + seg + DVF | ≈ 400–600 GB |
| **total** | **~0.8 TB** |

Time: A6 B-spline is ~27 min/phase (measured on VinDr). 200 studies × 2 moving phases
× 27 min ≈ **180 h single-threaded** — the dominant cost by far, and DEEDS/ANTs are the
same order. `run_all_resume.sh` already runs the jobs in parallel; even so, plan on
1–2 weeks wall-clock on the current server.

**If that does not fit the publication timeline**, cut the cohort, not the conditions:
sample **~60 studies** stratified by phase availability and slice thickness. VinDr reached
p = 1.5e-6 on 53 test studies, so 60 is ample power for the front-end effect, and it
brings A6 to ~54 h. Decide this explicitly and state the sampling rule in the paper —
a stratified 60 is defensible; a quietly truncated 60 is not.

## 7. Risks

| risk | mitigation |
|---|---|
| Tumour masks mostly single-phase → deliverable 2 dies | check first, 20 MB (§3.1) |
| TotalSegmentator degrades on 5 mm slices → sparse `seg_reg` | `combine_masks` already exits 2; stratify by thickness, report the failure rate |
| Numeric study IDs silently match nothing | `waw%04d` / `p{n}` scheme (§2) |
| Front-end gain does **not** transfer | that is a publishable negative with the geometry data to explain it; do not hide it |
| Compute overruns the timeline | stratified 60-study cohort decided up front, not mid-run |

## 8. Suggested order of work

1. Download metadata + tumour masks (20 MB); settle decision 3.1. ← start here
2. Write `prepare_waw_tace.py`; run Stage 0 on the local sample.
3. Make the 4 `MAIN_PATH` edits; confirm VinDr still runs unchanged (regression check).
4. Settle decisions 3.2–3.5, including cohort size, in writing.
5. Server: download, prepare, segment, align.
6. Registration grid on the cut condition list.
7. Evaluation, tumour table, stratification, cohort table.
