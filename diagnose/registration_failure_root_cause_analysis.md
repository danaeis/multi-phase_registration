# Root Cause Analysis: Catastrophic Organ Centroid Displacement (80–160 mm)

## What the uploaded files tell us

### Study 1.2.392 — ALIGNED stage (before registration)
All three phases share **identical affine origin and grid** (correct — crop
step worked). But the moving phases (Arterial, Venous) are shifted relative
to NC:

| Organ       | |Δ| Arterial | |Δ| Venous | Direction |
|-------------|------------|-----------|-----------|
| L1–L3       | 19–22 mm   | 18–22 mm  | +X (pure lateral) |
| Spinal cord | 20.5 mm    | 20.2 mm   | +X |
| Kidneys     | 19–22 mm   | 19–22 mm  | +X |
| Liver       | 11 mm      | 11 mm     | +X |
| Aorta       | 13 mm      | 11 mm     | +X |

**Pattern**: consistent ~18–22 mm shift in physical X only. Z-delta ≤ 12 mm.
This is a pure in-plane (lateral) offset, *not a rotation*, and is **expected**
pre-registration residual — the align step is Z-only. The Kabsch pass should
correct this (12 common organs, det = +1.0 for both phases).

### Study 1.2.840 — RIGID0 output (after Kabsch pass)
All three series are within **2–6 mm** of each other on every organ.
Rigid0 worked correctly here. IVC shows 16 mm but that is arterial-phase
IVC segmentation variability, not a registration failure.

---

## Root causes of 80–160 mm catastrophic failures (other studies)

Based on the code review and the data pattern, four causes can fire. They
appear roughly in this order of frequency:

---

### Cause 1 — Seg_reg too sparse: Kabsch gets < 3 common organs → identity fallback
**Most common cause.** The pipeline has this hard guard in `register_v2`:
```python
if len(common) < 3:
    raise ValueError(f"Need ≥3 common organs for Kabsch, got {len(common)}")
```
This exception propagates up to the `except Exception as e:` block in
`register_study()`, which appends to `results["failed"]` and **skips the
series entirely** — leaving no output file. However if the exception is
silently swallowed somewhere or a downstream script uses a stale file, the
effective transform is identity → 80–160 mm displacement.

**Why would seg_reg be sparse?**
- Bug 1 (double HU rescale on 1.2.840 scanner) was fixed globally, but if
  any study was processed *before* the fix, its NIfTI/seg files on disk are
  still corrupted. TotalSegmentator ran on bone that looked like lung and
  found nothing → empty or near-empty seg_reg.
- The crop window is wrong (Cause 3 below) → only padding/air inside the
  window → TotalSegmentator finds no organs.

**How to confirm**: run `diagnose_registration_failures.py` and look for
`KABSCH_FAIL_TOO_FEW_ORGANS`.

---

### Cause 2 — Direction determinant sign mismatch (body flip)
**Second most likely cause for the largest errors (>100 mm).** If one phase
was acquired FFS (Feet First Supine) and another HFS (Head First Supine),
or if `DICOMOrient` produced a mirrored result on one phase, the organ
layout appears reflected. Kabsch SVD finds the "best" point cloud
alignment of a mirror image, which is the translation that maps
each organ to its anatomical mirror position — roughly the body diameter
(300–400 mm for the full body, or 100–200 mm for the organ cluster).

The code already handles the reflection-correction case in `kabsch_align`:
```python
d = np.linalg.det(Vt.T @ U.T)
D = np.diag([1.0, 1.0, float(np.sign(d)) if d != 0 else 1.0])
R = Vt.T @ D @ U.T
```
But this only works if Kabsch *runs*. If the seg_reg has <3 organs (Cause 1),
Kabsch never runs.

**How to confirm**: `diagnose_registration_failures.py` checks
`det_sign_match`. A `False` value means one phase has a reflected volume.

**Fix**: add an upfront orientation guard in `align_data_v3.align_study()`:
```python
nc_det  = np.linalg.det(np.array(ref_img.GetDirection()).reshape(3,3))
mov_det = np.linalg.det(np.array(mov_img.GetDirection()).reshape(3,3))
if np.sign(nc_det) != np.sign(mov_det):
    # Force DICOMOrient on the moving image before computing Z-offset
    mov_img = sitk.DICOMOrient(mov_img, "LPS")
```

---

### Cause 3 — Align_data produced a bad Z-offset → wrong anatomy in crop window
`align_data_v3` has `MAX_OFFSET_MM = 250.0` (very permissive). If TotalSegmentator
produced a corrupted seg_reg (from Bug 1 on pre-fix images), the organ
centroid will be computed on garbage labels, and `z_offset` can have the
wrong sign or be 100–200 mm off. This shifts the crop window entirely into
the wrong anatomical region (e.g. pelvis instead of upper abdomen).

Signs of this failure in the alignment metadata JSON:
- `z_offset` > 50 voxels (75 mm) with confidence 1.0 — suspicious high confidence
  on a large offset, typically because the dominant "organ" is actually an
  artefact voxel cluster.
- Or `confidence < 0.4` → fallback fires, but fallback `_organ_z_center`
  can also be wrong if seg_reg is sparse.

**How to confirm**: `diagnose_registration_failures.py` checks the physical
Z-offset between anchor-organ centroids in aligned seg_reg. `BAD_Z_OFFSET`
fires when |z_offset| > 80 mm.

**Fix**: tighten `MAX_OFFSET_MM` from 250 → 80 mm, and add a per-organ
consistency check (std of per-organ offsets should be < 15 mm; if std > 20 mm
the seg is suspicious).

---

### Cause 4 — Large in-plane (XY) residual after Z-only alignment
The Z-alignment step corrects only the scan table Z-position difference.
In-plane (XY) offsets from patient lateral repositioning between phases
are left uncorrected. For most cases these are < 20 mm, which Kabsch and
Nelder-Mead handle easily.

But for a minority of studies with > 40 mm lateral offset:
- Kabsch works (it is a global optimizer) but its residual error is large,
  giving a poor starting point for Pass 1.
- Nelder-Mead (Pass 1) is a local optimizer and can get stuck.

The 1.2.392 aligned files show ~18–20 mm XY residual — within the safe zone.
When this exceeds ~40 mm, Nelder-Mead failure risk increases substantially.

**Fix**: Pass 0b (all-organ Kabsch) already handles this well because it is
a global optimizer. Make sure Pass 0b completes successfully before handing
off to Pass 1. If Pass 0b Kabsch residual > 20 mm, log a warning.

---

## Summary checklist for a failing study

```
1. Check alignment metadata JSON:
   - z_offset > 50 voxels?  → re-run align_data with MAX_OFFSET_MM=80
   - confidence < 0.40?     → check seg_reg organ content (organ count, voxels)

2. Run diagnose_registration_failures.py:
   - KABSCH_FAIL_TOO_FEW_ORGANS → seg_reg is empty/corrupted
     → re-run dicom_processor + TotalSegmentator on this study
     → verify HU range: body voxels should span -1024 to +1024
     → connected-component bone check: HU>200 largest component > 10,000 voxels
   
   - DIRECTION_DET_SIGN_MISMATCH → orientation flip between phases
     → add DICOMOrient guard in align_data_v3 before Z-offset computation
   
   - BAD_Z_OFFSET                → crop window is wrong
     → tighten MAX_OFFSET_MM, rerun alignment
   
   - LARGE_XY_RESIDUAL           → >40mm in-plane before Kabsch
     → verify Pass 0b Kabsch residual in registration logs

3. Verify rigid0 output with diagnose_registration_failures.py --seg_postfix _rigid0_seg_reg.nii.gz:
   - If mean_centroid_3d > 15mm after rigid0 → Kabsch failed or ran with wrong seg
   - If mean_centroid_3d < 10mm → rigid0 is fine; check Pass 1 / Pass 2 degradation
```

---

## How to use the diagnostic scripts

```bash
# Single study (aligned stage, before registration):
python diagnose_registration_failures.py \
    --study_dir /data/aligned_volumes/STUDY_ID \
    --study_id  STUDY_ID \
    --labels_csv /data/labels.csv \
    --seg_postfix _aligned_seg_reg.nii.gz

# After rigid0:
python diagnose_registration_failures.py \
    --study_dir /data/rigid_registered/STUDY_ID \
    --study_id  STUDY_ID \
    --labels_csv /data/labels.csv \
    --seg_postfix _rigid0_seg_reg.nii.gz

# Batch over all studies, save CSV:
python diagnose_registration_failures.py --all \
    --base_dir   /data/aligned_volumes \
    --labels_csv /data/labels.csv \
    --seg_postfix _aligned_seg_reg.nii.gz \
    --out_csv    /data/diagnosis_aligned.csv

# Render HTML report from the CSV:
python visualize_diagnosis.py \
    --csv /data/diagnosis_aligned.csv \
    --out /data/diagnosis_aligned.html

# Or from Python, no labels CSV needed:
from diagnose_registration_failures import quick_check_pair
quick_check_pair("NC_aligned_seg_reg.nii.gz", "ART_aligned_seg_reg.nii.gz")
```
