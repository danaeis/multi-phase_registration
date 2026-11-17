"""
evaluate_tier3.py
=================
Score registration on organs the registration objective NEVER SAW.

The objection this answers
--------------------------
Every organ in `evaluate_registration.ORGAN_LABELS` — liver, spleen, kidneys,
pancreas, gallbladder, aorta, IVC, L1-L3 — is in `seg_reg`, the mask the rigid
objective optimises overlap on. So the headline table grades the method with
its own ruler. A reviewer will say so, and they will be right.

`segmentation/combine_masks.py` already separates the two masks:

    seg_reg   tier 1-2, 71 organs  -> fed to the registration objective
    seg_full  + 20 tier-3 organs   -> commented in that file as
                                      "evaluation, not metric mask"

`seg_full` is written for every series and carried through every A-series
stage — the crops (A1/A2), the rigid passes (A3-A5 and the b-variants) and the
deformable pass (A6). Those labels have existed on every run ever done here and
have simply never been scored. Scoring them needs no new data and no
re-registration: this script reads the warped `seg_full` that is already on disk.

Scope, and why it is the right scope
------------------------------------
A-series in practice, and the script discovers that rather than assuming it: it
looks for the warped `seg_full` on disk per condition. The B-series runners
(`_common.py`) carry `seg_reg` alone, so DEEDS / ANTs / uniGradICON have none to
read, and they are reported as not applicable. That is not a gap:
their objectives are intensity-based and see no mask at all, so the circularity
charge does not apply to them in the first place. It applies to the A-series and
to R_organ / R_full, whose objectives contain an organ-overlap term — and those
are covered.

Extending it to the deformable baselines would mean warping `seg_full` through
their saved DVFs. Possible without re-registering, but a separate job; only
worth doing if someone asks for a tier-3 cross-method comparison.

The caveat that must travel with these numbers
----------------------------------------------
`seg_full` also widens the anatomy-guided crop bounding box, so tier-3 organs are
independent of the registration **objective** but NOT of the **crop**. State it.
It does not weaken the circularity answer, which is aimed at the organ-overlap
term; and the crop claim rests on failure-rate reduction in the intensity-based
engines, which never see a mask.

Organ choice
------------
Default set is stomach, esophagus, adrenal glands and duodenum-adjacent soft
tissue where available: abdominal, inside the crop, and genuinely never
optimised. Lung lobes and gluteal muscles are tier-3 too but sit at the crop
boundary where presence depends on the crop window rather than on registration
quality, so they are excluded by default. `--organs` overrides.

Usage
-----
  # One condition:
  python evaluate_tier3.py --conditions A5_pass012

  # The whole A-series, test split, both cohorts (run once per MAIN_PATH):
  MAIN_PATH=/abs/vindr_ds/     python evaluate_tier3.py --all --split test
  MAIN_PATH=/abs/waw_tace/derived/ python evaluate_tier3.py --all --split test

  # See which conditions have warped seg_full on disk, without scoring:
  python evaluate_tier3.py --all --probe

Outputs (under RESULTS_DIR)
---------------------------
  tier3_detail.csv    per (condition, study, phase, organ)
  tier3_summary.csv   per (condition, organ): median +/- IQR
  tier3_table.csv     per condition, pooled over tier-3 organs
  tier3_table.tex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "compare_registrtaion"))

import evaluate_registration as E          # noqa: E402
import compare_config as C                 # noqa: E402


# ---------------------------------------------------------------------------
# Tier-3 organs: in seg_full, NOT in seg_reg. Labels from combine_masks.py.
# ---------------------------------------------------------------------------
TIER3_ALL: Dict[int, str] = {
    100: "stomach",
    101: "esophagus",
    102: "urinary_bladder",
    103: "prostate",
    104: "adrenal_gland_left",
    105: "adrenal_gland_right",
    106: "kidney_cyst_left",
    107: "kidney_cyst_right",
    108: "gluteus_maximus_left",
    109: "gluteus_maximus_right",
    110: "gluteus_medius_left",
    111: "gluteus_medius_right",
    112: "gluteus_minimus_left",
    113: "gluteus_minimus_right",
    114: "lung_lower_lobe_left",
    115: "lung_lower_lobe_right",
    116: "lung_middle_lobe_right",
    117: "lung_upper_lobe_left",
    118: "lung_upper_lobe_right",
    119: "heart",
}

# Default: abdominal, well inside the crop, never optimised. Excludes the lung
# lobes and gluteal muscles, whose presence is decided by the crop window rather
# than by registration quality, and the cysts, which are incidental findings
# present in only a few patients.
TIER3_DEFAULT = (100, 101, 104, 105)

def seg_full_postfix_for(cond: C.Condition) -> Optional[str]:
    """
    Derive the condition's seg_full postfix from its seg_reg one.

    NOTE: this says nothing about whether such files exist. Every condition's
    seg_postfix ends in "_seg_reg.nii.gz", B-series included, so this derives
    "_deeds_seg_full.nii.gz" just as readily as "_rigid2_seg_full.nii.gz". The
    A-series writes both side by side and the B-series does not, but that is a
    fact about what is on disk, not about the naming — so it is detected by
    looking for the files (see evaluate_condition_tier3), never inferred here.

    Returns None only if the postfix does not follow the convention at all.
    """
    sp = cond.seg_postfix
    if not sp.endswith("_seg_reg.nii.gz"):
        return None
    return sp[: -len("_seg_reg.nii.gz")] + "_seg_full.nii.gz"


def is_b_series(cond: C.Condition) -> bool:
    """
    True for the external-algorithm conditions.

    compare_config sets input_key only on the B-series (algo x input tier); the
    A-series leaves it None. Used to explain a missing seg_full accurately
    rather than reporting it as something gone wrong.
    """
    return cond.input_key is not None


# ===========================================================================
# ONE SERIES
# ===========================================================================

def evaluate_series_tier3(
        study_dir: str,
        study_id: str,
        ref_sid: str,
        mov_sid: str,
        phase: str,
        vol_postfix: str,
        seg_full_postfix: str,
        organs: Dict[int, str],
        erosion_mm: float,
) -> List[dict]:
    """Metric rows for one moving series, scored on tier-3 organs only."""
    ref_seg, ref_vol = E.find_series_files(
        study_dir, study_id, ref_sid, seg_full_postfix, vol_postfix)
    mov_seg, mov_vol = E.find_series_files(
        study_dir, study_id, mov_sid, seg_full_postfix, vol_postfix)

    if not (ref_seg and ref_vol and mov_seg and mov_vol):
        return []

    seg_f = sitk.ReadImage(ref_seg)
    seg_m = sitk.ReadImage(mov_seg)
    vol_f = sitk.ReadImage(ref_vol)
    vol_m = sitk.ReadImage(mov_vol)

    seg_f_np = sitk.GetArrayFromImage(seg_f)
    seg_m_np = sitk.GetArrayFromImage(seg_m)
    vol_f_np = sitk.GetArrayFromImage(vol_f)
    vol_m_np = sitk.GetArrayFromImage(vol_m)

    if seg_f_np.shape != seg_m_np.shape:
        print(f"      shape mismatch {study_id}/{phase}: "
              f"{seg_f_np.shape} vs {seg_m_np.shape} — skipped")
        return []

    spacing_zyx = E.get_spacing_zyx(seg_f)

    rows: List[dict] = []
    for label, name in sorted(organs.items()):
        # evaluate_organ_pair returns None when the organ is absent or too
        # small in either mask. For tier-3 organs that is common and expected
        # (no prostate in a female patient, stomach outside a tight crop), so
        # it is a skip, not a failure.
        r = E.evaluate_organ_pair(
            label, name,
            seg_f_np, seg_m_np, vol_f_np, vol_m_np,
            spacing_zyx, erosion_mm,
        )
        if r is None:
            continue
        r.update(study_id=study_id, phase=phase, tier=3)
        rows.append(r)
    return rows


# ===========================================================================
# ONE CONDITION
# ===========================================================================

def evaluate_condition_tier3(
        cond: C.Condition,
        labels_df: pd.DataFrame,
        ref_phase: str,
        organs: Dict[int, str],
        erosion_mm: float,
        study_ids: Optional[set] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Returns (rows, status note)."""
    sfp = seg_full_postfix_for(cond)
    if sfp is None:
        return None, f"seg_postfix {cond.seg_postfix!r} does not follow the convention"
    if not cond.base_dir.exists():
        return None, f"base_dir absent: {cond.base_dir}"

    study_dirs = sorted(
        d for d in os.listdir(cond.base_dir)
        if (cond.base_dir / d).is_dir()
        and (study_ids is None or d in study_ids)
    )
    if not study_dirs:
        return None, "no study directories for this split"

    # Probe one series before walking the cohort: if seg_full was never written
    # here, say so once instead of returning a silent empty frame.
    n_with_segfull = 0
    for d in study_dirs[:10]:
        if list((cond.base_dir / d).glob(f"*{sfp}")):
            n_with_segfull += 1
    if n_with_segfull == 0:
        if is_b_series(cond):
            return None, ("B-series: its runner carries seg_reg only, so no warped "
                          "seg_full exists — and its objective sees no mask at all, "
                          "so the circularity charge does not apply to it")
        return None, (f"no *{sfp} files on disk, though this is an A-series "
                      f"condition that should write them — check the stage that "
                      f"produced {cond.base_dir.name}")

    rows: List[dict] = []
    for i, study_id in enumerate(study_dirs, 1):
        study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
        ref = study_rows[study_rows["Label"] == ref_phase]
        if ref.empty:
            continue
        ref_sid = ref.iloc[0]["SeriesInstanceUID"]
        for _, r in study_rows.iterrows():
            sid, phase = r["SeriesInstanceUID"], r["Label"]
            if phase == ref_phase:
                continue
            rows.extend(evaluate_series_tier3(
                str(cond.base_dir / study_id), study_id, ref_sid, sid, phase,
                cond.vol_postfix, sfp, organs, erosion_mm))
        if i % 10 == 0 or i == len(study_dirs):
            print(f"      {i}/{len(study_dirs)} studies, {len(rows)} rows")

    if not rows:
        return None, "seg_full present but no tier-3 organ met the size threshold"

    df = pd.DataFrame(rows)
    df.insert(0, "condition", cond.tag)
    return df, f"{len(df)} rows, {df['organ'].nunique()} organs"


# ===========================================================================
# TABLES
# ===========================================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition", "organ"])
    out = g.agg(
        n=("eroded_dice", "count"),
        dice_median=("eroded_dice", "median"),
        dice_iqr=("eroded_dice", E.iqr),
        centroid_median_mm=("centroid_mm", "median"),
        hd95_median_mm=("hd95_mm", "median"),
    ).reset_index()
    for c in ("dice_median", "dice_iqr"):
        out[c] = out[c].round(3)
    for c in ("centroid_median_mm", "hd95_median_mm"):
        out[c] = out[c].round(2)
    return out


def build_table(df: pd.DataFrame, fail_dice: float) -> pd.DataFrame:
    rows = []
    for tag, sub in df.groupby("condition"):
        d = sub["eroded_dice"].dropna()
        rows.append({
            "condition": tag,
            "label": C.ROW_LABELS.get(tag, tag),
            "n_rows": len(sub),
            "n_organs": sub["organ"].nunique(),
            "dice_median": round(float(d.median()), 3) if len(d) else np.nan,
            "dice_iqr": round(float(E.iqr(d)), 3) if len(d) else np.nan,
            "failure_pct": round(100.0 * float((d < fail_dice).mean()), 1) if len(d) else np.nan,
            "centroid_median_mm": round(float(sub["centroid_mm"].dropna().median()), 2),
        })
    return pd.DataFrame(rows).sort_values("dice_median", ascending=False)


def latex_table(df: pd.DataFrame, organs: Dict[int, str], fail_dice: float) -> str:
    def esc(x) -> str:
        s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
            s = s.replace(a, b)
        return s

    cols = ["label", "n_rows", "dice_median", "dice_iqr", "failure_pct",
            "centroid_median_mm"]
    lines = [
        "% Tier-3 organs: present in seg_full, absent from seg_reg, so never seen",
        "% by the registration objective. Organs scored: "
        + ", ".join(sorted(organs.values())),
        r"\begin{table}[ht]", r"\centering",
        r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\hline",
        " & ".join(esc(c) for c in cols) + r" \\", r"\hline",
    ]
    for _, row in df[cols].iterrows():
        lines.append(" & ".join(esc(v) for v in row.tolist()) + r" \\")
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Registration quality on tier-3 organs, which are excluded from "
        r"the mask the registration objective optimises. Failure: eroded Dice "
        rf"below {fail_dice:g}.}}",
        r"\label{tab:tier3}", r"\end{table}", "",
    ]
    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Score registration on organs the objective never saw.")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--all", action="store_true",
                   help="Every condition in compare_config.")
    p.add_argument("--probe", action="store_true",
                   help="Report which conditions have warped seg_full, score nothing.")
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--ref_phase", default=C.REF_PHASE)
    p.add_argument("--erosion_mm", type=float, default=C.EROSION_MM)
    p.add_argument("--split", choices=["all", "train", "test"], default="test")
    p.add_argument("--organs", nargs="*", type=int, default=list(TIER3_DEFAULT),
                   help=f"Tier-3 labels to score. Default {list(TIER3_DEFAULT)} "
                        f"(stomach, esophagus, adrenal glands). "
                        f"Available: {sorted(TIER3_ALL)}")
    p.add_argument("--fail_dice", type=float, default=0.50)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    organs = {k: v for k, v in TIER3_ALL.items() if k in set(args.organs)}
    unknown = set(args.organs) - set(TIER3_ALL)
    if unknown:
        print(f"ERROR: not tier-3 labels: {sorted(unknown)}")
        print(f"  tier-3 labels are {sorted(TIER3_ALL)}; anything else is in "
              f"seg_reg and would defeat the purpose of this script.")
        return 2
    if not organs:
        print("ERROR: no organs selected.")
        return 2

    print("=" * 78)
    print("evaluate_tier3.py — organs the registration objective never saw")
    print(f"  MAIN_PATH : {C.MAIN}")
    print(f"  organs    : " + ", ".join(f"{k} {v}" for k, v in sorted(organs.items())))
    print(f"  split     : {args.split}")
    print("=" * 78)

    study_ids = None
    if args.split != "all":
        from generate_split import load_split
        study_ids = load_split(args.split)
        print(f"split={args.split}: {len(study_ids)} studies")

    labels_df = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})

    tags = list(C.CONDITIONS) if (args.all or not args.conditions) else args.conditions

    if args.probe:
        print("\nseg_full availability per condition:")
        for tag in tags:
            cond = C.CONDITIONS.get(tag)
            if cond is None:
                continue
            sfp = seg_full_postfix_for(cond)
            if sfp is None:
                print(f"  --  {tag:<34} postfix off-convention: {cond.seg_postfix}")
                continue
            n = len(list(cond.base_dir.rglob(f"*{sfp}"))) if cond.base_dir.exists() else 0
            why = "" if n else ("  (B-series — seg_reg only, not applicable)"
                                if is_b_series(cond) else "  (expected for an A-series condition)")
            print(f"  {'ok' if n else '--'}  {tag:<34} {n:>5} files  *{sfp}{why}")
        return 0

    frames: List[pd.DataFrame] = []
    notes: List[Tuple[str, str]] = []
    for tag in tags:
        cond = C.CONDITIONS.get(tag)
        if cond is None:
            print(f"  unknown condition: {tag}")
            continue
        print(f"\n### {tag}")
        df, note = evaluate_condition_tier3(
            cond, labels_df, args.ref_phase, organs, args.erosion_mm, study_ids)
        print(f"    {note}")
        notes.append((tag, note))
        if df is not None:
            frames.append(df)

    if not frames:
        print("\nNo condition produced tier-3 rows. Notes above say why for each.")
        print("If every A-series condition says 'no *_seg_full files found', the")
        print("segmentation stage did not write seg_full — check combine_masks.py")
        print("output, not this script.")
        return 1

    detail = pd.concat(frames, ignore_index=True)
    summary = build_summary(detail)
    table = build_table(detail, args.fail_dice)

    detail.to_csv(out_dir / "tier3_detail.csv", index=False)
    summary.to_csv(out_dir / "tier3_summary.csv", index=False)
    table.to_csv(out_dir / "tier3_table.csv", index=False)
    (out_dir / "tier3_table.tex").write_text(
        latex_table(table, organs, args.fail_dice))

    print("\n" + "=" * 78)
    print("TIER-3 ORGANS — the objective never saw these")
    print("=" * 78)
    print(table[["label", "n_rows", "n_organs", "dice_median", "dice_iqr",
                 "failure_pct", "centroid_median_mm"]].to_string(index=False))

    print("\nper organ:")
    print(summary.to_string(index=False))

    skipped = [(t, n) for t, n in notes if "rows," not in n]
    if skipped:
        print(f"\nnot scored ({len(skipped)}):")
        for t, n in skipped:
            print(f"    {t:<34} {n}")

    print("\nCaveat to report with these numbers: seg_full also widens the")
    print("anatomy-guided crop bounding box, so tier-3 organs are independent of")
    print("the registration OBJECTIVE but not of the CROP.")
    print(f"\nwritten to {out_dir}: tier3_detail.csv, tier3_summary.csv, "
          f"tier3_table.csv/.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
