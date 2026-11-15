"""
failure_rates.py
================
Registration FAILURE RATE per condition, and the paired with/without-front-end
contrast, decomposed into the crop step and the z-align step.

Failure definition
------------------
A (study, phase, organ) triple is a failure when its eroded Dice falls below
--fail_dice (default 0.5). That is the threshold this project already reports
("failure rate Dice<0.5: A6 9.9%, A4 13.4%, ANTs 9.9%, DEEDS 3.8%"), so it is
reused here rather than a second, inconsistent definition.

On top of that, a triple the labels CSV says should exist but which the
condition never produced counts as `no_output` — a failure, not a missing row:

    failure = no_output OR (eroded_dice < fail_dice)

That distinction is the point of this script. `eval_compare_conditions.py`
reports median ± IQR over the rows a condition produced, which silently rewards
a registrar that crashes on hard cases: its median is computed over the easy
remainder. This project has already been bitten by exactly that — plain
uniGradICON's crop-only run covers 176 of 578 samples and those samples are an
easy subset (raw Dice 0.745 vs 0.590 for the rest). Counting absent output as
failure makes that visible instead of flattering.

The three contrasts
-------------------
Every external algorithm ran on three input tiers, so the front-end decomposes:

    raw      -> baseline   the anatomy-guided crop alone
    baseline -> aligned    the z-align stage alone
    raw      -> aligned    the whole front-end

Each is PAIRED over the (study, phase, organ) triples the two tiers share:

    McNemar exact   on the discordant failures  — did the step change failure?
    Wilcoxon        on Dice where both survived — did it change accuracy?

Both are reported because they answer different questions, and on this data they
disagree in an informative way: the front-end is a failure-mode eliminator, so
it moves the tail far more than the median.

Matching
--------
Triples are deduplicated with keep="first", matching how the thesis builds its
574 matched triples (some studies carry several "Other" series). Change this and
the numbers stop lining up with the manuscript.

Usage
-----
  python failure_rates.py --split test
  python failure_rates.py --split test --fail_dice 0.5
  python failure_rates.py --conditions A2_zalign B3_ants__aligned B3_ants__raw

Outputs (under RESULTS_DIR)
---------------------------
  failure_rates.csv        one row per condition
  failure_rates.tex        same, LaTeX
  failure_triples.csv      per (condition, study, phase, organ) outcome — the
                           raw material for the paired tests
  front_end_effect.csv     per algorithm x step, paired
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import compare_config as C

# The project's established failure threshold on eroded Dice.
DEFAULT_FAIL_DICE = 0.50

# Organs the comparison table is built on (evaluate_registration.ORGAN_LABELS).
# Restricting to these keeps the denominator identical across conditions even if
# a stray organ appears in one condition's CSV.
EVAL_ORGANS = (
    "liver", "spleen", "kidney_left", "kidney_right", "pancreas", "gallbladder",
    "aorta", "inferior_vena_cava", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
)

KEY = ["study_id", "phase", "organ"]

# A-series z-align contrast: the same registration stage fed aligned vs
# crop-only input, so the pair isolates z-align.
A_SERIES_ZALIGN_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("A3 Rigid P0",        "A3_pass0",   "A3b_pass0_noalign"),
    ("A4 Rigid P0+1",      "A4_pass01",  "A4b_pass01_noalign"),
    ("A5 Rigid P0+1+2",    "A5_pass012", "A5b_pass012_noalign"),
    ("A6 Full deformable", "A6_bspline", "A6_bspline_baseline"),
)

# Input tiers, weakest front-end first.
TIERS = ("raw", "baseline", "aligned")
TIER_STEPS: Tuple[Tuple[str, str, str], ...] = (
    ("crop",      "raw",      "baseline"),
    ("z-align",   "baseline", "aligned"),
    ("front-end", "raw",      "aligned"),
)


# ===========================================================================
# EXPECTED WORK  (the denominator)
# ===========================================================================

def expected_triples(labels_csv: Path,
                     ref_phase: str,
                     organs: Tuple[str, ...],
                     study_ids: Optional[set] = None) -> pd.DataFrame:
    """
    Every (study, phase, organ) triple that SHOULD have been produced.

    dtype=str so numeric-looking study IDs (WAW-TACE's waw0024 scheme aside,
    any bare-numeric cohort) stay strings and keep matching directory names.
    """
    df = pd.read_csv(labels_csv, dtype=str)
    col = "StudyInstanceUID" if "StudyInstanceUID" in df.columns else "study_id"
    df = df.rename(columns={col: "study_id", "Label": "phase"})
    if study_ids is not None:
        df = df[df["study_id"].isin(study_ids)]

    # A study with no reference phase has no fixed image, so nothing can be
    # registered for it. iter_pairs() skips those studies; counting them here
    # would charge every method for work the pipeline never had input for.
    eligible = set(df.loc[df["phase"] == ref_phase, "study_id"])
    n_all = df["study_id"].nunique()
    if len(eligible) < n_all:
        print(f"  eligible studies: {len(eligible)} of {n_all} "
              f"({n_all - len(eligible)} have no '{ref_phase}' phase)")
    df = df[df["study_id"].isin(eligible) & (df["phase"] != ref_phase)]

    pairs = df[["study_id", "phase"]].drop_duplicates()
    out = pairs.merge(pd.DataFrame({"organ": list(organs)}), how="cross")
    return out.reset_index(drop=True)


# ===========================================================================
# PER-CONDITION OUTCOMES
# ===========================================================================

def load_detail(cond: C.Condition,
                eligible: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Load one condition's eval_detail.csv, restricted to eligible triples.

    Returns None when there is no detail CSV — "never run" is not the same as
    "ran and succeeded", and must not be reported as 0% failure.
    """
    if not cond.detail_csv.exists():
        return None
    detail = pd.read_csv(cond.detail_csv, dtype={"study_id": str})
    if detail.empty or "eroded_dice" not in detail.columns:
        return None

    # Match the thesis's matching rule: several "Other" series per study would
    # otherwise contribute duplicate triples and silently reweight the test.
    detail = detail.drop_duplicates(subset=KEY, keep="first")
    cols = [c for c in ("eroded_dice", "centroid_mm", "hd95_mm")
            if c in detail.columns]
    return detail[KEY + cols].merge(eligible, on=KEY, how="inner")


def comparable_triples(details: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    The union of triples that at least one condition managed to evaluate.

    This is the right denominator for a failure rate. The label-derived product
    (pairs x organs) over-counts: TotalSegmentator legitimately finds no
    gallbladder in a patient who has had one removed, and charging every method
    for that is noise, not failure. A triple SOME method produced, though, is a
    triple this method could have produced — so a gap there is this method's
    failure, which is exactly the bias a median-only table hides.
    """
    if not details:
        return pd.DataFrame(columns=KEY)
    union = pd.concat([d[KEY] for d in details.values()], ignore_index=True)
    return union.drop_duplicates().reset_index(drop=True)


def condition_outcomes(detail: pd.DataFrame,
                       tag: str,
                       denominator: pd.DataFrame,
                       fail_dice: float) -> pd.DataFrame:
    """One row per triple in the denominator, with this condition's outcome."""
    out = denominator.merge(detail, on=KEY, how="left")
    cond = C.CONDITIONS[tag]

    out["condition"] = cond.tag
    out["no_output"] = out["eroded_dice"].isna()
    out["below_thresh"] = (~out["no_output"]) & (out["eroded_dice"] < fail_dice)
    out["failure"] = out["no_output"] | out["below_thresh"]
    return out


def summarise(outcomes: pd.DataFrame, tag: str) -> dict:
    n = len(outcomes)
    n_missing = int(outcomes["no_output"].sum())
    n_below = int(outcomes["below_thresh"].sum())
    n_fail = int(outcomes["failure"].sum())
    got = outcomes.loc[~outcomes["no_output"]]
    n_eval = len(got)

    row = {
        "condition": tag,
        "label": C.ROW_LABELS.get(tag, tag),
        "n_expected": n,
        "n_evaluated": n_eval,
        "coverage_pct": round(100.0 * n_eval / n, 1) if n else np.nan,
        "n_no_output": n_missing,
        "n_below_thresh": n_below,
        "n_failure": n_fail,
        "failure_pct": round(100.0 * n_fail / n, 1) if n else np.nan,
        "no_output_pct": round(100.0 * n_missing / n, 1) if n else np.nan,
        # Failure rate among only the triples that were produced — the number a
        # median-only table implicitly reports. Printed next to failure_pct so
        # the gap between them is visible.
        "failure_pct_of_evaluated":
            round(100.0 * n_below / n_eval, 1) if n_eval else np.nan,
        "dice_median": round(float(got["eroded_dice"].median()), 3) if n_eval else np.nan,
    }
    if "centroid_mm" in outcomes.columns and n_eval:
        row["centroid_median_mm"] = round(float(got["centroid_mm"].median()), 2)
    return row


# ===========================================================================
# PAIRED TESTS
# ===========================================================================

def _mcnemar(b: int, c: int) -> Tuple[float, str]:
    """
    Exact two-sided McNemar on discordant counts.

    b = failed WITHOUT the step but not with it   (the step rescued it)
    c = failed WITH the step but not without it   (the step broke it)
    """
    n = b + c
    if n == 0:
        return float("nan"), "no discordant triples"
    p = float("nan")
    try:
        from scipy.stats import binomtest
        p = float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        pass
    except Exception:
        pass
    if np.isnan(p):
        try:
            from scipy.stats import binom_test          # scipy < 1.7
            p = float(binom_test(b, n, 0.5, alternative="two-sided"))
        except Exception:
            from math import erfc, sqrt
            chi = (abs(b - c) - 1) ** 2 / n
            p = float(erfc(sqrt(max(chi, 0.0) / 2.0)))
    return p, f"b={b} rescued, c={c} broken"


def _wilcoxon(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d = d[d["x"] != d["y"]]
    if len(d) < 5:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d["x"], d["y"]).pvalue)
    except Exception:
        return float("nan")


def paired_contrast(algo: str,
                    step: str,
                    without_fe: pd.DataFrame,
                    with_fe: pd.DataFrame) -> dict:
    """Paired comparison of one step for one algorithm, over shared triples."""
    m = without_fe.merge(with_fe, on=KEY, suffixes=("_without", "_with"))
    if m.empty:
        return {"algorithm": algo, "step": step, "n_triples": 0,
                "note": "no shared triples"}

    f_without = m["failure_without"].astype(bool)
    f_with = m["failure_with"].astype(bool)

    b = int((f_without & ~f_with).sum())
    c = int((f_with & ~f_without).sum())
    p_mc, disc = _mcnemar(b, c)

    both = m.loc[~f_with & ~f_without]
    p_wx = _wilcoxon(both["eroded_dice_with"], both["eroded_dice_without"])

    # Mean paired Dice delta over every triple both tiers produced (including
    # failures), which is the quantity the transfer claim is stated in.
    produced = m.loc[m["eroded_dice_with"].notna() & m["eroded_dice_without"].notna()]
    d_mean = (float((produced["eroded_dice_with"]
                     - produced["eroded_dice_without"]).mean())
              if len(produced) else np.nan)

    return {
        "algorithm": algo,
        "step": step,
        "n_triples": len(m),
        "failure_pct_without": round(100.0 * f_without.mean(), 1),
        "failure_pct_with": round(100.0 * f_with.mean(), 1),
        "failure_pct_delta": round(100.0 * (f_with.mean() - f_without.mean()), 1),
        "rescued_b": b,
        "broken_c": c,
        "mcnemar_p": p_mc,
        "discordant": disc,
        "n_both_ok": len(both),
        "dice_median_without": round(float(both["eroded_dice_without"].median()), 3)
            if len(both) else np.nan,
        "dice_median_with": round(float(both["eroded_dice_with"].median()), 3)
            if len(both) else np.nan,
        "dice_mean_delta": round(d_mean, 3) if not np.isnan(d_mean) else np.nan,
        "n_paired_dice": len(produced),
        "wilcoxon_p": p_wx,
    }


def build_front_end_table(outcomes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []

    # B-series: each algorithm across the three input tiers.
    for algo in C.BASELINE_ALGOS.values():
        by_tier = {t: outcomes.get(C.condition_tag(algo.tag, t)) for t in TIERS}
        for step, lo, hi in TIER_STEPS:
            if by_tier.get(lo) is None or by_tier.get(hi) is None:
                continue
            rows.append(paired_contrast(algo.name, step, by_tier[lo], by_tier[hi]))

    # A-series: the front-end rows themselves (A1 crop-only -> A2 z-aligned)...
    a1, a2 = outcomes.get("A1_crop_only"), outcomes.get("A2_zalign")
    if a1 is not None and a2 is not None:
        rows.append(paired_contrast("A1/A2 front-end only", "z-align", a1, a2))

    # ...and each registration stage with vs without z-align.
    for name, aligned_tag, baseline_tag in A_SERIES_ZALIGN_PAIRS:
        a, b_ = outcomes.get(aligned_tag), outcomes.get(baseline_tag)
        if a is None or b_ is None:
            continue
        rows.append(paired_contrast(name, "z-align", b_, a))

    return pd.DataFrame(rows)


# ===========================================================================
# LATEX
# ===========================================================================

def _latex_table(df: pd.DataFrame, fail_dice: float) -> str:
    """
    Hand-rolled tabular.

    pandas' DataFrame.to_latex routes through Styler on this pandas version and
    so needs jinja2, an optional dependency that is not guaranteed on the run
    host. Writing the rows directly keeps the table working everywhere, and the
    shape is simple enough that there is nothing to gain from the dependency.
    """
    def esc(x) -> str:
        s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
            s = s.replace(a, b)
        return s

    lines = [
        f"% Failure: eroded Dice < {fail_dice:g}, or no output produced.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}",
        r"\hline",
        " & ".join(esc(c) for c in df.columns) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(v) for v in row.tolist()) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Registration failure rate. A (study, phase, organ) triple "
        rf"fails if its eroded Dice is below {fail_dice:g} or the method "
        r"produced no output for it.}",
        r"\label{tab:failure_rates}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Per-condition failure rate + paired front-end contrasts.")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Condition tags (default: all with a detail CSV).")
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--ref_phase", default=C.REF_PHASE)
    p.add_argument("--split", choices=["all", "train", "test"], default="test",
                   help="Restrict to a split (default: test — the eval sample).")
    p.add_argument("--fail_dice", type=float, default=DEFAULT_FAIL_DICE,
                   help=f"Eroded Dice below this is a failure "
                        f"(default {DEFAULT_FAIL_DICE}, the project's own threshold).")
    p.add_argument("--organs", nargs="*", default=list(EVAL_ORGANS),
                   help="Organs forming the denominator.")
    p.add_argument("--denominator", choices=["union", "labels"], default="union",
                   help="union (default): triples at least one condition "
                        "evaluated — a gap is then this method losing a triple "
                        "others got. labels: every (pair x organ) the labels CSV "
                        "implies, which also counts organs absent in the patient.")
    p.add_argument("--out_dir", default=None,
                   help="Default: compare_config.RESULTS_DIR.")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    study_ids = None
    if args.split != "all":
        from generate_split import load_split
        study_ids = load_split(args.split)
        print(f"split={args.split}: {len(study_ids)} studies")

    eligible = expected_triples(Path(args.labels_csv), args.ref_phase,
                                tuple(args.organs), study_ids)
    n_pairs = eligible[["study_id", "phase"]].drop_duplicates().shape[0]
    print(f"  label-derived: {n_pairs} (study, phase) pairs x "
          f"{len(args.organs)} organs = {len(eligible)} triples")
    if eligible.empty:
        print("Nothing eligible — check --split and labels.csv.")
        return 1

    print(f"failure: eroded Dice < {args.fail_dice}, or no output at all")

    tags = args.conditions if args.conditions else list(C.CONDITIONS)

    # Pass 1: load every condition that actually ran.
    details: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []
    for tag in tags:
        cond = C.CONDITIONS.get(tag)
        if cond is None:
            print(f"  unknown condition: {tag}")
            continue
        d = load_detail(cond, eligible)
        if d is None or d.empty:
            skipped.append(tag)
            continue
        details[tag] = d

    if not details:
        print("\nNo condition had an eval_detail.csv — "
              "run eval_compare_conditions.py first.")
        return 1

    # Pass 2: the denominator, then outcomes against it.
    if args.denominator == "union":
        denominator = comparable_triples(details)
        print(f"  denominator: union of triples any condition evaluated "
              f"= {len(denominator)} "
              f"({100.0 * len(denominator) / len(eligible):.0f}% of label-derived; "
              f"the remainder are organs no method could segment)")
    else:
        denominator = eligible
        print(f"  denominator: all label-derived triples = {len(denominator)}")

    outcomes: Dict[str, pd.DataFrame] = {}
    summaries: List[dict] = []
    for tag, d in details.items():
        out = condition_outcomes(d, tag, denominator, args.fail_dice)
        outcomes[tag] = out
        summaries.append(summarise(out, tag))

    summary_df = pd.DataFrame(summaries).sort_values("failure_pct")
    triples_df = pd.concat(outcomes.values(), ignore_index=True)
    fe_df = build_front_end_table(outcomes)

    summary_df.to_csv(out_dir / "failure_rates.csv", index=False)
    triples_df.to_csv(out_dir / "failure_triples.csv", index=False)
    if not fe_df.empty:
        fe_df.to_csv(out_dir / "front_end_effect.csv", index=False)

    tex_cols = [c for c in ("label", "n_expected", "coverage_pct", "no_output_pct",
                            "failure_pct", "dice_median") if c in summary_df.columns]
    (out_dir / "failure_rates.tex").write_text(
        _latex_table(summary_df[tex_cols], args.fail_dice))

    print("\n" + "=" * 86)
    print("FAILURE RATE PER CONDITION")
    print("=" * 86)
    show = [c for c in ("label", "n_expected", "coverage_pct", "no_output_pct",
                        "failure_pct", "failure_pct_of_evaluated", "dice_median")
            if c in summary_df.columns]
    print(summary_df[show].to_string(index=False))
    print("\n  failure_pct counts absent output as failure; "
          "failure_pct_of_evaluated ignores it.")
    print("  A large gap between the two means that condition's median is "
          "computed on a survivor subset.")

    if skipped:
        print(f"\nno eval_detail.csv — absent, NOT 0% failure ({len(skipped)}):")
        for t in skipped:
            print(f"    {t}")

    if not fe_df.empty:
        print("\n" + "=" * 86)
        print("WITH vs WITHOUT THE FRONT-END  (paired per study+phase+organ)")
        print("=" * 86)
        cols = [c for c in ("algorithm", "step", "n_triples",
                            "failure_pct_without", "failure_pct_with",
                            "failure_pct_delta", "rescued_b", "broken_c",
                            "mcnemar_p", "dice_mean_delta", "wilcoxon_p")
                if c in fe_df.columns]
        print(fe_df[cols].to_string(index=False))
        print("\n  failure_pct_delta < 0 means the step REDUCED the failure rate.")
        print("  McNemar is computed only on the discordant triples "
              "(rescued_b + broken_c); with few of those the p-value says little.")
    else:
        print("\nNo algorithm had two input tiers evaluated, so no paired "
              "contrast was computed.")

    print(f"\nwritten to {out_dir}:")
    print("  failure_rates.csv / .tex")
    print("  failure_triples.csv     (per-triple outcomes)")
    if not fe_df.empty:
        print("  front_end_effect.csv    (paired crop / z-align / front-end steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
