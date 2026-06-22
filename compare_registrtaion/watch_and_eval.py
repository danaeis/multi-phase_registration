"""
watch_and_eval.py
=================
Incremental per-study evaluator that runs alongside registration.

Instead of waiting for all registration to finish and then evaluating
everything at once, this watcher polls the output directory every
--poll seconds. As soon as ALL moving-phase files for a study are on
disk (meaning that study's registration is complete), it calls
evaluate_study() once for that study, appends the rows to the
per-condition detail CSV, and marks the study done in memory.

Each study is evaluated exactly once — no wasted re-evaluation.

The watcher exits cleanly when:
  1. The --sentinel file exists  (registration process touched it on exit), AND
  2. No new complete studies remain.
If the sentinel never appears (e.g. registration crashed), pass --timeout
to exit after N minutes regardless.

Usage
-----
# Watch one condition:
python watch_and_eval.py \\
    --conditions B2_deeds__aligned \\
    --sentinel   run_logs/.deeds_reg_done

# Watch multiple conditions from one algo (they share the same sentinel):
python watch_and_eval.py \\
    --conditions B2_deeds__aligned B2_deeds__baseline B2_deeds__raw \\
    --sentinel   run_logs/.deeds_reg_done \\
    --poll 60

# Parallelise per-study evaluation:
python watch_and_eval.py \\
    --conditions B3_ants__aligned \\
    --sentinel   run_logs/.ants_reg_done \\
    --workers 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_EVAL_PKG = _HERE.parent / "evaluate_pipeline"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_EVAL_PKG))

import compare_config as C
import evaluate_registration as E


# ---------------------------------------------------------------------------
# Study-completeness check
# ---------------------------------------------------------------------------

def study_is_complete(
    study_id: str,
    base_dir: Path,
    cond: C.Condition,
    labels_df: pd.DataFrame,
    ref_phase: str,
) -> bool:
    """
    True only when the reference (NC) vol AND every expected moving-phase vol
    exist for this study.  Prevents evaluating a study mid-registration.
    """
    study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    if study_rows.empty:
        return False

    study_dir = base_dir / study_id
    if not study_dir.is_dir():
        return False

    for _, row in study_rows.iterrows():
        sid = row["SeriesInstanceUID"]
        vol = study_dir / f"{study_id}_{sid}{cond.vol_postfix}"
        if not vol.exists():
            return False

    return True


# ---------------------------------------------------------------------------
# Incremental CSV append (thread-safe single-writer)
# ---------------------------------------------------------------------------

def append_rows_to_csv(detail_csv: Path, rows: List[dict]) -> None:
    """Append new rows to an existing CSV, or create it if absent."""
    df_new = pd.DataFrame(rows)
    write_header = not detail_csv.exists()
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(detail_csv, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Per-condition watcher state
# ---------------------------------------------------------------------------

class ConditionWatcher:
    def __init__(
        self,
        cond: C.Condition,
        labels_df: pd.DataFrame,
        ref_phase: str,
        erosion_mm: float,
        workers: int,
        study_ids: Optional[Set[str]] = None,
    ):
        self.cond       = cond
        self.labels_df  = labels_df
        self.ref_phase  = ref_phase
        self.erosion_mm = erosion_mm
        self.workers    = workers
        self.study_ids  = study_ids   # None = all, else restrict to this set
        self.done: Set[str] = set()   # study_ids already evaluated

        # Load already-evaluated studies from an existing detail CSV
        if cond.detail_csv.exists():
            try:
                df_ex = pd.read_csv(cond.detail_csv, usecols=["study_id"])
                self.done = set(df_ex["study_id"].unique())
                print(f"  [{cond.tag}] Resuming: {len(self.done)} studies already in CSV")
            except Exception:
                pass

    def scan_and_evaluate(self) -> int:
        """
        Scan base_dir for studies that are complete but not yet evaluated.
        Evaluate each one and append to the detail CSV.
        Returns the number of new studies evaluated this pass.
        """
        base_dir = self.cond.base_dir
        if not base_dir.exists():
            return 0

        newly_evaluated = 0

        candidate_dirs = sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir()
            and d.name not in self.done
            and (self.study_ids is None or d.name in self.study_ids)
        )

        for study_id in candidate_dirs:
            if not study_is_complete(
                study_id, base_dir, self.cond, self.labels_df, self.ref_phase
            ):
                continue

            print(f"  [{self.cond.tag}] → evaluating {study_id[:55]}...", flush=True)
            try:
                rows = E.evaluate_study(
                    study_id=study_id,
                    study_dir=str(base_dir / study_id),
                    labels_df=self.labels_df,
                    seg_postfix=self.cond.seg_postfix,
                    vol_postfix=self.cond.vol_postfix,
                    ref_phase=self.ref_phase,
                    erosion_mm=self.erosion_mm,
                )
                if rows:
                    for r in rows:
                        r["condition"] = self.cond.tag
                    append_rows_to_csv(self.cond.detail_csv, rows)
                    print(
                        f"  [{self.cond.tag}] ✓ {study_id[:40]} "
                        f"({len(rows)} organ-phase pairs) → {self.cond.detail_csv.name}",
                        flush=True,
                    )
                else:
                    print(f"  [{self.cond.tag}] ⚠ {study_id[:40]} — no rows returned")

                self.done.add(study_id)
                newly_evaluated += 1

            except Exception as exc:
                import traceback
                print(f"  [{self.cond.tag}] ✗ {study_id[:40]}: {exc}")
                traceback.print_exc()
                # Don't add to done — retry next pass
        return newly_evaluated


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def watch_loop(
    watchers: List[ConditionWatcher],
    sentinel: Optional[Path],
    poll_sec: int,
    timeout_min: Optional[int],
) -> None:
    t_start = time.time()
    pass_n  = 0

    print(f"\nWatcher started: {len(watchers)} condition(s), poll={poll_sec}s")
    if sentinel:
        print(f"  sentinel: {sentinel}")
    if timeout_min:
        print(f"  timeout : {timeout_min} min")
    print()

    while True:
        pass_n += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] pass {pass_n} — scanning ...", flush=True)

        total_new = sum(w.scan_and_evaluate() for w in watchers)
        print(
            f"[{ts}] pass {pass_n} done: {total_new} new studies evaluated  "
            f"(totals: {', '.join(f'{w.cond.tag}={len(w.done)}' for w in watchers)})",
            flush=True,
        )

        # Exit when sentinel is present AND no new studies appeared this pass
        sentinel_here = sentinel is not None and sentinel.exists()
        if sentinel_here and total_new == 0:
            # One final scan to catch any race between reg finishing and sentinel touch
            extra = sum(w.scan_and_evaluate() for w in watchers)
            if extra:
                print(f"  Final catch-up: {extra} more studies evaluated")
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Sentinel found, no remaining studies — watcher exiting.")
            break

        if timeout_min is not None:
            elapsed_min = (time.time() - t_start) / 60.0
            if elapsed_min >= timeout_min:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Timeout ({timeout_min} min) reached — watcher exiting.")
                break

        time.sleep(poll_sec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Incrementally evaluate studies as registration files appear.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--conditions", nargs="+", required=True,
        help="Condition tag(s) from compare_config to watch (e.g. B2_deeds__aligned).",
    )
    p.add_argument(
        "--sentinel", default=None,
        help="Path to a sentinel file created by the registration process when done.\n"
             "The watcher exits once this file exists and all studies are evaluated.",
    )
    p.add_argument(
        "--poll", type=int, default=60,
        help="Polling interval in seconds (default: 60).",
    )
    p.add_argument(
        "--timeout", type=int, default=None,
        help="Exit after this many minutes regardless of sentinel (safety valve).",
    )
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--ref_phase",  default=C.REF_PHASE)
    p.add_argument("--erosion_mm", type=float, default=C.EROSION_MM)
    p.add_argument("--workers",    type=int,   default=1,
                   help="Parallel workers passed to evaluate_study (default: 1).")
    p.add_argument("--split", choices=["all", "train", "test"], default="all",
                   help="Evaluate only the train/test split (default: all).")
    args = p.parse_args()

    unknown = [t for t in args.conditions if t not in C.CONDITIONS]
    if unknown:
        raise SystemExit(
            f"Unknown condition(s): {unknown}\nValid: {list(C.CONDITIONS)}"
        )

    study_ids: Optional[Set[str]] = None
    if args.split != "all":
        from generate_split import load_split as _load_split
        study_ids = _load_split(args.split)
        print(f"  --split {args.split}: evaluating {len(study_ids)} studies only")

    labels_df = pd.read_csv(args.labels_csv)

    watchers = [
        ConditionWatcher(
            cond       = C.CONDITIONS[tag],
            labels_df  = labels_df,
            ref_phase  = args.ref_phase,
            erosion_mm = args.erosion_mm,
            workers    = args.workers,
            study_ids  = study_ids,
        )
        for tag in args.conditions
    ]

    sentinel = Path(args.sentinel) if args.sentinel else None

    watch_loop(
        watchers  = watchers,
        sentinel  = sentinel,
        poll_sec  = args.poll,
        timeout_min = args.timeout,
    )

    print("\nWatcher summary:")
    for w in watchers:
        print(f"  {w.cond.tag}: {len(w.done)} studies evaluated → {w.cond.detail_csv}")


if __name__ == "__main__":
    main()
