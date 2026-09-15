"""
measure_resources.py — Re-measure wall time and peak memory for the two
registration methods whose original cost numbers are incomplete.

Why this exists
---------------
profiling.py samples the RSS of the *Python* process only. DEEDS runs as
external binaries (linearBCV, deedsBCV, applyBCVfloat), so its memory never
appeared in the profile, and its recorded time also included the dense
displacement-field reconstruction that exists only to measure folding.
The B-spline pass (registration/deformable_registration_v2.py) was timed by
its own logger and its memory was never recorded.

What it measures
----------------
For every phase pair:
  * wall time of the registration call, split into stages for DEEDS
    (linear pre-alignment, deformable, evaluation-only field reconstruction);
  * peak total resident memory of the WHOLE process tree — this process and
    every descendant — sampled every 0.1 s. For methods that run inside the
    Python process this equals the old per-process number; for DEEDS it is
    the only valid one.

Nothing is written into the evaluated result directories: DEEDS outputs are
discarded, and the B-spline pass writes to a temporary directory that is
deleted after each study.

Usage (on the server that ran the original experiments, ideally while idle)
--------------------------------------------------------------------------
    export DEEDS_BIN=.../deedsBCV LINEAR_BIN=.../linearBCV APPLYFLOAT_BIN=.../applyBCVfloat
    python measure_resources.py deeds   --input aligned --split test
    python measure_resources.py deeds   --input raw     --split test
    python measure_resources.py bspline --split test              # ~28 h for 26 studies
    python measure_resources.py bspline --split test --max-studies 6
    python measure_resources.py summary

Runs are resumable: pairs already present in the output CSV are skipped.

Outputs ({RESULTS_DIR}/profiling/)
    tree_B2_deeds_<input>.csv
    tree_A6_bspline_aligned.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import compare_config as C

BSPLINE_SCRIPT = HERE.parent / "registration" / "deformable_registration_v2.py"
OUT_DIR = C.RESULTS_DIR / "profiling"
SAMPLE_INTERVAL_S = 0.1

DEEDS_FIELDS = [
    "method", "input", "study", "phase", "series", "status",
    "wall_s", "registration_s", "linear_s", "deeds_s", "dvf_recon_s",
    "peak_tree_mb", "peak_tree_excl_recon_mb", "child_maxrss_cum_mb",
    "cpu_count", "loadavg_start", "loadavg_end", "host", "timestamp",
]
BSPLINE_FIELDS = [
    "method", "input", "study", "phase", "series", "status",
    "t_deform_s", "study_wall_s", "study_peak_tree_mb",
    "cpu_count", "loadavg_start", "loadavg_end", "host", "timestamp",
]


# ---------------------------------------------------------------------------
# Process-tree memory sampler
# ---------------------------------------------------------------------------

class TreeMemSampler:
    """Track the peak summed RSS of a process and all of its descendants.

    `set_stage(name)` attributes subsequent samples to a named stage so the
    peak can be reported per stage (e.g. excluding evaluation-only work).
    """

    def __init__(self, root_pid: int | None = None, interval: float = SAMPLE_INTERVAL_S):
        self.root = psutil.Process(root_pid or os.getpid())
        self.interval = interval
        self.peak_bytes = 0
        self.stage: str | None = None
        self.stage_peaks: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        try:
            procs = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            return 0
        total = 0
        for proc in procs:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _record(self) -> None:
        rss = self._sample()
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        if self.stage is not None and rss > self.stage_peaks.get(self.stage, 0):
            self.stage_peaks[self.stage] = rss

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval)

    def set_stage(self, name: str | None) -> None:
        self._record()
        self.stage = name

    def __enter__(self) -> "TreeMemSampler":
        self._record()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @staticmethod
    def mb(nbytes: int) -> float:
        return round(nbytes / 1024 ** 2, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loadavg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def _child_maxrss_mb() -> float:
    # Largest RSS of any terminated child so far; Linux reports kilobytes.
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)


def _host_fields() -> dict:
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "host": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(r["study"], r["series"]) for r in csv.DictReader(fh)}


def _append(path: Path, fields: list, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _study_filter(split: str, explicit) -> set | None:
    studies = set(explicit) if explicit else None
    if split != "all":
        from generate_split import load_split
        split_ids = load_split(split)
        studies = (studies & split_ids) if studies else split_ids
    return studies


# ---------------------------------------------------------------------------
# DEEDS
# ---------------------------------------------------------------------------

def measure_deeds(args) -> None:
    import pandas as pd
    import SimpleITK as sitk
    import _common as K
    import run_deeds as rd

    rd.RECON_DVF = not args.no_recon
    src = C.INPUTS[args.input]
    seg_dir = src.seg_dir if src.seg_dir is not None else src.dir
    labels_df = pd.read_csv(args.labels_csv)
    studies = _study_filter(args.split, args.studies)
    out_csv = OUT_DIR / f"tree_B2_deeds_{args.input}.csv"
    done = _done_keys(out_csv)

    state: dict = {"sampler": None, "times": {}}
    orig_run, orig_recon = rd._run, rd._reconstruct_dense_dvf

    def timed_run(cmd, timeout):
        sampler = state["sampler"]
        name = Path(cmd[0]).name
        in_recon = sampler.stage == "dvf_recon"
        stage = "dvf_recon" if in_recon else ("linear" if "linear" in name else "deeds")
        previous = sampler.stage
        sampler.set_stage(stage)
        t0 = time.perf_counter()
        try:
            return orig_run(cmd, timeout)
        finally:
            if not in_recon:
                state["times"][stage] = state["times"].get(stage, 0.0) + time.perf_counter() - t0
            sampler.set_stage(previous)

    def timed_recon(fixed, prefix, tmp):
        sampler = state["sampler"]
        previous = sampler.stage
        sampler.set_stage("dvf_recon")
        t0 = time.perf_counter()
        try:
            return orig_recon(fixed, prefix, tmp)
        finally:
            state["times"]["dvf_recon"] = time.perf_counter() - t0
            sampler.set_stage(previous)

    rd._run = timed_run
    rd._reconstruct_dense_dvf = timed_recon

    n_done = 0
    for item in K.iter_pairs(labels_df, src.dir, src.vol_postfix, src.seg_postfix,
                             seg_dir=seg_dir, flat=src.flat):
        if studies is not None and item["study"] not in studies:
            continue
        if (item["study"], item["mv_sid"]) in done:
            continue

        fixed = sitk.ReadImage(item["fixed_vol"])
        moving = sitk.ReadImage(item["moving_vol"])
        moving_seg = sitk.ReadImage(item["moving_seg"])

        state["times"] = {}
        load_start = _loadavg()
        status = "ok"
        with TreeMemSampler() as sampler:
            state["sampler"] = sampler
            sampler.set_stage("prepare")
            t0 = time.perf_counter()
            try:
                rd.register_deeds(fixed, moving, moving_seg, item)
            except Exception as exc:                      # keep measuring the rest
                status = f"error: {exc}"[:200]
            wall = time.perf_counter() - t0

        recon = state["times"].get("dvf_recon", 0.0)
        excl = max((v for k, v in sampler.stage_peaks.items() if k != "dvf_recon"), default=0)
        _append(out_csv, DEEDS_FIELDS, {
            "method": "B2_deeds", "input": args.input,
            "study": item["study"], "phase": item["phase"], "series": item["mv_sid"],
            "status": status,
            "wall_s": round(wall, 2),
            "registration_s": round(wall - recon, 2),
            "linear_s": round(state["times"].get("linear", 0.0), 2),
            "deeds_s": round(state["times"].get("deeds", 0.0), 2),
            "dvf_recon_s": round(recon, 2),
            "peak_tree_mb": TreeMemSampler.mb(sampler.peak_bytes),
            "peak_tree_excl_recon_mb": TreeMemSampler.mb(excl),
            "child_maxrss_cum_mb": _child_maxrss_mb(),
            "loadavg_start": load_start, "loadavg_end": _loadavg(),
            **_host_fields(),
        })
        n_done += 1
        print(f"  [{n_done}] {item['study'][-20:]} {item['phase']:<9} {status} "
              f"reg {wall - recon:.0f}s  peak {TreeMemSampler.mb(excl):.0f} MB", flush=True)
        del fixed, moving, moving_seg

    print(f"\nDEEDS ({args.input}): measured {n_done} new pairs -> {out_csv}")


# ---------------------------------------------------------------------------
# B-spline pass
# ---------------------------------------------------------------------------

def measure_bspline(args) -> None:
    import pandas as pd

    studies = _study_filter(args.split, args.studies)
    run_dir = C.MAIN / f"deformable_registered_{args.metric}"
    if not run_dir.exists():
        raise SystemExit(f"Evaluated B-spline run not found: {run_dir}")
    todo = sorted(p.name for p in run_dir.iterdir()
                  if p.is_dir() and (studies is None or p.name in studies))
    if args.max_studies:
        todo = todo[:args.max_studies]

    out_csv = OUT_DIR / "tree_A6_bspline_aligned.csv"
    done_studies = {study for study, _ in _done_keys(out_csv)}
    tmp_root = OUT_DIR / "_tmp_bspline"

    for i, study in enumerate(todo, 1):
        if study in done_studies:
            continue
        tmp = tmp_root / study
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        cmd = [sys.executable, str(BSPLINE_SCRIPT), study,
               "--metric", args.metric, "--sampling", str(args.sampling),
               "--iters", str(args.iters), "--grid", str(args.grid),
               "--out-dir", str(tmp)]

        load_start = _loadavg()
        t0 = time.perf_counter()
        with open(tmp_root / f"{study}.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=str(BSPLINE_SCRIPT.parent),
                                    stdout=log, stderr=subprocess.STDOUT)
            with TreeMemSampler(root_pid=proc.pid) as sampler:
                returncode = proc.wait()
        wall = time.perf_counter() - t0

        summary = tmp / f"batch_summary_{args.metric}.csv"
        phases = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
        status = "ok" if returncode == 0 and not phases.empty else f"exit {returncode}"
        if phases.empty:
            phases = pd.DataFrame([{"phase": "", "series_id": "", "t_deform_s": None}])
        for _, ph in phases.iterrows():
            _append(out_csv, BSPLINE_FIELDS, {
                "method": "A6_bspline", "input": "aligned",
                "study": study, "phase": ph.get("phase", ""), "series": ph.get("series_id", ""),
                "status": status,
                "t_deform_s": ph.get("t_deform_s"),
                "study_wall_s": round(wall, 1),
                "study_peak_tree_mb": TreeMemSampler.mb(sampler.peak_bytes),
                "loadavg_start": load_start, "loadavg_end": _loadavg(),
                **_host_fields(),
            })
        print(f"  [{i}/{len(todo)}] {study[-20:]} {status} wall {wall:.0f}s "
              f"peak {TreeMemSampler.mb(sampler.peak_bytes):.0f} MB", flush=True)
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nB-spline: results -> {out_csv}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(_args) -> None:
    import pandas as pd

    def describe(series: pd.Series) -> str:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "n/a"
        return f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}] (n={len(s)})"

    for path in sorted(OUT_DIR.glob("tree_*.csv")):
        df = pd.read_csv(path)
        ok = df[df.status == "ok"]
        print(f"\n{path.name}: {len(ok)} ok / {len(df)} rows, hosts={sorted(df.host.unique())}")
        cols = (["registration_s", "linear_s", "deeds_s", "dvf_recon_s",
                 "peak_tree_excl_recon_mb", "peak_tree_mb"]
                if "registration_s" in df else ["t_deform_s", "study_peak_tree_mb"])
        for col in cols:
            print(f"  {col:26s} {describe(ok[col])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="method", required=True)

    deeds = sub.add_parser("deeds", help="measure DEEDS per phase pair")
    deeds.add_argument("--input", choices=["aligned", "baseline", "raw"], default="aligned")
    deeds.add_argument("--no-recon", action="store_true",
                       help="skip the evaluation-only dense-field reconstruction")
    deeds.add_argument("--labels_csv", default=str(C.LABELS_CSV))

    bspline = sub.add_parser("bspline", help="measure the B-spline pass per study")
    bspline.add_argument("--metric", default="grad_ncc")
    bspline.add_argument("--sampling", type=float, default=0.15)
    bspline.add_argument("--iters", type=int, default=300)
    bspline.add_argument("--grid", type=float, default=25.0)
    bspline.add_argument("--max-studies", type=int, default=None)
    bspline.add_argument("--keep", action="store_true", help="keep temporary outputs")

    sub.add_parser("summary", help="print time and memory summaries")

    for p in (deeds, bspline):
        p.add_argument("--split", choices=["all", "train", "test"], default="test")
        p.add_argument("--studies", nargs="*", default=None)

    args = parser.parse_args()
    {"deeds": measure_deeds, "bspline": measure_bspline, "summary": summarize}[args.method](args)


if __name__ == "__main__":
    main()
