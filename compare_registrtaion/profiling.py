"""
profiling.py — Per-registration wall time, peak RAM, and peak GPU memory.

Used by every baseline runner via _common.run_baseline() and by the
VoxelMorph training loop. Results are appended as JSONL then aggregated into
a summary CSV by aggregate_profiles.py.

Requirements:
    pip install psutil        # CPU / RAM monitoring
    pip install thop          # FLOPs (optional, only for PyTorch models)

Output (one JSONL per algo×input, in {RESULTS_DIR}/profiling/):
    profiling_B2_deeds_aligned.jsonl
    profiling_B3_ants_aligned.jsonl
    ...

Each line:
    {"algo": "B2_deeds", "input": "aligned", "study": "...", "phase": "...",
     "elapsed_sec": 12.4, "peak_ram_mb": 2048.0, "peak_gpu_mb": null,
     "cpu_count": 8}
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import torch as _torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Background RAM poller — captures peak RSS during blocking calls
# ---------------------------------------------------------------------------

class _MemPoller:
    """Poll the current process RSS every `interval` seconds; track peak."""

    def __init__(self, interval: float = 0.25):
        self._interval = interval
        self._peak_bytes: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if _PSUTIL:
            self._proc = psutil.Process(os.getpid())
            self._peak_bytes = self._proc.memory_info().rss

    def start(self) -> None:
        if not _PSUTIL:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        """Stop polling and return peak RSS in MB (None if psutil unavailable)."""
        if not _PSUTIL:
            return None
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_bytes / (1024 ** 2)

    def _run(self) -> None:
        while self._running:
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:
                break
            time.sleep(self._interval)


# ---------------------------------------------------------------------------
# Main context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def profile_stage(
    algo: str,
    input_key: str,
    study: str,
    phase: str,
    profiling_dir: Path,
    use_gpu: bool = False,
):
    """
    Context manager that measures wall time, peak RAM, and peak GPU memory
    for one registration call.

    Usage:
        with profile_stage("B3_ants", "aligned", study, phase, profiling_dir):
            result = register_ants(...)

    The context manager yields a mutable dict `record` so the caller can
    annotate it with extra fields (e.g. {"flops_g": 42.1}):
        with profile_stage(...) as rec:
            rec["grid_spacing_mm"] = 25
            result = register_bspline(...)
    """
    poller = _MemPoller()

    # Baseline GPU state
    gpu_available = use_gpu and _TORCH and _torch.cuda.is_available()
    if gpu_available:
        _torch.cuda.reset_peak_memory_stats()

    cpu_count = psutil.cpu_count(logical=False) if _PSUTIL else None

    record: dict = {}
    poller.start()
    t0 = time.perf_counter()

    try:
        yield record
    finally:
        elapsed = time.perf_counter() - t0
        peak_ram_mb = poller.stop()

        peak_gpu_mb = None
        if gpu_available:
            try:
                peak_gpu_mb = _torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                pass

        record.update({
            "algo":         algo,
            "input":        input_key,
            "study":        study,
            "phase":        phase,
            "elapsed_sec":  round(elapsed, 3),
            "peak_ram_mb":  round(peak_ram_mb, 1) if peak_ram_mb is not None else None,
            "peak_gpu_mb":  round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None,
            "cpu_count":    cpu_count,
        })

        _append_record(profiling_dir, algo, input_key, record)


# ---------------------------------------------------------------------------
# Training profiler (VoxelMorph / LapIRN — whole-run summary)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def profile_training(
    algo: str,
    loss_type: str,
    n_pairs: int,
    epochs: int,
    steps_per_epoch: int,
    profiling_dir: Path,
    gpu: int = 0,
):
    """
    Context manager for the full training run of a learning-based method.

    Measures total wall time and peak GPU memory; computes derived stats:
        time_per_epoch_sec = elapsed / epochs
        time_per_step_sec  = elapsed / (epochs * steps_per_epoch)
    """
    gpu_available = _TORCH and torch_cuda_ok(gpu)
    if gpu_available:
        import torch
        torch.cuda.reset_peak_memory_stats(gpu)

    poller = _MemPoller()
    poller.start()
    t0 = time.perf_counter()

    record: dict = {}
    try:
        yield record
    finally:
        elapsed = time.perf_counter() - t0
        peak_ram_mb = poller.stop()

        peak_gpu_mb = None
        if gpu_available:
            try:
                import torch
                peak_gpu_mb = torch.cuda.max_memory_allocated(gpu) / (1024 ** 2)
            except Exception:
                pass

        total_steps = epochs * steps_per_epoch
        record.update({
            "algo":                  algo,
            "loss_type":             loss_type,
            "mode":                  "train",
            "n_training_pairs":      n_pairs,
            "epochs":                epochs,
            "steps_per_epoch":       steps_per_epoch,
            "total_steps":           total_steps,
            "total_elapsed_sec":     round(elapsed, 1),
            "time_per_epoch_sec":    round(elapsed / max(epochs, 1), 2),
            "time_per_step_sec":     round(elapsed / max(total_steps, 1), 4),
            "peak_ram_mb":           round(peak_ram_mb, 1) if peak_ram_mb is not None else None,
            "peak_gpu_mb":           round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None,
        })

        _append_record(profiling_dir, f"{algo}_training", loss_type, record)


# ---------------------------------------------------------------------------
# FLOPs helper (PyTorch models only)
# ---------------------------------------------------------------------------

def measure_flops(model, input_shape: tuple, device, extra_inputs=()) -> Optional[float]:
    """
    Return GFLOPs for one forward pass of `model` on a single volume of
    `input_shape` (Z, Y, X). Returns None if thop is not installed.

    Usage:
        gflops = measure_flops(vxm_model, (192, 192, 192), device,
                               extra_inputs=(dummy_fixed,))
    """
    try:
        from thop import profile as thop_profile
        import torch
        dummy = torch.zeros(1, 1, *input_shape).to(device)
        inputs = (dummy,) + tuple(extra_inputs) + (False,)
        flops, params = thop_profile(model, inputs=inputs, verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        return None
    except Exception:
        return None


def report_model_stats(model, input_shape: tuple, device) -> dict:
    """
    Return a dict with param count (M) and GFLOPs for a PyTorch model.
    Printed to stdout and returned for saving.
    """
    import torch
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = measure_flops(model, input_shape, device,
                           extra_inputs=(torch.zeros(1, 1, *input_shape).to(device),))
    stats = {
        "params_M":  round(params_m, 3),
        "gflops":    round(gflops, 2) if gflops is not None else None,
        "input_shape": list(input_shape),
    }
    print(f"  Model stats: params={params_m:.2f}M  "
          f"GFLOPs={gflops:.1f}" if gflops is not None
          else f"  Model stats: params={params_m:.2f}M  (install thop for FLOPs)",
          flush=True)
    return stats


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def _append_record(profiling_dir: Path, algo: str, key: str, record: dict) -> None:
    profiling_dir.mkdir(parents=True, exist_ok=True)
    log_path = profiling_dir / f"profiling_{algo}_{key}.jsonl"
    with open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def torch_cuda_ok(gpu: int = 0) -> bool:
    if not _TORCH:
        return False
    try:
        import torch
        return torch.cuda.is_available() and gpu < torch.cuda.device_count()
    except Exception:
        return False
