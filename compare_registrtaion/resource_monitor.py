"""
resource_monitor.py
===================
Lightweight context manager that measures wall time, peak process RSS,
and peak GPU VRAM allocation for a block of code.

Usage::

    from resource_monitor import ResourceMonitor, append_resource_row
    with ResourceMonitor() as m:
        some_heavy_work()
    print(m.wall_time_s, m.cpu_peak_mb, m.gpu_peak_mb)
    append_resource_row(csv_path, {"study_id": sid, "condition": tag, **m.as_dict})

Notes
-----
- CPU RSS is sampled every `poll_interval` seconds in a daemon thread via psutil.
  If psutil is absent, cpu_peak_mb is 0.
- GPU VRAM uses torch.cuda.max_memory_allocated() if torch+CUDA are available.
  For CPU-only methods (DEEDS, ANTs) gpu_peak_mb is 0 — that is expected.
- Subprocess memory (external binaries like deedsBCV) is NOT captured by RSS;
  wall_time_s is the reliable metric for those methods.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ResourceMonitor:
    """
    Context manager.  After __exit__, access:
        .wall_time_s  — elapsed wall-clock seconds
        .cpu_peak_mb  — peak RSS of this Python process (MB)
        .gpu_peak_mb  — peak CUDA allocated memory (MB)
        .as_dict      — dict with all three (rounded)
    """

    def __init__(self, poll_interval: float = 0.25):
        self._poll = poll_interval
        self.wall_time_s: float = 0.0
        self.cpu_peak_mb: float = 0.0
        self.gpu_peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ResourceMonitor":
        self._t0 = time.perf_counter()
        self._stop.clear()
        self.cpu_peak_mb = 0.0
        self.gpu_peak_mb = 0.0

        if _HAS_TORCH and _torch.cuda.is_available():
            _torch.cuda.reset_peak_memory_stats()

        if _HAS_PSUTIL:
            self._proc = _psutil.Process()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.wall_time_s = time.perf_counter() - self._t0
        if _HAS_TORCH and _torch.cuda.is_available():
            self.gpu_peak_mb = _torch.cuda.max_memory_allocated() / 1024 ** 2

    def _poll_loop(self):
        while not self._stop.wait(timeout=self._poll):
            try:
                rss = self._proc.memory_info().rss / 1024 ** 2
                if rss > self.cpu_peak_mb:
                    self.cpu_peak_mb = rss
            except Exception:
                break

    @property
    def as_dict(self) -> dict:
        return {
            "wall_time_s": round(self.wall_time_s, 3),
            "cpu_peak_mb": round(self.cpu_peak_mb, 1),
            "gpu_peak_mb": round(self.gpu_peak_mb, 1),
        }


def append_resource_row(csv_path: Path, row: dict) -> None:
    """Append one row to a resource CSV, creating the file + header if absent."""
    import pandas as pd
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
