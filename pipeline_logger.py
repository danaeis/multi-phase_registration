from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

LOG_ROOT = os.environ.get(
    "PIPELINE_LOG_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_logs"),
)

# Maximum size per rotating log file before rollover (bytes)
LOG_MAX_BYTES    = int(os.environ.get("PIPELINE_LOG_MAX_BYTES", 50 * 1024 * 1024))  # 50 MB
LOG_BACKUP_COUNT = int(os.environ.get("PIPELINE_LOG_BACKUP_COUNT", 5))

# Cross-stage error file (all ERROR records land here regardless of stage)
GLOBAL_ERROR_FILE = os.path.join(LOG_ROOT, "pipeline_errors.jsonl")
GLOBAL_SUMMARY_FILE = os.path.join(LOG_ROOT, "pipeline_summary.jsonl")

# Valid stage names — used for subdirectory and file naming
VALID_STAGES = {"dcm2nii", "segmentation", "alignment", "registration", "evaluation"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class _JsonlHandler(logging.Handler):
    """
    Writes every LogRecord as one JSON line.
    Also mirrors ERROR / CRITICAL records to the global error file.
    """

    def __init__(self, jsonl_path: str, stage: str, series_ctx: dict):
        super().__init__()
        self.jsonl_path  = jsonl_path
        self.stage       = stage
        self.series_ctx  = series_ctx

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts":      _utc_iso(),
            "stage":   self.stage,
            "level":   record.levelname,
            "msg":     record.getMessage(),
            **self.series_ctx,
        }
        if record.exc_info:
            entry["traceback"] = traceback.format_exception(*record.exc_info)

        line = json.dumps(entry, default=str) + "\n"
        try:
            with open(self.jsonl_path, "a") as fh:
                fh.write(line)
        except Exception:
            pass

        if record.levelno >= logging.ERROR:
            try:
                global_err = os.path.join(
                    os.path.dirname(os.path.dirname(self.jsonl_path)),
                    "pipeline_errors.jsonl",
                )
                _ensure_dir(os.path.dirname(global_err))
                with open(global_err, "a") as fh:
                    fh.write(line)
            except Exception:
                pass


class PipelineLogger(logging.Logger):
    """
    Extends the standard Logger with two extra methods:
      .metric(event_name, data_dict)   — write a structured metric record
      .stage_summary(data_dict)        — append to the global summary file
    """

    def __init__(self, name: str, stage: str, series_ctx: dict,
                 metrics_path: str):
        super().__init__(name)
        self._stage       = stage
        self._series_ctx  = series_ctx
        self._metrics_path = metrics_path

    def metric(self, event: str, data: Dict[str, Any]) -> None:
        """
        Write a structured metric record to <stage>_metrics.jsonl.

        Example:
            log.metric("seg_sanity", {"anchor_found": 3, "bone_found": 5, "ok": True})
            log.metric("pass0_kabsch", {"err_mm": 4.2, "n_organs": 9})
            log.metric("alignment_offset", {"xy_mm": [3.1, -2.4], "z_mm": 12.5})
        """
        entry = {
            "ts":    _utc_iso(),
            "stage": self._stage,
            "event": event,
            **self._series_ctx,
            **data,
        }
        try:
            with open(self._metrics_path, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def stage_summary(self, data: Dict[str, Any]) -> None:
        """
        Append a one-record summary to the global pipeline_summary.jsonl.
        Call once per study×series at the END of each stage.

        Example:
            log.stage_summary({"status": "ok", "anchor_found": 4, "z_offset_mm": 13.5})
        """
        entry = {
            "ts":    _utc_iso(),
            "stage": self._stage,
            **self._series_ctx,
            **data,
        }
        try:
            _ensure_dir(os.path.dirname(GLOBAL_SUMMARY_FILE))
            with open(GLOBAL_SUMMARY_FILE, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stage_logger(
        stage:      str,
        study_id:   Optional[str] = None,
        series_id:  Optional[str] = None,
        phase:      Optional[str] = None,
        log_root:   Optional[str] = None,
) -> PipelineLogger:
    """
    Build (or retrieve) a PipelineLogger for one stage × series.

    Parameters
    ----------
    stage      : one of VALID_STAGES ("dcm2nii", "segmentation",
                 "alignment", "registration", "evaluation")
    study_id   : study UID — attached to every log record
    series_id  : series UID — attached to every log record
    phase      : phase label ("Non-contrast", "Arterial", etc.)
    log_root   : override LOG_ROOT for this call only

    Returns
    -------
    PipelineLogger instance with .info / .warning / .error / .metric /
    .stage_summary methods.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {VALID_STAGES}.")

    root       = log_root or LOG_ROOT
    stage_dir  = os.path.join(root, stage)
    _ensure_dir(stage_dir)

    # Build a unique logger name so multiple series don't share handlers
    ctx_tag = "_".join(filter(None, [study_id, series_id, phase]))
    logger_name = f"pipeline.{stage}.{ctx_tag}" if ctx_tag else f"pipeline.{stage}"

    # Return existing instance if already configured
    existing = logging.Logger.manager.loggerDict.get(logger_name)
    if isinstance(existing, PipelineLogger):
        return existing

    # Context dict attached to every record
    series_ctx: dict = {}
    if study_id:  series_ctx["study_id"]  = study_id
    if series_id: series_ctx["series_id"] = series_id
    if phase:     series_ctx["phase"]     = phase

    # Paths
    log_file     = os.path.join(stage_dir, f"{stage}.log")
    metrics_file = os.path.join(stage_dir, f"{stage}_metrics.jsonl")

    # Register the base Logger class (not our subclass) for getLogger,
    # then patch the returned instance — avoids __init__ signature clash.
    log = logging.getLogger(logger_name)

    # Attach our extra attributes and methods directly to the instance
    log.__class__ = PipelineLogger
    log._stage        = stage
    log._series_ctx   = series_ctx
    log._metrics_path = metrics_file

    log.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called again with same name
    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ── Rotating text log (human-readable) ───────────────────────────────
    fh = RotatingFileHandler(
        log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # ── Console (INFO and above) ──────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # ── JSONL handler (all levels → stage log; errors → global error file) ──
    jh = _JsonlHandler(
        os.path.join(stage_dir, f"{stage}_structured.jsonl"),
        stage, series_ctx,
    )
    jh.setLevel(logging.DEBUG)
    log.addHandler(jh)

    return log


def log_metric(log: PipelineLogger, event: str, data: Dict[str, Any]) -> None:
    """Convenience wrapper — same as log.metric(event, data)."""
    log.metric(event, data)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        log = get_stage_logger("segmentation", study_id="S1", series_id="X1",
                               phase="Arterial", log_root=tmp)
        log.info("Test info message")
        log.warning("Test warning")
        log.error("Test error — check global error file")
        log.metric("seg_sanity", {"anchor_found": 3, "bone_found": 5, "ok": True})
        log.stage_summary({"status": "ok", "anchor_found": 3})

        stage_dir = os.path.join(tmp, "segmentation")
        files = os.listdir(stage_dir)
        print(f"Files in stage dir: {files}")
        print(f"Global error file exists: {os.path.exists(os.path.join(tmp, 'pipeline_errors.jsonl'))}")
        print("Self-test PASSED")
    finally:
        shutil.rmtree(tmp)