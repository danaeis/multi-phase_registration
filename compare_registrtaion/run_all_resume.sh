#!/usr/bin/env bash
# =============================================================================
# run_all_resume.sh — Resumable parallel orchestration for the thesis comparison
# =============================================================================
#
# Five algorithms run in parallel; each has a dedicated watcher process that
# evaluates each study as soon as its registration files are complete.
# Studies are evaluated exactly once — no re-scanning of already-done work.
#
# Jobs:
#   B2 DEEDS             — aligned + baseline + raw
#   B3 ANTs-SyN          — aligned + baseline + raw
#   B5 uniGradICON       — aligned + baseline + raw  (fast inference, no IO)
#   B5 uniGradICON IO-50 — aligned + baseline + raw  (50 instance-opt steps)
#   A6 B-spline deform.  — sobel_ncc, aligned input
#
# Safe to re-run: .run_state guards every registration step; the watcher
# resumes from whichever studies are already in the detail CSV.
#
# RECOMMENDED: launch inside tmux to survive SSH disconnects:
#   tmux new-session -s thesis 'bash run_all_resume.sh 2>&1 | tee run_logs/master.log'
#
# Monitor a job:
#   tail -f run_logs/job_deeds.log
#
# Reset one algorithm:
#   sed -i '/deeds/d' .run_state && bash run_all_resume.sh
#
# Reset everything:
#   rm -f .run_state && bash run_all_resume.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── DEEDS binaries ────────────────────────────────────────────────────────────
export DEEDS_BIN="${DEEDS_BIN:-$SCRIPT_DIR/deedsBCV/deedsBCV}"
export LINEAR_BIN="${LINEAR_BIN:-$SCRIPT_DIR/deedsBCV/linearBCV}"
export APPLYFLOAT_BIN="${APPLYFLOAT_BIN:-$SCRIPT_DIR/deedsBCV/applyBCVfloat}"

PYTHON="${PYTHON:-python3}"
LOG_DIR="$SCRIPT_DIR/run_logs"
STATE_FILE="$SCRIPT_DIR/.run_state"

# Watcher poll interval — lower = faster interim results, more CPU
POLL_SEC="${POLL_SEC:-60}"

# Safety timeout for watchers: exit after N minutes even without a sentinel
WATCH_TIMEOUT_MIN="${WATCH_TIMEOUT_MIN:-1200}"

mkdir -p "$LOG_DIR"
touch "$STATE_FILE"

# ── helpers ───────────────────────────────────────────────────────────────────
ts()       { date '+%Y-%m-%d %H:%M:%S'; }
mark_done(){ echo "$1" >> "$STATE_FILE"; }
is_done()  { grep -qxF "$1" "$STATE_FILE" 2>/dev/null; }

hdr() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1  [$(ts)]"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# run_reg <state_key> <log> <cmd...>
run_reg() {
    local key="$1" log="$2"; shift 2
    if is_done "reg_${key}"; then
        echo "[$(ts)] SKIP  reg_${key}"
        return 0
    fi
    echo "[$(ts)] START reg_${key}: $*"
    if "$@" >> "$log" 2>&1; then
        mark_done "reg_${key}"
        echo "[$(ts)] DONE  reg_${key}"
    else
        echo "[$(ts)] FAIL  reg_${key}  (see $log)"
        return 1
    fi
}

# ── Job functions — each spawns a registration subprocess and a watcher ───────
# Pattern:
#   1. Touch away any stale sentinel from a prior run.
#   2. Start the watcher (background) — it reads the sentinel path and waits.
#   3. Run registration (foreground, blocks) — touches sentinel on exit.
#   4. Wait for watcher to finish its final pass.
# The watcher sees the sentinel, does one last scan, then exits.

job_deeds() {
    hdr "B2 DEEDS  (aligned + baseline + raw)"
    local sentinel="$LOG_DIR/.deeds_reg_done"
    rm -f "$sentinel"

    $PYTHON watch_and_eval.py \
        --conditions B2_deeds__aligned B2_deeds__baseline B2_deeds__raw \
        --sentinel "$sentinel" \
        --poll "$POLL_SEC" \
        --timeout "$WATCH_TIMEOUT_MIN" \
        >> "$LOG_DIR/deeds_eval.log" 2>&1 &
    local watch_pid=$!

    run_reg "deeds_all" "$LOG_DIR/deeds_reg.log" \
        $PYTHON run_deeds.py --input all || true

    touch "$sentinel"
    wait $watch_pid
    echo "[$(ts)] job_deeds complete"
}

job_ants() {
    hdr "B3 ANTs-SyN  (aligned + baseline + raw)"
    local sentinel="$LOG_DIR/.ants_reg_done"
    rm -f "$sentinel"

    $PYTHON watch_and_eval.py \
        --conditions B3_ants__aligned B3_ants__baseline B3_ants__raw \
        --sentinel "$sentinel" \
        --poll "$POLL_SEC" \
        --timeout "$WATCH_TIMEOUT_MIN" \
        >> "$LOG_DIR/ants_eval.log" 2>&1 &
    local watch_pid=$!

    run_reg "ants_all" "$LOG_DIR/ants_reg.log" \
        $PYTHON run_ants.py --input all || true

    touch "$sentinel"
    wait $watch_pid
    echo "[$(ts)] job_ants complete"
}

job_unigradicon() {
    hdr "B5 uniGradICON  (fast inference, no IO)"
    local sentinel="$LOG_DIR/.ugi_reg_done"
    rm -f "$sentinel"

    $PYTHON watch_and_eval.py \
        --conditions B5_unigradicon__aligned B5_unigradicon__baseline B5_unigradicon__raw \
        --sentinel "$sentinel" \
        --poll "$POLL_SEC" \
        --timeout "$WATCH_TIMEOUT_MIN" \
        >> "$LOG_DIR/unigradicon_eval.log" 2>&1 &
    local watch_pid=$!

    run_reg "ugi_all" "$LOG_DIR/unigradicon_reg.log" \
        $PYTHON run_unigradicon.py --input all || true

    touch "$sentinel"
    wait $watch_pid
    echo "[$(ts)] job_unigradicon complete"
}

job_unigradicon_io() {
    hdr "B5 uniGradICON IO-50  (50 instance-optimisation steps)"
    # io_steps=50: after the network forward pass, 50 gradient-descent steps
    # fine-tune the displacement field for this specific image pair using the
    # GradICON loss.  ~3-5x slower than pure inference but better on hard cases.
    # Written to B5_unigradicon_io conditions (separate from the fast run).
    local sentinel="$LOG_DIR/.ugi_io_reg_done"
    rm -f "$sentinel"

    $PYTHON watch_and_eval.py \
        --conditions B5_unigradicon_io__aligned B5_unigradicon_io__baseline B5_unigradicon_io__raw \
        --sentinel "$sentinel" \
        --poll "$POLL_SEC" \
        --timeout "$WATCH_TIMEOUT_MIN" \
        >> "$LOG_DIR/unigradicon_io_eval.log" 2>&1 &
    local watch_pid=$!

    run_reg "ugi_io_all" "$LOG_DIR/unigradicon_io_reg.log" \
        $PYTHON run_unigradicon.py --input all \
            --algo_tag B5_unigradicon_io --io_steps 50 || true

    touch "$sentinel"
    wait $watch_pid
    echo "[$(ts)] job_unigradicon_io complete"
}

job_a6() {
    hdr "A6 B-spline deformable  (sobel_ncc, aligned input)"
    # deformable_registration_v2.py → deformable_registered_sobel_ncc/
    # prepare_a6_bspline.py hardlinks to all_baseline_algorithms/A6_bspline/
    # so compare_config CONDITIONS["A6_bspline"] finds the files.
    local sentinel="$LOG_DIR/.a6_reg_done"
    rm -f "$sentinel"

    $PYTHON watch_and_eval.py \
        --conditions A6_bspline \
        --sentinel "$sentinel" \
        --poll "$POLL_SEC" \
        --timeout "$WATCH_TIMEOUT_MIN" \
        >> "$LOG_DIR/a6_eval.log" 2>&1 &
    local watch_pid=$!

    run_reg "a6_sobelncc" "$LOG_DIR/a6_reg.log" \
        $PYTHON ../registration/deformable_registration_v2.py \
            --metric sobel_ncc --all --skip || true

    # Populate the compare_config expected location with hardlinks
    run_reg "a6_prepare" "$LOG_DIR/a6_prepare.log" \
        $PYTHON prepare_a6_bspline.py --metric sobel_ncc || true

    touch "$sentinel"
    wait $watch_pid
    echo "[$(ts)] job_a6 complete"
}

# ── Launch all jobs in parallel ────────────────────────────────────────────────

echo "################################################################"
echo "#  THESIS REGISTRATION — RESUME RUN  $(ts)"
echo "#"
echo "#  State file      : $STATE_FILE"
echo "#  Logs            : $LOG_DIR/"
echo "#  Poll interval   : ${POLL_SEC}s"
echo "#  Watcher timeout : ${WATCH_TIMEOUT_MIN} min"
echo "#"
echo "#  DEEDS_BIN       = $DEEDS_BIN"
echo "#  LINEAR_BIN      = $LINEAR_BIN"
echo "#  APPLYFLOAT_BIN  = $APPLYFLOAT_BIN"
echo "################################################################"
echo ""

job_deeds          > "$LOG_DIR/job_deeds.log"     2>&1 &  PID_D=$!
job_ants           > "$LOG_DIR/job_ants.log"      2>&1 &  PID_A=$!
job_unigradicon    > "$LOG_DIR/job_ugi.log"       2>&1 &  PID_U=$!
job_unigradicon_io > "$LOG_DIR/job_ugi_io.log"    2>&1 &  PID_UI=$!
job_a6             > "$LOG_DIR/job_a6.log"        2>&1 &  PID_A6=$!

echo "Background PIDs:"
echo "  DEEDS              = $PID_D    → tail -f $LOG_DIR/job_deeds.log"
echo "  ANTs               = $PID_A    → tail -f $LOG_DIR/job_ants.log"
echo "  uniGradICON        = $PID_U    → tail -f $LOG_DIR/job_ugi.log"
echo "  uniGradICON IO-50  = $PID_UI   → tail -f $LOG_DIR/job_ugi_io.log"
echo "  A6-bspline         = $PID_A6   → tail -f $LOG_DIR/job_a6.log"
echo ""
echo "Progress:  bash check_progress.sh"
echo ""
echo "Waiting for all jobs ..."

FAIL=0
wait $PID_D  || { echo "[$(ts)] WARN: DEEDS job non-zero";          FAIL=1; }
wait $PID_A  || { echo "[$(ts)] WARN: ANTs job non-zero";           FAIL=1; }
wait $PID_U  || { echo "[$(ts)] WARN: uniGradICON job non-zero";    FAIL=1; }
wait $PID_UI || { echo "[$(ts)] WARN: uniGradICON IO job non-zero"; FAIL=1; }
wait $PID_A6 || { echo "[$(ts)] WARN: A6 job non-zero";             FAIL=1; }

# ── Final aggregated comparison table ─────────────────────────────────────────
echo ""
echo "[$(ts)] Building final comparison table (all conditions) ..."
if $PYTHON eval_compare_conditions.py --all >> "$LOG_DIR/final_eval.log" 2>&1; then
    echo "[$(ts)] DONE: final table → $LOG_DIR/final_eval.log"
else
    echo "[$(ts)] FAIL: final table — see $LOG_DIR/final_eval.log"
    FAIL=1
fi

echo ""
echo "################################################################"
echo "#  FINISHED  $(ts)  (FAIL=$FAIL)"
echo "#  Results dir: $(python3 -c \"import sys; sys.path.insert(0,'$SCRIPT_DIR'); import compare_config as C; print(C.RESULTS_DIR)\" 2>/dev/null)"
echo "################################################################"
exit $FAIL
