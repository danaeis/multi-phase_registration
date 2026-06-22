#!/usr/bin/env bash
# =============================================================================
# run_all_resume.sh — Resumable parallel orchestration for the thesis comparison
# =============================================================================
#
# Parallel top-level jobs:
#
#   job_a1_a2              eval A1/A2 immediately (data already on disk)
#   job_a_series_aligned   rigid (aligned) → A3/A4/A5 eval →
#                            deformable (aligned) → A6_bspline eval
#   job_a_series_baseline  rigid (baseline) → A3b/A4b/A5b eval →
#                            deformable (baseline) → A6_bspline_baseline eval
#   job_deeds              DEEDS all inputs → watcher eval
#   job_ants               ANTs  all inputs → watcher eval
#   job_unigradicon        uniGradICON (no IO) → watcher eval
#   job_unigradicon_io     uniGradICON (IO-50) → watcher eval
#
# Within job_a_series_*: rigid and deformable run SEQUENTIALLY (deformable
# needs rigid output). The two A-series jobs run in PARALLEL with each other
# and with all B-series jobs.
#
# Each registration step is guarded by .run_state; watchers resume from
# already-evaluated studies in eval_detail.csv.
#
# RECOMMENDED:
#   # Clear the false A6 state from the previous run first:
#   sed -i '/^reg_a6/d' .run_state
#
#   # Then launch inside tmux:
#   tmux new-session -s thesis 'bash run_all_resume.sh 2>&1 | tee run_logs/master.log'
#
# Monitor:
#   tail -f run_logs/job_a_series_aligned.log
#   bash check_progress.sh
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

POLL_SEC="${POLL_SEC:-60}"
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

# run_reg <state_key> <log> <cmd ...>
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

# start_watcher <sentinel> <log> <conditions...> [-- <extra_args...>]
# Conditions are everything before '--'; extra args (e.g. --split test) after '--'.
# Launches watch_and_eval.py in the background; prints its PID.
start_watcher() {
    local sentinel="$1" log="$2"; shift 2
    local conditions=() extra=() in_extra=0
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then in_extra=1; continue; fi
        if [[ $in_extra -eq 0 ]]; then conditions+=("$arg")
        else extra+=("$arg"); fi
    done
    $PYTHON watch_and_eval.py \
        --conditions "${conditions[@]}" \
        --sentinel  "$sentinel" \
        --poll      "$POLL_SEC" \
        --timeout   "$WATCH_TIMEOUT_MIN" \
        "${extra[@]}" \
        >> "$log" 2>&1 &
    echo $!
}

# ── A1 / A2: static data already on disk — eval immediately ──────────────────

job_a1_a2() {
    hdr "A1 crop-only + A2 z-align  (data on disk, eval now on test split)"
    if is_done "eval_a1_a2"; then
        echo "[$(ts)] SKIP  eval_a1_a2"
        return 0
    fi
    echo "[$(ts)] START eval A1_crop_only + A2_zalign (test split)"
    if $PYTHON eval_compare_conditions.py \
            --conditions A1_crop_only A2_zalign \
            --split test \
            >> "$LOG_DIR/a1a2_eval.log" 2>&1; then
        mark_done "eval_a1_a2"
        echo "[$(ts)] DONE  eval_a1_a2"
    else
        echo "[$(ts)] FAIL  eval_a1_a2  (see run_logs/a1a2_eval.log)"
    fi
}

# ── A-series aligned: rigid → eval A3/A4/A5 → deformable → eval A6 ───────────

job_a_series_aligned() {
    hdr "A-series ALIGNED  (register_fixed → deformable_v2, sobel_ncc)"

    # ── Stage 1: rigid registration (aligned, test split only) ──────────
    local s_rigid="$LOG_DIR/.rigid_aligned_done"
    rm -f "$s_rigid"

    local w_rigid
    w_rigid=$(start_watcher "$s_rigid" "$LOG_DIR/a345_aligned_eval.log" \
        A3_pass0 A4_pass01 A5_pass012 -- --split test)

    run_reg "rigid_aligned" "$LOG_DIR/rigid_aligned_reg.log" \
        $PYTHON ../registration/register_fixed.py --all --skip --split test || true

    touch "$s_rigid"
    wait "$w_rigid"
    echo "[$(ts)] rigid aligned done + A3/A4/A5 evaluated (test split)"

    # ── Stage 2: deformable registration (aligned, sobel_ncc, test split) ─
    local s_deform="$LOG_DIR/.deform_aligned_done"
    rm -f "$s_deform"

    local w_deform
    w_deform=$(start_watcher "$s_deform" "$LOG_DIR/a6_aligned_eval.log" \
        A6_bspline -- --split test)

    run_reg "a6_sobelncc" "$LOG_DIR/a6_aligned_reg.log" \
        $PYTHON ../registration/deformable_registration_v2.py \
            --metric sobel_ncc --all --skip --split test || true

    run_reg "a6_prepare" "$LOG_DIR/a6_aligned_prepare.log" \
        $PYTHON prepare_a6_bspline.py --metric sobel_ncc || true

    touch "$s_deform"
    wait "$w_deform"
    echo "[$(ts)] deformable aligned done + A6_bspline evaluated (test split)"
}

# ── A-series baseline: rigid(baseline) → eval A3b/A4b/A5b → deform(baseline) → eval A6b

job_a_series_baseline() {
    hdr "A-series BASELINE  (register_fixed --baseline → deformable_v2 --baseline)"

    # ── Stage 1: rigid registration (baseline, test split only) ─────────
    local s_rigid_b="$LOG_DIR/.rigid_baseline_done"
    rm -f "$s_rigid_b"

    local w_rigid_b
    w_rigid_b=$(start_watcher "$s_rigid_b" "$LOG_DIR/a345_baseline_eval.log" \
        A3b_pass0_noalign A4b_pass01_noalign A5b_pass012_noalign -- --split test)

    run_reg "rigid_baseline" "$LOG_DIR/rigid_baseline_reg.log" \
        $PYTHON ../registration/register_fixed.py --all --skip --baseline \
            --split test || true

    touch "$s_rigid_b"
    wait "$w_rigid_b"
    echo "[$(ts)] rigid baseline done + A3b/A4b/A5b evaluated (test split)"

    # ── Stage 2: deformable registration (baseline, sobel_ncc, test split) ─
    local s_deform_b="$LOG_DIR/.deform_baseline_done"
    rm -f "$s_deform_b"

    local w_deform_b
    w_deform_b=$(start_watcher "$s_deform_b" "$LOG_DIR/a6_baseline_eval.log" \
        A6_bspline_baseline -- --split test)

    run_reg "a6_sobelncc_baseline" "$LOG_DIR/a6_baseline_reg.log" \
        $PYTHON ../registration/deformable_registration_v2.py \
            --metric sobel_ncc --all --skip --baseline --split test || true

    run_reg "a6_prepare_baseline" "$LOG_DIR/a6_baseline_prepare.log" \
        $PYTHON prepare_a6_bspline.py --metric sobel_ncc --baseline || true

    touch "$s_deform_b"
    wait "$w_deform_b"
    echo "[$(ts)] deformable baseline done + A6_bspline_baseline evaluated (test split)"
}

# ── B-series: all independent, each has its own watcher ──────────────────────

job_deeds() {
    hdr "B2 DEEDS  (aligned + baseline + raw — test split)"
    local sentinel="$LOG_DIR/.deeds_reg_done"
    rm -f "$sentinel"

    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/deeds_eval.log" \
        B2_deeds__aligned B2_deeds__baseline B2_deeds__raw -- --split test)

    run_reg "deeds_all" "$LOG_DIR/deeds_reg.log" \
        $PYTHON run_deeds.py --input all --split test || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_deeds complete"
}

job_ants() {
    hdr "B3 ANTs-SyN  (aligned + baseline + raw — test split)"
    local sentinel="$LOG_DIR/.ants_reg_done"
    rm -f "$sentinel"

    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/ants_eval.log" \
        B3_ants__aligned B3_ants__baseline B3_ants__raw -- --split test)

    run_reg "ants_all" "$LOG_DIR/ants_reg.log" \
        $PYTHON run_ants.py --input all --split test || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_ants complete"
}

job_unigradicon() {
    hdr "B5 uniGradICON  (fast inference — test split)"
    local sentinel="$LOG_DIR/.ugi_reg_done"
    rm -f "$sentinel"

    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/unigradicon_eval.log" \
        B5_unigradicon__aligned B5_unigradicon__baseline B5_unigradicon__raw \
        -- --split test)

    run_reg "ugi_all" "$LOG_DIR/unigradicon_reg.log" \
        $PYTHON run_unigradicon.py --input all --split test || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_unigradicon complete"
}

job_unigradicon_io() {
    hdr "B5 uniGradICON IO-50  (test split — instance optimisation)"
    local sentinel="$LOG_DIR/.ugi_io_reg_done"
    rm -f "$sentinel"

    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/unigradicon_io_eval.log" \
        B5_unigradicon_io__aligned B5_unigradicon_io__baseline \
        B5_unigradicon_io__raw -- --split test)

    run_reg "ugi_io_all" "$LOG_DIR/unigradicon_io_reg.log" \
        $PYTHON run_unigradicon.py --input all --split test \
            --algo_tag B5_unigradicon_io --io_steps 50 || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_unigradicon_io complete"
}

# ── Launch all top-level jobs in parallel ─────────────────────────────────────

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
echo "TIP: if re-running after a partial run, first clear stale A6 state:"
echo "  sed -i '/^reg_a6/d' .run_state"
echo ""

job_a1_a2              > "$LOG_DIR/job_a1a2.log"             2>&1 &  PID_A1A2=$!
job_a_series_aligned   > "$LOG_DIR/job_a_series_aligned.log" 2>&1 &  PID_ALN=$!
job_a_series_baseline  > "$LOG_DIR/job_a_series_baseline.log" 2>&1 & PID_BAS=$!
job_deeds              > "$LOG_DIR/job_deeds.log"            2>&1 &  PID_D=$!
job_ants               > "$LOG_DIR/job_ants.log"             2>&1 &  PID_A=$!
job_unigradicon        > "$LOG_DIR/job_ugi.log"              2>&1 &  PID_U=$!
job_unigradicon_io     > "$LOG_DIR/job_ugi_io.log"           2>&1 &  PID_UI=$!

echo "Background PIDs:"
echo "  A1/A2 eval          = $PID_A1A2  → tail -f $LOG_DIR/job_a1a2.log"
echo "  A-series aligned    = $PID_ALN   → tail -f $LOG_DIR/job_a_series_aligned.log"
echo "  A-series baseline   = $PID_BAS   → tail -f $LOG_DIR/job_a_series_baseline.log"
echo "  DEEDS               = $PID_D     → tail -f $LOG_DIR/job_deeds.log"
echo "  ANTs                = $PID_A     → tail -f $LOG_DIR/job_ants.log"
echo "  uniGradICON         = $PID_U     → tail -f $LOG_DIR/job_ugi.log"
echo "  uniGradICON IO-50   = $PID_UI    → tail -f $LOG_DIR/job_ugi_io.log"
echo ""
echo "Progress:  bash check_progress.sh"
echo ""
echo "Waiting for all jobs ..."

FAIL=0
wait $PID_A1A2 || { echo "[$(ts)] WARN: A1/A2 eval non-zero";           FAIL=1; }
wait $PID_ALN  || { echo "[$(ts)] WARN: A-series aligned non-zero";     FAIL=1; }
wait $PID_BAS  || { echo "[$(ts)] WARN: A-series baseline non-zero";    FAIL=1; }
wait $PID_D    || { echo "[$(ts)] WARN: DEEDS non-zero";                FAIL=1; }
wait $PID_A    || { echo "[$(ts)] WARN: ANTs non-zero";                 FAIL=1; }
wait $PID_U    || { echo "[$(ts)] WARN: uniGradICON non-zero";          FAIL=1; }
wait $PID_UI   || { echo "[$(ts)] WARN: uniGradICON IO non-zero";       FAIL=1; }

# ── Final comparison table ────────────────────────────────────────────────────
echo ""
echo "[$(ts)] Building final comparison table (all conditions, test split) ..."
if $PYTHON eval_compare_conditions.py --all --split test >> "$LOG_DIR/final_eval.log" 2>&1; then
    echo "[$(ts)] DONE: final table → $LOG_DIR/final_eval.log"
else
    echo "[$(ts)] FAIL: final table (see $LOG_DIR/final_eval.log)"
    FAIL=1
fi

echo ""
echo "################################################################"
echo "#  FINISHED  $(ts)  (FAIL=$FAIL)"
echo "#  Results: ncct_cect/vindr_ds/all_baseline_algorithms/"
echo "################################################################"
exit $FAIL
