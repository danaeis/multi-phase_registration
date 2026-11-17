#!/usr/bin/env bash
# =============================================================================
# run_waw_tace.sh — WAW-TACE external validation: registration grid + eval
# =============================================================================
#
# Same structure as run_all_resume.sh (resumable, state-guarded, parallel
# watchers), with two deliberate differences:
#
#   1. MAIN_PATH points at the WAW-TACE root instead of vindr_ds, so every
#      stage runs the SAME code on the second cohort. That is the whole claim.
#
#   2. CUT CONDITION LIST. VinDr ran ~30 conditions; external validation only
#      needs the ones the argument rests on:
#
#        A1_crop_only, A2_zalign        the front-end gain — the headline
#        A3b/A4b/A5b_noalign            isolates z-align out of distribution
#        A5_pass012                     best rigid, cheap
#        A6_bspline                     the full proposed method
#        R_full, R_mmi                  proposed rigid objective vs standard
#        B2_deeds, B3_ants,
#        B5_unigradicon_io              the three that tied/won on VinDr
#
#      Dropped: R_organ, R_sobel, B4_vxm, B5_unigradicon (no IO).
#
#      The --input raw arm is KEPT for the B-series, deliberately. The headline
#      transfer claim is the raw -> aligned contrast, and the crop step is
#      raw -> baseline; without a raw tier neither can be re-tested on this
#      cohort, which is the entire reason for the run. The saving comes from
#      the cohort size instead (generate_split.py --sample N).
#
# EVALUATION SAMPLE
#   Every job below runs with --split test, so the test split IS the sample.
#   Draw it once, before starting, with a recorded seed:
#       python generate_split.py --sample 40 --sample_seed 7
#   That marks exactly 40 studies as "test" and writes splits/eval_sample.csv.
#   Because every condition reads the same split, all algorithms and all states
#   are measured on the identical studies — which is what keeps the comparison
#   paired, and is the only honest way to cut compute. Sampling per algorithm
#   would compare methods on different data.
#
# PREREQUISITES
#   export MAIN_PATH=/abs/path/to/waw_tace/derived/     # trailing slash, ABSOLUTE
#   labels.csv, splits/train_test_split.csv, ts_segmentations/,
#   aligned_volumes/, baseline_volumes/ all present under $MAIN_PATH
#
# USAGE
#   export MAIN_PATH=/data/waw_tace/derived/
#   tmux new-session -s wawtace \
#     'bash run_waw_tace.sh 2>&1 | tee run_logs_waw/master.log'
#
#   bash check_progress.sh            # progress
#   tail -f run_logs_waw/job_a_series_aligned.log
#
# RESUME: just re-run it. Completed steps are skipped via .run_state_waw.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── MAIN_PATH is mandatory and must be absolute ───────────────────────────────
if [[ -z "${MAIN_PATH:-}" ]]; then
    echo "ERROR: MAIN_PATH is not set."
    echo "  export MAIN_PATH=/abs/path/to/waw_tace/derived/"
    exit 2
fi
if [[ "${MAIN_PATH:0:1}" != "/" ]]; then
    echo "ERROR: MAIN_PATH must be ABSOLUTE ('$MAIN_PATH' is relative)."
    echo "  Stages run from different working directories; a relative root"
    echo "  resolves differently in each one."
    exit 2
fi
[[ "${MAIN_PATH}" != */ ]] && export MAIN_PATH="${MAIN_PATH}/"
if [[ ! -f "${MAIN_PATH}labels.csv" ]]; then
    echo "ERROR: ${MAIN_PATH}labels.csv not found — run prepare_waw_tace.py first."
    exit 2
fi
export MAIN_PATH

# ── DEEDS binaries ────────────────────────────────────────────────────────────
export DEEDS_BIN="${DEEDS_BIN:-$SCRIPT_DIR/deedsBCV/deedsBCV}"
export LINEAR_BIN="${LINEAR_BIN:-$SCRIPT_DIR/deedsBCV/linearBCV}"
export APPLYFLOAT_BIN="${APPLYFLOAT_BIN:-$SCRIPT_DIR/deedsBCV/applyBCVfloat}"

PYTHON="${PYTHON:-python3}"
LOG_DIR="$SCRIPT_DIR/run_logs_waw"
STATE_FILE="$SCRIPT_DIR/.run_state_waw"

POLL_SEC="${POLL_SEC:-60}"
WATCH_TIMEOUT_MIN="${WATCH_TIMEOUT_MIN:-1200}"

mkdir -p "$LOG_DIR"
touch "$STATE_FILE"

# ── Preflight: verify every engine BEFORE launching 7 parallel jobs ───────────
# Each job shells out to a binary or imports a package. Without this, a missing
# one fails all 79 pairs of that condition in turn and the failure surfaces
# hours later as an empty results directory, mixed in with jobs that worked.
# Checking costs a second. SKIP_PREFLIGHT=1 overrides for a deliberate
# partial run.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    pf=0
    echo "preflight:"

    for v in DEEDS_BIN LINEAR_BIN APPLYFLOAT_BIN; do
        b="${!v}"
        if [[ -x "$b" ]]; then
            echo "  ok    $v = $b"
        else
            echo "  FAIL  $v = $b  (missing or not executable)"
            echo "        deedsBCV is a git submodule and must be BUILT on this host:"
            echo "            git submodule update --init compare_registrtaion/deedsBCV"
            echo "        Linux builds are required — macOS arm64 binaries will not run."
            pf=1
        fi
    done

    if command -v antsRegistration >/dev/null 2>&1; then
        echo "  ok    antsRegistration = $(command -v antsRegistration)"
    else
        echo "  FAIL  antsRegistration not on PATH (job_ants needs it)"
        pf=1
    fi

    if $PYTHON -c "import unigradicon" >/dev/null 2>&1; then
        echo "  ok    python can import unigradicon"
    else
        echo "  FAIL  $PYTHON cannot import unigradicon (job_unigradicon_io needs it)"
        echo "        pip install unigradicon — and its weights are a server-only"
        echo "        download, so check they are present too."
        pf=1
    fi

    if $PYTHON -c "import SimpleITK, pandas, numpy, scipy" >/dev/null 2>&1; then
        echo "  ok    python core (SimpleITK, pandas, numpy, scipy)"
    else
        echo "  FAIL  $PYTHON is missing one of SimpleITK/pandas/numpy/scipy"
        echo "        A background shell does not inherit an interactive venv"
        echo "        activation — source it inside the launched command."
        pf=1
    fi

    if [[ $pf -ne 0 ]]; then
        echo ""
        echo "Refusing to start. Fix the above, or set SKIP_PREFLIGHT=1 to run"
        echo "only the jobs whose engines are present. Nothing was written."
        exit 2
    fi
    echo ""
fi

# ── helpers (identical to run_all_resume.sh) ──────────────────────────────────
ts()       { date '+%Y-%m-%d %H:%M:%S'; }
mark_done(){ echo "$1" >> "$STATE_FILE"; }
is_done()  { grep -qxF "$1" "$STATE_FILE" 2>/dev/null; }

hdr() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1  [$(ts)]"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

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
    hdr "A1 crop-only + A2 z-align  (the headline front-end gain)"
    if is_done "eval_a1_a2"; then echo "[$(ts)] SKIP eval_a1_a2"; return 0; fi
    if $PYTHON eval_compare_conditions.py \
            --conditions A1_crop_only A2_zalign \
            --split test >> "$LOG_DIR/a1a2_eval.log" 2>&1; then
        mark_done "eval_a1_a2"; echo "[$(ts)] DONE eval_a1_a2"
    else
        echo "[$(ts)] FAIL eval_a1_a2 (see run_logs_waw/a1a2_eval.log)"
    fi
}

# ── A-series aligned: rigid → A5 eval → deformable → A6 eval ─────────────────

job_a_series_aligned() {
    hdr "A-series ALIGNED  (register_fixed → deformable_v2 sobel_ncc)"

    local s_rigid="$LOG_DIR/.rigid_aligned_done"; rm -f "$s_rigid"
    local w_rigid
    w_rigid=$(start_watcher "$s_rigid" "$LOG_DIR/a345_aligned_eval.log" \
        A3_pass0 A4_pass01 A5_pass012 -- --split test)

    run_reg "rigid_aligned" "$LOG_DIR/rigid_aligned_reg.log" \
        $PYTHON ../registration/register_fixed.py --all --skip --split test || true

    touch "$s_rigid"; wait "$w_rigid"
    echo "[$(ts)] rigid aligned done + A3/A4/A5 evaluated"

    local s_deform="$LOG_DIR/.deform_aligned_done"; rm -f "$s_deform"
    local w_deform
    w_deform=$(start_watcher "$s_deform" "$LOG_DIR/a6_aligned_eval.log" \
        A6_bspline -- --split test)

    run_reg "a6_sobelncc" "$LOG_DIR/a6_aligned_reg.log" \
        $PYTHON ../registration/deformable_registration_v2.py \
            --metric sobel_ncc --all --skip --split test || true

    run_reg "a6_prepare" "$LOG_DIR/a6_aligned_prepare.log" \
        $PYTHON prepare_a6_bspline.py --metric sobel_ncc || true

    touch "$s_deform"; wait "$w_deform"
    echo "[$(ts)] deformable aligned done + A6_bspline evaluated"
}

# ── A-series baseline (no z-align): the contrast that isolates the front-end ──

job_a_series_baseline() {
    hdr "A-series BASELINE  (no z-align — isolates the front-end contribution)"

    local s_rigid_b="$LOG_DIR/.rigid_baseline_done"; rm -f "$s_rigid_b"
    local w_rigid_b
    w_rigid_b=$(start_watcher "$s_rigid_b" "$LOG_DIR/a345_baseline_eval.log" \
        A3b_pass0_noalign A4b_pass01_noalign A5b_pass012_noalign -- --split test)

    run_reg "rigid_baseline" "$LOG_DIR/rigid_baseline_reg.log" \
        $PYTHON ../registration/register_fixed.py --all --skip --baseline \
            --split test || true

    touch "$s_rigid_b"; wait "$w_rigid_b"
    echo "[$(ts)] rigid baseline done + A3b/A4b/A5b evaluated"

    local s_deform_b="$LOG_DIR/.deform_baseline_done"; rm -f "$s_deform_b"
    local w_deform_b
    w_deform_b=$(start_watcher "$s_deform_b" "$LOG_DIR/a6_baseline_eval.log" \
        A6_bspline_baseline -- --split test)

    run_reg "a6_sobelncc_baseline" "$LOG_DIR/a6_baseline_reg.log" \
        $PYTHON ../registration/deformable_registration_v2.py \
            --metric sobel_ncc --all --skip --baseline --split test || true

    run_reg "a6_prepare_baseline" "$LOG_DIR/a6_baseline_prepare.log" \
        $PYTHON prepare_a6_bspline.py --metric sobel_ncc --baseline || true

    touch "$s_deform_b"; wait "$w_deform_b"
    echo "[$(ts)] deformable baseline done + A6_bspline_baseline evaluated"
}

# ── Rigid objective comparison: proposed vs the standard intensity baseline ───
# Only full + mmi. R_organ / R_sobel are ablations of the objective itself,
# already settled on VinDr; they are not external-validity questions.

job_rigid_variants() {
    hdr "R_full (proposed) vs R_mmi (standard MI)  — aligned input"

    local sentinel="$LOG_DIR/.rigid_variants_done"; rm -f "$sentinel"
    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/rigid_variants_eval.log" \
        R_full__aligned R_mmi__aligned -- --split test)

    for metric in full mmi; do
        run_reg "rigid_${metric}" "$LOG_DIR/rigid_${metric}_reg.log" \
            $PYTHON run_rigid.py --metric "$metric" --input aligned \
                --split test || true
    done

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_rigid_variants complete"
}

# ── B-series: all three input tiers ──────────────────────────────────────────
# raw / baseline / aligned give the two front-end steps separately:
#   raw -> baseline  = the crop        (VinDr: carries the whole effect)
#   baseline -> aligned = z-align      (VinDr: contributes nothing)
#   raw -> aligned   = the front-end   (the headline claim)
# failure_rates.py tests each one paired per (study, phase, organ).

job_deeds() {
    hdr "B2 DEEDS  (raw + baseline + aligned)"
    local sentinel="$LOG_DIR/.deeds_reg_done"; rm -f "$sentinel"
    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/deeds_eval.log" \
        B2_deeds__aligned B2_deeds__baseline B2_deeds__raw -- --split test)

    run_reg "deeds_all" "$LOG_DIR/deeds_reg.log" \
        $PYTHON run_deeds.py --input all --split test || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_deeds complete"
}

job_ants() {
    hdr "B3 ANTs-SyN  (raw + baseline + aligned)"
    local sentinel="$LOG_DIR/.ants_reg_done"; rm -f "$sentinel"
    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/ants_eval.log" \
        B3_ants__aligned B3_ants__baseline B3_ants__raw -- --split test)

    run_reg "ants_all" "$LOG_DIR/ants_reg.log" \
        $PYTHON run_ants.py --input all --split test || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_ants complete"
}

job_unigradicon_io() {
    hdr "B5 uniGradICON + IO-50  (raw + baseline + aligned)"
    local sentinel="$LOG_DIR/.ugi_io_reg_done"; rm -f "$sentinel"
    local w; w=$(start_watcher "$sentinel" "$LOG_DIR/unigradicon_io_eval.log" \
        B5_unigradicon_io__aligned B5_unigradicon_io__baseline \
        B5_unigradicon_io__raw -- --split test)

    run_reg "ugi_io_all" "$LOG_DIR/unigradicon_io_reg.log" \
        $PYTHON run_unigradicon.py --input all --split test \
            --algo_tag B5_unigradicon_io --io_steps 50 || true

    touch "$sentinel"; wait "$w"
    echo "[$(ts)] job_unigradicon_io complete"
}

# ── Launch ────────────────────────────────────────────────────────────────────

echo "################################################################"
echo "#  WAW-TACE EXTERNAL VALIDATION  $(ts)"
echo "#"
echo "#  MAIN_PATH  = $MAIN_PATH"
echo "#  studies    = $(tail -n +2 "${MAIN_PATH}labels.csv" | cut -d, -f1 | sort -u | wc -l)"
echo "#  series     = $(tail -n +2 "${MAIN_PATH}labels.csv" | wc -l)"
echo "#  state      = $STATE_FILE"
echo "#  logs       = $LOG_DIR/"
echo "#  DEEDS_BIN  = $DEEDS_BIN"
echo "################################################################"
echo ""

job_a1_a2              > "$LOG_DIR/job_a1a2.log"              2>&1 &  PID_A1A2=$!
job_a_series_aligned   > "$LOG_DIR/job_a_series_aligned.log"  2>&1 &  PID_ALN=$!
job_a_series_baseline  > "$LOG_DIR/job_a_series_baseline.log" 2>&1 &  PID_BAS=$!
job_rigid_variants     > "$LOG_DIR/job_rigid_variants.log"    2>&1 &  PID_RV=$!
job_deeds              > "$LOG_DIR/job_deeds.log"             2>&1 &  PID_D=$!
job_ants               > "$LOG_DIR/job_ants.log"              2>&1 &  PID_A=$!
job_unigradicon_io     > "$LOG_DIR/job_ugi_io.log"            2>&1 &  PID_UI=$!

echo "Background PIDs:"
echo "  A1/A2 eval        = $PID_A1A2"
echo "  A-series aligned  = $PID_ALN"
echo "  A-series baseline = $PID_BAS"
echo "  Rigid full/mmi    = $PID_RV"
echo "  DEEDS             = $PID_D"
echo "  ANTs              = $PID_A"
echo "  uniGradICON IO-50 = $PID_UI"
echo ""
echo "Progress:  bash check_progress.sh"
echo "Waiting for all jobs ..."

FAIL=0
wait $PID_A1A2 || { echo "[$(ts)] WARN: A1/A2 eval non-zero";        FAIL=1; }
wait $PID_ALN  || { echo "[$(ts)] WARN: A-series aligned non-zero";  FAIL=1; }
wait $PID_BAS  || { echo "[$(ts)] WARN: A-series baseline non-zero"; FAIL=1; }
wait $PID_RV   || { echo "[$(ts)] WARN: Rigid variants non-zero";    FAIL=1; }
wait $PID_D    || { echo "[$(ts)] WARN: DEEDS non-zero";             FAIL=1; }
wait $PID_A    || { echo "[$(ts)] WARN: ANTs non-zero";              FAIL=1; }
wait $PID_UI   || { echo "[$(ts)] WARN: uniGradICON IO non-zero";    FAIL=1; }

echo ""
echo "[$(ts)] Building final WAW-TACE comparison table ..."
if $PYTHON eval_compare_conditions.py --all --split test \
        >> "$LOG_DIR/final_eval.log" 2>&1; then
    echo "[$(ts)] DONE: final table → $LOG_DIR/final_eval.log"
else
    echo "[$(ts)] FAIL: final table (see $LOG_DIR/final_eval.log)"; FAIL=1
fi

echo ""
echo "[$(ts)] Failure rates + paired with/without-front-end contrast ..."
if $PYTHON failure_rates.py --split test >> "$LOG_DIR/failure_rates.log" 2>&1; then
    echo "[$(ts)] DONE: failure rates → $LOG_DIR/failure_rates.log"
    tail -40 "$LOG_DIR/failure_rates.log"
else
    echo "[$(ts)] FAIL: failure rates (see $LOG_DIR/failure_rates.log)"; FAIL=1
fi

echo ""
echo "################################################################"
echo "#  FINISHED  $(ts)  (FAIL=$FAIL)"
echo "#  Results: ${MAIN_PATH}all_baseline_algorithms/"
echo "################################################################"
exit $FAIL
