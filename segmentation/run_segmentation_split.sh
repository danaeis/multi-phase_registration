#!/usr/bin/env bash
# =============================================================================
# run_segmentation_split.sh — TotalSegmentator + combine_masks over one cohort,
#                             optionally restricted to a train/test split.
# =============================================================================
#
# Differs from run_segmentation.sh in four ways, each of which matters when the
# cohort is large enough that segmentation is the long pole:
#
#   1. Paths come from $MAIN_PATH, so the same script serves any cohort
#      (vindr_ds, waw_tace/derived, ...) instead of hardcoding one.
#
#   2. --split test restricts the run to the studies in
#      splits/train_test_split.csv. With a sampled evaluation set this is the
#      difference between ~120 series and ~600: everything downstream filters
#      on the same split, so segmenting the rest buys nothing until the sample
#      is widened.
#
#   3. It RESUMES. A series whose _seg_reg.nii.gz already exists is skipped, so
#      an interrupted run costs only the series it was mid-way through.
#
#   4. combine_masks.py exiting non-zero does NOT abort the loop. Exit 2 means
#      the sanity check found seg_reg too sparse — a per-series finding to count
#      and report (thick slices are where to expect it), not a reason to stop
#      the other 119. Failures are collected and listed at the end.
#
# USAGE
#   export MAIN_PATH=/abs/path/to/cohort/        # trailing slash
#   bash run_segmentation_split.sh                       # whole cohort
#   bash run_segmentation_split.sh --split test          # sampled studies only
#   bash run_segmentation_split.sh --split test --dry-run
#   bash run_segmentation_split.sh --exclude waw0540     # skip studies
#
# Long job — run it under tmux or nohup.
# =============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
SPLIT="all"
DRY_RUN=0
EXCLUDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)   SPLIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --exclude) EXCLUDE="$EXCLUDE $2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done

if [[ -z "${MAIN_PATH:-}" ]]; then
    echo "ERROR: MAIN_PATH is not set."
    echo "  export MAIN_PATH=/abs/path/to/cohort/"
    exit 2
fi
if [[ "${MAIN_PATH:0:1}" != "/" ]]; then
    echo "ERROR: MAIN_PATH must be ABSOLUTE ('$MAIN_PATH' is relative)."
    exit 2
fi
[[ "${MAIN_PATH}" != */ ]] && MAIN_PATH="${MAIN_PATH}/"

DATA_DIR="${MAIN_PATH}nifti_unprocessed_volumes"
OUT_DIR="${MAIN_PATH}ts_segmentations"
SPLIT_CSV="${MAIN_PATH}splits/train_test_split.csv"

[[ -d "$DATA_DIR" ]] || { echo "ERROR: no $DATA_DIR"; exit 2; }
mkdir -p "$OUT_DIR"

export LOG_ROOT="${PIPELINE_LOG_ROOT:-$SCRIPT_DIR/pipeline_logs}"
mkdir -p "$LOG_ROOT"

# ── Resolve the study filter ──────────────────────────────────────────────────
STUDY_FILTER=""
if [[ "$SPLIT" != "all" ]]; then
    [[ -f "$SPLIT_CSV" ]] || {
        echo "ERROR: --split $SPLIT needs $SPLIT_CSV"
        echo "  run: python generate_split.py --sample N --sample_seed S"
        exit 2
    }
    STUDY_FILTER=$(awk -F, -v want="$SPLIT" 'NR>1 && $2==want {print $1}' "$SPLIT_CSV")
    n_filter=$(echo "$STUDY_FILTER" | grep -c . || true)
    echo "split=$SPLIT: $n_filter studies"
fi

in_filter() {   # in_filter <study_id>
    [[ -z "$STUDY_FILTER" ]] && return 0
    grep -qxF "$1" <<< "$STUDY_FILTER"
}
is_excluded() {
    [[ -z "$EXCLUDE" ]] && return 1
    for e in $EXCLUDE; do [[ "$1" == "$e" ]] && return 0; done
    return 1
}

# ── Preflight: fail fast, loudly ──────────────────────────────────────────────
# Without this, a missing TotalSegmentator (the usual cause: the conda env is
# not active in a nohup shell) fails every single series in turn and the run
# ends minutes later having "processed" the whole cohort into the crash list.
# Better to refuse to start.
preflight_fail=0

if ! command -v TotalSegmentator >/dev/null 2>&1; then
    echo "ERROR: TotalSegmentator is not on PATH."
    echo ""
    echo "  A nohup/background shell does NOT inherit an interactive activation,"
    echo "  so activate INSIDE the backgrounded command:"
    echo ""
    echo "      nohup bash -c 'source /path/to/venv/bin/activate && \\"
    echo "            bash run_segmentation_split.sh --split test' > log 2>&1 &"
    echo ""
    echo "  On this project's server the venv is:"
    echo "      source /media/external20/saeedeh_danaei/ncct_env/bin/activate"
    preflight_fail=1
fi

if ! $PYTHON -c "import numpy, nibabel" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON cannot import numpy and nibabel, which combine_masks.py needs."
    preflight_fail=1
fi

if [[ ! -f "$SCRIPT_DIR/combine_masks.py" ]]; then
    echo "ERROR: combine_masks.py not found next to this script ($SCRIPT_DIR)."
    preflight_fail=1
fi

if [[ $preflight_fail -ne 0 ]]; then
    echo ""
    echo "Refusing to start: fix the above, then re-run. Nothing was written."
    exit 2
fi

echo "preflight ok: TotalSegmentator=$(command -v TotalSegmentator)"

echo "======================================================================"
echo "  MAIN_PATH : $MAIN_PATH"
echo "  volumes   : $DATA_DIR"
echo "  output    : $OUT_DIR"
echo "  split     : $SPLIT"
echo "  exclude   :${EXCLUDE:- (none)}"
echo "  dry-run   : $DRY_RUN"
echo "======================================================================"

# ── Build the work list first, so the count is known up front ─────────────────
declare -a TODO=()
n_total=0 n_skip_split=0 n_skip_excl=0 n_skip_done=0

for vol in "$DATA_DIR"/*_standardized.nii.gz; do
    [[ -e "$vol" ]] || continue
    n_total=$((n_total + 1))
    base=$(basename "$vol" .nii.gz); base="${base/_standardized/}"
    IFS='_' read -r study series _ <<< "$base"

    if ! in_filter "$study";  then n_skip_split=$((n_skip_split + 1)); continue; fi
    if   is_excluded "$study"; then n_skip_excl=$((n_skip_excl + 1));  continue; fi
    if [[ -f "${OUT_DIR}/${study}_${series}_seg_reg.nii.gz" ]]; then
        n_skip_done=$((n_skip_done + 1)); continue
    fi
    TODO+=("$vol")
done

echo "series: $n_total total | skip: $n_skip_split not-in-split, "\
"$n_skip_excl excluded, $n_skip_done already done | TO RUN: ${#TODO[@]}"
echo ""

if [[ ${#TODO[@]} -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi
if [[ $DRY_RUN -eq 1 ]]; then
    for v in "${TODO[@]}"; do echo "  would run: $(basename "$v")"; done
    exit 0
fi

# ── Run ───────────────────────────────────────────────────────────────────────
declare -a SPARSE=() CRASHED=()
i=0
t_start=$(date +%s)

for vol in "${TODO[@]}"; do
    i=$((i + 1))
    base=$(basename "$vol" .nii.gz); base="${base/_standardized/}"
    IFS='_' read -r study series _ <<< "$base"
    prefix="${OUT_DIR}/${study}_${series}"
    seg_dir="${prefix}_segs"
    mkdir -p "$seg_dir"

    echo "────────────────────────────────────────────────────────────────"
    echo "[$i/${#TODO[@]}] $base   $(date '+%H:%M:%S')"

    if ! TotalSegmentator -i "$vol" -o "$seg_dir"; then
        echo "  TotalSegmentator FAILED on $base"
        CRASHED+=("$base (TotalSegmentator)")
        # Three failures in the first three series is a systematic problem
        # (environment, GPU, weights), not three unlucky volumes. Stop rather
        # than burn the whole work list proving it.
        if [[ $i -le 3 && ${#CRASHED[@]} -ge 3 ]]; then
            echo ""
            echo "ABORTING: the first 3 series all failed — this is systematic,"
            echo "not data. Check the TotalSegmentator output above (GPU, model"
            echo "weights, environment) before re-running."
            exit 3
        fi
        continue
    fi

    if $PYTHON combine_masks.py "$seg_dir" "$prefix"; then
        echo "  combine_masks ok"
    else
        code=$?
        if [[ $code -eq 2 ]]; then
            echo "  combine_masks SANITY FAILED (seg_reg too sparse)"
            SPARSE+=("$base")
        else
            echo "  combine_masks CRASHED (exit $code)"
            CRASHED+=("$base (combine_masks exit $code)")
        fi
    fi
done

elapsed=$(( $(date +%s) - t_start ))
echo ""
echo "======================================================================"
printf "  done %d series in %dh%02dm\n" "${#TODO[@]}" $((elapsed/3600)) $(((elapsed%3600)/60))
echo "  sanity-failed (sparse seg_reg): ${#SPARSE[@]}"
for s in "${SPARSE[@]:-}";  do [[ -n "$s" ]] && echo "      $s"; done
echo "  crashed: ${#CRASHED[@]}"
for s in "${CRASHED[@]:-}"; do [[ -n "$s" ]] && echo "      $s"; done
echo ""
echo "  Both counts are reportable numbers — the sparse ones especially,"
echo "  since thick-slice series are where TotalSegmentator is expected to"
echo "  struggle and that is a cohort property worth stating."
echo "======================================================================"

[[ ${#CRASHED[@]} -eq 0 ]] || exit 1
