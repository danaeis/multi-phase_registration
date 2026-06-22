#!/usr/bin/env bash
# check_progress.sh — quick status snapshot for all comparison conditions
#
# Usage:
#   bash check_progress.sh
#   watch -n 60 bash check_progress.sh    # refresh every minute

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  REGISTRATION PROGRESS  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"

# ── State file ────────────────────────────────────────────────────────────────
echo ""
if [ -f .run_state ] && [ -s .run_state ]; then
    echo "  Completed steps (.run_state):"
    while IFS= read -r line; do
        echo "    ✓ $line"
    done < .run_state
else
    echo "  .run_state is empty (no steps completed yet)"
fi

# ── Running PIDs ──────────────────────────────────────────────────────────────
echo ""
echo "  Active Python processes (registration/evaluation):"
pgrep -la python3 2>/dev/null | grep -E "run_deeds|run_ants|run_unigradicon|deformable|eval_compare" \
    | sed 's/^/    /' || echo "    (none)"

# ── Per-condition file counts ─────────────────────────────────────────────────
echo ""
$PYTHON - "$SCRIPT_DIR" << 'PYEOF'
import sys, os
from pathlib import Path

script_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script_dir))
import compare_config as C
import pandas as pd

try:
    ldf = pd.read_csv(C.LABELS_CSV)
    total_studies = ldf["StudyInstanceUID"].nunique()
    total_pairs   = len(ldf[ldf["Label"] != C.REF_PHASE])
except Exception as e:
    total_studies = "?"
    total_pairs   = "?"

print(f"  Dataset: {total_studies} studies, {total_pairs} moving pairs\n")

# Group by algo for cleaner output
from collections import defaultdict
algo_groups = defaultdict(list)
for tag, cond in C.CONDITIONS.items():
    prefix = tag.split("__")[0]
    algo_groups[prefix].append((tag, cond))

col_w = 38
print(f"  {'Condition':<{col_w}} {'StudyDirs':>9}  {'VolFiles':>8}  {'EvalCSV':>7}")
print(f"  {'-'*(col_w+30)}")

for prefix, items in algo_groups.items():
    for tag, cond in items:
        n_dirs  = 0
        n_files = 0
        if cond.base_dir.exists():
            study_dirs = [d for d in cond.base_dir.iterdir() if d.is_dir()]
            n_dirs  = len(study_dirs)
            n_files = sum(len(list(d.glob(f"*{cond.vol_postfix}"))) for d in study_dirs)

        eval_ok = "✓" if cond.detail_csv.exists() else "—"
        status  = "✓" if n_files > 0 else "○"
        print(f"  {status} {tag:<{col_w}} {n_dirs:>9}  {n_files:>8}  {eval_ok:>7}")
    print()

PYEOF

# ── Recent log tails ──────────────────────────────────────────────────────────
if [ -d run_logs ]; then
    echo ""
    echo "  Recent log activity:"
    for log in run_logs/job_deeds.log run_logs/job_ants.log run_logs/job_ugi.log run_logs/job_a6.log; do
        if [ -f "$log" ]; then
            last=$(tail -1 "$log" 2>/dev/null)
            name=$(basename "$log" .log)
            printf "    %-20s  %s\n" "$name" "$last"
        fi
    done
fi

echo ""
