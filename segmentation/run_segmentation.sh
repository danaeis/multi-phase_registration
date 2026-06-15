#!/bin/bash
set -e
export LOG_ROOT="${PIPELINE_LOG_ROOT:-pipeline_logs}"
mkdir -p "$LOG_ROOT"


DATA_DIR="../../ncct_cect/vindr_ds/nifti_unprocessed_volumes"
output_dir="../../ncct_cect/vindr_ds/ts_segmentations"
mkdir -p "$output_dir"

# Get all NIfTI files using Python snippet
nifti_list=$(python3 <<EOF
import os
DATA_DIR = "$DATA_DIR"
nifti_paths = []
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".nii.gz"):
        nifti_paths.append(os.path.join(DATA_DIR, file))
print(" ".join(nifti_paths))
EOF
)

for vol in $nifti_list; do
    file_name=$(basename "$vol" .nii.gz)

    # Strip _standardized suffix if present so the prefix stays clean
    base_name="${file_name/_standardized/}"

    # Split study / series from the filename (first two underscore-delimited tokens)
    IFS='_' read -r study_name vol_name _ <<< "$base_name"

    seg_dir="${output_dir}/${study_name}_${vol_name}_segs"
    mkdir -p "$seg_dir"

    echo "──────────────────────────────────────────────────────────────────"
    echo "Volume     : $vol"
    echo "Seg dir    : $seg_dir"

    echo "Running TotalSegmentator..."
    TotalSegmentator -i "$vol" -o "$seg_dir"

    # combine_masks.py now writes TWO files:
    #   <prefix>_seg_full.nii.gz  — all organs (crop bbox)
    #   <prefix>_seg_reg.nii.gz   — bones + stable soft (registration metric)
    #
    # Pass the prefix (no extension) as the second argument.
    prefix="${output_dir}/${study_name}_${vol_name}"
    echo "Combining masks → ${prefix}_seg_full.nii.gz + _seg_reg.nii.gz"
    if python3 combine_masks.py "$seg_dir" "$prefix"; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [INFO] combine_masks ok: $file_name"
    else
        exit_code=$?
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        log_root="${LOG_ROOT:-pipeline_logs}"
        mkdir -p "$log_root"
        if [ "$exit_code" -eq 2 ]; then
            msg="Seg sanity FAILED: $file_name"
            echo "$ts [WARNING] $msg"
            echo "{\"ts\":\"$ts\",\"stage\":\"segmentation\",\"level\":\"WARNING\",\"msg\":\"$msg\"}" \
                >> "$log_root/pipeline_errors.jsonl"
        else
            msg="combine_masks CRASHED (exit $exit_code): $file_name"
            echo "$ts [ERROR] $msg"
            echo "{\"ts\":\"$ts\",\"stage\":\"segmentation\",\"level\":\"ERROR\",\"msg\":\"$msg\"}" \
                >> "$log_root/pipeline_errors.jsonl"
        fi
    fi
    echo "Done: $file_name"
done

echo "======================================================================"
echo "Segmentation + mask combination complete."
echo "======================================================================"
