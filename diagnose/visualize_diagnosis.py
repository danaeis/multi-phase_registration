"""
visualize_diagnosis.py
======================
Reads the CSV output of diagnose_registration_failures.py and renders a
color-coded per-study HTML report showing which root cause fired.

Usage
-----
python visualize_diagnosis.py --csv /data/diagnosis_report.csv \\
    --out /data/diagnosis_report.html
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


FLAG_DESCRIPTIONS = {
    "KABSCH_FAIL_TOO_FEW_ORGANS": (
        "red",
        "Kabsch SVD had <3 common organs → identity transform used → no correction applied."
        " Root cause: bad/empty seg_reg (TotalSegmentator failure), wrong crop window,"
        " or the DICOM HU bug (Bug 1) still active on this scanner."
    ),
    "KABSCH_BONE_TOO_FEW": (
        "orange",
        "Pass 0a bone-only Kabsch had <3 common bone labels. Bone Kabsch is the first"
        " init step — if it fails, all-organ Kabsch (Pass 0b) is still attempted,"
        " but a failed bone step is a warning that the seg is sparse."
    ),
    "DIRECTION_DET_SIGN_MISMATCH": (
        "red",
        "NC and moving seg_reg have opposite direction-matrix determinant signs."
        " This indicates a left-right or head-foot flip between phases (HFS vs FFS"
        " acquisition, or DICOMOrient produced a mirrored output). Kabsch will find"
        " the 'perfect' mirror alignment → all organs displaced by the body diameter."
    ),
    "BAD_Z_OFFSET": (
        "orange",
        "The physical Z-offset between anchor-organ centroids exceeds 80 mm."
        " Likely cause: align_data computed a wrong z_offset (sign flip or bad crop),"
        " placing the wrong anatomical region inside the crop window."
    ),
    "LARGE_XY_RESIDUAL": (
        "yellow",
        "Anchor-organ XY centroid residual >40 mm after alignment."
        " Z-alignment corrects only the Z axis. A large in-plane offset remains and"
        " may cause Nelder-Mead (Pass 1) to converge to a local minimum."
    ),
    "LARGE_CENTROID_DIST": (
        "orange",
        "One or more organs have >15 mm 3D centroid distance between NC and moving"
        " phase after this pipeline stage. Indicates residual misalignment."
    ),
}

STATUS_COLORS = {"OK": "#d4edda", "BAD": "#f8d7da", "file_missing": "#fff3cd"}


def classify_flags(flags_str: str):
    """Parse flags string and return list of (flag_key, color, description) tuples."""
    if not flags_str or pd.isna(flags_str):
        return []
    items = []
    for part in flags_str.split(" | "):
        part = part.strip()
        matched = False
        for key, (color, desc) in FLAG_DESCRIPTIONS.items():
            if key in part:
                items.append((part, color, desc))
                matched = True
                break
        if not matched:
            items.append((part, "gray", ""))
    return items


def render_html(df: pd.DataFrame, out_path: str):
    # Aggregate per study
    studies = df["study_id"].unique()

    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Registration Failure Diagnosis Report</title>
<style>
  body { font-family: monospace; font-size: 13px; margin: 20px; background: #f5f5f5; }
  h1   { font-size: 20px; }
  h2   { font-size: 15px; margin-top: 30px; color: #333; border-bottom: 1px solid #ccc; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
  th { background: #333; color: #fff; padding: 6px 8px; text-align: left; }
  td { padding: 5px 8px; border-bottom: 1px solid #ddd; vertical-align: top; }
  .ok      { background: #d4edda; }
  .bad     { background: #f8d7da; }
  .missing { background: #fff3cd; }
  .flag-red    { background: #dc3545; color: white; border-radius: 4px; padding: 2px 5px; margin: 1px; display: inline-block; }
  .flag-orange { background: #fd7e14; color: white; border-radius: 4px; padding: 2px 5px; margin: 1px; display: inline-block; }
  .flag-yellow { background: #ffc107; color: black; border-radius: 4px; padding: 2px 5px; margin: 1px; display: inline-block; }
  .flag-gray   { background: #6c757d; color: white; border-radius: 4px; padding: 2px 5px; margin: 1px; display: inline-block; }
  .tooltip-wrap { position: relative; display: inline-block; cursor: help; }
  .tooltip-wrap .tooltip-text {
    visibility: hidden; width: 350px; background: #222; color: #fff;
    padding: 8px; border-radius: 6px; position: absolute; z-index: 1;
    bottom: 120%; left: 0; font-size: 11px; line-height: 1.5;
  }
  .tooltip-wrap:hover .tooltip-text { visibility: visible; }
  .summary-bar { display: flex; gap: 20px; margin-bottom: 20px; }
  .summary-box { border: 1px solid #ccc; border-radius: 8px; padding: 10px 20px;
                 background: white; text-align: center; }
  .summary-box .num { font-size: 28px; font-weight: bold; }
  .summary-box .lbl { font-size: 12px; color: #666; }
</style>
</head>
<body>
<h1>Registration Failure Diagnosis Report</h1>
"""]

    n_total = len(df)
    n_bad   = (df["status"] == "BAD").sum()
    n_ok    = (df["status"] == "OK").sum()
    n_miss  = (df["status"] == "file_missing").sum()

    html_parts.append(f"""
<div class="summary-bar">
  <div class="summary-box"><div class="num">{n_total}</div><div class="lbl">Total series</div></div>
  <div class="summary-box" style="border-color:#28a745"><div class="num" style="color:#28a745">{n_ok}</div><div class="lbl">OK</div></div>
  <div class="summary-box" style="border-color:#dc3545"><div class="num" style="color:#dc3545">{n_bad}</div><div class="lbl">BAD ({100*n_bad/max(n_total,1):.0f}%)</div></div>
  <div class="summary-box" style="border-color:#ffc107"><div class="num" style="color:#888">{n_miss}</div><div class="lbl">File missing</div></div>
</div>
""")

    # Flag frequency summary
    flag_counts: dict = {}
    for flags_str in df.get("flags", []):
        if not flags_str or pd.isna(flags_str):
            continue
        for part in str(flags_str).split(" | "):
            for key in FLAG_DESCRIPTIONS:
                if key in part:
                    flag_counts[key] = flag_counts.get(key, 0) + 1

    if flag_counts:
        html_parts.append("<h2>Flag Frequency</h2><table><tr><th>Flag</th><th>Count</th><th>Meaning</th></tr>")
        for key, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
            color, desc = FLAG_DESCRIPTIONS[key]
            html_parts.append(
                f"<tr><td><span class='flag-{color}'>{key}</span></td>"
                f"<td>{cnt}</td><td>{desc[:120]}…</td></tr>"
            )
        html_parts.append("</table>")

    # Per-study tables
    html_parts.append("<h2>Per-Study Details</h2>")
    for study in sorted(studies):
        sdf = df[df["study_id"] == study]
        study_status = "BAD" if (sdf["status"] == "BAD").any() else \
                       ("file_missing" if (sdf["status"] == "file_missing").any() else "OK")
        css_class = {"BAD": "bad", "OK": "ok", "file_missing": "missing"}[study_status]

        html_parts.append(f"""
<details>
  <summary style="cursor:pointer; padding:6px; background:#{'#f8d7da' if study_status=='BAD' else '#d4edda' if study_status=='OK' else '#fff3cd'}; border-radius:4px; margin:4px 0;">
    <b>{study[:70]}</b>  —  {study_status}
  </summary>
  <table>
    <tr>
      <th>Phase</th><th>Status</th>
      <th>Common organs<br>(Kabsch / bone)</th>
      <th>det NC / MOV</th>
      <th>Z-offset (mm)</th>
      <th>XY residual (mm)</th>
      <th>Mean centroid<br>3D (mm)</th>
      <th>Flags</th>
    </tr>
""")
        for _, row in sdf.iterrows():
            rc = {"BAD": "bad", "OK": "ok", "file_missing": "missing"}.get(row["status"], "")
            flag_items = classify_flags(str(row.get("flags", "")))
            flag_html = ""
            for flag, color, desc in flag_items:
                tooltip = f'<span class="tooltip-text">{desc}</span>' if desc else ""
                flag_html += f'<span class="tooltip-wrap"><span class="flag-{color}">{flag}</span>{tooltip}</span> '

            nc_det  = f"{row.get('nc_det', '?'):.3f}"  if isinstance(row.get('nc_det'), float) else "?"
            mov_det = f"{row.get('mov_det', '?'):.3f}" if isinstance(row.get('mov_det'), float) else "?"
            det_sign_ok = "✓" if row.get("det_sign_match") else "✗"

            html_parts.append(f"""
    <tr class="{rc}">
      <td>{row['phase']}</td>
      <td><b>{row['status']}</b></td>
      <td>{row.get('common_kabsch_organs','?')} / {row.get('common_bone_organs','?')}</td>
      <td>{nc_det} / {mov_det} {det_sign_ok}</td>
      <td>{row.get('z_offset_mean_mm','?'):+.1f}</td>
      <td>{row.get('xy_mean_mm','?'):.1f}</td>
      <td>{row.get('mean_centroid_3d','?'):.1f}</td>
      <td>{flag_html}</td>
    </tr>
""")
        html_parts.append("</table></details>")

    html_parts.append("""
<hr>
<p style="color:#888; font-size:11px;">
Generated by visualize_diagnosis.py &nbsp;|&nbsp;
Flag colors: 🔴 red = likely root cause of catastrophic failure &nbsp;
🟠 orange = warning &nbsp; 🟡 yellow = performance risk<br>
Hover over a flag for its explanation.
</p>
</body></html>
""")

    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"✓ HTML report saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Output CSV from diagnose_registration_failures.py")
    p.add_argument("--out", default="diagnosis_report.html")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    render_html(df, args.out)


if __name__ == "__main__":
    main()
