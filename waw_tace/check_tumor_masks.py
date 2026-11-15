"""
check_tumor_masks.py
====================
Gate check, run BEFORE the 48 GB scan download.

WAW-TACE ships hand-drawn HCC tumour masks. Those are labels this pipeline's
registration objective never sees, which is the one evaluation that answers the
circularity objection against TotalSegmentator-mask Dice. But the masks were
drawn on whichever phase showed the lesion most distinctly, so a patient with a
mask in ONE phase only is useless for a registration Dice: you need the lesion
delineated in the fixed phase AND in a moving phase.

This script counts how many patients actually qualify, from two 20 MB files:
    tumor_masks_wawtace_v1_08_05_2024.zip   (2.3 MB)
    ct_hcc_metadata_v2.csv                  (16 kB)

Mask filenames are {case}_{phase}_{tumour_index}_tumor_seg.{nii.gz,nrrd}
e.g. 35_1_0_tumor_seg = patient 35, arterial phase, tumour #0.

Decide from the output:
  >= ~25 qualifying patients -> the independent-label table is viable.
  <  ~25                     -> demote to a qualitative figure; the paper rests
                                on the front-end transfer table alone.

Usage
-----
  python check_tumor_masks.py --masks /data/waw_tace/tumor_masks
  python check_tumor_masks.py --masks ... --metadata /data/waw_tace/ct_hcc_metadata_v2.csv
  python check_tumor_masks.py --masks ... --phases 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

PHASE_LABELS = {0: "Non-contrast", 1: "Arterial", 2: "Venous", 3: "Delayed"}
REF_PHASE_CODE = 0

# {case}_{phase}_{index}_tumor_seg.nii.gz / .nrrd / .nii
MASK_RE = re.compile(
    r"(?P<case>\d+)_(?P<phase>\d+)_(?P<idx>\d+)_tumor_seg\.(?P<ext>nii\.gz|nii|nrrd)$",
    re.I,
)


def scan_masks(root: Path) -> Tuple[Dict[int, Dict[int, Set[int]]], Counter, list]:
    """Return ({case: {phase: {tumour indices}}}, ext counter, unparsed names)."""
    per_case: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    exts: Counter = Counter()
    unparsed: list = []

    if not root.exists():
        raise SystemExit(f"--masks path does not exist: {root}")

    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        m = MASK_RE.search(f.name)
        if not m:
            if f.suffix.lower() in (".gz", ".nii", ".nrrd"):
                unparsed.append(f.name)
            continue
        case = int(m.group("case"))
        phase = int(m.group("phase"))
        per_case[case][phase].add(int(m.group("idx")))
        exts[m.group("ext").lower()] += 1

    return per_case, exts, unparsed


def main() -> int:
    p = argparse.ArgumentParser(description="Is the tumour-mask evaluation viable?")
    p.add_argument("--masks", required=True,
                   help="Directory holding the extracted tumour masks.")
    p.add_argument("--metadata", default=None,
                   help="Optional ct_hcc_metadata_v2.csv — columns are printed, "
                        "not assumed.")
    p.add_argument("--phases", nargs="*", type=int, default=[0, 1, 2],
                   help="Phase codes the registration run will use. Default 0 1 2.")
    p.add_argument("--out", default=None,
                   help="Optional CSV of qualifying patients.")
    args = p.parse_args()

    phases = set(args.phases)
    root = Path(args.masks).expanduser().resolve()

    print("=" * 74)
    print("check_tumor_masks.py")
    print(f"  masks  : {root}")
    print(f"  phases : {sorted(phases)}")
    print("=" * 74)

    per_case, exts, unparsed = scan_masks(root)

    if not per_case:
        print("\nNo tumour masks matched the expected naming pattern.")
        if unparsed:
            print("Image-like files that did not parse (first 10):")
            for n in unparsed[:10]:
                print(f"  {n}")
        return 1

    print(f"\nmask files by format: {dict(exts)}")
    if unparsed:
        print(f"unparsed image-like files: {len(unparsed)} "
              f"(e.g. {', '.join(unparsed[:3])})")

    # ---------------------------------------------------------------- counts
    n_patients = len(per_case)
    phase_hist = Counter()
    for case, byphase in per_case.items():
        for ph in byphase:
            phase_hist[ph] += 1

    print(f"\npatients with any tumour mask: {n_patients}")
    print("masks per phase:")
    for ph in sorted(phase_hist):
        print(f"  {ph} {PHASE_LABELS.get(ph, '?'):14s} {phase_hist[ph]:4d} patients")

    n_phases_hist = Counter(
        len([ph for ph in byphase if ph in phases]) for byphase in per_case.values()
    )
    print("\nphases-with-masks per patient (restricted to the run's phases):")
    for k in sorted(n_phases_hist):
        print(f"  {k} phase(s): {n_phases_hist[k]:4d} patients")

    # ------------------------------------------------------------ the verdict
    # Usable for a registration Dice: mask in the reference phase AND in at
    # least one moving phase that the run actually registers.
    moving = sorted(phases - {REF_PHASE_CODE})
    qualifying = []
    for case, byphase in sorted(per_case.items()):
        if REF_PHASE_CODE not in byphase:
            continue
        mv = [ph for ph in moving if ph in byphase]
        if not mv:
            continue
        qualifying.append({
            "case": case,
            "study_id": f"waw{case:04d}",
            "ref_phase_tumours": len(byphase[REF_PHASE_CODE]),
            "moving_phases": ",".join(str(ph) for ph in mv),
            "pairs": len(mv),
        })

    n_pairs = sum(q["pairs"] for q in qualifying)

    print("\n" + "-" * 74)
    print(f"QUALIFYING patients (mask in phase {REF_PHASE_CODE} AND in a moving "
          f"phase {moving}): {len(qualifying)}")
    print(f"evaluable (study, moving phase) pairs: {n_pairs}")
    print("-" * 74)

    if len(qualifying) >= 25:
        print("\nVERDICT: viable. The independent-label (tumour Dice / centroid)")
        print("         table can carry real weight — build evaluate_tumor.py.")
    elif qualifying:
        print(f"\nVERDICT: thin ({len(qualifying)} patients). Report it as a")
        print("         qualitative figure plus a caveated table; the paper rests")
        print("         on the front-end transfer result.")
    else:
        print("\nVERDICT: not viable. Masks never co-occur in the reference and a")
        print("         moving phase, so no registration Dice is computable from")
        print("         them. Drop deliverable 2 and say so explicitly.")

    if args.out and qualifying:
        out = Path(args.out).expanduser().resolve()
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qualifying[0]))
            w.writeheader()
            w.writerows(qualifying)
        print(f"\nqualifying patients -> {out}")

    # --------------------------------------------------------- metadata peek
    if args.metadata:
        mpath = Path(args.metadata).expanduser().resolve()
        if not mpath.exists():
            print(f"\n(metadata not found: {mpath})")
        else:
            with open(mpath, newline="") as f:
                rdr = csv.reader(f)
                header = next(rdr, [])
                rows = list(rdr)
            print(f"\nmetadata: {mpath.name}  {len(rows)} rows")
            print("  columns:")
            for c in header:
                print(f"    {c}")
            print("  first row:")
            if rows:
                for c, v in zip(header, rows[0]):
                    print(f"    {c:40s} = {v}")
            print("\n  Cross-check the phase-availability and tumour-count columns")
            print("  above against the mask scan before trusting either alone.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
