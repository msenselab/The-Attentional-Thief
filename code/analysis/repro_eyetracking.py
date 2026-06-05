"""
Reproduce the eye-tracking descriptives from derived data.

Reads ONLY ``data/derived/eyetracking/et_summary_all.csv`` (per subject x
condition fixation/saccade/pupil summaries, N = 23) and recomputes the group
mean (SD) for fixation count, mean fixation duration, and fixation rate per
condition.

Recomputed values are verified against the manuscript reference table
``data/derived/eyetracking/reference/et_descriptives.csv`` (formatted as
"mean (SD)" with Scrolling/Watching/Baseline columns).

Dependencies: numpy, pandas.

Run:
    uv run python code/analysis/repro_eyetracking.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ET = ROOT / "data" / "derived" / "eyetracking"
REF = ET / "reference"

# block_type -> manuscript condition label
LABELS = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}

# reference metric label -> (column, scale, decimals)
METRICS = {
    "Fixations per condition (n)": ("n_fix", 1.0, 1),
    "Mean fixation duration (ms)": ("mean_fix_dur", 1.0, 1),
    "Fixation rate (per s)": ("fix_rate", 1.0, 1),
}


def _parse(cell: str) -> tuple[float, float]:
    """Parse a 'mean (sd)' reference cell into floats."""
    m = re.match(r"\s*([-\d.]+)\s*\(([-\d.]+)\)", str(cell))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def main() -> None:
    df = pd.read_csv(ET / "et_summary_all.csv")
    print(f"Loaded {ET / 'et_summary_all.csv'}")
    print(f"  N subjects = {df['subject'].nunique()}, "
          f"conditions = {sorted(df.block_type.unique())}\n")

    ref = pd.read_csv(REF / "et_descriptives.csv").set_index("metric")

    print("Group mean (SD) per condition, recomputed vs reference:")
    worst, ok = 0.0, True
    for metric_label, (col, scale, dec) in METRICS.items():
        for block, label in LABELS.items():
            sub = df[df.block_type == block][col] * scale
            got_m, got_sd = round(sub.mean(), dec), round(sub.std(ddof=1), dec)
            ref_m, ref_sd = _parse(ref.loc[metric_label, label])
            dm, dsd = abs(got_m - ref_m), abs(got_sd - ref_sd)
            worst = max(worst, dm, dsd)
            flag = "" if max(dm, dsd) <= 10 ** (-dec) else "  <-- MISMATCH"
            if flag:
                ok = False
            print(f"  {metric_label:32s} {label:9s} "
                  f"got {got_m} ({got_sd})  ref {ref_m} ({ref_sd}){flag}")

    print(f"\n  max|Δ| = {worst:.4g}")
    print("=" * 60)
    if ok:
        print("REPRODUCTION OK — eye-tracking descriptives match the manuscript reference.")
    else:
        print("REPRODUCTION MISMATCH — see lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
