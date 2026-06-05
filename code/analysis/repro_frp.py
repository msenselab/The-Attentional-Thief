"""
Reproduce the fixation-related potential (FRP) condition contrasts from derived data.

Reads ONLY ``data/derived/frp/frp_summary_all.csv`` (per subject x condition x
cluster x window mean FRP amplitude) and recomputes, for every cluster x window,
the three pairwise condition contrasts (Scrolling/Watching/Baseline) with paired
t-tests and Cohen's dz.

Recomputed values are verified against the manuscript reference table
``data/derived/frp/frp_condition_comparisons.csv``. The occipital-lambda effect
(the headline FRP result) is additionally checked against
``data/derived/frp/reference/frp_lambda_dz_ci.csv``.

Dependencies: numpy, pandas, scipy.

Run:
    uv run python code/analysis/repro_frp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FRP = ROOT / "data" / "derived" / "frp"
REF = FRP / "reference"

CONTRASTS = [("active", "passive"), ("active", "constant"), ("passive", "constant")]


def _fmt(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float, np.floating)) else str(x)


def contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cluster, window), g in summary.groupby(["cluster", "window"]):
        m = g.pivot_table(index="subject", columns="condition", values="mean_amp_uv")
        for a, b in CONTRASTS:
            if a not in m or b not in m:
                continue
            pair = m[[a, b]].dropna()
            t, p = stats.ttest_rel(pair[a], pair[b])
            diff = pair[a] - pair[b]
            dz = diff.mean() / diff.std(ddof=1)
            rows.append(dict(cluster=cluster, window=window,
                             comparison=f"{a}_vs_{b}", n_pairs=int(pair.shape[0]),
                             mean_a_uv=pair[a].mean(), mean_b_uv=pair[b].mean(),
                             mean_diff_uv=diff.mean(), t_stat=float(t),
                             p_value=float(p), cohens_dz=float(dz)))
    return pd.DataFrame(rows)


def compare(label, got, ref, keys, cols, atol=1e-3) -> bool:
    merged = got.merge(ref, on=keys, suffixes=("_got", "_ref"))
    if merged.empty:
        print(f"  [FAIL] {label}: no overlapping rows on {keys}")
        return False
    ok, worst = True, 0.0
    for gc, rc in cols.items():
        left = merged[gc if gc in merged else f"{gc}_got"]
        right = merged[rc if rc in merged else f"{rc}_ref"]
        d = (left - right).abs()
        worst = max(worst, float(d.max()))
        if d.max() > atol:
            ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {len(merged)} rows, max|Δ| = {_fmt(worst)}")
    return ok


def main() -> None:
    summary = pd.read_csv(FRP / "frp_summary_all.csv")
    print(f"Loaded {FRP / 'frp_summary_all.csv'}")
    print(f"  N subjects = {summary['subject'].nunique()}, "
          f"clusters = {sorted(summary.cluster.unique())}, "
          f"windows = {sorted(summary.window.unique())}\n")
    ok = []

    got = contrasts(summary)
    ref = pd.read_csv(FRP / "frp_condition_comparisons.csv")
    print("All cluster x window x contrast comparisons:")
    ok.append(compare("FRP t / dz / mean_diff", got, ref,
                      ["cluster", "window", "comparison"],
                      {"t_stat": "t_stat", "cohens_dz": "cohens_dz",
                       "mean_diff_uv": "mean_diff_uv"}, atol=1e-3))

    # Headline: occipital lambda (Scrolling > Watching > Baseline)
    print("\nOccipital-lambda contrasts (headline FRP effect):")
    lam = got[(got.cluster == "occipital") & (got.window == "lambda")]
    print(lam.to_string(index=False, float_format=_fmt))
    ref_ci = pd.read_csv(REF / "frp_lambda_dz_ci.csv")
    lam_cmp = lam.rename(columns={"t_stat": "t", "cohens_dz": "dz"})
    lam_cmp["contrast"] = lam_cmp["comparison"].map(
        {"active_vs_passive": "Scrolling vs Watching",
         "active_vs_constant": "Scrolling vs Baseline",
         "passive_vs_constant": "Watching vs Baseline"})
    ok.append(compare("Occipital-lambda t / dz vs CI table", lam_cmp, ref_ci,
                      ["contrast"], {"t": "t", "dz": "dz"}, atol=1e-3))

    print("\n" + "=" * 60)
    if all(ok):
        print("REPRODUCTION OK — FRP values match the manuscript reference.")
    else:
        print("REPRODUCTION MISMATCH — see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
