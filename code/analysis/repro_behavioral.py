"""
Reproduce the headline behavioral (subjective-time) results from derived data.

This script reads ONLY the shipped derived data in ``data/derived/behavioral/``
(no raw EEG / eye-tracking required) and recomputes:

  * Repeated-measures ANOVA (session_type x n_pictures) for estimated duration,
    estimation ratio, and recognition accuracy.
  * Pairwise post-hoc paired t-tests across the three conditions, Holm-corrected,
    with Cohen's dz.
  * Condition descriptives (mean, SD, N).

Each recomputed table is compared against the corresponding manuscript
"reference" table in ``data/derived/behavioral/reference/``. A PASS/FAIL line is
printed per check so the reproduction is self-verifying.

Dependencies: numpy, pandas, scipy, statsmodels (all in pyproject.toml).

Run:
    uv run python code/analysis/repro_behavioral.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# --- locate derived data relative to this file (repo-root independent) --------
ROOT = Path(__file__).resolve().parents[2]
BEH = ROOT / "data" / "derived" / "behavioral"
REF = BEH / "reference"

CONDITIONS = ["active", "passive", "constant"]
CONTRASTS = [("active", "passive"), ("active", "constant"), ("passive", "constant")]
METRICS = ["estimated_duration_s", "estimation_ratio", "accuracy"]


def _fmt(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float, np.floating)) else str(x)


def rm_anova(cells: pd.DataFrame) -> pd.DataFrame:
    """Two-way repeated-measures ANOVA for each metric (statsmodels AnovaRM).

    Partial eta squared is derived from F and the degrees of freedom:
        pes = (df_effect * F) / (df_effect * F + df_error)
    """
    rows = []
    for metric in METRICS:
        res = AnovaRM(data=cells, depvar=metric, subject="sub",
                      within=["session_type", "n_pictures"],
                      aggregate_func=np.mean).fit()
        tab = res.anova_table
        for effect, r in tab.iterrows():
            f, ndf, ddf = r["F Value"], r["Num DF"], r["Den DF"]
            pes = (ndf * f) / (ndf * f + ddf)
            rows.append(dict(metric=metric, effect=effect, F=f,
                             num_df=ndf, den_df=ddf, p=r["Pr > F"], pes=pes))
    return pd.DataFrame(rows)


def condition_means(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Subject x condition means, collapsed over n_pictures."""
    return (cells.groupby(["sub", "session_type"])[metric]
            .mean().unstack("session_type"))


def descriptives(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        m = condition_means(cells, metric)
        for cond in CONDITIONS:
            s = m[cond].dropna()
            rows.append(dict(metric=metric, session_type=cond,
                             mean=s.mean(), sd=s.std(ddof=1), n=int(s.size)))
    return pd.DataFrame(rows)


def posthoc(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        m = condition_means(cells, metric)
        pvals, recs = [], []
        for a, b in CONTRASTS:
            pair = m[[a, b]].dropna()
            t, p = stats.ttest_rel(pair[a], pair[b])
            diff = pair[a] - pair[b]
            dz = diff.mean() / diff.std(ddof=1)
            recs.append(dict(metric=metric, contrast=f"{a}_vs_{b}", n=int(pair.shape[0]),
                             mean_a=pair[a].mean(), mean_b=pair[b].mean(),
                             t=float(t), p_raw=float(p), dz=float(dz)))
            pvals.append(float(p))
        holm = multipletests(pvals, method="holm")[1]
        for rec, p_holm in zip(recs, holm):
            rec["p_holm"] = float(p_holm)
            rows.append(rec)
    return pd.DataFrame(rows)


def compare(label: str, got: pd.DataFrame, ref: pd.DataFrame,
            keys: list[str], cols: dict[str, str], atol: float = 1e-3) -> bool:
    """Merge recomputed vs reference on keys; report max abs diff per column."""
    merged = got.merge(ref, on=keys, suffixes=("_got", "_ref"))
    if merged.empty:
        print(f"  [FAIL] {label}: no overlapping rows on {keys}")
        return False
    ok, worst = True, 0.0
    for got_col, ref_col in cols.items():
        left = merged[got_col if got_col in merged else f"{got_col}_got"]
        right = merged[ref_col if ref_col in merged else f"{ref_col}_ref"]
        d = (left - right).abs()
        worst = max(worst, float(d.max()))
        if d.max() > atol:
            ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {len(merged)} rows compared, max|Δ| = {_fmt(worst)}")
    return ok


def main() -> None:
    cells = pd.read_csv(BEH / "behavioral_cell_level.csv")
    print(f"Loaded {BEH / 'behavioral_cell_level.csv'}")
    print(f"  N subjects = {cells['sub'].nunique()}, "
          f"conditions = {sorted(cells['session_type'].unique())}\n")

    ok = []

    # --- RM-ANOVA ---
    print("Repeated-measures ANOVA (session_type x n_pictures):")
    aov = rm_anova(cells)
    ref_aov = pd.read_csv(REF / "anova_results.csv").rename(
        columns={"F Value": "F", "Num DF": "num_df", "Den DF": "den_df", "Pr > F": "p"})
    ref_aov["effect"] = ref_aov["effect"].str.replace(":", " * ", regex=False)
    ok.append(compare("ANOVA F / pes", aov, ref_aov, ["metric", "effect"],
                      {"F": "F", "pes": "pes"}, atol=1e-2))
    print(aov.to_string(index=False, float_format=_fmt))

    # --- descriptives ---
    print("\nCondition descriptives:")
    desc = descriptives(cells)
    ref_desc = pd.read_csv(REF / "condition_descriptives.csv")
    ok.append(compare("Descriptives mean / sd", desc, ref_desc,
                      ["metric", "session_type"], {"mean": "mean", "sd": "sd"}))

    # --- post-hoc ---
    print("\nPost-hoc paired t-tests (Holm-corrected):")
    ph = posthoc(cells)
    ref_ph = pd.read_csv(REF / "condition_posthoc.csv")
    ok.append(compare("Post-hoc t / p_holm / dz", ph, ref_ph,
                      ["metric", "contrast"],
                      {"t": "t", "p_holm": "p_holm", "dz": "dz"}, atol=1e-2))
    print(ph[ph.metric == "estimation_ratio"].to_string(index=False, float_format=_fmt))

    print("\n" + "=" * 60)
    if all(ok):
        print("REPRODUCTION OK — all recomputed values match the manuscript reference.")
    else:
        print("REPRODUCTION MISMATCH — see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
