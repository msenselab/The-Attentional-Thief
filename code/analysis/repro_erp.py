"""
Reproduce the picture-locked ERP results from derived data.

Reads ONLY ``data/derived/erp/erp_cell_means.csv`` (per subject x condition x
repetition x window cluster amplitudes) and recomputes:

  * Condition x repetition repeated-measures ANOVA per ERP component
    (N1, P2, P3). N = 22 (one subject lacks a complete repetition cell).
  * Collapsed Scrolling - Watching paired t-tests per component (N = 23).

Recomputed values are verified against the manuscript reference tables in
``data/derived/erp/reference/``.

Dependencies: numpy, pandas, scipy, statsmodels.

Run:
    uv run python code/analysis/repro_erp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

ROOT = Path(__file__).resolve().parents[2]
ERP = ROOT / "data" / "derived" / "erp"
REF = ERP / "reference"

WINDOWS = ["N1", "P2", "P3"]
# Manuscript naming: Scrolling = active, Watching = passive.


def _fmt(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float, np.floating)) else str(x)


def rm_anova(cells: pd.DataFrame) -> pd.DataFrame:
    """Condition x repetition RM-ANOVA per window, on complete-case subjects."""
    rows = []
    for win in WINDOWS:
        w = cells[cells["window"] == win].copy()
        # keep only subjects with a full condition x repetition grid
        ncells = w.drop_duplicates(["sub", "condition", "repetition"]).groupby("sub").size()
        full = ncells[ncells == ncells.max()].index
        w = w[w["sub"].isin(full)]
        res = AnovaRM(data=w, depvar="amplitude_uV", subject="sub",
                      within=["condition", "repetition"], aggregate_func=np.mean).fit()
        for effect, r in res.anova_table.iterrows():
            f, ndf, ddf = r["F Value"], r["Num DF"], r["Den DF"]
            pes = (ndf * f) / (ndf * f + ddf)
            rows.append(dict(window=win, effect=effect, F=f, ddof1=int(ndf),
                             ddof2=int(ddf), p_unc=r["Pr > F"], partial_eta2=pes,
                             n=w["sub"].nunique()))
    return pd.DataFrame(rows)


def collapsed_contrast(cells: pd.DataFrame) -> pd.DataFrame:
    """Scrolling - Watching paired t per window, collapsed over repetition."""
    rows = []
    for win in WINDOWS:
        w = cells[cells["window"] == win]
        m = w.groupby(["sub", "condition"])["amplitude_uV"].mean().unstack("condition")
        pair = m[["active", "passive"]].dropna()
        t, p = stats.ttest_rel(pair["active"], pair["passive"])
        diff = pair["active"] - pair["passive"]
        dz = diff.mean() / diff.std(ddof=1)
        rows.append(dict(window=win, n=int(pair.shape[0]),
                         mean_scrolling=pair["active"].mean(),
                         mean_watching=pair["passive"].mean(),
                         mean_diff_uV=diff.mean(), t=float(t), df=int(pair.shape[0] - 1),
                         p=float(p), dz=float(dz)))
    return pd.DataFrame(rows)


def compare(label, got, ref, keys, cols, atol=1e-2) -> bool:
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
    cells = pd.read_csv(ERP / "erp_cell_means.csv")
    print(f"Loaded {ERP / 'erp_cell_means.csv'}")
    print(f"  N subjects = {cells['sub'].nunique()}, windows = {sorted(cells.window.unique())}, "
          f"conditions = {sorted(cells.condition.unique())}\n")
    ok = []

    print("Condition x repetition RM-ANOVA (per component):")
    aov = rm_anova(cells)
    ref_aov = pd.read_csv(REF / "erp_rmanova.csv")
    ref_aov["effect"] = ref_aov["effect"].str.replace("condition * repetition",
                                                       "condition:repetition", regex=False)
    aov_cmp = aov.copy()
    ok.append(compare("RM-ANOVA F / partial_eta2", aov_cmp, ref_aov, ["window", "effect"],
                      {"F": "F", "partial_eta2": "partial_eta2"}, atol=1e-2))
    print(aov.to_string(index=False, float_format=_fmt))

    print("\nCollapsed Scrolling - Watching paired t-tests:")
    cc = collapsed_contrast(cells)
    ref_cc = pd.read_csv(REF / "erp_maineffect_collapsed.csv")
    ok.append(compare("Collapsed t / dz", cc, ref_cc, ["window"],
                      {"t": "t", "dz": "dz", "mean_diff_uV": "mean_diff_uV"}, atol=1e-2))
    print(cc.to_string(index=False, float_format=_fmt))

    print("\n" + "=" * 60)
    if all(ok):
        print("REPRODUCTION OK — ERP values match the manuscript reference.")
    else:
        print("REPRODUCTION MISMATCH — see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
