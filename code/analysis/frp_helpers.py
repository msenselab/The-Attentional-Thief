"""Shared FRP analysis helpers.

Pure data-processing utilities used by subject-level and group-level FRP scripts.
These functions operate on MNE-like objects but avoid plotting / file I/O so they
are easy to test.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_COMPARISONS = [
    ("active", "passive"),
    ("active", "constant"),
    ("passive", "constant"),
]


def _window_mask(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    start, end = window
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        raise ValueError(f"Window {window} does not overlap with available times")
    return mask


def _available_picks(ch_names: Iterable[str], requested: Iterable[str]) -> list[int]:
    ch_names = list(ch_names)
    return [ch_names.index(ch) for ch in requested if ch in ch_names]


def _cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    pooled = ((len(x) - 1) * var_x + (len(y) - 1) * var_y) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def _all_comparisons(conditions: Iterable[str]) -> list[tuple[str, str]]:
    unique = list(dict.fromkeys(conditions))
    return list(combinations(unique, 2))


def compute_subject_component_summary(
    evokeds: Mapping[str, object],
    windows: Mapping[str, tuple[float, float]],
    clusters: Mapping[str, list[str]],
    subject: str | None = None,
) -> pd.DataFrame:
    """Return condition × cluster × window mean amplitudes in µV."""
    rows: list[dict] = []
    for condition, evoked in evokeds.items():
        for cluster_name, cluster_channels in clusters.items():
            picks = _available_picks(evoked.ch_names, cluster_channels)
            if not picks:
                continue
            for window_name, window in windows.items():
                mask = _window_mask(np.asarray(evoked.times), window)
                mean_amp_uv = float(np.asarray(evoked.data)[np.ix_(picks, mask)].mean() * 1e6)
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "cluster": cluster_name,
                        "window": window_name,
                        "n": int(getattr(evoked, "nave", np.nan)),
                        "mean_amp_uv": round(mean_amp_uv, 6),
                    }
                )
    return pd.DataFrame(rows)


def build_epoch_amplitude_table(
    epochs,
    windows: Mapping[str, tuple[float, float]],
    clusters: Mapping[str, list[str]],
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return one row per epoch × cluster × window with mean amplitude in µV."""
    if conditions is None:
        conditions = list(epochs.event_id)

    rows: list[dict] = []
    for condition in conditions:
        if condition not in epochs.event_id or len(epochs[condition]) == 0:
            continue
        cond_epochs = epochs[condition]
        times = np.asarray(cond_epochs.times)
        for cluster_name, cluster_channels in clusters.items():
            picks = _available_picks(cond_epochs.ch_names, cluster_channels)
            if not picks:
                continue
            data = cond_epochs.get_data(picks=picks) * 1e6  # epochs × channels × times
            for window_name, window in windows.items():
                mask = _window_mask(times, window)
                mean_by_epoch = data[:, :, mask].mean(axis=(1, 2))
                for epoch_idx, amplitude_uv in enumerate(mean_by_epoch):
                    rows.append(
                        {
                            "condition": condition,
                            "cluster": cluster_name,
                            "window": window_name,
                            "epoch_index": epoch_idx,
                            "amplitude_uv": float(amplitude_uv),
                        }
                    )
    return pd.DataFrame(rows)


def compute_condition_comparisons(
    epoch_table: pd.DataFrame,
    comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Independent-samples condition comparisons on epoch-level amplitudes."""
    if epoch_table.empty:
        return pd.DataFrame()
    if comparisons is None:
        comparisons = _all_comparisons(epoch_table["condition"].unique())

    rows: list[dict] = []
    grouped = epoch_table.groupby(["cluster", "window"], sort=False)
    for (cluster, window), group_df in grouped:
        for cond_a, cond_b in comparisons:
            a = group_df.loc[group_df["condition"] == cond_a, "amplitude_uv"].to_numpy(dtype=float)
            b = group_df.loc[group_df["condition"] == cond_b, "amplitude_uv"].to_numpy(dtype=float)
            if len(a) == 0 or len(b) == 0:
                continue
            if len(a) < 2 or len(b) < 2:
                t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            rows.append(
                {
                    "cluster": cluster,
                    "window": window,
                    "comparison": f"{cond_a}_vs_{cond_b}",
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "mean_a_uv": float(a.mean()),
                    "mean_b_uv": float(b.mean()),
                    "mean_diff_uv": float(a.mean() - b.mean()),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": _cohens_d_independent(a, b),
                }
            )
    return pd.DataFrame(rows)


def compute_group_condition_comparisons(
    subject_summary: pd.DataFrame,
    comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Paired condition comparisons using one value per subject."""
    if subject_summary.empty:
        return pd.DataFrame()
    if comparisons is None:
        comparisons = _all_comparisons(subject_summary["condition"].unique())

    rows: list[dict] = []
    grouped = subject_summary.groupby(["cluster", "window"], sort=False)
    for (cluster, window), group_df in grouped:
        pivot = group_df.pivot_table(
            index="subject",
            columns="condition",
            values="mean_amp_uv",
            aggfunc="mean",
        )
        for cond_a, cond_b in comparisons:
            if cond_a not in pivot.columns or cond_b not in pivot.columns:
                continue
            paired = pivot[[cond_a, cond_b]].dropna()
            if paired.empty:
                continue
            a = paired[cond_a].to_numpy(dtype=float)
            b = paired[cond_b].to_numpy(dtype=float)
            if len(paired) < 2:
                t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = stats.ttest_rel(a, b)
            rows.append(
                {
                    "cluster": cluster,
                    "window": window,
                    "comparison": f"{cond_a}_vs_{cond_b}",
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_pairs": int(len(paired)),
                    "mean_a_uv": float(a.mean()),
                    "mean_b_uv": float(b.mean()),
                    "mean_diff_uv": float((a - b).mean()),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_dz": _cohens_d_paired(a, b),
                }
            )
    return pd.DataFrame(rows)
