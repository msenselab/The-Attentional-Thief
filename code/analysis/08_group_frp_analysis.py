"""
Attentional Thief — Step 8: Group FRP Analysis
==============================================
Loads fixation-locked epochs from all processed subjects, computes grand-average
FRPs, summarizes component amplitudes, runs paired condition comparisons, and
exports publication-ready waveforms/topomaps.

Output : analysis/output/group_frp/
           grand_average_frp.png          — grand-average cluster waveforms
           grand_average_topomaps.png     — condition + difference scalp maps
           frp_summary_all.csv            — subject-level component amplitudes
           frp_condition_comparisons.csv  — paired group statistics
           group_frp_report.txt           — text summary

Usage:
    python analysis/08_group_frp_analysis.py
    python analysis/08_group_frp_analysis.py --exclude 003 008
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLUSTERS, COLORS, DERIVATIVES_DIR, FRP_WINDOWS, OUT_DIR, ROOT, analysis_subjects, sub_deriv_dir
from frp_helpers import compute_group_condition_comparisons, compute_subject_component_summary

mne.set_log_level('WARNING')

CONDS = ['active', 'passive', 'constant']
COND_LABELS = {
    'active': 'Scrolling',
    'passive': 'Watching',
    'constant': 'Baseline',
}
DIFF_COMPARISONS = [('active', 'passive'), ('active', 'constant'), ('passive', 'constant')]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('attthief.group_frp')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description='Group FRP analysis for Attentional Thief')
    p.add_argument('--exclude', nargs='*', default=[], help='Subject IDs to exclude, e.g. 003 008')
    p.add_argument('--min-per-condition', type=int, default=40,
                   help='Minimum number of fixation epochs per condition (default: 40)')
    return p.parse_args()


def discover_processed_subjects(exclude: list[str], log: logging.Logger) -> list[str]:
    subs: list[str] = []
    for sub_id in analysis_subjects(exclude):
        epo_file = sub_deriv_dir(sub_id) / 'fixation_onset-epo.fif'
        if epo_file.exists():
            subs.append(sub_id)
        else:
            log.warning(f'  sub-{sub_id}: missing fixation_onset-epo.fif — skipping')
    return subs


def load_all_epochs(subjects: list[str], min_per_condition: int,
                    log: logging.Logger) -> tuple[dict[str, mne.Epochs], list[str]]:
    all_epochs: dict[str, mne.Epochs] = {}
    included: list[str] = []

    for sub in subjects:
        epo_path = sub_deriv_dir(sub) / 'fixation_onset-epo.fif'
        try:
            epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')
        except Exception as exc:
            log.error(f'  sub-{sub}: failed to load — {exc}')
            continue

        missing = [c for c in CONDS if c not in epochs.event_id]
        if missing:
            log.warning(f'  sub-{sub}: missing conditions {missing} — skipping')
            continue

        counts = {cond: len(epochs[cond]) for cond in CONDS}
        if any(count < min_per_condition for count in counts.values()):
            log.warning(
                f"  sub-{sub}: low FRP counts {counts} — skipping (min per condition={min_per_condition})"
            )
            continue

        all_epochs[sub] = epochs
        included.append(sub)
        log.info('  sub-%s: %s', sub, ', '.join(f'{cond}={counts[cond]}' for cond in CONDS))

    if all_epochs:
        common_tmin = max(ep.tmin for ep in all_epochs.values())
        common_tmax = min(ep.tmax for ep in all_epochs.values())
        log.info(f'  Common time window: [{common_tmin:.3f}, {common_tmax:.3f}] s')
        for sub in included:
            all_epochs[sub] = all_epochs[sub].copy().crop(tmin=common_tmin, tmax=common_tmax)

    return all_epochs, included


def compute_grand_averages(all_epochs: dict[str, mne.Epochs], log: logging.Logger) -> dict[str, mne.Evoked]:
    grand_avg: dict[str, mne.Evoked] = {}
    for cond in CONDS:
        evokeds = [epochs[cond].average() for epochs in all_epochs.values()]
        grand_avg[cond] = mne.grand_average(evokeds)
        log.info(f'  Grand average {cond}: {len(evokeds)} subjects')
    return grand_avg


def compute_subject_summary(all_epochs: dict[str, mne.Epochs], log: logging.Logger) -> pd.DataFrame:
    rows = []
    for sub, epochs in all_epochs.items():
        evokeds = {cond: epochs[cond].average() for cond in CONDS}
        rows.append(
            compute_subject_component_summary(
                evokeds=evokeds,
                windows=FRP_WINDOWS,
                clusters=CLUSTERS,
                subject=sub,
            )
        )
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    log.info(f'  Subject summary rows: {len(df)}')
    return df


def _shade_windows(ax):
    colors = {
        'lambda': '#bdbdbd',
        'P1_frp': '#c6dbef',
        'N1_frp': '#fdd0a2',
        'P2_frp': '#c7e9c0',
        'Late_frp': '#fcbba1',
    }
    for name, (t0, t1) in FRP_WINDOWS.items():
        ax.axvspan(t0 * 1000, t1 * 1000, color=colors.get(name, '#eeeeee'), alpha=0.14, zorder=0)


def plot_grand_average(grand_avg: dict[str, mne.Evoked], out_dir: Path,
                       n_subjects: int, log: logging.Logger) -> None:
    plot_clusters = ['occipital', 'posterior', 'parietal', 'frontal']
    fig, axes = plt.subplots(len(plot_clusters), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Grand-Average FRP (N={n_subjects})', fontsize=14, fontweight='bold')

    if len(plot_clusters) == 1:
        axes = [axes]

    for ax, cluster_name in zip(axes, plot_clusters):
        ch_list = CLUSTERS[cluster_name]
        for cond in CONDS:
            evoked = grand_avg[cond]
            picks = [evoked.ch_names.index(ch) for ch in ch_list if ch in evoked.ch_names]
            if not picks:
                continue
            data = evoked.data[picks].mean(axis=0) * 1e6
            ax.plot(evoked.times * 1000, data, color=COLORS[cond],
                    label=COND_LABELS[cond], linewidth=1.8)
        _shade_windows(ax)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='k', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_ylabel(f'{cluster_name}\nµV')
        ax.set_title(cluster_name.replace('_', ' ').title(), fontsize=10)

    axes[0].legend(frameon=False, ncol=3, fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Time from fixation onset (ms)')
    axes[-1].set_xlim(grand_avg[CONDS[0]].times[[0, -1]] * 1000)
    plt.tight_layout()
    out_path = out_dir / 'grand_average_frp.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {out_path.relative_to(ROOT)}')


def plot_topomaps(grand_avg: dict[str, mne.Evoked], out_dir: Path,
                  n_subjects: int, log: logging.Logger) -> None:
    diff_evokeds = {
        f'{a}-{b}': mne.combine_evoked([grand_avg[a], grand_avg[b]], weights=[1, -1])
        for a, b in DIFF_COMPARISONS
    }
    all_rows = list(grand_avg.items()) + list(diff_evokeds.items())
    windows = list(FRP_WINDOWS.items())

    fig, axes = plt.subplots(len(all_rows), len(windows), figsize=(3 * len(windows), 2.8 * len(all_rows)))
    fig.suptitle(f'Group FRP Topomaps (N={n_subjects})', fontsize=14, fontweight='bold')
    if len(all_rows) == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (label, evoked) in enumerate(all_rows):
        for col_idx, (window_name, (t0, t1)) in enumerate(windows):
            ax = axes[row_idx, col_idx]
            topo = evoked.copy().crop(tmin=t0, tmax=t1).data.mean(axis=1) * 1e6
            mne.viz.plot_topomap(topo, evoked.info, axes=ax, show=False,
                                 cmap='RdBu_r', vlim=(-4, 4))
            if row_idx == 0:
                ax.set_title(f'{window_name}\n{int(t0*1000)}–{int(t1*1000)} ms', fontsize=8)
        axes[row_idx, 0].set_ylabel(label, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / 'grand_average_topomaps.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {out_path.relative_to(ROOT)}')


POST_CHS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']


def plot_difference_waves(grand_avg: dict[str, mne.Evoked], out_dir: Path,
                          n_subjects: int, log: logging.Logger) -> None:
    """Pairwise difference wave time-courses (posterior channels)."""
    comparisons = [
        ('active', 'passive', 'Scrolling − Watching'),
        ('active', 'constant', 'Scrolling − Baseline'),
        ('passive', 'constant', 'Watching − Baseline'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'FRP Difference Waves — Posterior (N={n_subjects})',
                 fontsize=13, fontweight='bold')

    for ax, (cond_a, cond_b, title) in zip(axes, comparisons):
        if cond_a not in grand_avg or cond_b not in grand_avg:
            ax.text(0.5, 0.5, f'{title}: insufficient data',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            ax.set_title(title, fontsize=10)
            continue

        evk_a, evk_b = grand_avg[cond_a], grand_avg[cond_b]
        picks_a = [evk_a.ch_names.index(ch) for ch in POST_CHS if ch in evk_a.ch_names]
        picks_b = [evk_b.ch_names.index(ch) for ch in POST_CHS if ch in evk_b.ch_names]
        diff = evk_a.data[picks_a].mean(axis=0) * 1e6 - evk_b.data[picks_b].mean(axis=0) * 1e6

        ax.plot(evk_a.times * 1000, diff, 'k-', linewidth=1.5)
        ax.fill_between(evk_a.times * 1000, diff, 0,
                        where=diff > 0, alpha=0.15, color='red')
        ax.fill_between(evk_a.times * 1000, diff, 0,
                        where=diff < 0, alpha=0.15, color='blue')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        _shade_windows(ax)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time from fixation onset (ms)')
        ax.set_xlim(-200, 800)

    axes[0].set_ylabel('Amplitude (µV)')
    plt.tight_layout()
    out_path = out_dir / 'frp_difference_waves.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {out_path.relative_to(ROOT)}')


def plot_bar_summary(summary_df: pd.DataFrame, out_dir: Path,
                     log: logging.Logger) -> None:
    """Bar plot: mean amplitude per condition × FRP window (posterior/occipital avg)."""
    df = summary_df[summary_df['cluster'].isin(['posterior', 'occipital'])].copy()
    if df.empty:
        return

    # Average posterior + occipital per subject
    df = (df.groupby(['subject', 'condition', 'window'])['mean_amp_uv']
          .mean().reset_index())

    windows = list(FRP_WINDOWS.keys())
    x = np.arange(len(windows))
    conds_present = [c for c in CONDS if c in df['condition'].unique()]
    n_conds = len(conds_present)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(conds_present):
        vals = df[df['condition'] == cond]
        summary = vals.groupby('window')['mean_amp_uv'].agg(['mean', 'sem']).reindex(windows)
        ax.bar(x + i * width, summary['mean'], width, yerr=summary['sem'],
               label=COND_LABELS.get(cond, cond), color=COLORS[cond],
               alpha=0.8, capsize=3)

    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(windows)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('FRP Mean Amplitude by Condition × Component (Posterior/Occipital)')
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    out_path = out_dir / 'frp_bar_summary.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {out_path.relative_to(ROOT)}')


def write_report(subjects: list[str], stats_df: pd.DataFrame, out_dir: Path,
                 log: logging.Logger) -> None:
    lines = [
        'Attentional Thief — Group FRP Report',
        '====================================',
        f'Included subjects (N={len(subjects)}): {", ".join(subjects)}',
        '',
    ]

    if stats_df.empty:
        lines.append('No pairwise FRP statistics were available.')
    else:
        sig = stats_df.loc[stats_df['p_value'] < 0.05].sort_values(['p_value', 'cluster', 'window'])
        if sig.empty:
            lines.append('No FRP condition comparison survived p < .05 in the current paired tests.')
        else:
            lines.append('Significant paired condition comparisons (uncorrected p < .05):')
            for _, row in sig.iterrows():
                lines.append(
                    f"- {row['cluster']} / {row['window']}: {row['comparison']} | "
                    f"Δ={row['mean_diff_uv']:.3f} µV, t={row['t_stat']:.3f}, "
                    f"p={row['p_value']:.4f}, dz={row['cohens_dz']:.3f}, n={int(row['n_pairs'])}"
                )

    out_path = out_dir / 'group_frp_report.txt'
    out_path.write_text('\n'.join(lines) + '\n')
    log.info(f'  Saved: {out_path.relative_to(ROOT)}')


def main():
    args = parse_args()
    log = setup_logging()
    out_dir = OUT_DIR / 'group_frp'
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('Discovering subjects with fixation_onset-epo.fif ...')
    subjects = discover_processed_subjects(args.exclude, log)
    log.info(f'Found {len(subjects)} candidate subjects')
    if not subjects:
        raise SystemExit('No fixation_onset-epo.fif files found. Run analysis/03_frp.py first.')

    log.info('Loading epochs ...')
    all_epochs, included = load_all_epochs(subjects, args.min_per_condition, log)
    if not included:
        raise SystemExit('No subjects passed group FRP inclusion criteria.')

    log.info('Computing grand averages ...')
    grand_avg = compute_grand_averages(all_epochs, log)

    log.info('Computing subject summaries ...')
    summary_df = compute_subject_summary(all_epochs, log)
    summary_path = out_dir / 'frp_summary_all.csv'
    summary_df.to_csv(summary_path, index=False)
    log.info(f'  Saved: {summary_path.relative_to(ROOT)}')

    log.info('Running paired condition comparisons ...')
    stats_df = compute_group_condition_comparisons(summary_df, comparisons=DIFF_COMPARISONS)
    stats_path = out_dir / 'frp_condition_comparisons.csv'
    stats_df.to_csv(stats_path, index=False)
    log.info(f'  Saved: {stats_path.relative_to(ROOT)}')

    log.info('Plotting figures ...')
    plot_grand_average(grand_avg, out_dir, len(included), log)
    plot_topomaps(grand_avg, out_dir, len(included), log)
    plot_difference_waves(grand_avg, out_dir, len(included), log)
    plot_bar_summary(summary_df, out_dir, log)
    write_report(included, stats_df, out_dir, log)
    log.info('Done.')


if __name__ == '__main__':
    main()
