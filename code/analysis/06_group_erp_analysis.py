"""
Attentional Thief — Step 6: Group ERP Analysis (Scrolling vs Watching)
===================================================================
Loads picture-onset epochs from all preprocessed subjects, computes
grand-average ERPs, runs cluster-based permutation tests, and generates
publication-ready figures.

Output : analysis/output/group_erp/
           grand_average_erp.png       — grand-average waveforms (scrolling vs watching)
           grand_average_topomaps.png  — topographic maps at ERP windows
           cluster_permutation.png     — cluster-based permutation t-test results
           erp_summary_all.csv        — per-subject mean amplitudes per window
           grand_average_report.txt   — text summary

Usage:
    python analysis/06_group_erp_analysis.py
    python analysis/06_group_erp_analysis.py --exclude 003 008
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from mne.stats import spatio_temporal_cluster_test
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ROOT, DERIVATIVES_DIR, CLUSTERS, ERP_WINDOWS, COLORS,
    sub_deriv_dir, sub_out_dir, OUT_DIR, analysis_subjects,
)

mne.set_log_level('WARNING')

# ── Parameters ───────────────────────────────────────────────────────────────
CONDS = ['active', 'passive']
COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching'}
CLUSTER_CHANNELS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2',
                    'P3', 'Pz', 'P4', 'CPz']
N_PERMUTATIONS = 5000
ALPHA = 0.05


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('attthief.group_erp')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    return logger


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Group ERP analysis for Attentional Thief')
    p.add_argument('--exclude', nargs='*', default=[],
                   help='Subject IDs to exclude, e.g. 003 008')
    p.add_argument('--min-epochs', type=int, default=50,
                   help='Minimum total epochs per subject (default: 50)')
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────
def discover_processed_subjects(exclude: list[str], log: logging.Logger) -> list[str]:
    """Find canonical analysis subjects with preprocessed epochs."""
    subs = []
    for sub_id in analysis_subjects(exclude):
        epo_file = sub_deriv_dir(sub_id) / 'picture_onset-epo.fif'
        if epo_file.exists():
            subs.append(sub_id)
        else:
            log.warning(f'  sub-{sub_id}: missing picture_onset-epo.fif — skipping')
    return subs


def load_all_epochs(subjects: list[str], min_epochs: int,
                    log: logging.Logger) -> tuple[dict, list[str]]:
    """
    Load epochs for all subjects.
    Crops all epochs to the common time window to ensure compatibility.
    Returns dict of {sub: epochs} and list of included subjects.
    """
    all_epochs = {}
    included = []

    for sub in subjects:
        epo_path = sub_deriv_dir(sub) / 'picture_onset-epo.fif'
        try:
            epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')

            # Check both conditions exist
            missing = [c for c in CONDS if c not in epochs.event_id]
            if missing:
                log.warning(f'  sub-{sub}: missing conditions {missing} — skipping')
                continue

            n_total = sum(len(epochs[c]) for c in CONDS)
            if n_total < min_epochs:
                log.warning(f'  sub-{sub}: only {n_total} epochs — skipping (min={min_epochs})')
                continue

            all_epochs[sub] = epochs
            included.append(sub)
            n_active = len(epochs['active'])
            n_passive = len(epochs['passive'])
            log.info(f'  sub-{sub}: scrolling={n_active}, watching={n_passive} '
                     f'(tmin={epochs.tmin:.3f}, tmax={epochs.tmax:.3f})')

        except Exception as e:
            log.error(f'  sub-{sub}: failed to load — {e}')

    # Crop all to common time window
    if all_epochs:
        common_tmin = max(ep.tmin for ep in all_epochs.values())
        common_tmax = min(ep.tmax for ep in all_epochs.values())
        log.info(f'  Common time window: [{common_tmin:.3f}, {common_tmax:.3f}] s')
        for sub in list(all_epochs.keys()):
            all_epochs[sub] = all_epochs[sub].crop(tmin=common_tmin, tmax=common_tmax)

    return all_epochs, included


# ── Analysis ─────────────────────────────────────────────────────────────────
def compute_grand_averages(all_epochs: dict, log: logging.Logger) -> dict:
    """Compute grand-average evokeds per condition."""
    evokeds = {c: [] for c in CONDS}

    for sub, epochs in all_epochs.items():
        for cond in CONDS:
            evokeds[cond].append(epochs[cond].average())

    grand_avg = {}
    for cond in CONDS:
        grand_avg[cond] = mne.grand_average(evokeds[cond])
        log.info(f'  Grand average {cond}: {len(evokeds[cond])} subjects')

    return grand_avg


def compute_subject_amplitudes(all_epochs: dict, log: logging.Logger) -> pd.DataFrame:
    """Extract mean amplitudes per subject × condition × ERP window × cluster."""
    rows = []

    for sub, epochs in all_epochs.items():
        for cond in CONDS:
            evoked = epochs[cond].average()
            for win_name, (t_start, t_end) in ERP_WINDOWS.items():
                for clust_name, ch_list in CLUSTERS.items():
                    # Pick channels that exist
                    picks = [ch for ch in ch_list if ch in evoked.ch_names]
                    if not picks:
                        continue
                    idx = [evoked.ch_names.index(ch) for ch in picks]
                    t_mask = (evoked.times >= t_start) & (evoked.times <= t_end)
                    amp = evoked.data[np.ix_(idx, t_mask)].mean() * 1e6  # µV

                    rows.append(dict(
                        sub=sub, condition=cond,
                        window=win_name, cluster=clust_name,
                        amplitude_uV=round(amp, 3),
                    ))

    df = pd.DataFrame(rows)
    log.info(f'  Amplitude table: {len(df)} rows')
    return df


def run_cluster_permutation(all_epochs: dict, log: logging.Logger) -> dict:
    """
    Cluster-based permutation test: scrolling vs watching across all channels/times.
    Returns dict with results.
    """
    log.info('Running cluster-based permutation test...')

    # Get common EEG-only channels across subjects
    sample_epochs = list(all_epochs.values())[0]
    eeg_picks = mne.pick_types(sample_epochs.info, eeg=True, exclude=[])
    eeg_ch_set = set(sample_epochs.ch_names[i] for i in eeg_picks)

    common_chs = eeg_ch_set.copy()
    for epochs in all_epochs.values():
        eeg_idx = mne.pick_types(epochs.info, eeg=True, exclude=[])
        common_chs &= set(epochs.ch_names[i] for i in eeg_idx)
    common_chs = sorted(common_chs)

    # Build data arrays: (n_subjects, n_times, n_channels)
    active_data = []
    passive_data = []
    ch_names_used = common_chs

    for sub, epochs in all_epochs.items():
        active_evk = epochs['active'].average()
        passive_evk = epochs['passive'].average()

        active_picks = [active_evk.ch_names.index(ch) for ch in ch_names_used]
        passive_picks = [passive_evk.ch_names.index(ch) for ch in ch_names_used]

        active_data.append(active_evk.data[active_picks].T)   # (n_times, n_channels)
        passive_data.append(passive_evk.data[passive_picks].T)

    X_active = np.array(active_data)    # (n_subjects, n_times, n_channels)
    X_passive = np.array(passive_data)

    log.info(f'  Data shape: {X_active.shape} (subjects × times × channels)')

    # Compute adjacency from channel positions
    info_temp = mne.pick_info(sample_epochs.info,
                              [sample_epochs.ch_names.index(ch) for ch in ch_names_used])
    adjacency, _ = mne.channels.find_ch_adjacency(info_temp, ch_type='eeg')

    # Run test
    X = [X_active, X_passive]
    t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
        X, n_permutations=N_PERMUTATIONS, threshold=None,
        adjacency=adjacency, n_jobs=-1, verbose=False,
        out_type='mask',
    )

    # Find significant clusters
    sig_clusters = [(i, pv) for i, pv in enumerate(cluster_pv) if pv < ALPHA]
    log.info(f'  Significant clusters: {len(sig_clusters)} (α={ALPHA})')
    for i, pv in sig_clusters:
        log.info(f'    Cluster {i}: p={pv:.4f}')

    times = sample_epochs.times if hasattr(sample_epochs, 'times') else list(all_epochs.values())[0]['active'].times

    return dict(
        t_obs=t_obs,
        clusters=clusters,
        cluster_pv=cluster_pv,
        sig_clusters=sig_clusters,
        ch_names=ch_names_used,
        times=list(all_epochs.values())[0]['active'].times,
        n_subjects=len(all_epochs),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_grand_average(grand_avg: dict, out_dir: Path,
                       n_subjects: int, log: logging.Logger) -> None:
    """Plot grand-average ERP waveforms for key channel clusters."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Grand-Average ERP: Scrolling vs Watching (N={n_subjects})',
                 fontsize=14, fontweight='bold')

    plot_clusters = {
        'Posterior (PO7/PO3/POz/PO4/PO8)': ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],
        'Occipital (O1/Oz/O2)': ['O1', 'Oz', 'O2'],
        'Parietal (P3/Pz/P4)': ['P3', 'Pz', 'P4'],
        'Central-Parietal (CPz)': ['CPz'],
        'Frontal (F3/Fz/F4)': ['F3', 'Fz', 'F4'],
        'Central (Cz)': ['Cz'],
    }

    for ax, (title, ch_list) in zip(axes.flat, plot_clusters.items()):
        for cond in CONDS:
            evk = grand_avg[cond]
            picks = [evk.ch_names.index(ch) for ch in ch_list if ch in evk.ch_names]
            if not picks:
                continue
            data = evk.data[picks].mean(axis=0) * 1e6  # µV
            ax.plot(evk.times * 1000, data, color=COLORS[cond],
                    label=COND_LABELS.get(cond, cond.capitalize()), linewidth=1.5)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()  # Negative up
        ax.set_xlim(-200, 1000)

        # Shade ERP windows
        for win_name, (t0, t1) in ERP_WINDOWS.items():
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.05, color='blue')

    axes[0, 0].legend(loc='upper left', fontsize=9)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 2].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[1, 0].set_ylabel('Amplitude (µV)')

    plt.tight_layout()
    fig_path = out_dir / 'grand_average_erp.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {fig_path.relative_to(ROOT)}')


def plot_topomaps(grand_avg: dict, out_dir: Path,
                  n_subjects: int, log: logging.Logger) -> None:
    """Plot topographic maps at each ERP window for both conditions."""
    n_wins = len(ERP_WINDOWS)
    fig, axes = plt.subplots(2, n_wins, figsize=(3 * n_wins, 6))
    fig.suptitle(f'Topographic Maps: Scrolling vs Watching (N={n_subjects})',
                 fontsize=14, fontweight='bold')

    for i, cond in enumerate(CONDS):
        evk = grand_avg[cond]
        for j, (win_name, (t0, t1)) in enumerate(ERP_WINDOWS.items()):
            ax = axes[i, j]
            evk.plot_topomap(times=[(t0 + t1) / 2], axes=ax,
                             show=False, colorbar=False,
                             vlim=(-5, 5))
            if i == 0:
                ax.set_title(f'{win_name}\n({int(t0*1000)}–{int(t1*1000)} ms)',
                             fontsize=9)
        axes[i, 0].set_ylabel(COND_LABELS.get(cond, cond.capitalize()), fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig_path = out_dir / 'grand_average_topomaps.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {fig_path.relative_to(ROOT)}')


def plot_difference_wave(grand_avg: dict, out_dir: Path,
                         cluster_results: dict, n_subjects: int,
                         log: logging.Logger) -> None:
    """Plot scrolling–watching difference wave with significant clusters highlighted."""
    evk_diff = mne.combine_evoked(
        [grand_avg['active'], grand_avg['passive']], weights=[1, -1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Scrolling − Watching Difference (N={n_subjects})',
                 fontsize=14, fontweight='bold')

    # Left: Posterior cluster difference wave
    post_chs = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
    picks = [evk_diff.ch_names.index(ch) for ch in post_chs if ch in evk_diff.ch_names]
    diff_data = evk_diff.data[picks].mean(axis=0) * 1e6

    axes[0].plot(evk_diff.times * 1000, diff_data, 'k-', linewidth=1.5)
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].fill_between(evk_diff.times * 1000, diff_data, 0,
                         where=diff_data > 0, alpha=0.15, color='red')
    axes[0].fill_between(evk_diff.times * 1000, diff_data, 0,
                         where=diff_data < 0, alpha=0.15, color='blue')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title('Posterior Channels')
    axes[0].set_xlim(-200, 1000)

    # Right: t-statistic map (if cluster test was run)
    if cluster_results and cluster_results['t_obs'] is not None:
        t_obs = cluster_results['t_obs']
        times = cluster_results['times']
        ch_names = cluster_results['ch_names']

        # Average t across posterior channels
        post_idx = [ch_names.index(ch) for ch in post_chs if ch in ch_names]
        if post_idx:
            t_post = t_obs[:, post_idx].mean(axis=1)
            axes[1].plot(times * 1000, t_post, 'k-', linewidth=1.5)
            axes[1].axhline(0, color='gray', linewidth=0.5)
            axes[1].axvline(0, color='gray', linewidth=0.5, linestyle='--')

            # Highlight significant clusters
            for ci, pv in cluster_results['sig_clusters']:
                mask = cluster_results['clusters'][ci]
                # mask is (n_times, n_channels)
                t_mask = mask[:, post_idx].any(axis=1) if len(mask.shape) > 1 else mask
                sig_times = times[t_mask] * 1000
                if len(sig_times) > 0:
                    axes[1].axvspan(sig_times.min(), sig_times.max(),
                                    alpha=0.2, color='green',
                                    label=f'p={pv:.3f}')

            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('t-statistic')
            axes[1].set_title('Cluster Permutation Test')
            axes[1].set_xlim(-200, 1000)
            if cluster_results['sig_clusters']:
                axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'Cluster test not run\n(need ≥ 2 subjects)',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    fig_path = out_dir / 'difference_wave.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {fig_path.relative_to(ROOT)}')


def plot_bar_summary(amp_df: pd.DataFrame, out_dir: Path,
                     log: logging.Logger) -> None:
    """Bar plot of mean amplitudes per condition × window for posterior cluster."""
    df = amp_df[amp_df['cluster'] == 'posterior'].copy()
    if df.empty:
        df = amp_df[amp_df['cluster'] == 'occipital'].copy()
    if df.empty:
        log.warning('  No data for bar summary plot')
        return

    # Aggregate across subjects
    summary = df.groupby(['condition', 'window']).agg(
        mean=('amplitude_uV', 'mean'),
        sem=('amplitude_uV', lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    windows = list(ERP_WINDOWS.keys())
    x = np.arange(len(windows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(CONDS):
        d = summary[summary['condition'] == cond]
        d = d.set_index('window').reindex(windows)
        ax.bar(x + i * width, d['mean'], width, yerr=d['sem'],
               label=COND_LABELS.get(cond, cond.capitalize()), color=COLORS[cond], alpha=0.8,
               capsize=3)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(windows)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Mean ERP Amplitude by Condition × Window (Posterior)')
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig_path = out_dir / 'erp_bar_summary.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: {fig_path.relative_to(ROOT)}')


# ── Report ───────────────────────────────────────────────────────────────────
def write_report(out_dir: Path, subjects: list[str], amp_df: pd.DataFrame,
                 cluster_results: dict, log: logging.Logger) -> None:
    """Write text summary report."""
    lines = [
        f'Group ERP Analysis Report — Attentional Thief\n',
        f'{"=" * 50}\n',
        f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n',
        f'Subjects included: {len(subjects)}\n',
        f'Subject IDs: {", ".join(subjects)}\n\n',
    ]

    # Per-window stats
    lines.append('Mean Amplitudes (µV) — Posterior Cluster\n')
    lines.append(f'{"Window":<8} {"Scrolling":>10} {"Watching":>10} {"Diff":>10} {"t":>8} {"p":>8}\n')
    lines.append('-' * 60 + '\n')

    # Average posterior + occipital amplitudes per subject before t-test
    post_occ_df = amp_df[amp_df['cluster'].isin(['posterior', 'occipital'])]
    post_occ_avg = (post_occ_df.groupby(['sub', 'condition', 'window'])['amplitude_uV']
                    .mean().reset_index())
    for win in ERP_WINDOWS:
        active = post_occ_avg[(post_occ_avg['condition'] == 'active') & (post_occ_avg['window'] == win)]['amplitude_uV']
        passive = post_occ_avg[(post_occ_avg['condition'] == 'passive') & (post_occ_avg['window'] == win)]['amplitude_uV']
        if len(active) > 1 and len(passive) > 1:
            t_val, p_val = stats.ttest_rel(active.values, passive.values)
            lines.append(f'{win:<8} {active.mean():>10.2f} {passive.mean():>10.2f} '
                         f'{(active.mean()-passive.mean()):>10.2f} {t_val:>8.2f} {p_val:>8.4f}\n')
        else:
            lines.append(f'{win:<8} {active.mean():>10.2f} {passive.mean():>10.2f} '
                         f'{(active.mean()-passive.mean()):>10.2f} {"—":>8} {"—":>8}\n')

    # Cluster permutation results
    if cluster_results and cluster_results['sig_clusters']:
        lines.append(f'\nCluster Permutation Test (N={cluster_results["n_subjects"]}, '
                     f'{N_PERMUTATIONS} permutations)\n')
        for ci, pv in cluster_results['sig_clusters']:
            mask = cluster_results['clusters'][ci]
            times = cluster_results['times']
            if len(mask.shape) > 1:
                t_mask = mask.any(axis=1)
            else:
                t_mask = mask
            t_range = times[t_mask] * 1000
            lines.append(f'  Cluster {ci}: p={pv:.4f}, '
                         f'time={t_range.min():.0f}–{t_range.max():.0f} ms\n')
    else:
        lines.append('\nNo significant clusters found in permutation test.\n')

    report_path = out_dir / 'grand_average_report.txt'
    report_path.write_text(''.join(lines))
    log.info(f'  Saved: {report_path.relative_to(ROOT)}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log = setup_logging()

    log.info('═' * 50)
    log.info('GROUP ERP ANALYSIS — Attentional Thief')
    log.info('═' * 50)

    # Output directory
    out_dir = OUT_DIR / 'group_erp'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    exclude = [s.zfill(3) for s in args.exclude]
    subjects = discover_processed_subjects(exclude, log)
    log.info(f'Found {len(subjects)} preprocessed subjects')

    # Load epochs
    log.info('Loading epochs...')
    all_epochs, included = load_all_epochs(subjects, args.min_epochs, log)
    log.info(f'Included: {len(included)} subjects')

    if len(included) < 2:
        log.error('Need at least 2 subjects for group analysis')
        sys.exit(1)

    # Compute grand averages
    log.info('Computing grand averages...')
    grand_avg = compute_grand_averages(all_epochs, log)

    # Extract amplitudes per subject
    log.info('Extracting mean amplitudes...')
    amp_df = compute_subject_amplitudes(all_epochs, log)
    amp_df.to_csv(out_dir / 'erp_summary_all.csv', index=False)
    log.info(f'  Saved: {(out_dir / "erp_summary_all.csv").relative_to(ROOT)}')

    # Cluster permutation test
    cluster_results = {}
    if len(included) >= 6:
        cluster_results = run_cluster_permutation(all_epochs, log)
    else:
        log.warning(f'  Skipping cluster permutation (need ≥6 subjects, have {len(included)})')

    # Plots
    log.info('Generating figures...')
    plot_grand_average(grand_avg, out_dir, len(included), log)
    plot_topomaps(grand_avg, out_dir, len(included), log)
    plot_difference_wave(grand_avg, out_dir, cluster_results, len(included), log)
    plot_bar_summary(amp_df, out_dir, log)

    # Report
    write_report(out_dir, included, amp_df, cluster_results, log)

    log.info('═' * 50)
    log.info('Done!')
    log.info(f'Results: {out_dir.relative_to(ROOT)}/')
    log.info('═' * 50)


if __name__ == '__main__':
    main()
