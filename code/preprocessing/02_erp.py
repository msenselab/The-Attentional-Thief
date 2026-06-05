"""
Attentional Thief — Step 2: Picture-onset ERP
==============================================
Input  : derivatives/sub-{sub}/picture_onset-epo.fif
Output : sub-{sub}/erp_channels.png        — per-channel waveforms
         sub-{sub}/erp_clusters.png        — cluster-average waveforms
         sub-{sub}/erp_topomaps.png        — scalp topomaps at key latencies
         sub-{sub}/erp_summary.csv         — mean amplitudes per window

Notes:
- Scrolling vs Watching are the main comparison (baseline has too few pic-onset
  trials; baseline is analysed via FRP in 03_frp.py)
- Negative is plotted UP (standard ERP convention)

Usage:
    python analysis/02_erp.py
    python analysis/02_erp.py --sub 002
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLUSTERS, ERP_WINDOWS, COLORS, sub_out_dir, sub_deriv_dir

mne.set_log_level('WARNING')

CHANNEL_PICKS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8',   # occipital/posterior
                 'P3',  'Pz',  'P4',                    # parietal
                 'CPz', 'Cz']                            # central


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sub', default='001')
    return p.parse_args()


# ── plotting helpers ─────────────────────────────────────────────────────────

def _shade_windows(ax, windows=None):
    if windows is None:
        windows = ERP_WINDOWS
    colors = {'P1': '#aaaaaa', 'N1': '#aaaaaa',
              'P2': '#c6dbef', 'P3': '#fdae6b', 'Late': '#f28e2b'}
    for name, (a, b) in windows.items():
        ax.axvspan(a * 1000, b * 1000, color=colors.get(name, '#eeeeee'),
                   alpha=0.15, zorder=0)


def plot_channels(evokeds: dict, channels: list, out_path: Path,
                  title: str = '') -> None:
    conds = [c for c in ['active', 'passive', 'constant'] if c in evokeds]
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.8 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        for cond in conds:
            ev = evokeds[cond]
            if ch not in ev.ch_names:
                continue
            y = ev.copy().pick(ch).data[0] * 1e6
            ax.plot(ev.times * 1000, y,
                    color=COLORS[cond], lw=1.8,
                    label=f'{cond} (n={ev.nave})')
        _shade_windows(ax)
        ax.axvline(0, color='k', lw=0.8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.invert_yaxis()          # negative up
        ax.set_title(ch, fontsize=9, pad=2)
        ax.set_ylabel('µV', fontsize=8)

    axes[0].legend(frameon=False, ncol=len(conds), fontsize=8,
                   loc='upper right')
    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(evokeds[conds[0]].times[[0, -1]] * 1000)
    fig.suptitle(title, y=1.001, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_clusters(evokeds: dict, out_path: Path, title: str = '') -> None:
    conds = [c for c in ['active', 'passive', 'constant'] if c in evokeds]
    cluster_items = [(k, v) for k, v in CLUSTERS.items()
                     if k in ('occipital', 'posterior', 'parietal')]
    n = len(cluster_items)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, chs) in zip(axes, cluster_items):
        for cond in conds:
            ev = evokeds[cond]
            picks = [c for c in chs if c in ev.ch_names]
            if not picks:
                continue
            y = ev.copy().pick(picks).data.mean(axis=0) * 1e6
            ax.plot(ev.times * 1000, y,
                    color=COLORS[cond], lw=2.2,
                    label=f'{cond} (n={ev.nave})')
        _shade_windows(ax)
        ax.axvline(0, color='k', lw=0.8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.invert_yaxis()
        ax.set_title(label.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('µV')

    axes[0].legend(frameon=False, ncol=len(conds), fontsize=9)
    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(evokeds[conds[0]].times[[0, -1]] * 1000)
    fig.suptitle(title, y=1.001, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_topomaps(evokeds: dict, out_path: Path, title: str = '') -> None:
    conds = [c for c in ['active', 'passive'] if c in evokeds]
    times_ms = [100, 150, 200, 300, 500, 800]
    times_s = [t / 1000 for t in times_ms]

    n_cond = len(conds)
    n_t = len(times_s)
    fig, axes = plt.subplots(n_cond, n_t,
                             figsize=(2.4 * n_t, 2.6 * n_cond))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    for r, cond in enumerate(conds):
        ev = evokeds[cond]
        for c, t in enumerate(times_s):
            if t > ev.times[-1]:
                axes[r, c].axis('off')
                continue
            mne.viz.plot_topomap(
                ev.copy().crop(t - 0.025, t + 0.025).data.mean(axis=1) * 1e6,
                ev.info,
                axes=axes[r, c],
                show=False,
                cmap='RdBu_r',
                vlim=(-4, 4),
            )
            if r == 0:
                axes[r, c].set_title(f'{times_ms[c]} ms', fontsize=9)
        axes[r, 0].set_ylabel(cond, fontsize=9)

    fig.suptitle(title, y=1.001, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def compute_summary(evokeds: dict) -> pd.DataFrame:
    conds = list(evokeds.keys())
    rows = []
    for cluster_name, chs in CLUSTERS.items():
        for win_name, (a, b) in ERP_WINDOWS.items():
            for cond in conds:
                ev = evokeds[cond]
                picks = [c for c in chs if c in ev.ch_names]
                if not picks:
                    continue
                t = ev.times
                mask = (t >= a) & (t <= b)
                mean_amp = ev.copy().pick(picks).data[:, mask].mean() * 1e6
                rows.append(dict(
                    condition=cond, cluster=cluster_name,
                    window=win_name, n=ev.nave,
                    mean_amp_uv=round(mean_amp, 4),
                ))
    return pd.DataFrame(rows)


# ── Cluster permutation test: scrolling vs watching ───────────────────────────────

def run_cluster_permutation(epochs: mne.Epochs, evokeds: dict,
                            out_path: Path, title: str = '',
                            n_permutations: int = 1000) -> None:
    """
    Spatio-temporal cluster permutation test (scrolling vs watching).
    Uses MNE's spatio_temporal_cluster_test with channel adjacency.
    Outputs:
      - t-statistic time-course (mean over all EEG channels)
      - topomap at peak of each significant cluster (p < 0.05)
      - CSV with cluster summary
    """
    if 'active' not in epochs.event_id or 'passive' not in epochs.event_id:
        print('  [skip] cluster permutation needs scrolling + watching')
        return

    epo = epochs.copy().pick('eeg')

    # Data arrays: (n_epochs, n_ch, n_times) → transpose to (n_epochs, n_times, n_ch)
    X_active  = epo['active'].get_data().transpose(0, 2, 1)
    X_passive = epo['passive'].get_data().transpose(0, 2, 1)

    # Channel adjacency from montage
    adjacency, ch_names_adj = mne.channels.find_ch_adjacency(epo.info, 'eeg')

    n1, n2 = X_active.shape[0], X_passive.shape[0]

    print(f'  Running cluster permutation '
          f'(active n={n1}, passive n={n2}, TFCE, '
          f'{n_permutations} permutations)...')

    T_obs, clusters, cluster_p, _ = mne.stats.spatio_temporal_cluster_test(
        [X_active, X_passive],
        adjacency=adjacency,
        threshold=None,   # TFCE — avoids arbitrary threshold choice
        n_permutations=n_permutations,
        tail=0,
        n_jobs=-1,
        out_type='mask',
        verbose=False,
    )
    # T_obs shape: (n_times, n_ch)

    times = epo.times
    sig_clusters = [(c, p) for c, p in zip(clusters, cluster_p) if p < 0.05]
    print(f'  Significant clusters (p<0.05): {len(sig_clusters)}')

    # ── figure ────────────────────────────────────────────────────────────────
    n_sig = len(sig_clusters)
    n_topo_rows = max(1, n_sig)
    fig = plt.figure(figsize=(11, 3 + 2.5 * n_topo_rows))
    gs = fig.add_gridspec(1 + n_topo_rows, 1, hspace=0.55)

    # Top panel: difference wave + significant cluster shading
    ax_diff = fig.add_subplot(gs[0])
    diff = (evokeds['active'].data - evokeds['passive'].data) * 1e6  # (n_ch, n_t)
    # global mean difference
    ax_diff.plot(times * 1000, diff.mean(axis=0),
                 color='#333333', lw=1.8, label='scrolling − watching (mean all ch)')
    ax_diff.axhline(0, color='gray', lw=0.5, ls='--')
    ax_diff.axvline(0, color='k', lw=0.8)

    for mask, p in sig_clusters:
        # mask is (n_times, n_ch) boolean
        t_mask = mask.any(axis=1)
        t_indices = np.where(t_mask)[0]
        if len(t_indices) == 0:
            continue
        t_start = times[t_indices.min()] * 1000
        t_end   = times[t_indices.max()] * 1000
        ax_diff.axvspan(t_start, t_end, color='#e74c3c', alpha=0.18,
                        label=f'p={p:.3f}')

    ax_diff.set_xlabel('Time (ms)')
    ax_diff.set_ylabel('µV')
    ax_diff.set_title('scrolling − watching  (red = significant cluster, p<0.05)')
    ax_diff.legend(frameon=False, fontsize=8, loc='upper right')
    ax_diff.set_xlim(times[[0, -1]] * 1000)

    # Bottom rows: topomap at peak t for each significant cluster
    for row_i, (mask, p) in enumerate(sig_clusters):
        # peak time = time point with max |T_obs| within the cluster
        masked_T = np.where(mask, np.abs(T_obs), 0)
        peak_t_idx = masked_T.mean(axis=1).argmax()
        peak_ms = int(times[peak_t_idx] * 1000)

        ax_topo = fig.add_subplot(gs[1 + row_i])
        topo_data = T_obs[peak_t_idx]          # t-stat at each channel (n_ch,)
        mne.viz.plot_topomap(
            topo_data, epo.info,
            axes=ax_topo, show=False,
            cmap='RdBu_r',
        )
        ax_topo.set_title(f'Cluster {row_i+1}  peak={peak_ms} ms  p={p:.3f}',
                          fontsize=9)

    if n_sig == 0:
        ax_null = fig.add_subplot(gs[1])
        ax_null.text(0.5, 0.5, 'No significant clusters (p<0.05)',
                     ha='center', va='center', transform=ax_null.transAxes,
                     fontsize=11, color='gray')
        ax_null.axis('off')

    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = []
    for ci, (mask, p) in enumerate(sig_clusters):
        t_mask = mask.any(axis=1)
        ch_mask = mask.any(axis=0)
        t_indices = np.where(t_mask)[0]
        rows.append(dict(
            cluster=ci + 1,
            p_value=round(float(p), 4),
            t_start_ms=round(float(times[t_indices.min()] * 1000), 1),
            t_end_ms=round(float(times[t_indices.max()] * 1000), 1),
            n_channels=int(ch_mask.sum()),
            n_timepoints=int(t_mask.sum()),
        ))
    if rows:
        import pandas as _pd
        _pd.DataFrame(rows).to_csv(out_path.with_suffix('.csv'), index=False)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sub = args.sub
    deriv = sub_deriv_dir(sub)
    out = sub_out_dir(sub)
    epo_path = deriv / 'picture_onset-epo.fif'

    if not epo_path.exists():
        sys.exit(f'Epochs not found: {epo_path}\nRun 01_preprocess.py first.')

    print(f'\n── sub-{sub} ERP ────────────────────────')
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')

    conds = [c for c in ['active', 'passive', 'constant']
             if c in epochs.event_id and len(epochs[c]) >= 5]
    print(f'  Conditions: ' +
          ', '.join(f'{c}(n={len(epochs[c])})' for c in conds))

    evokeds = {c: epochs[c].average() for c in conds}

    # -- channel plot (scrolling + watching; note baseline for reference if enough)
    print('  Plotting per-channel waveforms...')
    ch_path = out / 'erp_channels.png'
    plot_channels(evokeds, CHANNEL_PICKS, ch_path,
                  title=f'sub-{sub} — picture-onset ERP (negative up)')
    print(f'  Saved: {ch_path.relative_to(ch_path.parents[3])}')

    # -- cluster plot
    print('  Plotting cluster averages...')
    cl_path = out / 'erp_clusters.png'
    plot_clusters(evokeds, cl_path,
                  title=f'sub-{sub} — cluster-average ERP')
    print(f'  Saved: {cl_path.relative_to(cl_path.parents[3])}')

    # -- topomaps (scrolling vs watching only)
    print('  Plotting topomaps...')
    topo_path = out / 'erp_topomaps.png'
    plot_topomaps(
        {c: evokeds[c] for c in ['active', 'passive'] if c in evokeds},
        topo_path,
        title=f'sub-{sub} — scalp topomaps',
    )
    print(f'  Saved: {topo_path.relative_to(topo_path.parents[3])}')

    # -- numeric summary
    df = compute_summary(evokeds)
    csv_path = out / 'erp_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'  Saved: {csv_path.relative_to(csv_path.parents[3])}')

    # -- cluster permutation test: scrolling vs watching
    print('  Cluster permutation test (scrolling vs watching)...')
    perm_path = out / 'erp_cluster_perm.png'
    run_cluster_permutation(
        epochs, evokeds, perm_path,
        title=f'sub-{sub} — scrolling vs watching  spatio-temporal cluster permutation',
    )
    print(f'  Saved: {perm_path.relative_to(perm_path.parents[3])}')

    print('  Done.\n')


if __name__ == '__main__':
    main()
