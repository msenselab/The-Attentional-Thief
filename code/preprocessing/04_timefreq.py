"""
Attentional Thief — Step 4: Time-Frequency Analysis
====================================================
Input  : derivatives/sub-{sub}/picture_onset-epo.fif
         data/sub-{sub}/eeg/{sub}_EC.vhdr   (eyes-closed rest, optional)

Output : sub-{sub}/tfr_oz_heatmap.png      — TFR heatmaps at Oz per condition
         sub-{sub}/tfr_alpha_timecourse.png — alpha (8-12 Hz) time course
         sub-{sub}/tfr_alpha_topomaps.png   — alpha topomap at key windows
         sub-{sub}/tfr_iaf.png              — IAF from resting PSD (if EC exists)
         sub-{sub}/tfr_data/               — raw TFR arrays (.npy) per condition
         sub-{sub}/tfr_summary.csv          — mean alpha power per window

Usage:
    python analysis/04_timefreq.py
    python analysis/04_timefreq.py --sub 002
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
from config import (
    TFR, CLUSTERS, COLORS, PREPROC,
    sub_out_dir, sub_deriv_dir, eeg_ec_vhdr,
)

mne.set_log_level('WARNING')

CONDS = ['active', 'passive', 'constant']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sub', default='001')
    return p.parse_args()


# ── TFR computation ──────────────────────────────────────────────────────────

def compute_tfr(epochs: mne.Epochs, cond: str) -> mne.time_frequency.AverageTFRArray | None:
    ep = epochs[cond] if cond in epochs.event_id else None
    if ep is None or len(ep) < 5:
        return None

    freqs = np.arange(*TFR['freqs'])
    n_cycles = freqs / TFR['n_cycles_div']

    power = ep.compute_tfr(
        method='morlet',
        freqs=freqs,
        n_cycles=n_cycles,
        decim=TFR['decim'],
        average=True,
        return_itc=False,
        verbose='ERROR',
    )
    power.apply_baseline(
        baseline=TFR['baseline'],
        mode=TFR['baseline_mode'],
        verbose='ERROR',
    )
    return power


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_heatmaps(tfr_dict: dict, out_path: Path, sub: str) -> None:
    """One TFR heatmap per condition at Oz."""
    conds = [c for c in CONDS if tfr_dict.get(c) is not None]
    if not conds:
        return

    fig, axes = plt.subplots(1, len(conds),
                              figsize=(5.5 * len(conds), 4.5),
                              sharey=True)
    if len(conds) == 1:
        axes = [axes]

    freqs = np.arange(*TFR['freqs'])
    vmax = 0.8   # ±80 % power change

    for ax, cond in zip(axes, conds):
        power = tfr_dict[cond]
        if 'Oz' not in power.ch_names:
            ax.axis('off')
            continue
        oz_idx = power.ch_names.index('Oz')
        data = power.data[oz_idx]   # (n_freqs, n_times)

        im = ax.imshow(
            data,
            aspect='auto', origin='lower',
            extent=[power.times[0] * 1000, power.times[-1] * 1000,
                    freqs[0], freqs[-1]],
            vmin=-vmax, vmax=vmax,
            cmap='RdBu_r',
        )
        ax.axvline(0, color='k', lw=1)
        # alpha band markers
        ax.axhline(TFR['alpha_band'][0], color='white', lw=0.9,
                   ls='--', alpha=0.8)
        ax.axhline(TFR['alpha_band'][1], color='white', lw=0.9,
                   ls='--', alpha=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'{cond}  (n={power.nave})',
                     color=COLORS.get(cond, 'k'), fontweight='bold')

    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=axes, label='Power change (%)',
                 fraction=0.015, pad=0.02)
    fig.suptitle(f'sub-{sub} — TFR at Oz | picture onset', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_alpha_timecourse(tfr_dict: dict, out_path: Path, sub: str) -> None:
    """Alpha (8-12 Hz) power time course averaged across occipital cluster."""
    conds = [c for c in CONDS if tfr_dict.get(c) is not None]
    if not conds:
        return

    freqs = np.arange(*TFR['freqs'])
    alpha_mask = (freqs >= TFR['alpha_band'][0]) & (freqs <= TFR['alpha_band'][1])
    occ_chs = CLUSTERS['occipital']

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # top: occipital cluster
    for cond in conds:
        power = tfr_dict[cond]
        picks = [power.ch_names.index(c)
                 for c in occ_chs if c in power.ch_names]
        if not picks:
            continue
        alpha_tc = power.data[picks][:, alpha_mask, :].mean(axis=(0, 1)) * 100
        axes[0].plot(power.times * 1000, alpha_tc,
                     color=COLORS.get(cond, '#888888'), lw=2,
                     label=f'{cond} (n={power.nave})')

    axes[0].axvline(0, color='k', lw=1, ls='--')
    axes[0].axhline(0, color='gray', lw=0.6)
    axes[0].set_ylabel('Alpha power change (%)')
    axes[0].set_title('Occipital cluster (O1, Oz, O2)')
    axes[0].legend(frameon=False, ncol=len(conds), fontsize=9)

    # bottom: posterior cluster
    post_chs = CLUSTERS['posterior']
    for cond in conds:
        power = tfr_dict[cond]
        picks = [power.ch_names.index(c)
                 for c in post_chs if c in power.ch_names]
        if not picks:
            continue
        alpha_tc = power.data[picks][:, alpha_mask, :].mean(axis=(0, 1)) * 100
        axes[1].plot(power.times * 1000, alpha_tc,
                     color=COLORS.get(cond, '#888888'), lw=2,
                     label=cond)

    axes[1].axvline(0, color='k', lw=1, ls='--')
    axes[1].axhline(0, color='gray', lw=0.6)
    axes[1].set_ylabel('Alpha power change (%)')
    axes[1].set_title('Posterior cluster (PO3/4/z/7/8)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_xlim(tfr_dict[conds[0]].times[[0, -1]] * 1000)

    fig.suptitle(f'sub-{sub} — Alpha (8–12 Hz) desynchronisation', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_alpha_topomaps(tfr_dict: dict, out_path: Path, sub: str) -> None:
    """Scalp topomaps of alpha power at key time windows."""
    conds = [c for c in CONDS if tfr_dict.get(c) is not None]
    if not conds:
        return

    freqs = np.arange(*TFR['freqs'])
    alpha_mask = (freqs >= TFR['alpha_band'][0]) & (freqs <= TFR['alpha_band'][1])

    windows_ms = [(0, 300), (300, 700), (700, 1200), (1200, 1500)]
    n_t = len(windows_ms)
    n_c = len(conds)

    fig, axes = plt.subplots(n_c, n_t,
                              figsize=(3.0 * n_t, 2.8 * n_c))
    if n_c == 1:
        axes = axes[np.newaxis, :]
    if n_t == 1:
        axes = axes[:, np.newaxis]

    for r, cond in enumerate(conds):
        power = tfr_dict[cond]
        for c, (t0_ms, t1_ms) in enumerate(windows_ms):
            t0, t1 = t0_ms / 1000, t1_ms / 1000
            t_mask = (power.times >= t0) & (power.times <= t1)
            if t_mask.sum() == 0 or t1 > power.times[-1]:
                axes[r, c].axis('off')
                continue
            # mean alpha power across time window
            alpha_map = power.data[:, alpha_mask, :][:, :, t_mask].mean(
                axis=(1, 2)) * 100
            mne.viz.plot_topomap(
                alpha_map, power.info,
                axes=axes[r, c], show=False,
                cmap='RdBu_r', vlim=(-30, 30),
            )
            if r == 0:
                axes[r, c].set_title(
                    f'{t0_ms}–{t1_ms} ms', fontsize=9)
        axes[r, 0].set_ylabel(cond, fontsize=9)

    fig.suptitle(f'sub-{sub} — Alpha power topomaps', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def analyse_iaf(ec_vhdr: Path, sub: str, out_path: Path) -> float | None:
    """Compute individual alpha frequency from eyes-closed resting PSD."""
    print('  Computing IAF from eyes-closed rest...')
    raw_ec = mne.io.read_raw_brainvision(
        str(ec_vhdr), preload=True, verbose='ERROR')
    montage = mne.channels.make_standard_montage(PREPROC['montage'])
    raw_ec.set_montage(montage, match_case=False, on_missing='ignore')
    raw_ec.filter(1.0, 40.0, verbose='ERROR')
    raw_ec.set_eeg_reference('average', verbose='ERROR')

    occ_picks = [c for c in CLUSTERS['occipital'] if c in raw_ec.ch_names]
    if not occ_picks:
        print('  [warn] No occipital channels found in EC recording.')
        return None

    psd = raw_ec.compute_psd(picks=occ_picks, fmin=1, fmax=30,
                              verbose='ERROR')
    freqs_psd = psd.freqs
    psds_mean = psd.get_data().mean(axis=0)     # average across channels

    iaf_range = (freqs_psd >= TFR['iaf_search'][0]) & \
                (freqs_psd <= TFR['iaf_search'][1])
    iaf = freqs_psd[iaf_range][np.argmax(psds_mean[iaf_range])]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(freqs_psd, psds_mean * 1e12,
                color='steelblue', lw=2, label='mean occipital PSD')
    ax.axvline(iaf, color='red', ls='--', lw=1.8,
               label=f'IAF = {iaf:.1f} Hz')
    ax.axvspan(*TFR['alpha_band'], alpha=0.12, color='orange',
               label=f'alpha band {TFR["alpha_band"]}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (µV²/Hz)')
    ax.set_title(f'sub-{sub} — Occipital PSD (eyes closed)')
    ax.set_xlim(1, 30)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    print(f'  IAF = {iaf:.1f} Hz')
    return float(iaf)


# ── summary table ─────────────────────────────────────────────────────────────

def compute_tfr_summary(tfr_dict: dict) -> pd.DataFrame:
    freqs = np.arange(*TFR['freqs'])
    alpha_mask = (freqs >= TFR['alpha_band'][0]) & (freqs <= TFR['alpha_band'][1])

    windows = {
        'early':  (0.0,  0.3),
        'mid':    (0.3,  0.7),
        'late':   (0.7,  1.2),
        'very_late': (1.2, 1.5),
    }
    rows = []
    for cond, power in tfr_dict.items():
        if power is None:
            continue
        for cl_name, chs in CLUSTERS.items():
            picks = [power.ch_names.index(c)
                     for c in chs if c in power.ch_names]
            if not picks:
                continue
            for win_name, (a, b) in windows.items():
                t_mask = (power.times >= a) & (power.times <= b)
                if t_mask.sum() == 0 or b > power.times[-1]:
                    continue
                mean_pct = (power.data[picks][:, alpha_mask, :][:, :, t_mask]
                            .mean() * 100)
                rows.append(dict(
                    condition=cond, cluster=cl_name,
                    window=win_name, n=power.nave,
                    mean_alpha_pct=round(mean_pct, 3),
                ))
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sub = args.sub
    deriv = sub_deriv_dir(sub)
    out = sub_out_dir(sub)
    epo_path = deriv / 'picture_onset-epo.fif'

    if not epo_path.exists():
        sys.exit(f'Epochs not found: {epo_path}\nRun 01_preprocess.py first.')

    print(f'\n── sub-{sub} TFR ────────────────────────')

    # 1. Load epochs
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')
    conds = [c for c in CONDS
             if c in epochs.event_id and len(epochs[c]) >= 5]
    print(f'  Conditions: ' +
          ', '.join(f'{c}(n={len(epochs[c])})' for c in conds))

    # 2. Compute TFR per condition
    print('  Computing Morlet TFR...')
    tfr_dict = {}
    for cond in conds:
        print(f'    {cond}...', end=' ', flush=True)
        tfr_dict[cond] = compute_tfr(epochs, cond)
        print('done')

    # 3. Save raw TFR arrays
    tfr_dir = out / 'tfr_data'
    tfr_dir.mkdir(exist_ok=True)
    for cond, power in tfr_dict.items():
        if power is not None:
            power.save(str(tfr_dir / f'{cond}-tfr.h5'), overwrite=True)
    print(f'  Saved TFR arrays → {tfr_dir.name}/')

    # 4. Plots
    print('  Plotting heatmaps...')
    plot_heatmaps(tfr_dict, out / 'tfr_oz_heatmap.png', sub)
    print(f'  Saved: tfr_oz_heatmap.png')

    print('  Plotting alpha time course...')
    plot_alpha_timecourse(tfr_dict, out / 'tfr_alpha_timecourse.png', sub)
    print(f'  Saved: tfr_alpha_timecourse.png')

    print('  Plotting alpha topomaps...')
    plot_alpha_topomaps(tfr_dict, out / 'tfr_alpha_topomaps.png', sub)
    print(f'  Saved: tfr_alpha_topomaps.png')

    # 5. IAF from eyes-closed rest (optional)
    ec_vhdr = eeg_ec_vhdr(sub)
    iaf = None
    if ec_vhdr.exists():
        iaf = analyse_iaf(ec_vhdr, sub, out / 'tfr_iaf.png')
        print(f'  Saved: tfr_iaf.png')
    else:
        print(f'  [skip] No eyes-closed recording found ({ec_vhdr.name})')

    # 6. Summary CSV
    df = compute_tfr_summary(tfr_dict)
    if iaf is not None:
        df['iaf_hz'] = iaf
    df.to_csv(out / 'tfr_summary.csv', index=False)
    print(f'  Saved: tfr_summary.csv')

    print('  Done.\n')


if __name__ == '__main__':
    main()
