"""
Attentional Thief — Step 5: Group ERP / FRP / SRP comparison
=============================================================
Plots picture-onset ERP, fixation-onset FRP, and saccade-onset SRP
for sub-001 and sub-002 side by side.

Output : analysis/output/group_erp_frp_srp.png

Usage:
    python analysis/05_group_erp_frp_srp.py
"""
import gc
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import eyelinkio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLUSTERS, COLORS, DERIVATIVES_DIR, analysis_subjects, et_edf, normalize_subject_id, sub_out_dir, sub_deriv_dir
from et_eeg_sync import build_sync_pairs

mne.set_log_level('WARNING')

# ── parameters ────────────────────────────────────────────────────────────────
SUBS   = None  # auto-discovered below from derivatives/
CONDS  = ['active', 'passive', 'constant']
COND_COLORS = COLORS  # scrolling=black, watching=red, baseline=blue

# Epoch windows
FRP_TMIN, FRP_TMAX = -0.2, 0.8
FRP_BASELINE       = (-0.2, 0.0)
FRP_REJECT         = 150e-6

SRP_TMIN, SRP_TMAX = -0.2, 0.5   # saccade-onset shorter window
SRP_BASELINE       = (-0.2, 0.0)
SRP_REJECT         = 150e-6

SAC_DUR_MIN_MS = 10
SAC_DUR_MAX_MS = 150

# Trigger maps (for re-building SRP sync)
BLOCK_START = {10: 'active', 20: 'passive', 30: 'constant'}
BLOCK_END   = {19: 'active', 29: 'passive', 39: 'constant'}
PIC_ONSET   = {12: 'active', 22: 'passive', 32: 'constant'}

# EDF path helper (sub-001 uses sub001.EDF, sub-002 uses sub002.EDF)
def _et_edf(sub):
    return et_edf(sub)


# ── EEG helpers (same logic as 03_frp.py) ────────────────────────────────────

def _stim_code(desc):
    m = re.search(r'S\s*(\d+)', desc)
    return int(m.group(1)) if m else None


def segment_eeg_blocks(raw):
    sfreq = raw.info['sfreq']
    blocks, current = [], None
    for ann in raw.annotations:
        code = _stim_code(ann['description'])
        if code is None:
            continue
        onset_s = ann['onset']
        if code in BLOCK_START:
            current = dict(block_type=BLOCK_START[code], start_s=onset_s,
                           end_s=None, pic_samples=[])
        elif code in PIC_ONSET and current is not None:
            current['pic_samples'].append(int(round(onset_s * sfreq)))
        elif code in BLOCK_END and current is not None:
            current['end_s'] = onset_s
            blocks.append(current)
            current = None
    return blocks


def parse_et_data(edf_path):
    """Return (et_blocks, fix_df, sac_df) from EDF."""
    edf = eyelinkio.read_edf(str(edf_path))
    msgs = [(t, m.decode(errors='replace').strip())
            for t, m in edf.discrete['messages']]

    blocks, current = [], None
    for t, msg in msgs:
        if msg.startswith('BLOCK_START'):
            m = re.match(
                r'BLOCK_START\s+(\w+)\s+rep=(\d+)\s+session=(\d+)\s+n_pics=(\d+)', msg)
            if m and int(m.group(2)) > 0:
                current = dict(
                    block_type=m.group(1).lower().replace('geometry', 'constant'),
                    rep=int(m.group(2)), session=int(m.group(3)),
                    n_pics=int(m.group(4)), images=[])
        elif msg.startswith('IMAGE_ON') and current is not None:
            m2 = re.search(r'pic=(\d+)/(\d+)', msg)
            pic_idx = int(m2.group(1)) if m2 else len(current['images']) + 1
            current['images'].append(dict(et_on=float(t), et_off=None, pic_idx=pic_idx))
        elif msg.startswith('IMAGE_OFF') and current is not None:
            if current['images'] and current['images'][-1]['et_off'] is None:
                current['images'][-1]['et_off'] = float(t)
        elif msg.startswith('BLOCK_END') and current is not None:
            blocks.append(current)
            current = None

    # fixations
    fix = pd.DataFrame(edf.discrete['fixations'])
    fix = fix.rename(columns={c: n for c, n in
                               [('axp','x'),('ayp','y'),('sttime','stime'),('entime','etime')]
                               if c in fix.columns})
    fix['dur_ms'] = (fix['etime'] - fix['stime']) * 1000
    fix = fix[(fix['dur_ms'] >= 80) & (fix['dur_ms'] <= 1500)].reset_index(drop=True)

    # saccades
    sac = pd.DataFrame(edf.discrete['saccades'])
    sac = sac.rename(columns={c: n for c, n in
                               [('sttime','stime'),('entime','etime')]
                               if c in sac.columns})
    sac['dur_ms'] = (sac['etime'] - sac['stime']) * 1000
    sac = sac[(sac['dur_ms'] >= SAC_DUR_MIN_MS) &
              (sac['dur_ms'] <= SAC_DUR_MAX_MS)].reset_index(drop=True)

    return blocks, fix, sac


def assign_et_events(et_df, pairs, sfreq, time_col='stime', skip_first=False):
    """
    Map ET event onsets (fixations or saccades) to EEG samples.
    skip_first=True: exclude the first event per image window (for FRP,
    to avoid mixing picture-onset ERP into the first fixation).

    Vectorized via searchsorted on (et_on+0.08) — assumes per-image
    windows are non-overlapping, which is true for sequential blocks.
    """
    if len(et_df) == 0 or len(pairs) == 0:
        out = et_df.iloc[0:0].copy()
        out['block_type'] = pd.Series(dtype=object)
        out['onset_sample'] = pd.Series(dtype=int)
        return out.reset_index(drop=True)

    p_sorted = pairs.sort_values('et_on').reset_index(drop=True)
    p_on = p_sorted['et_on'].to_numpy(dtype=float)
    p_off = p_sorted['et_off'].to_numpy(dtype=float)
    p_eeg = p_sorted['eeg_pic_s'].to_numpy(dtype=float)
    p_block = p_sorted['block_type'].to_numpy()

    win_lo = p_on + 0.08
    win_hi = p_off - 0.08
    ev_t = et_df[time_col].to_numpy(dtype=float)

    # Rightmost window starting at or before each event.
    idx = np.searchsorted(win_lo, ev_t, side='right') - 1
    in_range = (idx >= 0) & (idx < len(win_lo))
    valid = np.zeros_like(in_range)
    if in_range.any():
        cand = np.where(in_range)[0]
        keep = ev_t[cand] <= win_hi[idx[cand]]
        valid[cand[keep]] = True

    if skip_first:
        # First valid event (in et_df order) per image window — drop it.
        v_pos = np.where(valid)[0]
        if len(v_pos):
            df = pd.DataFrame({'pos': v_pos, 'img': idx[v_pos]})
            first_pos = df.drop_duplicates('img', keep='first')['pos'].to_numpy()
            valid[first_pos] = False

    sel = np.where(valid)[0]
    if len(sel) == 0:
        out = et_df.iloc[0:0].copy()
        out['block_type'] = pd.Series(dtype=object)
        out['onset_sample'] = pd.Series(dtype=int)
        return out.reset_index(drop=True)

    sel_img = idx[sel]
    dt = ev_t[sel] - p_on[sel_img]
    onset_sample = np.round((p_eeg[sel_img] + dt) * sfreq).astype(int)

    out = et_df.iloc[sel].copy().reset_index(drop=True)
    out['block_type'] = p_block[sel_img]
    out['onset_sample'] = onset_sample
    return out


def make_epochs(raw, assigned_df, tmin, tmax, baseline, reject):
    code_map = {'active': 1, 'passive': 2, 'constant': 3}
    evs = np.column_stack([
        assigned_df['onset_sample'].to_numpy(),
        np.zeros(len(assigned_df), dtype=int),
        assigned_df['block_type'].map(code_map).to_numpy(),
    ])
    epo = mne.Epochs(raw, evs, event_id=code_map,
                     tmin=tmin, tmax=tmax, baseline=baseline,
                     preload=True, reject=dict(eeg=reject),
                     reject_by_annotation=True, verbose='ERROR')
    epo.drop_bad(verbose='ERROR')
    return epo


# ── load all data ─────────────────────────────────────────────────────────────

def _evokeds_per_cond(epo, conds):
    """Return {cond: Evoked or None} and {cond: n_epochs}, then caller can drop epo."""
    evokeds, counts = {}, {}
    for c in conds:
        if c in epo.event_id and len(epo[c]) > 0:
            evokeds[c] = epo[c].average()
            counts[c] = len(epo[c])
        else:
            evokeds[c] = None
            counts[c] = 0
    return evokeds, counts


def load_sub(sub):
    """
    Returns three (evokeds, counts) tuples — one per epoch type.
    Heavy objects (Raw, Epochs) are freed before returning so that
    `all_data` only holds light Evokeds across subjects.
    """
    deriv = sub_deriv_dir(sub)

    # 1. ERP (pre-built) — average then drop.
    erp_epo = mne.read_epochs(str(deriv / 'picture_onset-epo.fif'),
                               preload=True, verbose='ERROR')
    all_conds = list(set(CONDS) | set(ERP_CONDS))
    erp_evokeds, erp_counts = _evokeds_per_cond(erp_epo, all_conds)
    print(f'sub-{sub}  ERP: {dict((c, erp_counts[c]) for c in CONDS if c in erp_counts)}')
    del erp_epo
    gc.collect()

    # 2+3. FRP (first-fixation excluded) + SRP — build from EDF + preprocessed raw.
    raw = mne.io.read_raw_fif(str(deriv / 'preprocessed-raw.fif'),
                               preload=True, verbose='ERROR')
    eeg_blocks = segment_eeg_blocks(raw)
    et_blocks, fix_df, sac_df = parse_et_data(_et_edf(sub))
    pairs = build_sync_pairs(eeg_blocks, et_blocks)
    sfreq = raw.info['sfreq']

    fix_assigned = assign_et_events(fix_df, pairs, sfreq,
                                    time_col='stime', skip_first=True)
    del fix_df
    frp_epo = make_epochs(raw, fix_assigned,
                          FRP_TMIN, FRP_TMAX, FRP_BASELINE, FRP_REJECT)
    del fix_assigned
    frp_evokeds, frp_counts = _evokeds_per_cond(frp_epo, CONDS)
    print(f'sub-{sub}  FRP (skip1st): {frp_counts}')
    del frp_epo
    gc.collect()

    sac_assigned = assign_et_events(sac_df, pairs, sfreq, time_col='stime')
    del sac_df
    srp_epo = make_epochs(raw, sac_assigned,
                          SRP_TMIN, SRP_TMAX, SRP_BASELINE, SRP_REJECT)
    del sac_assigned, raw
    srp_evokeds, srp_counts = _evokeds_per_cond(srp_epo, CONDS)
    print(f'sub-{sub}  SRP: {srp_counts}')
    del srp_epo
    gc.collect()

    return (erp_evokeds, erp_counts), (frp_evokeds, frp_counts), (srp_evokeds, srp_counts)


# ── plotting ──────────────────────────────────────────────────────────────────

PLOT_CLUSTERS = ['occipital', 'posterior', 'parietal', 'frontal']

# Individual channels to inspect for ERP
ERP_CHANNELS = [
    'O1', 'Oz', 'O2',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'P3', 'Pz', 'P4',
    'CPz', 'Cz',
    'F3', 'Fz', 'F4',
]
ERP_CONDS = ['active', 'passive']   # no baseline for ERP

def _evoked_cluster(ev, cluster):
    """Return (cluster-mean trace in µV, n_epochs). ev may be None."""
    if ev is None:
        return None, 0
    picks = [c for c in CLUSTERS[cluster] if c in ev.ch_names]
    if not picks:
        return None, ev.nave
    pick_idx = [ev.ch_names.index(c) for c in picks]
    y = ev.data[pick_idx].mean(axis=0) * 1e6
    return y, ev.nave


def plot_panel(ax, sub_evokeds, sub_counts, cluster, tmin, tmax, conds, title):
    """Plot one cluster panel: multiple conditions overlaid."""
    times = None
    for cond in conds:
        ev = sub_evokeds.get(cond)
        n = sub_counts.get(cond, 0)
        if ev is None or n < 5:
            continue
        y, _ = _evoked_cluster(ev, cluster)
        if y is None:
            continue
        if times is None:
            times = ev.times * 1000
        ax.plot(times, y, color=COND_COLORS[cond], lw=1.8,
                label=f'{cond} (n={n})')
    ax.axvline(0, color='k', lw=0.8)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    if times is not None:
        ax.set_xlim(tmin * 1000, tmax * 1000)


def make_erp_channels_figure(all_data):
    """Per-channel ERP, scrolling vs watching only, all subjects side by side."""
    subs = sorted(all_data.keys())
    n_rows = len(ERP_CHANNELS)
    n_cols = len(subs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 2.2 * n_rows),
                             sharey='row', sharex=True)
    fig.patch.set_facecolor('white')
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for sub_i, sub in enumerate(subs):
        sub_evokeds, sub_counts = all_data[sub][0]   # ERP evokeds
        for r, ch in enumerate(ERP_CHANNELS):
            ax = axes[r, sub_i]
            for cond in ERP_CONDS:
                ev = sub_evokeds.get(cond)
                n = sub_counts.get(cond, 0)
                if ev is None or n < 5:
                    continue
                if ch not in ev.ch_names:
                    ax.text(0.5, 0.5, f'{ch} missing', transform=ax.transAxes,
                            ha='center', va='center', fontsize=7, color='gray')
                    continue
                y = ev.data[ev.ch_names.index(ch)] * 1e6
                ax.plot(ev.times * 1000, y, color=COND_COLORS[cond],
                        lw=1.6, label=f'{cond} (n={n})')
            ax.axvline(0, color='k', lw=0.7)
            ax.axhline(0, color='gray', lw=0.4, ls='--')
            ax.invert_yaxis()
            ax.set_xlim(-200, 1500)
            if sub_i == 0:
                ax.set_ylabel(ch, fontsize=8)
            if r == 0:
                ax.set_title(f'sub-{sub}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=7, frameon=False, loc='upper right')
            if r == n_rows - 1:
                ax.set_xlabel('Time (ms)', fontsize=9)

    fig.suptitle('ERP per channel — scrolling vs watching  (negative up)',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / 'output' / 'group_erp_channels.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)
    return None


def make_one_figure(all_data, type_name, type_lbl, tmin, tmax, epo_idx, fname,
                    conds=None):
    """
    One figure per epoch type (ERP / FRP / SRP).
    Rows = clusters, Columns = subjects.
    sharey='row' gives unified y-axis per cluster row.
    """
    if conds is None:
        conds = CONDS
    subs = sorted(all_data.keys())
    n_rows = len(PLOT_CLUSTERS)
    n_cols = len(subs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.2 * n_rows),
                             sharey='row', sharex='row')
    fig.patch.set_facecolor('white')
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for sub_i, sub in enumerate(subs):
        sub_evokeds, sub_counts = all_data[sub][epo_idx]

        for r, cluster in enumerate(PLOT_CLUSTERS):
            ax = axes[r, sub_i]
            plot_panel(ax, sub_evokeds, sub_counts, cluster,
                       tmin, tmax, conds, title='')

            if r == 0:
                ax.set_title(f'sub-{sub}', fontsize=11, fontweight='bold')
            if r == n_rows - 1:
                ax.set_xlabel('Time (ms)', fontsize=9)
            if sub_i == 0:
                ax.set_ylabel(f'{cluster}\nµV', fontsize=9)

            # legend only on top-left panel
            if r == 0 and sub_i == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper right')

    fig.suptitle(f'{type_name} — {type_lbl} locked  (negative up)',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / 'output' / fname
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def _discover_subjects():
    """Find canonical analysis subjects with both preprocessed EEG and ET data."""
    subs = []
    for sub_id in analysis_subjects():
        deriv = sub_deriv_dir(sub_id)
        raw_path = deriv / 'preprocessed-raw.fif'
        epo_path = deriv / 'picture_onset-epo.fif'
        edf_path = _et_edf(sub_id)
        if raw_path.exists() and epo_path.exists() and edf_path.exists():
            subs.append(sub_id)
    return subs


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--exclude', nargs='*', default=[],
                   help='Subject IDs to exclude, e.g. 015 016')
    args = p.parse_args()
    exclude = {normalize_subject_id(s) for s in args.exclude}

    subs = [s for s in _discover_subjects() if s not in exclude]
    print(f'Found {len(subs)} subjects with EEG + ET data '
          f'(excluded: {sorted(exclude) or "none"})')

    all_data = {}
    for sub in subs:
        print(f'\nLoading sub-{sub}...')
        try:
            all_data[sub] = load_sub(sub)
        except Exception as e:
            print(f'  SKIPPED sub-{sub}: {e}')

    print('\nPlotting...')
    make_erp_channels_figure(all_data)
    figs = [
        ('ERP', 'picture onset',  -0.2, 1.5, 0, 'group_erp.png',  ERP_CONDS),
        ('FRP', 'fixation onset', -0.2, 0.8, 1, 'group_frp.png',  CONDS),
        ('SRP', 'saccade onset',  -0.2, 0.5, 2, 'group_srp.png',  CONDS),
    ]
    for args in figs:
        make_one_figure(all_data, *args)
    print('Done.')


if __name__ == '__main__':
    main()
