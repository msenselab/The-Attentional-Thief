"""
Baseline Condition: Joint Eye-Tracking + EEG Analysis
======================================================
Analyses:
  1. Fixation & saccade descriptives by condition (scrolling/watching/baseline)
  2. FRP (fixation-related potential) comparison across conditions
  3. Within-baseline habituation: FRP amplitude by fixation ordinal
  4. Fixation duration × ERP amplitude correlation
  5. SRP (saccade-related potential) comparison

Output: analysis/output/constant_joint/

Usage:
    python analysis/code/constant_et_eeg_joint.py
    python analysis/code/constant_et_eeg_joint.py --exclude 003
"""
import argparse
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import mne
import eyelinkio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ROOT, DERIVATIVES_DIR, CLUSTERS, ERP_WINDOWS, COLORS,
    analysis_subjects, normalize_subject_id, sub_deriv_dir, sub_out_dir, OUT_DIR, et_edf,
)

mne.set_log_level('WARNING')

# ── Parameters ───────────────────────────────────────────────────────────────
BLOCK_START = {10: 'active', 20: 'passive', 30: 'constant'}
BLOCK_END   = {19: 'active', 29: 'passive', 39: 'constant'}
PIC_ONSET   = {12: 'active', 22: 'passive', 32: 'constant'}

FRP_TMIN, FRP_TMAX = -0.2, 0.8
FRP_BASELINE       = (-0.2, 0.0)
FRP_REJECT         = 150e-6
FIX_DUR_MIN_MS     = 80
FIX_DUR_MAX_MS     = 1500

SRP_TMIN, SRP_TMAX = -0.2, 0.5
SRP_BASELINE       = (-0.2, 0.0)
SRP_REJECT         = 150e-6
SAC_DUR_MIN_MS     = 10
SAC_DUR_MAX_MS     = 150

POST_CHS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
OUT_PATH = OUT_DIR / 'constant_joint'
COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching', 'constant': 'Baseline'}

# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging():
    log = logging.getLogger('constant_joint')
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(ch)
    return log


# ── Reuse sync functions from 03_frp.py ─────────────────────────────────────
def _stim_code(desc):
    m = re.search(r'S\s*(\d+)', desc)
    return int(m.group(1)) if m else None

def segment_eeg_blocks(raw):
    sfreq = raw.info['sfreq']
    blocks, current = [], None
    for ann in raw.annotations:
        code = _stim_code(ann['description'])
        if code is None: continue
        sample = int(round(ann['onset'] * sfreq))
        if code in BLOCK_START:
            current = dict(block_type=BLOCK_START[code], start_s=ann['onset'],
                           end_s=None, pic_samples=[])
        elif code in PIC_ONSET and current is not None:
            current['pic_samples'].append(sample)
        elif code in BLOCK_END and current is not None:
            current['end_s'] = ann['onset']
            blocks.append(current)
            current = None
    return blocks

def parse_et_data(edf_path):
    edf = eyelinkio.read_edf(str(edf_path))
    msgs = [(t, m.decode(errors='replace').strip()) for t, m in edf.discrete['messages']]
    
    blocks, current = [], None
    for t, msg in msgs:
        if msg.startswith('BLOCK_START'):
            m = re.match(r'BLOCK_START\s+(\w+)\s+rep=(\d+)\s+session=(\d+)\s+n_pics=(\d+)', msg)
            if m and int(m.group(2)) > 0:
                current = dict(block_type=m.group(1).lower().replace('geometry','constant'),
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

    # Fixations
    fix = pd.DataFrame(edf.discrete['fixations'])
    rename = {}
    for old, new in [('axp','x'),('ayp','y'),('sttime','stime'),('entime','etime')]:
        if old in fix.columns: rename[old] = new
    fix = fix.rename(columns=rename)
    fix['dur_ms'] = (fix['etime'] - fix['stime']) * 1000
    fix_filtered = fix[(fix['dur_ms'] >= FIX_DUR_MIN_MS) & (fix['dur_ms'] <= FIX_DUR_MAX_MS)].copy()

    # Saccades
    sac = pd.DataFrame(edf.discrete['saccades'])
    sac_rename = {}
    for old, new in [('sttime','stime'),('entime','etime')]:
        if old in sac.columns: sac_rename[old] = new
    sac = sac.rename(columns=sac_rename)
    sac['dur_ms'] = (sac['etime'] - sac['stime']) * 1000
    sac_filtered = sac[(sac['dur_ms'] >= SAC_DUR_MIN_MS) & (sac['dur_ms'] <= SAC_DUR_MAX_MS)].copy()

    return blocks, fix_filtered.reset_index(drop=True), sac_filtered.reset_index(drop=True)

def build_sync_pairs(eeg_blocks, et_blocks):
    eeg_by = defaultdict(list)
    et_by = defaultdict(list)
    for b in eeg_blocks: eeg_by[b['block_type']].append(b)
    for b in et_blocks: et_by[b['block_type']].append(b)
    rows = []
    for cond in ['active','passive','constant']:
        n = min(len(eeg_by[cond]), len(et_by[cond]))
        for bi in range(n):
            eb, tb = eeg_by[cond][bi], et_by[cond][bi]
            for p in range(min(len(eb['pic_samples']), len(tb['images']))):
                img = tb['images'][p]
                if img['et_off'] is None: continue
                rows.append(dict(block_type=cond, rep=tb['rep'], session=tb['session'],
                                 pic_idx=img['pic_idx'], et_on=img['et_on'], et_off=img['et_off'],
                                 eeg_pic_s=eb['pic_samples'][p]/1000.0, eeg_pic_sample=eb['pic_samples'][p]))
    return pd.DataFrame(rows)

def assign_events_to_images(event_df, pairs, sfreq, time_col='stime'):
    """Assign fixations or saccades to images, compute EEG onset + ordinal within image."""
    assigned = []
    for _, ev in event_df.iterrows():
        for _, img in pairs.iterrows():
            if img['et_on'] + 0.05 <= ev[time_col] <= img['et_off'] - 0.05:
                dt = ev[time_col] - img['et_on']
                eeg_s = img['eeg_pic_s'] + dt
                assigned.append(dict(
                    block_type=img['block_type'], rep=img['rep'], session=img['session'],
                    pic_idx=img['pic_idx'], eeg_onset_s=eeg_s,
                    eeg_onset_sample=int(round(eeg_s * sfreq)),
                    dur_ms=ev['dur_ms'], time_in_image_s=dt))
                break
    return pd.DataFrame(assigned) if assigned else pd.DataFrame()


# ── Per-subject processing ───────────────────────────────────────────────────
def process_subject(sub, log):
    """Process one subject: return dict with ET descriptives + FRP/SRP epochs."""
    raw_path = sub_deriv_dir(sub) / 'preprocessed-raw.fif'
    edf_path = et_edf(sub)
    if not raw_path.exists() or not edf_path.exists():
        return None

    log.info(f'  sub-{sub}: loading...')
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose='ERROR')
    sfreq = raw.info['sfreq']

    eeg_blocks = segment_eeg_blocks(raw)
    et_blocks, fix_df, sac_df = parse_et_data(edf_path)
    pairs = build_sync_pairs(eeg_blocks, et_blocks)

    if pairs.empty:
        log.warning(f'  sub-{sub}: no sync pairs')
        return None

    # Assign fixations & saccades
    fix_assigned = assign_events_to_images(fix_df, pairs, sfreq, 'stime')
    sac_assigned = assign_events_to_images(sac_df, pairs, sfreq, 'stime')

    # Add fixation ordinal within each image
    if not fix_assigned.empty:
        fix_assigned['fix_ordinal'] = fix_assigned.groupby(
            ['block_type','rep','session','pic_idx']).cumcount() + 1

    result = dict(sub=sub)

    # ET descriptives per condition
    for cond in ['active','passive','constant']:
        f = fix_assigned[fix_assigned['block_type'] == cond] if not fix_assigned.empty else pd.DataFrame()
        s = sac_assigned[sac_assigned['block_type'] == cond] if not sac_assigned.empty else pd.DataFrame()
        result[f'n_fix_{cond}'] = len(f)
        result[f'fix_dur_{cond}'] = f['dur_ms'].mean() if len(f) > 0 else np.nan
        result[f'n_sac_{cond}'] = len(s)
        result[f'sac_dur_{cond}'] = s['dur_ms'].mean() if len(s) > 0 else np.nan

    # Build FRP epochs
    frp_data = {}
    if not fix_assigned.empty:
        code_map = {'active': 1, 'passive': 2, 'constant': 3}
        valid = fix_assigned[fix_assigned['block_type'].isin(code_map)].copy()
        if not valid.empty:
            frp_events = np.column_stack([
                valid['eeg_onset_sample'].values,
                np.zeros(len(valid), dtype=int),
                valid['block_type'].map(code_map).values])
            try:
                epochs = mne.Epochs(raw, frp_events, event_id=code_map,
                                    tmin=FRP_TMIN, tmax=FRP_TMAX, baseline=FRP_BASELINE,
                                    preload=True, reject=dict(eeg=FRP_REJECT),
                                    reject_by_annotation=True, verbose='ERROR')
                for c in ['active','passive','constant']:
                    if c in epochs.event_id and len(epochs[c]) >= 5:
                        frp_data[c] = epochs[c].average()
                        log.info(f'    FRP {c}: {epochs[c].__len__()} epochs')
            except Exception as e:
                log.warning(f'    FRP failed: {e}')

    # Build SRP epochs
    srp_data = {}
    if not sac_assigned.empty:
        code_map = {'active': 1, 'passive': 2, 'constant': 3}
        valid = sac_assigned[sac_assigned['block_type'].isin(code_map)].copy()
        if not valid.empty:
            srp_events = np.column_stack([
                valid['eeg_onset_sample'].values,
                np.zeros(len(valid), dtype=int),
                valid['block_type'].map(code_map).values])
            try:
                epochs = mne.Epochs(raw, srp_events, event_id=code_map,
                                    tmin=SRP_TMIN, tmax=SRP_TMAX, baseline=SRP_BASELINE,
                                    preload=True, reject=dict(eeg=SRP_REJECT),
                                    reject_by_annotation=True, verbose='ERROR')
                for c in ['active','passive','constant']:
                    if c in epochs.event_id and len(epochs[c]) >= 5:
                        srp_data[c] = epochs[c].average()
                        log.info(f'    SRP {c}: {epochs[c].__len__()} epochs')
            except Exception as e:
                log.warning(f'    SRP failed: {e}')

    # Baseline habituation: FRP by fixation ordinal (1st, 2nd, 3rd, 4th+)
    hab_data = {}
    if not fix_assigned.empty:
        const_fix = fix_assigned[fix_assigned['block_type'] == 'constant'].copy()
        if len(const_fix) > 20:
            ordinal_bins = {1: 'fix_1st', 2: 'fix_2nd', 3: 'fix_3rd'}
            const_fix['ord_bin'] = const_fix['fix_ordinal'].apply(
                lambda x: ordinal_bins.get(x, 'fix_4th+'))
            
            code_map_hab = {'fix_1st': 1, 'fix_2nd': 2, 'fix_3rd': 3, 'fix_4th+': 4}
            valid = const_fix[const_fix['ord_bin'].isin(code_map_hab)].copy()
            if not valid.empty:
                hab_events = np.column_stack([
                    valid['eeg_onset_sample'].values,
                    np.zeros(len(valid), dtype=int),
                    valid['ord_bin'].map(code_map_hab).values])
                try:
                    epochs = mne.Epochs(raw, hab_events, event_id=code_map_hab,
                                        tmin=FRP_TMIN, tmax=FRP_TMAX, baseline=FRP_BASELINE,
                                        preload=True, reject=dict(eeg=FRP_REJECT),
                                        reject_by_annotation=True, verbose='ERROR')
                    for label in code_map_hab:
                        if label in epochs.event_id and len(epochs[label]) >= 3:
                            hab_data[label] = epochs[label].average()
                            log.info(f'    Hab {label}: {epochs[label].__len__()} epochs')
                except Exception as e:
                    log.warning(f'    Habituation FRP failed: {e}')

    result['frp'] = frp_data
    result['srp'] = srp_data
    result['hab'] = hab_data
    result['fix_assigned'] = fix_assigned

    return result


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_group_frp(all_frp, out_dir, n_subs, log):
    """Grand-average FRP: Scrolling vs Watching vs Baseline."""
    conds = ['active', 'passive', 'constant']
    grand = {}
    for c in conds:
        evks = [r['frp'][c] for r in all_frp if c in r['frp']]
        if len(evks) >= 3:
            grand[c] = mne.grand_average(evks)

    if len(grand) < 2:
        log.warning('  Not enough FRP data for group plot')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    clusters = [('Posterior', POST_CHS), ('Parietal', ['P3','Pz','P4','CPz']), ('Frontal', ['F3','Fz','F4'])]
    
    for ax, (title, chs) in zip(axes, clusters):
        for c in conds:
            if c not in grand: continue
            evk = grand[c]
            picks = [ch for ch in chs if ch in evk.ch_names]
            if not picks: continue
            data = evk.copy().pick(picks).data.mean(axis=0) * 1e6
            ax.plot(evk.times * 1000, data, color=COLORS.get(c, 'gray'),
                    lw=2, label=f"{COND_LABELS.get(c, c.capitalize())} (n={evk.nave})")
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Time from fixation onset (ms)')
        ax.set_xlim(-200, 800)
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].legend(fontsize=9)
    fig.suptitle(f'Grand-Average FRP by Condition (N={n_subs})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'group_frp_by_condition.png', dpi=150)
    plt.close(fig)
    log.info(f'  Saved: group_frp_by_condition.png')


def plot_group_srp(all_data, out_dir, n_subs, log):
    """Grand-average SRP."""
    conds = ['active', 'passive', 'constant']
    grand = {}
    for c in conds:
        evks = [r['srp'][c] for r in all_data if c in r['srp']]
        if len(evks) >= 3:
            grand[c] = mne.grand_average(evks)

    if len(grand) < 2:
        log.warning('  Not enough SRP data')
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in conds:
        if c not in grand: continue
        evk = grand[c]
        picks = [ch for ch in POST_CHS if ch in evk.ch_names]
        if not picks: continue
        data = evk.copy().pick(picks).data.mean(axis=0) * 1e6
        ax.plot(evk.times * 1000, data, color=COLORS.get(c, 'gray'),
                lw=2, label=f"{COND_LABELS.get(c, c.capitalize())} (n={evk.nave})")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.invert_yaxis()
    ax.set_xlabel('Time from saccade onset (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Grand-Average SRP by Condition (N={n_subs})')
    ax.legend()
    ax.set_xlim(-200, 500)
    plt.tight_layout()
    fig.savefig(out_dir / 'group_srp_by_condition.png', dpi=150)
    plt.close(fig)
    log.info(f'  Saved: group_srp_by_condition.png')


def plot_habituation(all_data, out_dir, n_subs, log):
    """FRP habituation within baseline: 1st vs 2nd vs 3rd vs 4th+ fixation."""
    labels = ['fix_1st', 'fix_2nd', 'fix_3rd', 'fix_4th+']
    hab_colors = {'fix_1st': '#d62728', 'fix_2nd': '#ff7f0e', 'fix_3rd': '#2ca02c', 'fix_4th+': '#1f77b4'}
    
    grand = {}
    for label in labels:
        evks = [r['hab'][label] for r in all_data if label in r['hab']]
        if len(evks) >= 3:
            grand[label] = mne.grand_average(evks)

    if len(grand) < 2:
        log.warning('  Not enough habituation data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Waveform
    ax = axes[0]
    for label in labels:
        if label not in grand: continue
        evk = grand[label]
        picks = [ch for ch in POST_CHS if ch in evk.ch_names]
        if not picks: continue
        data = evk.copy().pick(picks).data.mean(axis=0) * 1e6
        ax.plot(evk.times * 1000, data, color=hab_colors[label],
                lw=2, label=f'{label.replace("fix_","")} (n={evk.nave})')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.invert_yaxis()
    ax.set_xlabel('Time from fixation onset (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Baseline: FRP by Fixation Ordinal')
    ax.legend(fontsize=9)
    ax.set_xlim(-200, 800)

    # Bar: mean amplitude in N1 window by ordinal
    ax2 = axes[1]
    n1_amps = {}
    for label in labels:
        if label not in grand: continue
        evk = grand[label]
        picks = [ch for ch in POST_CHS if ch in evk.ch_names]
        if not picks: continue
        t0, t1 = ERP_WINDOWS.get('N1', (0.14, 0.20))
        t_mask = (evk.times >= t0) & (evk.times <= t1)
        n1_amps[label] = evk.copy().pick(picks).data[:, t_mask].mean() * 1e6

    if n1_amps:
        x = range(len(n1_amps))
        ax2.bar(x, n1_amps.values(), color=[hab_colors[l] for l in n1_amps.keys()])
        ax2.set_xticks(x)
        ax2.set_xticklabels([l.replace('fix_','') for l in n1_amps.keys()])
        ax2.set_ylabel('N1 Amplitude (µV)')
        ax2.set_title('N1 Window: Habituation Effect')

    fig.suptitle(f'Baseline Condition — Fixation Habituation (N={n_subs})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'constant_habituation.png', dpi=150)
    plt.close(fig)
    log.info(f'  Saved: constant_habituation.png')


def plot_et_descriptives(all_data, out_dir, log):
    """Bar plot: fixation duration + count by condition."""
    rows = []
    for r in all_data:
        for c in ['active','passive','constant']:
            rows.append(dict(sub=r['sub'], condition=c,
                             n_fix=r.get(f'n_fix_{c}', 0),
                             fix_dur=r.get(f'fix_dur_{c}', np.nan),
                             n_sac=r.get(f'n_sac_{c}', 0),
                             sac_dur=r.get(f'sac_dur_{c}', np.nan)))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, (measure, label) in zip(axes, [('fix_dur', 'Fixation Duration (ms)'),
                                            ('n_fix', 'Number of Fixations'),
                                            ('sac_dur', 'Saccade Duration (ms)')]):
        means = df.groupby('condition')[measure].mean()
        sems = df.groupby('condition')[measure].sem()
        conds = ['active','passive','constant']
        x = range(len(conds))
        colors = [COLORS.get(c, 'gray') for c in conds]
        ax.bar(x, [means.get(c, 0) for c in conds],
               yerr=[sems.get(c, 0) for c in conds],
               color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c, c.capitalize()) for c in conds])
        ax.set_ylabel(label)

    fig.suptitle(f'Eye-Tracking Descriptives by Condition (N={len(all_data)})', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / 'et_descriptives.png', dpi=150)
    plt.close(fig)
    log.info(f'  Saved: et_descriptives.png')

    # Stats
    print('\n  ET Condition Comparisons (paired t-tests):')
    for measure, label in [('fix_dur', 'Fix duration'), ('n_fix', 'N fixations'), ('sac_dur', 'Sac duration')]:
        pivot = df.pivot(index='sub', columns='condition', values=measure).dropna()
        if len(pivot) > 3:
            for c1, c2 in [('active','passive'), ('active','constant'), ('passive','constant')]:
                if c1 in pivot.columns and c2 in pivot.columns:
                    t, p = stats.ttest_rel(pivot[c1], pivot[c2])
                    print(f'    {label}: {c1} vs {c2}: t={t:.2f}, p={p:.4f}')

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', nargs='*', default=[])
    args = parser.parse_args()
    exclude = [normalize_subject_id(s) for s in args.exclude]

    log = setup_logging()
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    log.info('═' * 50)
    log.info('BASELINE CONDITION: Joint ET + EEG Analysis')
    log.info('═' * 50)

    # Discover subjects
    subs = analysis_subjects(exclude)
    log.info(f'Processing {len(subs)} canonical subjects...')

    all_data = []
    for sub in subs:
        result = process_subject(sub, log)
        if result:
            all_data.append(result)

    log.info(f'\nIncluded: {len(all_data)} subjects')

    if len(all_data) < 3:
        log.error('Need ≥3 subjects')
        sys.exit(1)

    # Plots
    log.info('Generating group figures...')
    plot_group_frp(all_data, OUT_PATH, len(all_data), log)
    plot_group_srp(all_data, OUT_PATH, len(all_data), log)
    plot_habituation(all_data, OUT_PATH, len(all_data), log)
    et_df = plot_et_descriptives(all_data, OUT_PATH, log)
    et_df.to_csv(OUT_PATH / 'et_descriptives.csv', index=False)

    log.info('═' * 50)
    log.info(f'Results: {OUT_PATH.relative_to(ROOT)}/')
    log.info('Done!')


if __name__ == '__main__':
    main()
