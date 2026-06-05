"""
Attentional Thief — Step 7: Subsequent Memory Effect (Dm) Analysis
===================================================================
Links recognition outcomes to picture-onset EEG epochs and computes
the Dm effect (remembered vs forgotten) per condition.

Design:
  - Each block has ONE recognition test for ONE "old" image.
  - "Old" images were shown during the block → have a corresponding epoch.
  - Hit  (old + responded "old") → Remembered
  - Miss (old + responded "new") → Forgotten
  - "New" foils have no EEG epoch and are excluded.

Output : analysis/output/dm_analysis/
           dm_trial_counts.csv          — per-subject trial counts
           dm_grand_average.png         — group Dm waveforms per condition
           dm_topomaps.png              — Dm difference topographies
           dm_bar_summary.png           — mean amplitude Dm × condition
           dm_statistics.csv            — paired t-tests and interaction
           dm_report.txt                — text summary

Usage:
    python analysis/07_dm_analysis.py
    python analysis/07_dm_analysis.py --exclude 003 008
"""
import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ROOT, DATA_DIR, DERIVATIVES_DIR, CLUSTERS, ERP_WINDOWS, COLORS,
    PIC_ONSET_ID, analysis_subjects, sub_deriv_dir, OUT_DIR,
)

mne.set_log_level('WARNING')

# ── Parameters ───────────────────────────────────────────────────────────────
CONDS = ['active', 'passive']
COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching'}
DM_LABELS = {'hit': 'Remembered', 'miss': 'Forgotten'}
DM_COLORS = {'hit': '#2ca02c', 'miss': '#9467bd'}  # green / purple

POST_CHS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
MIN_TRIALS = 3  # minimum trials per Dm bin to include a subject

# EEG trigger → condition mapping (for block segmentation)
BLOCK_START = {10: 'active', 20: 'passive', 30: 'constant'}
BLOCK_END = {19: 'active', 29: 'passive', 39: 'constant'}
PIC_ONSET = {12: 'active', 22: 'passive', 32: 'constant'}


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('attthief.dm')
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
        description='Subsequent Memory Effect (Dm) analysis')
    p.add_argument('--exclude', nargs='*', default=[],
                   help='Subject IDs to exclude, e.g. 003 008')
    return p.parse_args()


# ── Bracket-aware CSV parser ────────────────────────────────────────────────
def _parse_csv_row(line: str) -> list[str]:
    """Split CSV line on commas, respecting brackets and quotes."""
    fields = []
    current = ''
    depth = 0
    in_quote = False
    for ch in line.strip():
        if ch == '"' and depth == 0:
            in_quote = not in_quote
            current += ch
        elif ch == '[':
            depth += 1
            current += ch
        elif ch == ']':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0 and not in_quote:
            fields.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        fields.append(current.strip())
    return fields


def _parse_image_list(raw: str) -> list[str]:
    """Parse image_files field like '["human_056.jpg", "social_118.jpg"]'."""
    raw = raw.strip().strip('"')
    if not raw or raw == '[]':
        return []
    return [m.group(1) for m in re.finditer(r'"([^"]+\.jpg)"', raw)]


# ── Behavioral data ─────────────────────────────────────────────────────────
def load_recognition_data(sub: str) -> pd.DataFrame | None:
    """
    Parse behavioral CSV and return one row per block with:
      session_type, block_order, image_files, recognition_image,
      recognition_is_old, recognition_response, recognition_correct
    """
    beh_dir = DATA_DIR / f'sub-{sub}' / 'beh'
    csvs = sorted(beh_dir.glob('*.csv'))
    if not csvs:
        return None

    with open(csvs[0], encoding='utf-8-sig') as f:
        lines = f.readlines()

    rows = []
    for line in lines[1:]:
        fields = _parse_csv_row(line)
        if len(fields) < 20:
            continue
        cond = fields[4].strip()
        if cond not in ('active', 'passive'):
            continue

        recog_is_old = fields[17].strip()
        if recog_is_old != 'True':
            # Only old items have corresponding EEG epochs
            continue

        image_list = _parse_image_list(fields[15])
        recog_image = fields[16].strip()
        recog_response = fields[18].strip()

        if not image_list or not recog_image:
            continue

        # Find the position of the tested image within the block
        try:
            pic_idx = image_list.index(recog_image)
        except ValueError:
            continue

        dm_label = 'hit' if recog_response == 'old' else 'miss'

        rows.append(dict(
            session_type=cond,
            rep=int(fields[3]),
            block_order=int(fields[6]),
            n_pictures=int(fields[7]),
            recog_image=recog_image,
            pic_idx_in_block=pic_idx,
            dm=dm_label,
        ))

    return pd.DataFrame(rows) if rows else None


# ── EEG block segmentation ──────────────────────────────────────────────────
def _stim_code(desc: str) -> int | None:
    m = re.search(r'S\s*(\d+)', desc)
    return int(m.group(1)) if m else None


def segment_eeg_blocks(raw: mne.io.Raw) -> list[dict]:
    """Parse annotations into blocks: {block_type, pic_samples: [int]}."""
    sfreq = raw.info['sfreq']
    blocks, current = [], None
    for ann in raw.annotations:
        code = _stim_code(ann['description'])
        if code is None:
            continue
        onset_s = ann['onset']
        sample = int(round(onset_s * sfreq))
        if code in BLOCK_START:
            current = dict(block_type=BLOCK_START[code], pic_samples=[])
        elif code in PIC_ONSET and current is not None:
            current['pic_samples'].append(sample)
        elif code in BLOCK_END and current is not None:
            blocks.append(current)
            current = None
    return blocks


# ── Map recognition outcomes to epoch indices ────────────────────────────────
def label_epochs(epochs: mne.Epochs, raw: mne.io.Raw,
                 recog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match recognition trials to specific epoch indices.

    Strategy:
      1. Segment EEG into blocks by condition.
      2. Match behavioral blocks to EEG blocks (by condition + sequential order).
      3. Within each matched block, the Nth pic-onset sample = Nth image.
      4. Find which epoch index corresponds to that sample.

    Returns DataFrame with: epoch_idx, condition, dm ('hit'/'miss'), recog_image
    """
    eeg_blocks = segment_eeg_blocks(raw)

    # Group EEG blocks by condition, preserving order
    eeg_by_cond = {}
    for b in eeg_blocks:
        eeg_by_cond.setdefault(b['block_type'], []).append(b)

    # Group behavioral blocks by condition, preserving order
    beh_by_cond = {}
    for _, row in recog_df.iterrows():
        beh_by_cond.setdefault(row['session_type'], []).append(row)

    # Build set of epoch event samples for quick lookup
    # epochs.events: (n_epochs, 3) — [sample, 0, event_id]
    event_samples = epochs.events[:, 0]

    labels = []
    for cond in CONDS:
        eeg_blks = eeg_by_cond.get(cond, [])
        beh_blks = beh_by_cond.get(cond, [])

        n_match = min(len(eeg_blks), len(beh_blks))
        for bi in range(n_match):
            eb = eeg_blks[bi]
            bb = beh_blks[bi]

            pic_idx = bb['pic_idx_in_block']
            if pic_idx >= len(eb['pic_samples']):
                continue

            target_sample = eb['pic_samples'][pic_idx]

            # Find the epoch whose event sample matches
            # Allow small tolerance (±2 samples) for rounding
            diffs = np.abs(event_samples - target_sample)
            best = diffs.argmin()
            if diffs[best] <= 2:
                labels.append(dict(
                    epoch_idx=int(best),
                    condition=cond,
                    dm=bb['dm'],
                    recog_image=bb['recog_image'],
                ))

    return pd.DataFrame(labels)


# ── Single subject Dm ────────────────────────────────────────────────────────
def compute_subject_dm(sub: str, log: logging.Logger) -> dict | None:
    """
    Load epochs + behavioral data, link recognition to epochs,
    compute per-condition Dm evokeds and amplitudes.
    """
    deriv = sub_deriv_dir(sub)
    epo_path = deriv / 'picture_onset-epo.fif'
    raw_path = deriv / 'preprocessed-raw.fif'

    if not epo_path.exists() or not raw_path.exists():
        log.warning(f'  sub-{sub}: missing preprocessed files — skipping')
        return None

    recog_df = load_recognition_data(sub)
    if recog_df is None or len(recog_df) == 0:
        log.warning(f'  sub-{sub}: no recognition data — skipping')
        return None

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')
    raw = mne.io.read_raw_fif(str(raw_path), preload=False, verbose='ERROR')

    labels_df = label_epochs(epochs, raw, recog_df)
    if labels_df.empty:
        log.warning(f'  sub-{sub}: no epochs matched to recognition — skipping')
        return None

    result = {'sub': sub, 'labels': labels_df}

    # Trial counts
    counts = {}
    for cond in CONDS:
        for dm in ['hit', 'miss']:
            n = len(labels_df[(labels_df['condition'] == cond) &
                              (labels_df['dm'] == dm)])
            counts[f'{cond}_{dm}'] = n
        counts[f'{cond}_total'] = counts[f'{cond}_hit'] + counts[f'{cond}_miss']

    result['counts'] = counts
    log.info(f'  sub-{sub}: ' + ', '.join(
        f'{c} H{counts[f"{c}_hit"]}/M{counts[f"{c}_miss"]}'
        for c in CONDS))

    # Check minimum trials
    usable_conds = []
    for cond in CONDS:
        if counts[f'{cond}_hit'] >= MIN_TRIALS and counts[f'{cond}_miss'] >= MIN_TRIALS:
            usable_conds.append(cond)

    if not usable_conds:
        log.warning(f'  sub-{sub}: too few trials in both conditions — skipping')
        return None

    result['usable_conds'] = usable_conds

    # Compute evokeds per condition × dm
    evokeds = {}
    amplitudes = []
    for cond in usable_conds:
        for dm in ['hit', 'miss']:
            idx = labels_df[(labels_df['condition'] == cond) &
                            (labels_df['dm'] == dm)]['epoch_idx'].values
            if len(idx) == 0:
                continue
            evk = epochs[idx].average()
            evokeds[f'{cond}_{dm}'] = evk

            # Extract mean amplitudes per window × cluster
            for win_name, (t0, t1) in ERP_WINDOWS.items():
                for cl_name, ch_list in CLUSTERS.items():
                    picks = [ch for ch in ch_list if ch in evk.ch_names]
                    if not picks:
                        continue
                    ch_idx = [evk.ch_names.index(ch) for ch in picks]
                    t_mask = (evk.times >= t0) & (evk.times <= t1)
                    amp = evk.data[np.ix_(ch_idx, t_mask)].mean() * 1e6
                    amplitudes.append(dict(
                        sub=sub, condition=cond, dm=dm,
                        window=win_name, cluster=cl_name,
                        amplitude_uV=round(amp, 3),
                        n_trials=len(idx),
                    ))

    result['evokeds'] = evokeds
    result['amplitudes'] = pd.DataFrame(amplitudes)
    return result


# ── Group analysis ───────────────────────────────────────────────────────────
def run_group_dm(all_results: list[dict], out_dir: Path,
                 log: logging.Logger) -> None:
    """Grand-average Dm, statistics, and figures."""

    # ── 1. Trial count summary ──────────────────────────────────────���────────
    count_rows = []
    for r in all_results:
        row = {'sub': r['sub']}
        row.update(r['counts'])
        count_rows.append(row)
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(out_dir / 'dm_trial_counts.csv', index=False)
    log.info(f'  Saved: dm_trial_counts.csv')

    # ── 2. Amplitude table ───────────────────────────────────────────────────
    amp_df = pd.concat([r['amplitudes'] for r in all_results], ignore_index=True)
    amp_df.to_csv(out_dir / 'dm_amplitudes.csv', index=False)

    # ── 3. Grand-average evokeds ─────────────────────────────────────────────
    # Collect per-subject evokeds per condition × dm
    grand = {}
    for cond in CONDS:
        for dm in ['hit', 'miss']:
            key = f'{cond}_{dm}'
            evk_list = [r['evokeds'][key] for r in all_results
                        if key in r['evokeds']]
            if len(evk_list) >= 2:
                grand[key] = mne.grand_average(evk_list)
                grand[key].comment = f'{cond}_{dm} (N={len(evk_list)})'

    # Also compute collapsed Dm (across conditions)
    for dm in ['hit', 'miss']:
        evk_list = []
        for r in all_results:
            # Average across conditions within subject, then collect
            sub_evks = [r['evokeds'][f'{c}_{dm}'] for c in CONDS
                        if f'{c}_{dm}' in r['evokeds']]
            if sub_evks:
                evk_list.append(mne.combine_evoked(sub_evks, weights='equal'))
        if len(evk_list) >= 2:
            grand[f'collapsed_{dm}'] = mne.grand_average(evk_list)

    # ── 4. Figures ───────────────────────────────────────────────────────────
    _plot_grand_average(grand, out_dir, len(all_results), log)
    _plot_topomaps(grand, out_dir, len(all_results), log)
    _plot_bar_summary(amp_df, out_dir, log)

    # ── 5. Statistics ────────────────────────────────────────────────────────
    stats_df = _compute_statistics(amp_df, log)
    stats_df.to_csv(out_dir / 'dm_statistics.csv', index=False)
    log.info(f'  Saved: dm_statistics.csv')

    # ── 6. Report ────────────────────────────────────────────────────────────
    _write_report(out_dir, all_results, amp_df, stats_df, count_df, log)


# ── Plotting ─────────────────────────────────────────────────────────────────
def _plot_grand_average(grand: dict, out_dir: Path,
                        n_subjects: int, log: logging.Logger) -> None:
    """Grand-average Dm waveforms: posterior channels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Subsequent Memory Effect (Dm) — Grand Average (N={n_subjects})',
                 fontsize=13, fontweight='bold')

    panels = [
        ('Scrolling', 'active'),
        ('Watching', 'passive'),
        ('Collapsed', 'collapsed'),
    ]

    for ax, (title, prefix) in zip(axes, panels):
        for dm in ['hit', 'miss']:
            key = f'{prefix}_{dm}'
            if key not in grand:
                continue
            evk = grand[key]
            picks = [evk.ch_names.index(ch) for ch in POST_CHS
                     if ch in evk.ch_names]
            if not picks:
                continue
            data = evk.data[picks].mean(axis=0) * 1e6
            ax.plot(evk.times * 1000, data, color=DM_COLORS[dm],
                    linewidth=1.8, label=DM_LABELS[dm])

        # Plot difference wave (hit - miss)
        hit_key = f'{prefix}_hit'
        miss_key = f'{prefix}_miss'
        if hit_key in grand and miss_key in grand:
            evk_h = grand[hit_key]
            evk_m = grand[miss_key]
            picks_h = [evk_h.ch_names.index(ch) for ch in POST_CHS
                       if ch in evk_h.ch_names]
            picks_m = [evk_m.ch_names.index(ch) for ch in POST_CHS
                       if ch in evk_m.ch_names]
            diff = (evk_h.data[picks_h].mean(axis=0)
                    - evk_m.data[picks_m].mean(axis=0)) * 1e6
            ax.plot(evk_h.times * 1000, diff, color='k', linewidth=1.2,
                    linestyle='--', alpha=0.6, label='Dm (R−F)')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(-200, 1000)

        # Shade ERP windows
        for win_name, (t0, t1) in ERP_WINDOWS.items():
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.05, color='blue')

    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / 'dm_grand_average.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: dm_grand_average.png')


def _plot_topomaps(grand: dict, out_dir: Path,
                   n_subjects: int, log: logging.Logger) -> None:
    """Dm difference topographies per window."""
    panels = [
        ('Scrolling Dm', 'active'),
        ('Watching Dm', 'passive'),
        ('Collapsed Dm', 'collapsed'),
    ]

    n_wins = len(ERP_WINDOWS)
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, n_wins, figsize=(3 * n_wins, 3 * n_rows))
    fig.suptitle(f'Dm Topography (Remembered − Forgotten, N={n_subjects})',
                 fontsize=13, fontweight='bold')

    for i, (title, prefix) in enumerate(panels):
        hit_key = f'{prefix}_hit'
        miss_key = f'{prefix}_miss'
        if hit_key not in grand or miss_key not in grand:
            for j in range(n_wins):
                axes[i, j].axis('off')
            axes[i, 0].text(0.5, 0.5, f'{title}: insufficient data',
                            transform=axes[i, 0].transAxes, ha='center')
            continue

        diff_evk = mne.combine_evoked(
            [grand[hit_key], grand[miss_key]], weights=[1, -1])

        for j, (win_name, (t0, t1)) in enumerate(ERP_WINDOWS.items()):
            ax = axes[i, j]
            t_mid = (t0 + t1) / 2
            mne.viz.plot_topomap(
                diff_evk.copy().crop(t0, t1).data.mean(axis=1) * 1e6,
                diff_evk.info, axes=ax, show=False,
                cmap='RdBu_r', vlim=(-3, 3))
            if i == 0:
                ax.set_title(f'{win_name}\n({int(t0*1000)}–{int(t1*1000)} ms)',
                             fontsize=9)
        axes[i, 0].set_ylabel(title, fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_dir / 'dm_topomaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: dm_topomaps.png')


def _plot_bar_summary(amp_df: pd.DataFrame, out_dir: Path,
                      log: logging.Logger) -> None:
    """Bar plot: mean Dm amplitude per condition × window (posterior)."""
    df = amp_df[amp_df['cluster'].isin(['posterior', 'occipital'])].copy()
    if df.empty:
        return

    # Average posterior+occipital per subject first
    df = df.groupby(['sub', 'condition', 'dm', 'window'])['amplitude_uV'].mean().reset_index()

    windows = list(ERP_WINDOWS.keys())
    x = np.arange(len(windows))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Dm Effect: Posterior/Occipital Mean Amplitude',
                 fontsize=13, fontweight='bold')

    for ax_i, (ax, (cond_label, cond)) in enumerate(
            zip(axes[:2], [('Scrolling', 'active'), ('Watching', 'passive')])):
        for di, dm in enumerate(['hit', 'miss']):
            vals = df[(df['condition'] == cond) & (df['dm'] == dm)]
            summary = vals.groupby('window')['amplitude_uV'].agg(['mean', 'sem'])
            summary = summary.reindex(windows)
            ax.bar(x + di * width, summary['mean'], width,
                   yerr=summary['sem'], label=DM_LABELS[dm],
                   color=DM_COLORS[dm], alpha=0.8, capsize=3)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(windows)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(cond_label)
        ax.legend()
        ax.axhline(0, color='gray', linewidth=0.5)

    # Third panel: Dm difference (hit - miss) per condition
    ax = axes[2]
    cond_colors = {'active': '#111111', 'passive': '#d62728'}
    for ci, cond in enumerate(CONDS):
        hit_df = df[(df['condition'] == cond) & (df['dm'] == 'hit')].set_index(['sub', 'window'])
        miss_df = df[(df['condition'] == cond) & (df['dm'] == 'miss')].set_index(['sub', 'window'])
        common = hit_df.index.intersection(miss_df.index)
        if common.empty:
            continue
        diff = hit_df.loc[common, 'amplitude_uV'] - miss_df.loc[common, 'amplitude_uV']
        diff = diff.reset_index()
        summary = diff.groupby('window')['amplitude_uV'].agg(['mean', 'sem']).reindex(windows)
        ax.bar(x + ci * width, summary['mean'], width,
               yerr=summary['sem'], label=COND_LABELS[cond],
               color=cond_colors[cond], alpha=0.8, capsize=3)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(windows)
    ax.set_ylabel('Dm Amplitude (µV)')
    ax.set_title('Dm Difference (R − F)')
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_dir / 'dm_bar_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Saved: dm_bar_summary.png')


# ── Statistics ───────────────────────────────────────────────────────────────
def _compute_statistics(amp_df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Per window (posterior/occipital):
      1. Dm effect per condition: paired t-test hit vs miss
      2. Dm × Condition interaction: paired t-test on (hit-miss) difference
    """
    df = amp_df[amp_df['cluster'].isin(['posterior', 'occipital'])].copy()
    df = df.groupby(['sub', 'condition', 'dm', 'window'])['amplitude_uV'].mean().reset_index()

    results = []

    for win in ERP_WINDOWS:
        # 1. Dm effect within each condition
        for cond in CONDS:
            hit = df[(df['condition'] == cond) & (df['dm'] == 'hit') &
                     (df['window'] == win)].set_index('sub')['amplitude_uV']
            miss = df[(df['condition'] == cond) & (df['dm'] == 'miss') &
                      (df['window'] == win)].set_index('sub')['amplitude_uV']
            common = hit.index.intersection(miss.index)
            if len(common) < 3:
                continue
            t_val, p_val = stats.ttest_rel(hit.loc[common], miss.loc[common])
            d = (hit.loc[common] - miss.loc[common]).mean() / (hit.loc[common] - miss.loc[common]).std()
            results.append(dict(
                test='Dm_effect', condition=cond, window=win,
                mean_hit=round(hit.loc[common].mean(), 3),
                mean_miss=round(miss.loc[common].mean(), 3),
                mean_diff=round((hit.loc[common] - miss.loc[common]).mean(), 3),
                t=round(t_val, 3), p=round(p_val, 4),
                cohens_d=round(d, 3), n=len(common),
            ))

        # 2. Dm × Condition interaction
        dm_diffs = {}
        for cond in CONDS:
            hit = df[(df['condition'] == cond) & (df['dm'] == 'hit') &
                     (df['window'] == win)].set_index('sub')['amplitude_uV']
            miss = df[(df['condition'] == cond) & (df['dm'] == 'miss') &
                      (df['window'] == win)].set_index('sub')['amplitude_uV']
            common = hit.index.intersection(miss.index)
            if len(common) >= 3:
                dm_diffs[cond] = (hit.loc[common] - miss.loc[common])

        if len(dm_diffs) == 2:
            common_subs = dm_diffs['active'].index.intersection(
                dm_diffs['passive'].index)
            if len(common_subs) >= 3:
                t_val, p_val = stats.ttest_rel(
                    dm_diffs['active'].loc[common_subs],
                    dm_diffs['passive'].loc[common_subs])
                d = ((dm_diffs['active'].loc[common_subs]
                      - dm_diffs['passive'].loc[common_subs]).mean()
                     / (dm_diffs['active'].loc[common_subs]
                        - dm_diffs['passive'].loc[common_subs]).std())
                results.append(dict(
                    test='Dm_x_Condition', condition='interaction', window=win,
                    mean_hit=round(dm_diffs['active'].loc[common_subs].mean(), 3),
                    mean_miss=round(dm_diffs['passive'].loc[common_subs].mean(), 3),
                    mean_diff=round((dm_diffs['active'].loc[common_subs]
                                     - dm_diffs['passive'].loc[common_subs]).mean(), 3),
                    t=round(t_val, 3), p=round(p_val, 4),
                    cohens_d=round(d, 3), n=len(common_subs),
                ))

    return pd.DataFrame(results)


# ── Report ───────────────────────────────────────────────────────────────────
def _write_report(out_dir: Path, all_results: list[dict],
                  amp_df: pd.DataFrame, stats_df: pd.DataFrame,
                  count_df: pd.DataFrame, log: logging.Logger) -> None:
    lines = [
        f'Subsequent Memory Effect (Dm) Analysis — Attentional Thief\n',
        f'{"=" * 60}\n',
        f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n',
        f'Subjects included: {len(all_results)}\n',
        f'Subject IDs: {", ".join(r["sub"] for r in all_results)}\n',
        f'Minimum trials per bin: {MIN_TRIALS}\n\n',
    ]

    # Trial counts
    lines.append('TRIAL COUNTS\n')
    lines.append('-' * 50 + '\n')
    for r in all_results:
        c = r['counts']
        lines.append(f'  sub-{r["sub"]}: '
                     f'active H{c["active_hit"]}/M{c["active_miss"]}, '
                     f'passive H{c["passive_hit"]}/M{c["passive_miss"]}\n')

    # Statistics
    lines.append(f'\nSTATISTICS (posterior/occipital average)\n')
    lines.append(f'{"Test":<20} {"Cond":<12} {"Window":<6} '
                 f'{"Hit":>7} {"Miss":>7} {"Diff":>7} '
                 f'{"t":>7} {"p":>7} {"d":>6} {"N":>3}\n')
    lines.append('-' * 90 + '\n')
    for _, row in stats_df.iterrows():
        sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else ''
        lines.append(f'{row["test"]:<20} {row["condition"]:<12} {row["window"]:<6} '
                     f'{row["mean_hit"]:>7.2f} {row["mean_miss"]:>7.2f} '
                     f'{row["mean_diff"]:>7.2f} {row["t"]:>7.2f} '
                     f'{row["p"]:>7.4f} {row["cohens_d"]:>6.2f} {row["n"]:>3} {sig}\n')

    report_path = out_dir / 'dm_report.txt'
    report_path.write_text(''.join(lines))
    log.info(f'  Saved: dm_report.txt')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log = setup_logging()

    log.info('=' * 60)
    log.info('SUBSEQUENT MEMORY EFFECT (Dm) ANALYSIS')
    log.info('=' * 60)

    out_dir = OUT_DIR / 'dm_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    exclude = [s.zfill(3) for s in args.exclude]
    subs = [
        sub_id for sub_id in analysis_subjects(exclude)
        if (sub_deriv_dir(sub_id) / 'picture_onset-epo.fif').exists()
    ]

    log.info(f'Found {len(subs)} canonical preprocessed subjects')

    # Process each subject
    all_results = []
    for sub in subs:
        result = compute_subject_dm(sub, log)
        if result is not None:
            all_results.append(result)

    log.info(f'\nIncluded in group analysis: {len(all_results)} subjects')

    if len(all_results) < 3:
        log.error('Need at least 3 subjects for group Dm analysis')
        sys.exit(1)

    # Group analysis
    log.info('Running group Dm analysis...')
    run_group_dm(all_results, out_dir, log)

    log.info('=' * 60)
    log.info('Done!')
    log.info(f'Results: {out_dir.relative_to(ROOT)}/')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
