"""
Attentional Thief — Step 3: Fixation-Related Potential (FRP)
=============================================================
Input  : derivatives/sub-{sub}/preprocessed-raw.fif
         data/sub-{sub}/et/sub{N}.EDF

Output : sub-{sub}/frp_sync_diagnostics.png    — ET↔EEG sync quality
         sub-{sub}/frp_sync_pairs.csv           — matched image-onset pairs
         sub-{sub}/frp_clusters.png             — FRP cluster waveforms
         sub-{sub}/frp_topomaps.png             — FRP scalp maps by condition/window
         derivatives/sub-{sub}/fixation_onset-epo.fif — fixation-locked epochs
         sub-{sub}/frp_summary.csv              — component means per condition/cluster/window
         sub-{sub}/frp_condition_comparisons.csv— epoch-level condition contrasts

Sync strategy (fixes the sequential-pairing bug):
  1. Segment EEG into blocks using block_start/block_end triggers.
  2. Skip any block that has no corresponding ET entry (e.g. test blocks at
     the start of the recording where EyeLink rep=0 messages were sent).
  3. Match ET blocks → EEG blocks by condition and within-condition order.
  4. Within each matched block pair, pair image onsets by picture order.
  5. Convert fixation ET times to EEG times using the LOCAL image-onset
     anchor: eeg_fix_s = eeg_pic_s + (et_fix_s − et_pic_s).

Usage:
    python analysis/03_frp.py
    python analysis/03_frp.py --sub 002
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import eyelinkio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLUSTERS, FRP_WINDOWS, COLORS,
    et_edf, sub_out_dir, sub_deriv_dir,
)
from frp_helpers import (
    build_epoch_amplitude_table,
    compute_condition_comparisons,
    compute_subject_component_summary,
)
from et_eeg_sync import build_sync_pairs

mne.set_log_level('WARNING')

# Trigger codes
BLOCK_START = {10: 'active', 20: 'passive', 30: 'constant'}
BLOCK_END   = {19: 'active', 29: 'passive', 39: 'constant'}
PIC_ONSET   = {12: 'active', 22: 'passive', 32: 'constant'}

FRP_TMIN, FRP_TMAX = -0.2, 0.8
FRP_BASELINE       = (-0.2, 0.0)
FRP_REJECT         = 150e-6
FIX_DUR_MIN_MS     = 80
FIX_DUR_MAX_MS     = 1500


# ── EEG block segmentation ───────────────────────────────────────────────────

def _stim_code(annotation_desc: str) -> int | None:
    """Extract numeric trigger code from BrainVision annotation label."""
    m = re.search(r'S\s*(\d+)', annotation_desc)
    return int(m.group(1)) if m else None


def segment_eeg_blocks(raw: mne.io.Raw) -> list[dict]:
    """
    Parse a (possibly FIF-reloaded) Raw into a list of blocks using
    annotations directly, so that remapped integer codes are not an issue.

    Each block dict: {block_type, start_s, end_s, pic_samples: list[int]}
    """
    sfreq = raw.info['sfreq']
    blocks = []
    current = None

    for ann in raw.annotations:
        onset_s = ann['onset']
        sample = int(round(onset_s * sfreq))
        code = _stim_code(ann['description'])
        if code is None:
            continue

        if code in BLOCK_START:
            current = dict(
                block_type=BLOCK_START[code],
                start_s=onset_s,
                end_s=None,
                pic_samples=[],
            )
        elif code in PIC_ONSET and current is not None:
            current['pic_samples'].append(sample)
        elif code in BLOCK_END and current is not None:
            current['end_s'] = onset_s
            blocks.append(current)
            current = None

    return blocks


# ── ET parsing ───────────────────────────────────────────────────────────────

def parse_et_blocks(edf_path: Path) -> list[dict]:
    """
    Parse EyeLink EDF into a list of blocks (rep > 0 only).
    Each block dict: {block_type, rep, session, n_pics,
                      images: [{et_on, et_off, pic_idx}]}
    """
    edf = eyelinkio.read_edf(str(edf_path))
    msgs = [(t, m.decode(errors='replace').strip())
            for t, m in edf.discrete['messages']]

    blocks = []
    current = None
    for t, msg in msgs:
        if msg.startswith('BLOCK_START'):
            m = re.match(
                r'BLOCK_START\s+(\w+)\s+rep=(\d+)\s+session=(\d+)\s+n_pics=(\d+)',
                msg)
            if m and int(m.group(2)) > 0:       # skip rep=0 test runs
                current = dict(
                    block_type=m.group(1).lower().replace('geometry', 'constant'),
                    rep=int(m.group(2)),
                    session=int(m.group(3)),
                    n_pics=int(m.group(4)),
                    images=[],
                )
        elif msg.startswith('IMAGE_ON') and current is not None:
            m2 = re.search(r'pic=(\d+)/(\d+)', msg)
            pic_idx = int(m2.group(1)) if m2 else len(current['images']) + 1
            current['images'].append(
                dict(et_on=float(t), et_off=None, pic_idx=pic_idx))
        elif msg.startswith('IMAGE_OFF') and current is not None:
            if current['images'] and current['images'][-1]['et_off'] is None:
                current['images'][-1]['et_off'] = float(t)
        elif msg.startswith('BLOCK_END') and current is not None:
            blocks.append(current)
            current = None

    # attach fixations
    fix_df = _parse_fixations(edf)
    return blocks, fix_df


def _parse_fixations(edf) -> pd.DataFrame:
    fix = pd.DataFrame(edf.discrete['fixations'])
    # eyelinkio column names vary by version — handle both
    rename = {}
    for old, new in [('axp', 'x'), ('ayp', 'y'),
                     ('sttime', 'stime'), ('entime', 'etime')]:
        if old in fix.columns:
            rename[old] = new
    fix = fix.rename(columns=rename)
    fix['dur_ms'] = (fix['etime'] - fix['stime']) * 1000
    fix = fix[(fix['dur_ms'] >= FIX_DUR_MIN_MS) &
              (fix['dur_ms'] <= FIX_DUR_MAX_MS)].copy().reset_index(drop=True)
    return fix


# ── fixation → EEG time mapping ──────────────────────────────────────────────

def assign_fixations(fix_df: pd.DataFrame,
                     pairs: pd.DataFrame,
                     sfreq: float) -> pd.DataFrame:
    """
    For each fixation, find which image it belongs to and compute its EEG
    onset sample via the local image-onset anchor.
    """
    assigned = []
    for _, fix in fix_df.iterrows():
        for _, img in pairs.iterrows():
            # fixation must fall inside the image window (with small guard)
            if (img['et_on'] + 0.08 <= fix['stime'] <=
                    img['et_off'] - 0.08):
                # LOCAL anchor: ET offset from image onset → EEG sample
                dt = fix['stime'] - img['et_on']
                eeg_fix_s = img['eeg_pic_s'] + dt
                assigned.append(dict(
                    block_type     = img['block_type'],
                    rep            = img['rep'],
                    session        = img['session'],
                    pic_idx        = img['pic_idx'],
                    fix_onset_eeg_s= eeg_fix_s,
                    fix_onset_sample=int(round(eeg_fix_s * sfreq)),
                    dur_ms         = fix['dur_ms'],
                ))
                break
        else:
            assigned.append(None)

    fix_df = fix_df.copy()
    fix_df['_match'] = assigned
    fix_df = fix_df[fix_df['_match'].notna()].copy()
    extra = pd.DataFrame(fix_df['_match'].tolist(), index=fix_df.index)
    fix_df = pd.concat([fix_df.drop(columns=['_match']), extra], axis=1)
    return fix_df.reset_index(drop=True)


# ── sync diagnostics ─────────────────────────────────────────────────────────

def plot_sync_diagnostics(pairs: pd.DataFrame, out_path: Path,
                          sub: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. scatter ET vs EEG image onset times
    code_map = {'active': 0, 'passive': 1, 'constant': 2}
    c = pairs['block_type'].map(code_map)
    sc = axes[0].scatter(pairs['et_on'], pairs['eeg_pic_s'],
                         c=c, s=12, cmap='Set1', alpha=0.7)
    axes[0].set_xlabel('ET image onset (s)')
    axes[0].set_ylabel('EEG image onset (s)')
    axes[0].set_title('Image onset: ET vs EEG')
    for bt, idx in code_map.items():
        sub_p = pairs[pairs['block_type'] == bt]
        if len(sub_p) < 2:
            continue
        x, y = sub_p['et_on'].to_numpy(), sub_p['eeg_pic_s'].to_numpy()
        sl, ic = np.polyfit(x, y, 1)
        axes[0].plot([x.min(), x.max()],
                     [sl * x.min() + ic, sl * x.max() + ic],
                     color=plt.cm.Set1(idx / 3), lw=1.5, label=bt)
    axes[0].legend(fontsize=8, frameon=False)

    # 2. per-block within-block RMSE
    rmse_rows = []
    for (rep, session, bt), g in pairs.groupby(['rep', 'session', 'block_type']):
        if len(g) < 2:
            continue
        x, y = g['et_on'].to_numpy(), g['eeg_pic_s'].to_numpy()
        sl, ic = np.polyfit(x, y, 1)
        resid_ms = (y - (sl * x + ic)) * 1000
        rmse_rows.append(dict(
            label=f'r{rep}s{session}-{bt[:3]}',
            rmse_ms=np.sqrt(np.mean(resid_ms ** 2)),
            n=len(g),
        ))
    if rmse_rows:
        rdf = pd.DataFrame(rmse_rows).sort_values('rmse_ms')
        axes[1].barh(rdf['label'], rdf['rmse_ms'],
                     color=['#d62728' if v > 100 else '#1f77b4'
                            for v in rdf['rmse_ms']])
        axes[1].set_xlabel('Within-block RMSE (ms)')
        axes[1].set_title('Block-wise sync quality')
        axes[1].axvline(20, color='gray', ls='--', lw=1, label='20 ms')
        axes[1].legend(fontsize=8, frameon=False)

    # 3. ET→EEG offset per block (should be stable)
    offsets = []
    for (rep, session, bt), g in pairs.groupby(['rep', 'session', 'block_type']):
        offset_ms = (g['eeg_pic_s'] - g['et_on']).mean() * 1000
        offsets.append(dict(
            label=f'r{rep}s{session}-{bt[:3]}',
            offset_ms=offset_ms,
            block_type=bt,
        ))
    if offsets:
        odf = pd.DataFrame(offsets)
        for bt, grp in odf.groupby('block_type'):
            axes[2].scatter(range(len(grp)), grp['offset_ms'],
                            label=bt, s=40,
                            color=COLORS.get(bt, '#888888'))
        axes[2].set_xlabel('Block index')
        axes[2].set_ylabel('Mean ET→EEG offset (ms)')
        axes[2].set_title('ET→EEG offset per block')
        axes[2].legend(fontsize=8, frameon=False)

    fig.suptitle(f'sub-{sub} ET↔EEG sync diagnostics', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── FRP plotting ─────────────────────────────────────────────────────────────

def _shade_windows(ax, windows=None):
    if windows is None:
        windows = FRP_WINDOWS
    colors = {
        'lambda': '#bdbdbd',
        'P1_frp': '#c6dbef',
        'N1_frp': '#fdd0a2',
        'P2_frp': '#c7e9c0',
        'Late_frp': '#fcbba1',
    }
    for name, (a, b) in windows.items():
        ax.axvspan(a * 1000, b * 1000, color=colors.get(name, '#eeeeee'),
                   alpha=0.16, zorder=0)


def plot_frp_clusters(evokeds: dict, out_path: Path, sub: str) -> None:
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
        ax.set_title(label.title())
        ax.set_ylabel('µV')

    axes[0].legend(frameon=False, ncol=len(conds), fontsize=9)
    axes[-1].set_xlabel('Time from fixation onset (ms)')
    axes[-1].set_xlim(evokeds[conds[0]].times[[0, -1]] * 1000)
    fig.suptitle(f'sub-{sub} — Fixation-locked ERP (FRP)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_frp_topomaps(evokeds: dict, out_path: Path, title: str = '') -> None:
    conds = [c for c in ['active', 'passive', 'constant'] if c in evokeds]
    windows = list(FRP_WINDOWS.items())
    n_cond = len(conds)
    n_wins = len(windows)
    fig, axes = plt.subplots(n_cond, n_wins,
                             figsize=(2.5 * n_wins, 2.5 * n_cond))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    for r, cond in enumerate(conds):
        ev = evokeds[cond]
        for c, (window_name, (t0, t1)) in enumerate(windows):
            ax = axes[r, c]
            center = (t0 + t1) / 2
            if center > ev.times[-1]:
                ax.axis('off')
                continue
            topo = ev.copy().crop(tmin=t0, tmax=t1).data.mean(axis=1) * 1e6
            mne.viz.plot_topomap(
                topo,
                ev.info,
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(-4, 4),
            )
            if r == 0:
                ax.set_title(f'{window_name}\n{int(t0*1000)}–{int(t1*1000)} ms', fontsize=8)
        axes[r, 0].set_ylabel(cond, fontsize=9)

    fig.suptitle(title, y=1.001, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sub = args.sub
    deriv = sub_deriv_dir(sub)
    out = sub_out_dir(sub)
    raw_path = deriv / 'preprocessed-raw.fif'
    edf_path = et_edf(sub)

    for p, name in [(raw_path, 'preprocessed-raw.fif (run 01_preprocess.py)'),
                    (edf_path, f'ET EDF file')]:
        if not p.exists():
            sys.exit(f'Not found: {p}\n({name})')

    print(f'\n── sub-{sub} FRP ────────────────────────')

    # 1. Load EEG
    print('  Loading preprocessed EEG...')
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose='ERROR')
    sfreq = raw.info['sfreq']
    events, _ = mne.events_from_annotations(raw, verbose='ERROR')

    # 2. Segment EEG into blocks
    print('  Segmenting EEG blocks...')
    eeg_blocks = segment_eeg_blocks(raw)
    for bt in ['active', 'passive', 'constant']:
        n = sum(b['block_type'] == bt for b in eeg_blocks)
        pics = sum(len(b['pic_samples'])
                   for b in eeg_blocks if b['block_type'] == bt)
        print(f'    EEG {bt}: {n} blocks, {pics} pic triggers')

    # 3. Parse ET
    print('  Parsing ET EDF...')
    et_blocks, fix_df = parse_et_blocks(edf_path)
    for bt in ['active', 'passive', 'constant']:
        n = sum(b['block_type'] == bt for b in et_blocks)
        pics = sum(len(b['images']) for b in et_blocks
                   if b['block_type'] == bt)
        print(f'    ET  {bt}: {n} blocks, {pics} images')
    print(f'    ET fixations (after duration filter): {len(fix_df)}')

    # 4. Build sync pairs
    print('  Building sync pairs...')
    pairs = build_sync_pairs(eeg_blocks, et_blocks)
    print(f'    Paired image onsets: {len(pairs)}')
    for bt in ['active', 'passive', 'constant']:
        n = (pairs['block_type'] == bt).sum()
        print(f'      {bt}: {n}')

    # 5. Diagnostics
    sync_plot = out / 'frp_sync_diagnostics.png'
    plot_sync_diagnostics(pairs, sync_plot, sub)
    pairs.to_csv(out / 'frp_sync_pairs.csv', index=False)
    print(f'  Saved: {sync_plot.relative_to(sync_plot.parents[3])}')

    # 6. Assign fixations to images
    print('  Assigning fixations to images...')
    fix_assigned = assign_fixations(fix_df, pairs, sfreq)
    for bt in ['active', 'passive', 'constant']:
        n = (fix_assigned['block_type'] == bt).sum()
        print(f'    {bt}: {n} fixations assigned')

    # 7. Build FRP epochs
    print('  Building FRP epochs...')
    fix_assigned = fix_assigned.sort_values('fix_onset_sample').reset_index(drop=True)

    # Deduplicate: two fixations can map to the same sample at 1000 Hz
    n_before_dedup = len(fix_assigned)
    fix_assigned = fix_assigned.drop_duplicates(subset='fix_onset_sample', keep='first')
    n_dropped = n_before_dedup - len(fix_assigned)
    if n_dropped:
        print(f'    Dropped {n_dropped} duplicate-sample fixations')

    code_map = {'active': 1, 'passive': 2, 'constant': 3}
    frp_events = np.column_stack([
        fix_assigned['fix_onset_sample'].to_numpy(),
        np.zeros(len(fix_assigned), dtype=int),
        fix_assigned['block_type'].map(code_map).to_numpy(),
    ])

    epochs = mne.Epochs(
        raw, frp_events,
        event_id=code_map,
        tmin=FRP_TMIN, tmax=FRP_TMAX,
        baseline=FRP_BASELINE,
        preload=True,
        reject=dict(eeg=FRP_REJECT),
        reject_by_annotation=True,
        verbose='ERROR',
    )
    epochs.drop_bad(verbose='ERROR')

    conds_kept = [c for c in ['active', 'passive', 'constant']
                  if c in epochs.event_id and len(epochs[c]) >= 5]
    for c in conds_kept:
        print(f'    {c}: {len(epochs[c])} epochs kept')

    # 8. Save epochs
    epo_path = deriv / 'fixation_onset-epo.fif'
    epochs.save(str(epo_path), overwrite=True, verbose='ERROR')
    print(f'  Saved: {epo_path.relative_to(epo_path.parents[2])}')

    # 9. FRP plots
    evokeds = {c: epochs[c].average() for c in conds_kept}
    frp_plot = out / 'frp_clusters.png'
    plot_frp_clusters(evokeds, frp_plot, sub)
    print(f'  Saved: {frp_plot.relative_to(frp_plot.parents[3])}')

    topo_path = out / 'frp_topomaps.png'
    plot_frp_topomaps(evokeds, topo_path,
                      title=f'sub-{sub} — FRP topomaps by component window')
    print(f'  Saved: {topo_path.relative_to(topo_path.parents[3])}')

    # 10. Summary CSV
    summary_df = compute_subject_component_summary(
        evokeds=evokeds,
        windows=FRP_WINDOWS,
        clusters=CLUSTERS,
        subject=sub,
    )
    summary_df.to_csv(out / 'frp_summary.csv', index=False)
    print('  Saved: frp_summary.csv')

    # 11. Epoch-level condition comparisons
    epoch_table = build_epoch_amplitude_table(
        epochs,
        windows=FRP_WINDOWS,
        clusters=CLUSTERS,
        conditions=conds_kept,
    )
    comparison_df = compute_condition_comparisons(epoch_table)
    comparison_df.to_csv(out / 'frp_condition_comparisons.csv', index=False)
    print('  Saved: frp_condition_comparisons.csv')
    print('  Done.\n')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sub', default='001')
    return p.parse_args()


if __name__ == '__main__':
    main()
