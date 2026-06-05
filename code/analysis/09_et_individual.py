"""
Attentional Thief — Step 9: Individual eye-tracking analysis
=============================================================
Input  : data/sub-{sub}/et/sub{N}.EDF

Output : sub-{sub}/et_summary.csv        — per-condition fixation/saccade/pupil summary
         sub-{sub}/et_fixations.csv      — fixations with block_type/triplet
         sub-{sub}/et_saccades.csv       — saccades with block_type/triplet
         sub-{sub}/et_pupil.csv          — per-block pupil-size mean
         sub-{sub}/et_metrics.png        — per-subject bar plot (fix/sacc/pupil)

Logic mirrors analysis/notebooks/et_analysis.ipynb but writes per-subject
artifacts, so the eye-tracking analysis sits next to the EEG ERP/FRP outputs.

Usage:
    python analysis/09_et_individual.py --sub 022
    python analysis/09_et_individual.py --all --skip-existing
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eyelinkio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COLORS, DATA_DIR, analysis_subjects, et_edf, normalize_subject_id, sub_out_dir

BLOCK_TYPES = ['active', 'passive', 'constant']


# ── EDF parsing ──────────────────────────────────────────────────────────────

def get_block_intervals(edf) -> pd.DataFrame:
    msgs = [(t, m.decode(errors='replace').strip())
            for t, m in edf.discrete['messages']]
    blocks, cur, in_practice = [], {}, False
    for t, msg in msgs:
        if msg == 'PRACTICE_START':
            in_practice = True
        elif msg == 'PRACTICE_END':
            in_practice = False
        elif msg.startswith('BLOCK_START'):
            parts = msg.split()
            triplet = next((p.split('=')[1] for p in parts
                            if p.startswith('triplet=')), None)
            cur = dict(block_type=parts[1].lower(), triplet=triplet,
                       onset=t, is_practice=in_practice)
        elif msg.startswith('BLOCK_END') and cur:
            cur['offset'] = t
            blocks.append(cur.copy())
            cur = {}
    df = pd.DataFrame(blocks)
    if df.empty:
        return df
    return df[~df['is_practice']].reset_index(drop=True)


def assign_to_blocks(df_events: pd.DataFrame, df_blocks: pd.DataFrame,
                     time_col: str = 'stime') -> pd.DataFrame:
    bt_list, trip_list = [], []
    onsets = df_blocks['onset'].to_numpy()
    offsets = df_blocks['offset'].to_numpy()
    bt_arr = df_blocks['block_type'].to_numpy()
    trip_arr = df_blocks['triplet'].to_numpy()
    for t in df_events[time_col].to_numpy():
        idx = np.where((onsets <= t) & (offsets >= t))[0]
        if len(idx):
            bt_list.append(bt_arr[idx[0]])
            trip_list.append(trip_arr[idx[0]])
        else:
            bt_list.append(None)
            trip_list.append(None)
    out = df_events.copy()
    out['block_type'] = bt_list
    out['triplet'] = trip_list
    return out


# ── Per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(sub: str) -> dict:
    edf_path = et_edf(sub)
    if not edf_path.exists():
        raise FileNotFoundError(f'EDF not found: {edf_path}')

    edf = eyelinkio.read_edf(edf_path)
    df_blocks = get_block_intervals(edf)
    if df_blocks.empty:
        raise RuntimeError(f'sub-{sub}: no main blocks found in EDF')

    # Block durations
    df_blocks = df_blocks.copy()
    df_blocks['duration_s'] = df_blocks['offset'] - df_blocks['onset']
    dur_by_cond = (df_blocks.groupby('block_type')['duration_s']
                   .sum().reset_index())

    # Fixations
    fix = pd.DataFrame(edf.discrete['fixations'])
    fix.columns = ['eye', 'stime', 'etime', 'x', 'y']
    fix['duration_ms'] = (fix['etime'] - fix['stime']) * 1000
    fix = assign_to_blocks(fix, df_blocks)

    # Saccades
    sacc = pd.DataFrame(edf.discrete['saccades'])
    sacc.columns = ['eye', 'stime', 'etime', 'sx', 'sy', 'ex', 'ey', 'pv']
    sacc['duration_ms'] = (sacc['etime'] - sacc['stime']) * 1000
    sacc['amplitude'] = np.sqrt((sacc['ex'] - sacc['sx']) ** 2
                                + (sacc['ey'] - sacc['sy']) ** 2)
    sacc = assign_to_blocks(sacc, df_blocks)

    # Pupil per block
    sfreq = edf.info['sfreq']
    times = np.arange(edf._samples.shape[1]) / sfreq
    ps = edf._samples[2]
    pupil_rows = []
    for _, blk in df_blocks.iterrows():
        mask = (times >= blk['onset']) & (times <= blk['offset']) & (ps > 0)
        if mask.sum():
            pupil_rows.append(dict(
                block_type=blk['block_type'],
                triplet=blk['triplet'],
                pupil_mean=float(ps[mask].mean()),
                pupil_sd=float(ps[mask].std()),
                n_samples=int(mask.sum()),
            ))
    df_pup = pd.DataFrame(pupil_rows)

    # Per-condition summary
    fix_main = fix.dropna(subset=['block_type'])
    fix_sum = (fix_main.groupby('block_type')
               .agg(n_fix=('duration_ms', 'count'),
                    mean_fix_dur=('duration_ms', 'mean'))
               .reset_index()
               .merge(dur_by_cond, on='block_type'))
    fix_sum['fix_rate'] = fix_sum['n_fix'] / fix_sum['duration_s']

    sacc_main = sacc.dropna(subset=['block_type'])
    sacc_sum = (sacc_main.groupby('block_type')
                .agg(n_sacc=('duration_ms', 'count'),
                     mean_sacc_amp=('amplitude', 'mean'),
                     mean_sacc_pv=('pv', 'mean'))
                .reset_index()
                .merge(dur_by_cond, on='block_type'))
    sacc_sum['sacc_rate'] = sacc_sum['n_sacc'] / sacc_sum['duration_s']

    if not df_pup.empty:
        pup_summary = (df_pup.groupby('block_type')
                       .apply(lambda g: pd.Series({
                           'pupil_mean': float(np.average(g['pupil_mean'],
                                                          weights=g['n_samples'])),
                           'pupil_sd': float(g['pupil_sd'].mean()),
                       }))
                       .reset_index())
    else:
        pup_summary = pd.DataFrame(columns=['block_type', 'pupil_mean', 'pupil_sd'])

    summary = (fix_sum.merge(sacc_sum, on=['block_type', 'duration_s'],
                             how='outer')
               .merge(pup_summary, on='block_type', how='outer'))
    summary.insert(0, 'subject', sub)

    return dict(
        summary=summary,
        fixations=fix,
        saccades=sacc,
        pupil=df_pup,
        n_blocks=len(df_blocks),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_subject(summary: pd.DataFrame, sub: str, out_path: Path) -> None:
    metrics = [
        ('n_fix', 'count', 'N fixations'),
        ('mean_fix_dur', 'ms', 'Mean fix duration'),
        ('fix_rate', 'per sec', 'Fixation rate'),
        ('n_sacc', 'count', 'N saccades'),
        ('mean_sacc_amp', 'px', 'Mean sacc amplitude'),
        ('sacc_rate', 'per sec', 'Saccade rate'),
        ('pupil_mean', 'au', 'Pupil mean'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for ax, (col, ylab, title) in zip(axes, metrics):
        if col not in summary.columns:
            ax.axis('off')
            continue
        rows = summary.set_index('block_type').reindex(BLOCK_TYPES)
        ys = rows[col].to_numpy()
        colors = [COLORS[bt] for bt in BLOCK_TYPES]
        ax.bar(range(3), ys, color=colors, alpha=0.85)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Scrolling', 'Watching', 'Baseline'], fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        for i, v in enumerate(ys):
            if not np.isnan(v):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    # Hide trailing empty axis
    for ax in axes[len(metrics):]:
        ax.axis('off')

    fig.suptitle(f'sub-{sub} — eye-tracking by condition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────

def discover_subjects() -> list[str]:
    return [sub for sub in analysis_subjects() if et_edf(sub).exists()]


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--sub', nargs='+', help='Subject IDs (e.g. 003 022)')
    g.add_argument('--all', action='store_true', help='All subjects in data/')
    p.add_argument('--exclude', nargs='*', default=[],
                   help='Subject IDs to skip')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip subjects with et_summary.csv already present')
    return p.parse_args()


def main():
    args = parse_args()
    if args.all or not args.sub:
        subs = discover_subjects()
    else:
        subs = [normalize_subject_id(s) for s in args.sub]
    exclude = {normalize_subject_id(s) for s in args.exclude}
    subs = [s for s in subs if s not in exclude]

    print(f'Processing {len(subs)} subjects: {subs}')
    failed = []
    for sub in subs:
        out_dir = sub_out_dir(sub)
        summary_csv = out_dir / 'et_summary.csv'
        if args.skip_existing and summary_csv.exists():
            print(f'  sub-{sub}: skipped (existing)')
            continue
        try:
            print(f'  sub-{sub}: processing…', flush=True)
            res = process_subject(sub)
            res['summary'].to_csv(summary_csv, index=False)
            res['fixations'].to_csv(out_dir / 'et_fixations.csv', index=False)
            res['saccades'].to_csv(out_dir / 'et_saccades.csv', index=False)
            res['pupil'].to_csv(out_dir / 'et_pupil.csv', index=False)
            plot_subject(res['summary'], sub, out_dir / 'et_metrics.png')
            print(f'  sub-{sub}: done ({res["n_blocks"]} main blocks)')
        except Exception as e:
            print(f'  sub-{sub}: FAILED — {type(e).__name__}: {e}')
            failed.append(sub)

    if failed:
        print(f'\nFailed subjects: {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
