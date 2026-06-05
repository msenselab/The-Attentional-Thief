"""
Full Behavioral Analysis — All Subjects
========================================
Analyzes time estimation and recognition data across all conditions.

Measures:
  1. Time estimation ratio (estimated/actual) by condition
  2. Recognition accuracy and RT by condition (scrolling/watching only)
  3. Per-picture viewing duration patterns
  4. Condition × pic_count interactions

Output: analysis/output/behavioral_all/

Usage:
    python analysis/code/behavioral_analysis.py
"""
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, DATA_DIR, OUT_DIR, COLORS

OUT = OUT_DIR / 'behavioral_all'
OUT.mkdir(parents=True, exist_ok=True)
COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching', 'constant': 'Baseline'}

# ── Bracket-aware CSV parser ────────────────────────────────────────────────
def parse_behavioral_csv(csv_path):
    """Parse CSV with embedded arrays (brackets) that contain commas."""
    rows = []
    with open(csv_path) as fh:
        lines = fh.readlines()

    if not lines:
        return pd.DataFrame()

    header_line = lines[0].strip().rstrip(',')
    # Parse header properly
    header = header_line.split(',')

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Bracket-aware split: track [] and "" separately
        fields = []
        current = []
        in_quotes = False
        bracket_depth = 0
        for ch in line:
            if ch == '"' and bracket_depth == 0:
                in_quotes = not in_quotes
                current.append(ch)
            elif ch == '[' and in_quotes:
                bracket_depth += 1
                current.append(ch)
            elif ch == ']' and bracket_depth > 0:
                bracket_depth -= 1
                current.append(ch)
            elif ch == ',' and not in_quotes and bracket_depth == 0:
                fields.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        fields.append(''.join(current).strip())

        # Build row dict
        row = {}
        for i, h in enumerate(header):
            h = h.strip()
            if i < len(fields):
                row[h] = fields[i]
            else:
                row[h] = ''
        rows.append(row)

    return pd.DataFrame(rows)


def load_all_behavioral():
    """Load behavioral data for all subjects."""
    all_rows = []

    for i in range(1, 30):
        sub = f'{i:03d}'
        # Check both standard and nested layout
        beh_dirs = [
            DATA_DIR / f'sub-{sub}' / 'beh',
            DATA_DIR / f'sub-{sub}' / 'eeg' / 'beh',
        ]
        csv_path = None
        for bd in beh_dirs:
            if bd.exists():
                csvs = sorted(bd.glob('attention_thief_ET_*.csv'))
                if csvs:
                    csv_path = csvs[0]
                    break

        if csv_path is None:
            continue

        df = parse_behavioral_csv(csv_path)
        if df.empty:
            continue

        df['sub'] = sub
        all_rows.append(df)
        print(f'  sub-{sub}: {len(df)} rows from {csv_path.name}')

    if not all_rows:
        print('ERROR: No behavioral data found!')
        sys.exit(1)

    return pd.concat(all_rows, ignore_index=True)


# ── Data cleaning ────────────────────────────────────────────────────────────
def clean_data(df):
    """Convert types and filter to relevant rows."""
    # Filter to time_estimation phase
    est = df[df['phase'] == 'time_estimation'].copy()

    # Convert numeric columns
    for col in ['actual_duration', 'estimated_duration', 'estimation_ratio',
                'estimation_error', 'n_pictures', 'rep']:
        if col in est.columns:
            est[col] = pd.to_numeric(est[col], errors='coerce')

    # Clean session_type
    est['session_type'] = est['session_type'].str.strip().str.lower()

    # Filter to valid conditions
    est = est[est['session_type'].isin(['active', 'passive', 'constant'])].copy()

    # Recognition data — embedded in the estimation rows (not separate rows)
    # Filter to rows that have recognition data (scrolling/watching only)
    recog = est[est['recognition_image'].astype(str).str.strip().ne('')].copy()
    if not recog.empty:
        for col in ['recognition_rt']:
            if col in recog.columns:
                recog[col] = pd.to_numeric(recog[col], errors='coerce')

        # Parse boolean columns
        for col in ['recognition_is_old', 'recognition_correct']:
            if col in recog.columns:
                recog[col] = recog[col].astype(str).str.strip().str.lower().map(
                    {'true': True, 'false': False, '1': True, '0': False})

        recog['recognition_response'] = recog['recognition_response'].astype(str).str.strip().str.lower()

    return est, recog


# ── Analysis functions ───────────────────────────────────────────────────────
def estimation_analysis(est, out_dir):
    """Time estimation ratio by condition."""
    print('\n' + '=' * 60)
    print('TIME ESTIMATION ANALYSIS')
    print('=' * 60)

    # Per-subject means
    sub_means = est.groupby(['sub', 'session_type'])['estimation_ratio'].mean().reset_index()
    pivot = sub_means.pivot(index='sub', columns='session_type', values='estimation_ratio')
    n_subs = len(pivot)

    print(f'\nN = {n_subs} subjects')
    print('\nCondition means (estimation ratio = estimated/actual):')
    for cond in ['active', 'passive', 'constant']:
        if cond in pivot.columns:
            m, s = pivot[cond].mean(), pivot[cond].sem()
            print(f'  {cond:>10}: M = {m:.3f} (SEM = {s:.3f})')

    # One-sample t-tests vs 1.0 (accurate estimation)
    print('\nOne-sample t-tests vs 1.0 (no distortion):')
    for cond in ['active', 'passive', 'constant']:
        if cond in pivot.columns:
            t, p = stats.ttest_1samp(pivot[cond].dropna(), 1.0)
            d = (pivot[cond].mean() - 1.0) / pivot[cond].std()
            print(f'  {cond:>10}: t({len(pivot[cond].dropna())-1}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}')

    # Paired comparisons
    print('\nPaired t-tests:')
    conds = [c for c in ['active', 'passive', 'constant'] if c in pivot.columns]
    comp_results = []
    for c1, c2 in combinations(conds, 2):
        valid = pivot[[c1, c2]].dropna()
        if len(valid) >= 3:
            t, p = stats.ttest_rel(valid[c1], valid[c2])
            d = (valid[c1] - valid[c2]).mean() / (valid[c1] - valid[c2]).std()
            print(f'  {c1} vs {c2}: t({len(valid)-1}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}')
            comp_results.append(dict(comparison=f'{c1}_vs_{c2}', t=t, p=p, d=d, n=len(valid)))

    # Repeated-measures ANOVA (if scipy has it, else F-test via manual computation)
    if len(conds) >= 3:
        valid3 = pivot[conds].dropna()
        if len(valid3) >= 3:
            F, p = stats.friedmanchisquare(*[valid3[c] for c in conds])
            print(f'\n  Friedman test (non-parametric): χ²({len(conds)-1}) = {F:.2f}, p = {p:.4f}')

    # ── Figure: estimation ratio by condition ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Bar plot with individual dots
    ax = axes[0]
    cond_colors = [COLORS.get(c, 'gray') for c in conds]
    means = [pivot[c].mean() for c in conds]
    sems = [pivot[c].sem() for c in conds]
    x = range(len(conds))
    bars = ax.bar(x, means, yerr=sems, color=cond_colors, alpha=0.7, capsize=5, width=0.6)

    # Individual data points
    for i, c in enumerate(conds):
        jitter = np.random.normal(0, 0.06, len(pivot[c].dropna()))
        ax.scatter(i + jitter, pivot[c].dropna(), color='black', alpha=0.4, s=20, zorder=5)

    ax.axhline(1.0, color='gray', ls='--', lw=1, label='Veridical')
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c.capitalize()) for c in conds])
    ax.set_ylabel('Estimation Ratio (estimated / actual)')
    ax.set_title('Time Estimation by Condition')
    ax.legend()

    # By n_pictures
    ax = axes[1]
    for cond in conds:
        cond_data = est[est['session_type'] == cond]
        pic_means = cond_data.groupby('n_pictures')['estimation_ratio'].mean()
        pic_sems = cond_data.groupby('n_pictures')['estimation_ratio'].sem()
        ax.errorbar(pic_means.index, pic_means.values, yerr=pic_sems.values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond.capitalize()), capsize=3)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Number of Pictures')
    ax.set_ylabel('Estimation Ratio')
    ax.set_title('Estimation by Pic Count')
    ax.legend()

    # By rep
    ax = axes[2]
    for cond in conds:
        cond_data = est[est['session_type'] == cond]
        rep_means = cond_data.groupby('rep')['estimation_ratio'].mean()
        rep_sems = cond_data.groupby('rep')['estimation_ratio'].sem()
        ax.errorbar(rep_means.index, rep_means.values, yerr=rep_sems.values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond.capitalize()), capsize=3)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Repetition')
    ax.set_ylabel('Estimation Ratio')
    ax.set_title('Estimation by Repetition')
    ax.legend()

    fig.suptitle(f'Time Estimation Analysis (N={n_subs})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'estimation_by_condition.png', dpi=150)
    plt.close(fig)
    print(f'\n  Saved: estimation_by_condition.png')

    return pivot, comp_results


def recognition_analysis(recog, out_dir):
    """Recognition accuracy and RT analysis."""
    print('\n' + '=' * 60)
    print('RECOGNITION ANALYSIS')
    print('=' * 60)

    if recog.empty:
        print('  No recognition data found!')
        return None, []

    recog = recog[recog['session_type'].isin(['active', 'passive'])].copy()

    # Per-subject accuracy
    acc = recog.groupby(['sub', 'session_type'])['recognition_correct'].mean().reset_index()
    acc.columns = ['sub', 'session_type', 'accuracy']
    pivot_acc = acc.pivot(index='sub', columns='session_type', values='accuracy')

    # Per-subject RT (correct trials only)
    correct_only = recog[recog['recognition_correct'] == True].copy()
    rt = correct_only.groupby(['sub', 'session_type'])['recognition_rt'].mean().reset_index()
    rt.columns = ['sub', 'session_type', 'rt']
    pivot_rt = rt.pivot(index='sub', columns='session_type', values='rt')

    n_subs = len(pivot_acc)
    print(f'\nN = {n_subs} subjects')

    print('\nAccuracy:')
    for cond in ['active', 'passive']:
        if cond in pivot_acc.columns:
            m, s = pivot_acc[cond].mean(), pivot_acc[cond].sem()
            print(f'  {cond:>10}: M = {m:.3f} (SEM = {s:.3f})')

    print('\nRT (correct trials):')
    for cond in ['active', 'passive']:
        if cond in pivot_rt.columns:
            m, s = pivot_rt[cond].mean(), pivot_rt[cond].sem()
            print(f'  {cond:>10}: M = {m:.0f} ms (SEM = {s:.0f})')

    # Paired t-tests
    comp_results = []
    for measure, pivot_m, label in [('accuracy', pivot_acc, 'Accuracy'),
                                      ('rt', pivot_rt, 'RT')]:
        valid = pivot_m[['active', 'passive']].dropna()
        if len(valid) >= 3:
            t, p = stats.ttest_rel(valid['active'], valid['passive'])
            d = (valid['active'] - valid['passive']).mean() / (valid['active'] - valid['passive']).std()
            print(f'\n  {label}: scrolling vs watching: t({len(valid)-1}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}')
            comp_results.append(dict(measure=label, t=t, p=p, d=d, n=len(valid)))

    # Signal detection: hit rate, false alarm rate, d'
    print('\nSignal Detection:')
    dprime_rows = []
    for sub in recog['sub'].unique():
        for cond in ['active', 'passive']:
            sub_cond = recog[(recog['sub'] == sub) & (recog['session_type'] == cond)]
            if sub_cond.empty:
                continue
            old_trials = sub_cond[sub_cond['recognition_is_old'] == True]
            new_trials = sub_cond[sub_cond['recognition_is_old'] == False]

            hit_rate = (old_trials['recognition_response'] == 'old').mean() if len(old_trials) > 0 else np.nan
            fa_rate = (new_trials['recognition_response'] == 'old').mean() if len(new_trials) > 0 else np.nan

            # Correct for 0/1 rates
            n_old, n_new = len(old_trials), len(new_trials)
            if hit_rate == 1.0: hit_rate = 1 - 1/(2*n_old)
            if hit_rate == 0.0: hit_rate = 1/(2*n_old)
            if fa_rate == 1.0: fa_rate = 1 - 1/(2*n_new)
            if fa_rate == 0.0: fa_rate = 1/(2*n_new)

            d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate) if not (np.isnan(hit_rate) or np.isnan(fa_rate)) else np.nan
            criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate)) if not (np.isnan(hit_rate) or np.isnan(fa_rate)) else np.nan

            dprime_rows.append(dict(sub=sub, session_type=cond,
                                    hit_rate=hit_rate, fa_rate=fa_rate,
                                    d_prime=d_prime, criterion=criterion))

    dprime_df = pd.DataFrame(dprime_rows)
    pivot_dp = dprime_df.pivot(index='sub', columns='session_type', values='d_prime')

    for cond in ['active', 'passive']:
        if cond in pivot_dp.columns:
            m, s = pivot_dp[cond].mean(), pivot_dp[cond].sem()
            print(f"  d' {cond:>10}: M = {m:.2f} (SEM = {s:.2f})")

    valid_dp = pivot_dp[['active', 'passive']].dropna()
    if len(valid_dp) >= 3:
        t, p = stats.ttest_rel(valid_dp['active'], valid_dp['passive'])
        d = (valid_dp['active'] - valid_dp['passive']).mean() / (valid_dp['active'] - valid_dp['passive']).std()
        print(f"  d' scrolling vs watching: t({len(valid_dp)-1}) = {t:.2f}, p = {p:.4f}, d = {d:.2f}")

    # Criterion
    pivot_c = dprime_df.pivot(index='sub', columns='session_type', values='criterion')
    for cond in ['active', 'passive']:
        if cond in pivot_c.columns:
            m, s = pivot_c[cond].mean(), pivot_c[cond].sem()
            print(f"  c  {cond:>10}: M = {m:.2f} (SEM = {s:.2f})")

    valid_c = pivot_c[['active', 'passive']].dropna()
    if len(valid_c) >= 3:
        t, p = stats.ttest_rel(valid_c['active'], valid_c['passive'])
        print(f"  c scrolling vs watching: t({len(valid_c)-1}) = {t:.2f}, p = {p:.4f}")

    # ── Figure ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Accuracy
    ax = axes[0]
    for i, cond in enumerate(['active', 'passive']):
        vals = pivot_acc[cond].dropna()
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(i + jitter, vals, color=COLORS.get(cond, 'gray'), alpha=0.5, s=25)
        ax.bar(i, vals.mean(), yerr=vals.sem(), color=COLORS.get(cond, 'gray'),
               alpha=0.4, capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Scrolling', 'Watching'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Recognition Accuracy')
    ax.set_ylim(0.5, 1.05)

    # RT
    ax = axes[1]
    for i, cond in enumerate(['active', 'passive']):
        if cond in pivot_rt.columns:
            vals = pivot_rt[cond].dropna()
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(i + jitter, vals, color=COLORS.get(cond, 'gray'), alpha=0.5, s=25)
            ax.bar(i, vals.mean(), yerr=vals.sem(), color=COLORS.get(cond, 'gray'),
                   alpha=0.4, capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Scrolling', 'Watching'])
    ax.set_ylabel('RT (s)')
    ax.set_title('Recognition RT')

    # d'
    ax = axes[2]
    for i, cond in enumerate(['active', 'passive']):
        if cond in pivot_dp.columns:
            vals = pivot_dp[cond].dropna()
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(i + jitter, vals, color=COLORS.get(cond, 'gray'), alpha=0.5, s=25)
            ax.bar(i, vals.mean(), yerr=vals.sem(), color=COLORS.get(cond, 'gray'),
                   alpha=0.4, capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Scrolling', 'Watching'])
    ax.set_ylabel("d'")
    ax.set_title("Sensitivity (d')")

    # Criterion
    ax = axes[3]
    for i, cond in enumerate(['active', 'passive']):
        if cond in pivot_c.columns:
            vals = pivot_c[cond].dropna()
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(i + jitter, vals, color=COLORS.get(cond, 'gray'), alpha=0.5, s=25)
            ax.bar(i, vals.mean(), yerr=vals.sem(), color=COLORS.get(cond, 'gray'),
                   alpha=0.4, capsize=5, width=0.5)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Scrolling', 'Watching'])
    ax.set_ylabel('Criterion (c)')
    ax.set_title('Response Bias')

    fig.suptitle(f'Recognition Performance (N={n_subs})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'recognition_performance.png', dpi=150)
    plt.close(fig)
    print(f'\n  Saved: recognition_performance.png')

    # Save d' table
    dprime_df.to_csv(out_dir / 'dprime_table.csv', index=False)

    return pivot_acc, comp_results


def viewing_duration_analysis(est, out_dir):
    """Analyze per-picture viewing durations."""
    print('\n' + '=' * 60)
    print('VIEWING DURATION ANALYSIS')
    print('=' * 60)

    # Parse per_picture_durations from the array column
    dur_rows = []
    for _, row in est.iterrows():
        ppd = row.get('per_picture_durations', '')
        if not isinstance(ppd, str) or not ppd.strip():
            continue
        # Parse "[1.23, 4.56, ...]"
        ppd_clean = ppd.strip().strip('"').strip('[]')
        try:
            durs = [float(x.strip()) for x in ppd_clean.split(',') if x.strip()]
        except ValueError:
            continue
        for i, d in enumerate(durs):
            dur_rows.append(dict(sub=row['sub'], session_type=row['session_type'],
                                 rep=row['rep'], n_pictures=row['n_pictures'],
                                 pic_idx=i+1, pic_duration=d))

    if not dur_rows:
        print('  No per-picture duration data parsed!')
        return

    dur_df = pd.DataFrame(dur_rows)
    dur_df['pic_duration'] = pd.to_numeric(dur_df['pic_duration'], errors='coerce')
    dur_df = dur_df.dropna(subset=['pic_duration'])

    # Normalize pic position within block (0-1)
    dur_df['n_pictures'] = pd.to_numeric(dur_df['n_pictures'], errors='coerce')
    dur_df['norm_position'] = (dur_df['pic_idx'] - 1) / (dur_df['n_pictures'] - 1)

    print(f'\n  Total viewing episodes: {len(dur_df)}')

    # Mean duration by condition
    print('\n  Mean per-picture viewing duration:')
    for cond in ['active', 'passive', 'constant']:
        cond_data = dur_df[dur_df['session_type'] == cond]['pic_duration']
        if len(cond_data) > 0:
            print(f'    {cond:>10}: M = {cond_data.mean():.2f}s (SD = {cond_data.std():.2f})')

    # ── Figure: duration across position ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By absolute position (first 18 pictures)
    ax = axes[0]
    for cond in ['active', 'passive', 'constant']:
        cond_data = dur_df[dur_df['session_type'] == cond]
        pos_means = cond_data.groupby('pic_idx')['pic_duration'].mean()
        pos_sems = cond_data.groupby('pic_idx')['pic_duration'].sem()
        pos_means = pos_means[pos_means.index <= 18]
        pos_sems = pos_sems[pos_sems.index <= 18]
        ax.errorbar(pos_means.index, pos_means.values, yerr=pos_sems.values,
                    color=COLORS.get(cond, 'gray'), marker='.', lw=1.5,
                    label=COND_LABELS.get(cond, cond.capitalize()), capsize=2, alpha=0.8)
    ax.set_xlabel('Picture Position in Block')
    ax.set_ylabel('Viewing Duration (s)')
    ax.set_title('Duration by Picture Position')
    ax.legend()

    # By normalized position (binned)
    ax = axes[1]
    dur_df['pos_bin'] = pd.cut(dur_df['norm_position'], bins=5, labels=['1st','2nd','3rd','4th','5th'])
    for cond in ['active', 'passive', 'constant']:
        cond_data = dur_df[dur_df['session_type'] == cond]
        bin_means = cond_data.groupby('pos_bin', observed=True)['pic_duration'].mean()
        bin_sems = cond_data.groupby('pos_bin', observed=True)['pic_duration'].sem()
        ax.errorbar(range(len(bin_means)), bin_means.values, yerr=bin_sems.values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond.capitalize()), capsize=3)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Early', '', 'Mid', '', 'Late'])
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Viewing Duration (s)')
    ax.set_title('Duration by Block Position (normalized)')
    ax.legend()

    fig.suptitle(f'Per-Picture Viewing Duration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'viewing_durations.png', dpi=150)
    plt.close(fig)
    print(f'  Saved: viewing_durations.png')


def save_summary(est, recog, out_dir):
    """Save summary CSV with per-subject measures."""
    rows = []
    for sub in sorted(est['sub'].unique()):
        row = {'sub': sub}
        for cond in ['active', 'passive', 'constant']:
            sub_cond = est[(est['sub'] == sub) & (est['session_type'] == cond)]
            row[f'est_ratio_{cond}'] = sub_cond['estimation_ratio'].mean() if len(sub_cond) > 0 else np.nan
            row[f'n_blocks_{cond}'] = len(sub_cond)

        # Recognition
        if not recog.empty:
            for cond in ['active', 'passive']:
                sub_cond = recog[(recog['sub'] == sub) & (recog['session_type'] == cond)]
                if len(sub_cond) > 0:
                    row[f'recog_acc_{cond}'] = sub_cond['recognition_correct'].mean()
                    correct_rt = sub_cond[sub_cond['recognition_correct'] == True]['recognition_rt']
                    row[f'recog_rt_{cond}'] = pd.to_numeric(correct_rt, errors='coerce').mean()

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'behavioral_summary.csv', index=False)
    print(f'\n  Saved: behavioral_summary.csv ({len(summary)} subjects)')
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('BEHAVIORAL ANALYSIS — ALL SUBJECTS')
    print('=' * 60)

    print('\nLoading behavioral data...')
    df = load_all_behavioral()
    print(f'\n  Total rows: {len(df)}')
    print(f'  Subjects: {sorted(df["sub"].unique())}')

    est, recog = clean_data(df)
    print(f'  Estimation rows: {len(est)}')
    print(f'  Recognition rows: {len(recog)}')

    # Run analyses
    pivot_est, est_results = estimation_analysis(est, OUT)
    pivot_acc, recog_results = recognition_analysis(recog, OUT)
    viewing_duration_analysis(est, OUT)
    summary = save_summary(est, recog, OUT)

    # Save all stats
    all_stats = []
    for r in est_results:
        all_stats.append({**r, 'measure': 'estimation_ratio'})
    for r in recog_results:
        all_stats.append(r)
    pd.DataFrame(all_stats).to_csv(OUT / 'statistical_tests.csv', index=False)

    print('\n' + '=' * 60)
    print(f'All results saved to: {OUT.relative_to(ROOT)}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
