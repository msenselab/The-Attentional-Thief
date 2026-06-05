"""
Brain-Behavior Correlation: ERP components vs behavioral outcomes
=================================================================
Computes per-subject behavioral measures (time estimation ratio,
recognition accuracy) and correlates them with ERP amplitudes.

Output: analysis/output/brain_behavior/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ROOT, DERIVATIVES_DIR, CLUSTERS, ERP_WINDOWS, COLORS,
    analysis_subjects, sub_deriv_dir, OUT_DIR,
)

mne.set_log_level('WARNING')

DATA_DIR = ROOT / 'data'
OUT = OUT_DIR / 'brain_behavior'
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ['active', 'passive']
COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching', 'constant': 'Baseline'}
POST_CHS = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']


# ── Load behavioral data ────────────────────────────────────────────────────
def load_behavioral(sub: str) -> dict | None:
    """Load behavioral CSV and extract per-condition measures.
    
    The CSVs have embedded arrays with commas (per_picture_durations, etc.)
    that break standard parsing. We extract the needed columns from the
    first 13 comma-separated fields (before the array columns) and the
    fields after 'image_files' using a bracket-aware parser.
    """
    beh_dir = DATA_DIR / f'sub-{sub}' / 'beh'
    csvs = sorted(beh_dir.glob('*.csv'))
    if not csvs:
        return None

    # Parse with bracket-aware approach
    rows = []
    with open(csvs[0]) as fh:
        lines = fh.readlines()
    
    header = lines[0].strip().split(',')
    # Columns 0-12 are safe (before array columns)
    # We need: session_type(4), estimation_ratio(11)
    # For recognition columns, parse from the end
    for line in lines[1:]:
        # Split only the first 13 fields
        parts = line.strip().split(',')
        if len(parts) < 13:
            continue
        
        session_type = parts[4].strip()
        try:
            est_ratio = float(parts[11]) if parts[11].strip() else np.nan
        except (ValueError, IndexError):
            est_ratio = np.nan
        
        # Parse recognition fields from the end of the line
        # participant is last real field, then recognition fields before it
        # Work backwards: ..., recognition_correct(19), recognition_rt(20), constant_image(21), participant(22)
        # But array columns shift positions. Use reverse approach:
        rev_parts = line.strip().rsplit(',', 6)  # last 6 fields are safe
        # rev_parts = [everything_before, recog_image_or_stuff, recog_is_old, recog_response, recog_correct, recog_rt, constant_image, participant]
        try:
            # Last field is participant (or empty)
            # Work from the end
            tail = line.strip().rsplit(',', 5)  # split into [big_chunk, last5_fields]
            tail_fields = tail[-5:]  # recognition_rt, constant_image, participant, maybe empties
            # More reliable: count from end
            all_parts_rev = line.strip().split(',')
            # participant is at -1 or -2 (might have trailing comma)
            # recognition_correct is typically 4 from end
            recog_correct = None
            for candidate in all_parts_rev[-6:-1]:
                c = candidate.strip().lower()
                if c in ('true', 'false'):
                    recog_correct = c == 'true'
                    break
        except Exception:
            recog_correct = None
        
        rows.append({
            'session_type': session_type,
            'estimation_ratio': est_ratio,
            'recognition_correct': recog_correct,
        })
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)

    result = {'sub': sub}

    # Time estimation ratio per condition
    for cond in ['active', 'passive', 'constant']:
        mask = df['session_type'] == cond
        ratios = df.loc[mask, 'estimation_ratio'].dropna()
        if len(ratios) > 0:
            result[f'est_ratio_{cond}'] = ratios.mean()
            result[f'est_ratio_{cond}_sd'] = ratios.std()
            result[f'est_n_{cond}'] = len(ratios)
        else:
            result[f'est_ratio_{cond}'] = np.nan
            result[f'est_ratio_{cond}_sd'] = np.nan
            result[f'est_n_{cond}'] = 0

    # Scrolling - Watching estimation difference
    if not np.isnan(result.get('est_ratio_active', np.nan)) and not np.isnan(result.get('est_ratio_passive', np.nan)):
        result['est_ratio_diff_ap'] = result['est_ratio_active'] - result['est_ratio_passive']
    else:
        result['est_ratio_diff_ap'] = np.nan

    # Recognition accuracy per condition
    for cond in ['active', 'passive']:
        mask = df['session_type'] == cond
        correct = df.loc[mask, 'recognition_correct'].dropna()
        if len(correct) > 0:
            result[f'recog_acc_{cond}'] = correct.mean()
            result[f'recog_n_{cond}'] = len(correct)
        else:
            result[f'recog_acc_{cond}'] = np.nan
            result[f'recog_n_{cond}'] = 0

    # Recognition accuracy difference
    if not np.isnan(result.get('recog_acc_active', np.nan)) and not np.isnan(result.get('recog_acc_passive', np.nan)):
        result['recog_diff_ap'] = result['recog_acc_active'] - result['recog_acc_passive']
    else:
        result['recog_diff_ap'] = np.nan

    return result


# ── Load ERP amplitudes ─────────────────────────────────────────────────────
def load_erp_amplitudes(sub: str) -> dict | None:
    """Extract mean ERP amplitude per condition × window from posterior channels."""
    epo_path = sub_deriv_dir(sub) / 'picture_onset-epo.fif'
    if not epo_path.exists():
        return None

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose='ERROR')
    epochs = epochs.crop(tmin=-0.2, tmax=1.0)

    result = {'sub': sub}

    for cond in CONDS:
        if cond not in epochs.event_id or len(epochs[cond]) < 10:
            continue
        evk = epochs[cond].average()
        picks = [evk.ch_names.index(ch) for ch in POST_CHS if ch in evk.ch_names]

        for win, (t0, t1) in ERP_WINDOWS.items():
            t_mask = (evk.times >= t0) & (evk.times <= t1)
            amp = evk.data[np.ix_(picks, t_mask)].mean() * 1e6
            result[f'erp_{cond}_{win}'] = amp

    # Scrolling - Watching ERP difference per window
    for win in ERP_WINDOWS:
        a_key = f'erp_active_{win}'
        p_key = f'erp_passive_{win}'
        if a_key in result and p_key in result:
            result[f'erp_diff_{win}'] = result[a_key] - result[p_key]

    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Discover subjects
    subs = [
        sub for sub in analysis_subjects()
        if (sub_deriv_dir(sub) / 'picture_onset-epo.fif').exists()
    ]

    print(f'Found {len(subs)} canonical preprocessed subjects')

    # Load data
    beh_rows = []
    erp_rows = []
    for sub in subs:
        beh = load_behavioral(sub)
        erp = load_erp_amplitudes(sub)
        if beh and erp:
            beh_rows.append(beh)
            erp_rows.append(erp)
            print(f'  sub-{sub}: ✓')
        else:
            reason = []
            if not beh: reason.append('no behavioral')
            if not erp: reason.append('no ERP')
            print(f'  sub-{sub}: skipped ({", ".join(reason)})')

    beh_df = pd.DataFrame(beh_rows)
    erp_df = pd.DataFrame(erp_rows)
    merged = beh_df.merge(erp_df, on='sub')

    print(f'\nMerged: {len(merged)} subjects')
    print()

    # ── Summary table ────────────────────────────────────────────────────────
    print('═' * 70)
    print('BEHAVIORAL SUMMARY')
    print('═' * 70)
    for cond in ['active', 'passive', 'constant']:
        col = f'est_ratio_{cond}'
        if col in merged.columns:
            vals = merged[col].dropna()
            print(f'  Time estimation ratio ({cond}): M={vals.mean():.3f}, SD={vals.std():.3f}, N={len(vals)}')
    for cond in ['active', 'passive']:
        col = f'recog_acc_{cond}'
        if col in merged.columns:
            vals = merged[col].dropna()
            print(f'  Recognition accuracy ({cond}): M={vals.mean():.3f}, SD={vals.std():.3f}, N={len(vals)}')

    # ── Scrolling vs Watching behavioral t-tests ─────────────────────────────────
    print()
    print('BEHAVIORAL COMPARISONS (paired t-test)')
    print('-' * 50)
    for measure, label in [('est_ratio', 'Time estimation ratio'),
                           ('recog_acc', 'Recognition accuracy')]:
        a_col = f'{measure}_active'
        p_col = f'{measure}_passive'
        if a_col in merged.columns and p_col in merged.columns:
            a = merged[a_col].dropna()
            p = merged[p_col].dropna()
            common = merged[[a_col, p_col]].dropna()
            if len(common) > 2:
                t, pv = stats.ttest_rel(common[a_col], common[p_col])
                d = (common[a_col] - common[p_col]).mean() / (common[a_col] - common[p_col]).std()
                print(f'  {label}: t({len(common)-1})={t:.2f}, p={pv:.4f}, d={d:.2f}')

    # ── Brain-behavior correlations ──────────────────────────────────────────
    print()
    print('═' * 70)
    print('BRAIN-BEHAVIOR CORRELATIONS')
    print('═' * 70)

    corr_results = []

    # 1. ERP amplitude (per condition) vs estimation ratio (per condition)
    print('\n1. ERP amplitude vs Time Estimation Ratio (within condition)')
    print('-' * 60)
    for cond in CONDS:
        for win in ERP_WINDOWS:
            erp_col = f'erp_{cond}_{win}'
            beh_col = f'est_ratio_{cond}'
            if erp_col in merged.columns and beh_col in merged.columns:
                valid = merged[[erp_col, beh_col]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid[erp_col], valid[beh_col])
                    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                    print(f'  {cond:>8} {win:<6}: r={r:>6.3f}, p={p:.4f} {sig}')
                    corr_results.append(dict(
                        comparison='ERP_vs_EstRatio', condition=cond,
                        window=win, r=r, p=p, n=len(valid)))

    # 2. ERP difference (Scrolling-Watching) vs estimation ratio difference
    print('\n2. ERP difference (A-P) vs Estimation Ratio difference (A-P)')
    print('-' * 60)
    for win in ERP_WINDOWS:
        erp_col = f'erp_diff_{win}'
        beh_col = 'est_ratio_diff_ap'
        if erp_col in merged.columns and beh_col in merged.columns:
            valid = merged[[erp_col, beh_col]].dropna()
            if len(valid) > 5:
                r, p = stats.pearsonr(valid[erp_col], valid[beh_col])
                sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                print(f'  {win:<6}: r={r:>6.3f}, p={p:.4f} {sig}')
                corr_results.append(dict(
                    comparison='ERPdiff_vs_EstRatioDiff', condition='A-P',
                    window=win, r=r, p=p, n=len(valid)))

    # 3. ERP amplitude vs Recognition accuracy
    print('\n3. ERP amplitude vs Recognition Accuracy (within condition)')
    print('-' * 60)
    for cond in CONDS:
        for win in ERP_WINDOWS:
            erp_col = f'erp_{cond}_{win}'
            beh_col = f'recog_acc_{cond}'
            if erp_col in merged.columns and beh_col in merged.columns:
                valid = merged[[erp_col, beh_col]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid[erp_col], valid[beh_col])
                    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                    print(f'  {cond:>8} {win:<6}: r={r:>6.3f}, p={p:.4f} {sig}')
                    corr_results.append(dict(
                        comparison='ERP_vs_RecogAcc', condition=cond,
                        window=win, r=r, p=p, n=len(valid)))

    # Save correlation results with FDR correction
    corr_df = pd.DataFrame(corr_results)
    if len(corr_df) > 1:
        reject, p_fdr, _, _ = multipletests(corr_df['p'], method='fdr_bh')
        corr_df['p_fdr'] = p_fdr
        corr_df['sig_fdr'] = reject
        print(f'\nFDR correction (Benjamini-Hochberg) applied across {len(corr_df)} tests')
        sig_count = reject.sum()
        print(f'  Significant after FDR: {sig_count}/{len(corr_df)}')
        if sig_count > 0:
            for _, row in corr_df[corr_df['sig_fdr']].iterrows():
                print(f'    {row["comparison"]} {row["condition"]} {row["window"]}: '
                      f'r={row["r"]:.3f}, p={row["p"]:.4f}, p_fdr={row["p_fdr"]:.4f}')
    corr_df.to_csv(OUT / 'brain_behavior_correlations.csv', index=False)

    # Save merged data
    merged.to_csv(OUT / 'merged_erp_behavioral.csv', index=False)

    # ── Scatter plots ────────────────────────────────────────────────────────
    print('\nGenerating scatter plots...')

    # Plot 1: ERP per window vs estimation ratio (scrolling condition)
    fig, axes = plt.subplots(1, len(ERP_WINDOWS), figsize=(4*len(ERP_WINDOWS), 4))
    fig.suptitle('Scrolling ERP vs Time Estimation Ratio (per subject)', fontsize=13)
    for ax, (win, (t0, t1)) in zip(axes, ERP_WINDOWS.items()):
        x = merged[f'erp_active_{win}'].values
        y = merged['est_ratio_active'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            ax.scatter(x[mask], y[mask], c='k', s=30, alpha=0.7)
            r, p = stats.pearsonr(x[mask], y[mask])
            # Regression line
            m, b = np.polyfit(x[mask], y[mask], 1)
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xr, m*xr + b, 'r--', alpha=0.5)
            ax.set_title(f'{win}\nr={r:.2f}, p={p:.3f}', fontsize=10)
        ax.set_xlabel('ERP amplitude (µV)')
        ax.set_ylabel('Estimation ratio') if ax == axes[0] else None
    plt.tight_layout()
    fig.savefig(OUT / 'scatter_erp_vs_estimation_active.png', dpi=150)
    plt.close(fig)

    # Plot 2: ERP difference vs estimation difference
    fig, axes = plt.subplots(1, len(ERP_WINDOWS), figsize=(4*len(ERP_WINDOWS), 4))
    fig.suptitle('ERP Difference (A−P) vs Estimation Ratio Difference (A−P)', fontsize=13)
    for ax, (win, (t0, t1)) in zip(axes, ERP_WINDOWS.items()):
        x_col = f'erp_diff_{win}'
        y_col = 'est_ratio_diff_ap'
        if x_col in merged.columns and y_col in merged.columns:
            x = merged[x_col].values
            y = merged[y_col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 3:
                ax.scatter(x[mask], y[mask], c='k', s=30, alpha=0.7)
                r, p = stats.pearsonr(x[mask], y[mask])
                m, b = np.polyfit(x[mask], y[mask], 1)
                xr = np.linspace(x[mask].min(), x[mask].max(), 50)
                ax.plot(xr, m*xr + b, 'r--', alpha=0.5)
                ax.set_title(f'{win}\nr={r:.2f}, p={p:.3f}', fontsize=10)
                ax.axhline(0, color='gray', lw=0.5)
                ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('ΔERP (µV)')
        ax.set_ylabel('ΔEstimation ratio') if ax == axes[0] else None
    plt.tight_layout()
    fig.savefig(OUT / 'scatter_erp_diff_vs_estimation_diff.png', dpi=150)
    plt.close(fig)

    # Plot 3: ERP vs recognition accuracy
    fig, axes = plt.subplots(2, len(ERP_WINDOWS), figsize=(4*len(ERP_WINDOWS), 7))
    fig.suptitle('ERP vs Recognition Accuracy (per subject)', fontsize=13)
    for i, cond in enumerate(CONDS):
        for j, (win, _) in enumerate(ERP_WINDOWS.items()):
            ax = axes[i, j]
            x_col = f'erp_{cond}_{win}'
            y_col = f'recog_acc_{cond}'
            if x_col in merged.columns and y_col in merged.columns:
                x = merged[x_col].values
                y = merged[y_col].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 3:
                    ax.scatter(x[mask], y[mask], c=COLORS[cond], s=30, alpha=0.7)
                    r, p = stats.pearsonr(x[mask], y[mask])
                    m, b = np.polyfit(x[mask], y[mask], 1)
                    xr = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(xr, m*xr + b, 'r--', alpha=0.5)
                    ax.set_title(f'{cond} {win}\nr={r:.2f}, p={p:.3f}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{COND_LABELS.get(cond, cond.capitalize())}\nRecog. accuracy")
            if i == 1:
                ax.set_xlabel('ERP amplitude (µV)')
    plt.tight_layout()
    fig.savefig(OUT / 'scatter_erp_vs_recognition.png', dpi=150)
    plt.close(fig)

    print(f'\nResults saved to: {OUT.relative_to(ROOT)}/')
    print('Done!')


if __name__ == '__main__':
    main()
