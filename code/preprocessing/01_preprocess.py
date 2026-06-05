"""
Attentional Thief — Step 1: Preprocessing
==========================================
Cap    : EasyCap 64Ch actiCAP  (FCz = online ref, AFz = ground)
Input  : data/sub-{sub}/eeg/{sub}.vhdr
Output : derivatives/sub-{sub}/preprocessed-raw.fif    — cleaned continuous EEG (65 ch, FCz recovered)
         derivatives/sub-{sub}/picture_onset-epo.fif   — autoreject-cleaned epochs
         derivatives/sub-{sub}/ica-solution-ica.fif    — ICA for manual review
         derivatives/sub-{sub}/preprocess_report.txt   — QC summary

Pipeline:
  1. Load + montage
  2. Recover FCz (online reference)
  3. Bad channel detection (protect TP9/TP10/VEOG)
  4. Bandpass 0.1–30 Hz + notch 50 Hz
  5. Mastoid re-reference (TP9 + TP10)
  6. ICA fit  (1 Hz HP copy)
  7. ICA label  (ICLabel: Eye ≥ 0.8; Muscle/Heart/LineNoise/ChanNoise ≥ 0.9)
  8. ICA apply
  9. Interpolate bad channels
  10. Epoch (picture onset, −200–1000 ms)
  11. Autoreject
  12. Save + QC report

Usage:
    python analysis/01_preprocess.py --all                # process all subjects
    python analysis/01_preprocess.py --sub 001            # single subject
    python analysis/01_preprocess.py --sub 001 002 005    # multiple subjects
    python analysis/01_preprocess.py --all --skip-existing # skip already processed
    python analysis/01_preprocess.py --sub 002 --no-autoreject
    python analysis/01_preprocess.py --all --dry-run      # show what would run
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PREPROC, EPOCH, PIC_ONSET_ID, ROOT, DERIVATIVES_DIR,
    eeg_vhdr, sub_deriv_dir,
)
from mne_icalabel import label_components

mne.set_log_level('WARNING')


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging: INFO to console, DEBUG to file."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'preprocess_{timestamp}.log'

    logger = logging.getLogger('attthief.preprocess')
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        logger.handlers.clear()

    # Console: INFO
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    # File: DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(funcName)s | %(message)s'))
    logger.addHandler(fh)

    logger.info(f'Log file: {log_file}')
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Preprocess EEG data for Attentional Thief study')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--sub', nargs='+',
                   help='Subject ID(s), e.g. 001 002 005')
    g.add_argument('--all', action='store_true',
                   help='Process all subjects found in data/')
    p.add_argument('--no-autoreject', action='store_true',
                   help='Skip autoreject, use fixed 150 µV threshold')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip subjects with existing preprocessed-raw.fif')
    p.add_argument('--dry-run', action='store_true',
                   help='Show subjects to process without running')
    p.add_argument('--noisy-thresh-uv', type=float, default=None,
                   help='Override PREPROC["noisy_thresh"] (µV). Use a higher '
                        'value (e.g. 1000) for noisy raw recordings where the '
                        'default 200 µV flags too many channels.')
    return p.parse_args()


def discover_subjects() -> list[str]:
    """Find all sub-XXX directories that contain a .vhdr file."""
    data_dir = ROOT / 'data'
    subs = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.startswith('sub-'):
            sub_id = d.name.replace('sub-', '')
            if eeg_vhdr(sub_id).exists():
                subs.append(sub_id)
    return subs


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS (unchanged logic, now with logger)
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(vhdr: Path, log: logging.Logger) -> mne.io.Raw:
    """Step 1 & 2: Load BrainVision + recover FCz."""
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose='ERROR')

    ref_ch = PREPROC['online_ref']
    if ref_ch not in raw.ch_names:
        raw = mne.add_reference_channels(raw, ref_channels=ref_ch)
        log.info(f'  Recovered online reference: {ref_ch}')
    else:
        log.debug(f'  {ref_ch} already present')

    montage = mne.channels.make_standard_montage(PREPROC['montage'])
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


def find_bad_channels(raw: mne.io.Raw, log: logging.Logger) -> list[str]:
    """Step 3: Detect flat / noisy channels on pre-filter data."""
    eeg_idx = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_idx)
    std = data.std(axis=1)

    flat_thresh = PREPROC['flat_thresh']
    noisy_thresh = PREPROC['noisy_thresh']
    skip = set(PREPROC['protect_channels'] + [PREPROC['online_ref']])

    bads = []
    for i, s in enumerate(std):
        ch = raw.ch_names[eeg_idx[i]]
        if ch in skip:
            continue
        if s < flat_thresh:
            bads.append(ch)
            log.info(f'  [bad-flat]  {ch}  std={s*1e6:.2f} µV')
        elif s > noisy_thresh:
            bads.append(ch)
            log.info(f'  [bad-noisy] {ch}  std={s*1e6:.1f} µV')

    if not bads:
        log.info('  No bad channels detected')
    elif len(bads) > PREPROC['max_bad_channels']:
        log.warning(f'  {len(bads)} bad channels — check cap setup!')

    return bads


def run_ica(raw_filtered: mne.io.Raw, log: logging.Logger) -> mne.preprocessing.ICA:
    """Step 6–7: Fit ICA on 1 Hz HP copy; label with ICLabel."""
    raw_ica = raw_filtered.copy().filter(
        PREPROC['ica_l_freq'], None, verbose='ERROR')

    if 'VEOG' in raw_ica.ch_names:
        raw_ica = raw_ica.copy().drop_channels(['VEOG'])
        log.debug('  Dropped VEOG for ICA fitting')

    ica = mne.preprocessing.ICA(
        n_components=PREPROC['ica_n_components'],
        method='fastica',
        random_state=PREPROC['ica_random_state'],
        max_iter='auto',
    )
    ica.fit(raw_ica, reject=dict(eeg=PREPROC['ica_fit_reject']), verbose='ERROR')

    ic_labels = label_components(raw_ica, ica, method='iclabel')
    probs = ic_labels['y_pred_proba']
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    eye_thresh = PREPROC['iclabel_eye_thresh']
    artifact_thresh = PREPROC['iclabel_artifact_thresh']

    exclude = []
    for i in range(len(probs)):
        if probs[i, 2] >= eye_thresh:
            exclude.append(i)
        elif (probs[i, 1] >= artifact_thresh or probs[i, 3] >= artifact_thresh
              or probs[i, 4] >= artifact_thresh or probs[i, 5] >= artifact_thresh):
            exclude.append(i)

    ica.exclude = exclude
    log.info(f'  ICLabel excluded {len(exclude)} components: {exclude}')

    if len(ica.exclude) < PREPROC['min_ica_excluded']:
        log.warning(f'  Only {len(ica.exclude)} ICA components excluded '
                    f'(expected ≥ {PREPROC["min_ica_excluded"]}) — consider manual review')

    return ica


def make_epochs(raw_clean: mne.io.Raw, log: logging.Logger) -> tuple[mne.Epochs, dict]:
    """Step 10: Epoch around picture onsets."""
    events, event_id_found = mne.events_from_annotations(raw_clean, verbose='ERROR')

    event_id = {}
    for label, code in PIC_ONSET_ID.items():
        for k, v in event_id_found.items():
            if v == code or k.strip().endswith(str(code)):
                event_id[label] = v
                break

    missing = [c for c in PIC_ONSET_ID if c not in event_id]
    if missing:
        log.warning(f'  Triggers not found: {missing}')

    epochs = mne.Epochs(
        raw_clean, events, event_id=event_id,
        tmin=EPOCH['tmin'], tmax=EPOCH['tmax'],
        baseline=EPOCH['baseline'],
        preload=True,
        reject_by_annotation=True,
        verbose='ERROR',
    )
    return epochs, event_id


def apply_autoreject(epochs: mne.Epochs, log: logging.Logger) -> tuple[mne.Epochs, object]:
    """Step 11: Autoreject-based epoch cleaning."""
    try:
        from autoreject import AutoReject
    except ImportError:
        log.warning('  autoreject not installed — falling back to 150 µV threshold')
        epochs.drop_bad(reject=dict(eeg=150e-6), verbose='ERROR')
        return epochs, None

    picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
    ch_pos = epochs.info['chs']
    valid_picks = [p for p in picks
                   if np.all(np.isfinite(ch_pos[p]['loc'][:3]))
                   and not np.allclose(ch_pos[p]['loc'][:3], 0.0)]

    ar = AutoReject(
        picks=valid_picks,
        n_interpolate=[1, 2, 4],
        random_state=PREPROC['ica_random_state'],
        n_jobs=-1,
        verbose=False,
    )
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, reject_log


def apply_fixed_reject(epochs: mne.Epochs, thresh: float = 150e-6) -> mne.Epochs:
    epochs.drop_bad(reject=dict(eeg=thresh), verbose='ERROR')
    return epochs


# ═══════════════════════════════════════════════════════════════════════════════
# QC REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(path: Path, sub: str, raw: mne.io.Raw,
                 bad_channels: list, ica: mne.preprocessing.ICA,
                 epochs_before: int, epochs_clean: mne.Epochs,
                 use_autoreject: bool) -> None:
    pct_rejected = 100 * (1 - len(epochs_clean) / epochs_before) if epochs_before else 0

    lines = [
        f'Preprocessing report — sub-{sub}\n',
        f'{"=" * 40}\n',
        f'Recording duration : {raw.times[-1] / 60:.1f} min\n',
        f'Channels (total)   : {len(raw.ch_names)}  (incl. recovered FCz)\n',
        f'Sampling rate      : {raw.info["sfreq"]:.0f} Hz\n',
        f'\nFilter\n',
        f'  bandpass : {PREPROC["l_freq"]} – {PREPROC["h_freq"]} Hz\n',
        f'  notch    : {PREPROC["notch_freq"]} Hz\n',
        f'  reference: {PREPROC["reference"]}\n',
        f'\nBad channels ({len(bad_channels)}) — interpolated\n',
        f'  {bad_channels if bad_channels else "none"}\n',
        f'\nICA\n',
        f'  n_components : {ica.n_components_}\n',
        f'  excluded     : {ica.exclude}\n',
        f'  method       : ICLabel (eye≥{PREPROC["iclabel_eye_thresh"]}, artifact≥{PREPROC["iclabel_artifact_thresh"]})\n',
        f'\nEpoching\n',
        f'  window      : {EPOCH["tmin"]} to {EPOCH["tmax"]} s\n',
        f'  baseline    : {EPOCH["baseline"]} s\n',
        f'  method      : {"autoreject" if use_autoreject else "fixed 150 µV"}\n',
        f'  before AR   : {epochs_before}\n',
        f'  after AR    : {len(epochs_clean)}\n',
        f'  rejected    : {pct_rejected:.1f}%',
    ]
    if pct_rejected > PREPROC['max_rejection_pct']:
        lines.append(f'  ⚠ > {PREPROC["max_rejection_pct"]:.0f}% — review ICA quality')
    lines.append('\n\nTrial counts\n')

    for cond in PIC_ONSET_ID:
        if cond in epochs_clean.event_id:
            n = len(epochs_clean[cond])
            lines.append(f'  {cond:<10}: {n}\n')
        else:
            lines.append(f'  {cond:<10}: not found\n')

    path.write_text(''.join(lines))

    qc = dict(
        sub=sub,
        recording_min=round(float(raw.times[-1] / 60), 1),
        n_channels=len(raw.ch_names),
        bad_channels=bad_channels,
        ica_excluded=[int(x) for x in ica.exclude],
        n_ica_excluded=len(ica.exclude),
        epochs_before_ar=int(epochs_before),
        epochs_after_ar=int(len(epochs_clean)),
        pct_rejected=round(float(pct_rejected), 1),
        use_autoreject=use_autoreject,
        epochs_per_condition={
            c: int(len(epochs_clean[c]))
            for c in PIC_ONSET_ID if c in epochs_clean.event_id
        },
        flags={
            'few_ica_components': len(ica.exclude) < PREPROC['min_ica_excluded'],
            'many_bad_channels':  len(bad_channels) > PREPROC['max_bad_channels'],
            'high_rejection':     pct_rejected > PREPROC['max_rejection_pct'],
        },
    )
    (path.parent / 'preprocess_qc.json').write_text(json.dumps(qc, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-SUBJECT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_subject(sub: str, use_ar: bool, log: logging.Logger) -> dict:
    """
    Run full preprocessing for one subject.
    Returns a summary dict for the cumulative processing log.
    """
    vhdr = eeg_vhdr(sub)
    out = sub_deriv_dir(sub)
    summary = dict(sub=sub, status='success', errors=[])

    if not vhdr.exists():
        log.error(f'EEG file not found: {vhdr}')
        summary['status'] = 'error'
        summary['errors'].append(f'File not found: {vhdr}')
        return summary

    log.info(f'{"=" * 50}')
    log.info(f'sub-{sub}')
    log.info(f'{"=" * 50}')

    try:
        # 1+2. Load + recover FCz
        log.info('Loading raw data...')
        raw = load_raw(vhdr, log)
        log.info(f'  {len(raw.ch_names)} ch, {raw.times[-1]/60:.1f} min, '
                 f'{raw.info["sfreq"]:.0f} Hz')
        summary['recording_min'] = round(float(raw.times[-1] / 60), 1)

        # 3. Bad channel detection
        log.info('Detecting bad channels (pre-filter)...')
        bad_channels = find_bad_channels(raw, log)
        raw.info['bads'] = bad_channels
        summary['bad_channels'] = bad_channels

        # 4. Filter
        log.info(f'Filtering {PREPROC["l_freq"]}–{PREPROC["h_freq"]} Hz + '
                 f'{PREPROC["notch_freq"]} Hz notch...')
        raw.filter(PREPROC['l_freq'], PREPROC['h_freq'], verbose='ERROR')
        raw.notch_filter(PREPROC['notch_freq'], verbose='ERROR')

        # 5. Mastoid re-reference
        log.info(f'Re-referencing (mastoid: {PREPROC["reference"]})...')
        raw.set_eeg_reference(PREPROC['reference'], verbose='ERROR')

        # 6+7. ICA
        log.info('Fitting ICA...')
        ica = run_ica(raw, log)

        ica_path = out / 'ica-solution-ica.fif'
        ica.save(str(ica_path), overwrite=True)
        log.debug(f'  Saved ICA: {ica_path.name}')
        summary['ica_excluded'] = [int(x) for x in ica.exclude]

        # 8. Apply ICA
        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose='ERROR')

        # 9. Interpolate bad channels
        if bad_channels:
            raw_clean.info['bads'] = bad_channels
            interp, drop = [], []
            for ch in bad_channels:
                idx = raw_clean.ch_names.index(ch)
                loc = raw_clean.info['chs'][idx]['loc'][:3]
                if np.all(np.isfinite(loc)) and not np.allclose(loc, 0.0):
                    interp.append(ch)
                else:
                    drop.append(ch)
            if interp:
                raw_clean.info['bads'] = interp
                good_picks = mne.pick_types(raw_clean.info, eeg=True, exclude=interp)
                no_pos = [raw_clean.ch_names[p] for p in good_picks
                          if not (np.all(np.isfinite(raw_clean.info['chs'][p]['loc'][:3]))
                                  and not np.allclose(raw_clean.info['chs'][p]['loc'][:3], 0.0))]
                raw_clean.interpolate_bads(reset_bads=True, exclude=no_pos, verbose='ERROR')
                log.info(f'  Interpolated bad channels: {interp}')
            if drop:
                raw_clean.drop_channels(drop)
                log.info(f'  Dropped (no montage pos): {drop}')

        # Save cleaned continuous raw
        raw_path = out / 'preprocessed-raw.fif'
        raw_clean.save(str(raw_path), overwrite=True, verbose='ERROR')
        log.info(f'  Saved: {raw_path.relative_to(ROOT)}')

        # 10. Epoch
        log.info('Epoching...')
        epochs, event_id = make_epochs(raw_clean, log)
        n_before = len(epochs)
        log.info(f'  {n_before} epochs before rejection')

        # 11. Autoreject
        if use_ar:
            log.info('Running autoreject...')
            epochs_clean, reject_log = apply_autoreject(epochs, log)
            if reject_log is not None:
                np.save(str(out / 'autoreject_log.npy'), reject_log.labels)
        else:
            log.info('Applying fixed 150 µV threshold...')
            epochs_clean = apply_fixed_reject(epochs)

        for cond in PIC_ONSET_ID:
            if cond in epochs_clean.event_id:
                log.info(f'  {cond:<10}: {len(epochs_clean[cond])} epochs')

        pct = 100 * (1 - len(epochs_clean) / n_before) if n_before else 0
        log.info(f'  Rejection rate: {pct:.1f}%')
        if pct > PREPROC['max_rejection_pct']:
            log.warning(f'  > {PREPROC["max_rejection_pct"]:.0f}% rejected — review ICA!')

        summary['epochs_before'] = n_before
        summary['epochs_after'] = len(epochs_clean)
        summary['pct_rejected'] = round(pct, 1)

        # 12. Save epochs + report
        epo_path = out / 'picture_onset-epo.fif'
        epochs_clean.save(str(epo_path), overwrite=True, verbose='ERROR')
        log.info(f'  Saved: {epo_path.relative_to(ROOT)}')

        report_path = out / 'preprocess_report.txt'
        write_report(report_path, sub, raw_clean, bad_channels, ica,
                     n_before, epochs_clean, use_ar)
        log.info(f'  Saved: {report_path.relative_to(ROOT)}')
        log.info(f'  sub-{sub} ✓')

    except Exception as e:
        log.error(f'FAILED sub-{sub}: {e}')
        summary['status'] = 'error'
        summary['errors'].append(str(e))
        import traceback
        log.debug(traceback.format_exc())

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CUMULATIVE PROCESSING LOG
# ═══════════════════════════════════════════════════════════════════════════════

def update_processing_log(summaries: list[dict]) -> None:
    """Append results to derivatives/processing_log.json."""
    log_file = DERIVATIVES_DIR / 'processing_log.json'

    if log_file.exists():
        with open(log_file) as f:
            log_data = json.load(f)
    else:
        log_data = {
            'pipeline': 'attentional-thief-preprocess',
            'pipeline_version': '2.0',
            'parameters': {k: str(v) if not isinstance(v, (int, float, bool, str, list, type(None)))
                           else v for k, v in PREPROC.items()},
            'subjects': {},
        }

    log_data['last_run'] = datetime.now().isoformat()

    for s in summaries:
        log_data['subjects'][f'sub-{s["sub"]}'] = {
            'status': s['status'],
            'processed': datetime.now().isoformat(),
            'ica_excluded': s.get('ica_excluded', []),
            'bad_channels': s.get('bad_channels', []),
            'epochs_before': s.get('epochs_before'),
            'epochs_after': s.get('epochs_after'),
            'pct_rejected': s.get('pct_rejected'),
            'errors': s.get('errors', []),
        }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    use_ar = not args.no_autoreject

    # Set up logging
    log_dir = DERIVATIVES_DIR / 'logs'
    log = setup_logging(log_dir)

    if args.noisy_thresh_uv is not None:
        prev = PREPROC['noisy_thresh']
        PREPROC['noisy_thresh'] = args.noisy_thresh_uv * 1e-6
        log.info(f'Override noisy_thresh: {prev*1e6:.0f} → '
                 f'{args.noisy_thresh_uv:.0f} µV')

    # Determine subjects
    if args.all:
        subjects = discover_subjects()
        log.info(f'Discovered {len(subjects)} subjects with EEG data')
    else:
        subjects = [s.zfill(3) for s in args.sub]

    # Optionally skip existing
    if args.skip_existing:
        before = len(subjects)
        subjects = [s for s in subjects
                    if not (sub_deriv_dir(s) / 'preprocessed-raw.fif').exists()]
        skipped = before - len(subjects)
        if skipped:
            log.info(f'Skipping {skipped} already-processed subjects')

    log.info(f'Subjects to process: {subjects}')
    log.info(f'Autoreject: {"ON" if use_ar else "OFF (fixed 150 µV)"}')

    if args.dry_run:
        log.info('DRY RUN — exiting without processing')
        return

    if not subjects:
        log.info('Nothing to do.')
        return

    # Process
    summaries = []
    for i, sub in enumerate(subjects, 1):
        log.info(f'\n[{i}/{len(subjects)}] Processing sub-{sub}...')
        summary = preprocess_subject(sub, use_ar, log)
        summaries.append(summary)

    # Update cumulative log
    update_processing_log(summaries)
    log.info(f'Updated: {DERIVATIVES_DIR / "processing_log.json"}')

    # Final summary
    n_ok = sum(1 for s in summaries if s['status'] == 'success')
    n_err = sum(1 for s in summaries if s['status'] == 'error')

    log.info('')
    log.info('═' * 50)
    log.info('SUMMARY')
    log.info('═' * 50)
    log.info(f'Total: {len(summaries)}  |  Success: {n_ok}  |  Errors: {n_err}')

    if n_err:
        log.warning('Failed subjects:')
        for s in summaries:
            if s['status'] == 'error':
                log.warning(f'  sub-{s["sub"]}: {s["errors"]}')

    # Print table for quick overview
    for s in summaries:
        if s['status'] == 'success':
            log.info(
                f'  sub-{s["sub"]}  |  '
                f'ICA excl: {s.get("ica_excluded", [])}  |  '
                f'epochs: {s.get("epochs_after", "?")}/{s.get("epochs_before", "?")}  |  '
                f'rejected: {s.get("pct_rejected", "?")}%')

    log.info('Pipeline complete.')


if __name__ == '__main__':
    main()
