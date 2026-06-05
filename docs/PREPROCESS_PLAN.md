# EEG Preprocessing Plan: Attentional Thief Study

**Project:** attentional-thief
**Date:** 2026-03-17
**Cap:** EasyCap 64Ch actiCAP (standard_1020)

---

## 1. Hardware Setup

| Parameter | Value |
|-----------|-------|
| System | Brain Products actiCAP |
| Channels | 64 EEG |
| Sampling rate | 1000 Hz |
| Online reference | **FCz** (hard-wired, blue holder) |
| Ground | **AFz** (black holder) |
| Hardware LP filter | 250 Hz |
| Dedicated EOG | None — use scalp proxies |
| EOG (vertical) | **VEOG** (dedicated channel) |
| EOG (horizontal) | **F9, F10** (outer frontal) |

---

## 2. Pipeline Overview

```
Raw (.vhdr)
  │
  ├─ 1. Load + set channel types
  ├─ 2. Recover FCz (online reference → data channel)
  ├─ 3. Bad channel detection (amplitude-based or RANSAC)
  ├─ 4. Bandpass filter  0.1–30 Hz  +  notch 50 Hz
  ├─ 5. Average re-reference
  ├─ 6. ICA fit  (on 1 Hz HP copy)
  ├─ 7. ICA label  (VEOG dedicated + HEOG via F9/F10)
  ├─ 8. ICA apply → cleaned continuous raw
  ├─ 9. Interpolate bad channels
  ├─ 10. Epoch  (ERP / FRP / SRP)
  ├─ 11. Autoreject  (data-driven threshold + interpolation)
  └─ 12. Save + QC report
```

---

## 3. Step-by-Step

### Step 1: Load Raw Data

```python
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')
```

**Notes:**
- BrainVision annotations carry trigger codes as `'S  X'` strings
- Keep all 64 channels at load time

---

### Step 2: Recover FCz

FCz was the online reference — its signal is all zeros in the raw file.
Add it back as a flat channel, then re-reference to the average so it
picks up its true signal.

```python
raw = mne.add_reference_channels(raw, ref_channels='FCz')
```

⚠️ Must be done **before** re-referencing. FCz is an important channel
for fronto-central components (CNV, N2, P3).

---

### Step 3: Bad Channel Detection

Identify noisy/flat channels automatically, then mark for later interpolation.
Do this **before** ICA so bad channels don't waste ICA components.

```python
# Option A — amplitude-based heuristic (fast)
def find_bad_channels_simple(raw, flat_thresh=1e-7, noisy_thresh=200e-6):
    data = raw.get_data(picks='eeg')
    std = data.std(axis=1)
    flat   = [raw.ch_names[i] for i, s in enumerate(std) if s < flat_thresh]
    noisy  = [raw.ch_names[i] for i, s in enumerate(std) if s > noisy_thresh]
    return flat + noisy

# Option B — RANSAC (more robust, slower)
from autoreject import Ransac
ransac = Ransac(n_jobs=-1, random_state=42)
epochs_for_ransac = mne.make_fixed_length_epochs(raw, duration=1.0)
ransac.fit(epochs_for_ransac)
bad_channels = ransac.bad_chs_
```

**Log bad channels per subject in QC report.**

---

### Step 4: Filtering

```python
# Bandpass: 0.1 Hz preserves slow waves (CNV); 30 Hz removes line noise
raw.filter(l_freq=0.1, h_freq=30.0)

# Notch: EU power line
raw.notch_filter(freqs=50.0)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| High-pass | 0.1 Hz | Preserve P3, slow cortical potentials |
| Low-pass | 30.0 Hz | Remove muscle & line noise |
| Notch | 50 Hz | EU power line |

---

### Step 5: Mastoid Re-reference

```python
raw.set_eeg_reference(['TP9', 'TP10'])
```

Re-referenced to linked mastoids (TP9/TP10), matching the original MATLAB pipeline.
Applied after FCz recovery so FCz contributes as a data channel.
Final channel count: **65 EEG** (64 original + recovered FCz).

---

### Step 6: ICA — Fit

Fit ICA on a 1 Hz high-passed copy for numerical stability.

```python
raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None)

ica = mne.preprocessing.ICA(
    n_components=25,          # ~25 for 64-ch; adjust if rank-deficient
    method='fastica',
    random_state=42,
    max_iter='auto',
)
ica.fit(raw_ica, reject=dict(eeg=250e-6))
```

---

### Step 7: ICA — Label & Exclude

Automatically detect artifact components using EOG proxies.
Then **manually inspect** topographies + time courses to confirm.

```python
# VEOG is a dedicated channel — set its type before ICA labelling
raw_ica.set_channel_types({'VEOG': 'eog'})

exclude = set()

# Vertical EOG (blinks) — dedicated VEOG channel
idx, _ = ica.find_bads_eog(raw_ica, ch_name='VEOG', threshold=2.5)
exclude.update(idx)

# Horizontal EOG (saccades) — F9/F10 outer frontal proxies
for ch in ['F9', 'F10']:
    if ch in raw_ica.ch_names:
        idx, _ = ica.find_bads_eog(raw_ica, ch_name=ch, threshold=2.5)
        exclude.update(idx)

ica.exclude = sorted(exclude)
```

**Verified component scores (empirical, sub-001 / sub-002):**

| Component type | Detection channel | sub-001 | sub-002 |
|---------------|-------------------|---------|---------|
| Vertical EOG (blink) | VEOG | comp0 (r=0.92) | comp1 (r=0.64) |
| Horizontal EOG (saccade) | F9 / F10 | comp3 (r=0.91) | comp2 (r=0.80) |
| Cardiac (ECG) | Manual inspection | — | — |

⚠️ **Previous error:** `FT9/FT10` do not exist in this dataset; `Fp1/Fp2` are inferior
to the dedicated VEOG channel. The mistake caused only 1 component to be excluded
and ~75% epoch rejection in sub-001.

⚠️ **Always save ICA solution** for reproducibility and manual review:

```python
ica.save(out_dir / 'ica-solution.fif', overwrite=True)
```

**Flag for manual review if fewer than 2 components excluded.**

---

### Step 8: ICA — Apply

```python
raw_clean = raw.copy()
ica.apply(raw_clean)
```

Apply to the **0.1 Hz filtered** data (not the 1 Hz copy).

---

### Step 9: Interpolate Bad Channels

```python
raw_clean.info['bads'] = bad_channels   # from Step 3
raw_clean.interpolate_bads(reset_bads=True)
```

Done **after** ICA so bad channels are not modelled by ICA components.

---

### Step 10: Epoch — Picture Onset Locked

```python
PIC_ONSET_ID = {'active': 12, 'passive': 22, 'constant': 32}

events, event_id_found = mne.events_from_annotations(raw_clean)
# flexible trigger matching (BrainVision 'S 12' etc.)

epochs = mne.Epochs(
    raw_clean, events, event_id=event_id,
    tmin=-0.2, tmax=1.5,
    baseline=(-0.2, 0.0),
    preload=True,
    reject_by_annotation=True,
)
```

---

### Step 11: Autoreject — Data-driven Epoch Rejection

```python
from autoreject import AutoReject

ar = AutoReject(
    n_interpolate=[1, 2, 4],
    random_state=42,
    n_jobs=-1,
    verbose=False,
)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
```

**Rejection logging:**
- Save `reject_log` per subject
- Report % epochs rejected
- ⚠️ Flag subjects with >30% rejection for manual review

---

### Step 12: Save + QC Report

```python
# Epochs
epochs_clean.save(out_dir / 'picture_onset-epo.fif', overwrite=True)

# ICA
ica.save(out_dir / 'ica-solution.fif', overwrite=True)

# QC report
report = {
    'sub': sub,
    'recording_duration_min': raw.times[-1] / 60,
    'n_channels': len(raw.ch_names),
    'bad_channels': bad_channels,
    'ica_excluded': ica.exclude,
    'n_epochs_before_ar': len(epochs),
    'n_epochs_after_ar': len(epochs_clean),
    'pct_rejected': 100 * (1 - len(epochs_clean) / len(epochs)),
    'epochs_per_condition': {c: len(epochs_clean[c])
                             for c in PIC_ONSET_ID if c in epochs_clean.event_id},
}
```

---

## 4. QC Checkpoints

| Check | Pass Criterion | Action if Fail |
|-------|---------------|----------------|
| FCz recovered | Channel exists, std > 0 | Re-check load step |
| Bad channels | ≤ 5 channels interpolated | Review cap setup |
| ICA components excluded | ≥ 2 (VEOG + HEOG) | Manual review of topographies |
| Epoch rejection rate | < 30% | Review ICA quality |
| Epochs per condition (scrolling/watching) | ≥ 40 | Consider relaxing AR threshold |

---

## 5. Parameters Summary

```python
PREPROC = dict(
    # Filter
    l_freq              = 0.1,
    h_freq              = 30.0,
    notch_freq          = 50.0,
    # ICA
    ica_l_freq          = 1.0,
    ica_n_components    = 25,
    ica_random_state    = 42,
    ica_fit_reject      = 250e-6,
    # EOG detection
    eog_channels_v      = ['VEOG'],        # vertical — dedicated channel
    eog_channels_h      = ['F9', 'F10'],   # horizontal — outer frontal proxies
    eog_channel_type    = {'VEOG': 'eog'}, # set before ICA labelling
    eog_threshold_v     = 2.5,
    eog_threshold_h     = 2.5,
    # Bad channel detection
    flat_thresh         = 1e-7,
    noisy_thresh        = 200e-6,
    # Epoching
    tmin                = -0.2,
    tmax                = 1.5,
    baseline            = (-0.2, 0.0),
    # Autoreject
    ar_n_interpolate    = [1, 2, 4],
    ar_random_state     = 42,
)
```


## 6. FRP / SRP Epoching Parameters

In addition to picture-onset ERPs, the pipeline supports:

### Fixation-Related Potentials (FRP)

Locked to fixation onset from eye-tracking data (requires ET↔EEG synchronization).

```python
FRP_PARAMS = dict(
    tmin    = -0.1,     # 100 ms pre-fixation
    tmax    = 0.8,      # 800 ms post-fixation
    baseline = (-0.1, 0.0),
)
```

**Key components:** Lambda response (~80 ms), P1 (~100 ms), N1 (~170 ms)

### Saccade-Related Potentials (SRP)

Locked to saccade onset.

```python
SRP_PARAMS = dict(
    tmin    = -0.2,     # 200 ms pre-saccade
    tmax    = 0.6,      # 600 ms post-saccade
    baseline = (-0.2, -0.1),  # pre-saccade baseline
)
```

**Key components:** Spike potential (SP), presaccadic negativity

### Trigger Codes for Eye Events

| Event | Source | Notes |
|-------|--------|-------|
| Fixation onset | ET (via sync mapping) | Use `map_et_eeg_sync_v2.py` |
| Saccade onset | ET (via sync mapping) | Filter by amplitude if needed |

---

## 7. Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.2 | 2026-03-18 | Fixed pipeline diagram (EOG text). Removed incorrect Maxwell reference. Added Section 6: FRP/SRP epoching parameters. Added RANSAC example code. |
| 1.1 | 2026-03-17 | Fixed EOG channels: VEOG (dedicated) + F9/F10 for HEOG. Previous config used Fp1/Fp2 + FT9/FT10 (FT9/FT10 not in dataset), causing only 1 ICA component to be excluded and 75% epoch rejection in sub-001. |
| 1.0 | 2026-03-17 | Initial version |

*Version: 1.2 — 2026-03-18*
