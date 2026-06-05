# Attentional-Thief Analysis Pipeline

**Project:** attentional-thief  
**Last updated:** 2026-04-03  
**N:** 18 subjects (sub-001 to sub-018)

---

## 1. Behavioral Data

### 1.1 Data Source
- **Files:** `/data/sub-XXX/beh/*.csv`
- **Format:** PsychoPy CSV output with nested lists

### 1.2 Trial Structure
| Phase | Conditions | Trials per condition |
|-------|------------|---------------------|
| Time estimation | Scrolling, Watching, Baseline | 4 reps × 3 conditions = 12 blocks |
| Recognition | Scrolling, Watching, Baseline | 12 images per condition |

### 1.3 Exclusion Criteria

#### Practice trials
- **Criterion:** `rep = 0`
- **Status:** ✅ Automatically excluded (not in CSV, only rep=1,2,3,4)

#### Time estimation outliers
- **Criterion:** `estimation_ratio < 0.1` or `estimation_ratio > 3.0`
- **Rationale:** Extreme values likely indicate errors or misunderstanding

#### Recognition trials
- **Criterion:** None excluded (all valid responses)

### 1.4 Derived Measures

| Measure | Formula | Description |
|---------|---------|-------------|
| `estimation_ratio` | estimated_duration / actual_duration | <1 = underestimation |
| `estimation_error` | estimated_duration - actual_duration | Absolute error (seconds) |
| `recognition_accuracy` | correct / total × 100 | Overall accuracy (%) |
| `hit_rate` | hits / old_total × 100 | OLD items correctly identified (%) |
| `correct_rejection` | CR / new_total × 100 | NEW items correctly rejected (%) |

### 1.5 CSV Parsing Notes
⚠️ **Important:** The CSV contains nested lists (e.g., `per_picture_durations`, `image_files`) that break standard CSV parsing. Use regex-based extraction:

```python
# Time estimation
match = re.search(r'(active|passive|constant).*?,(\d+),.*?,(\d+\.\d+),(\d+\.\d+),([+-]?\d+\.\d+),(\d+\.\d+)', line)

# Recognition
match = re.search(r'(\w+\.jpg),(True|False),(old|new),(True|False),', line)
```

---

## 2. Eye Movement Data

### 2.1 Data Source
- **Files:** `/data/sub-XXX/et/subXXX.EDF`
- **Format:** EyeLink EDF (parsed with `eyelinkio`)
- **Sampling rate:** 1000 Hz

### 2.2 Event Markers in EDF

| Marker | Description |
|--------|-------------|
| `BLOCK_START ACTIVE rep=X` | Scrolling block start (rep=0 is practice) |
| `BLOCK_START PASSIVE rep=X` | Watching block start |
| `BLOCK_START CONSTANT rep=X` | Baseline block start |
| `BLOCK_END` | Block end |
| `IMAGE_ON <filename>` | Image onset |
| `IMAGE_OFF <filename> duration=X` | Image offset |

### 2.3 Exclusion Criteria

#### Practice trials
- **Criterion:** `rep = 0` in `BLOCK_START` message
- **Implementation:** `int(rep) > 0` filter
- **Status:** ✅ Excluded

#### Invalid saccades
- **Criterion:** Amplitude > 1000 px
- **Rationale:** Likely blinks or tracking artifacts
- **Implementation:** `sac_amp < 1000`

#### Invalid pupil samples
- **Criterion:** Pupil size < 100 or > 10000
- **Rationale:** Blinks produce zero/extreme values

### 2.4 Event Assignment
Fixations and saccades are assigned to images based on timing:

```python
fix_mask = (fix_times >= img_start) & (fix_times <= img_end)
sac_mask = (sac_times >= img_start) & (sac_times <= img_end)
```

### 2.5 Derived Measures (per image)

| Measure | Unit | Description |
|---------|------|-------------|
| `fix_n` | count | Number of fixations |
| `fix_dur` | ms | Mean fixation duration |
| `sac_n` | count | Number of saccades |
| `sac_amp` | px | Mean saccade amplitude |
| `pupil_mean` | AU | Mean pupil size |

### 2.6 Aggregation
1. **Per-image metrics** → computed for each IMAGE_ON/OFF period
2. **Per-subject means** → averaged across images within condition
3. **Group statistics** → paired t-tests on per-subject means

---

## 3. EEG Data

### 3.1 Data Source
- **Files:** `/data/sub-XXX/eeg/XXX.vhdr` (BrainVision format)
- **System:** Brain Products actiCAP 64-channel
- **Sampling rate:** 1000 Hz (downsampled to 500 Hz)

### 3.2 Trigger Codes

| Code | Event |
|------|-------|
| S12 | Scrolling picture onset |
| S22 | Watching picture onset |
| S32 | Baseline picture onset |
| S40 | Time estimation start |
| S50 | Recognition trial |

### 3.3 Preprocessing Pipeline

```
Raw (.vhdr)
  │
  ├─ 1. Load + set montage (standard_1020)
  ├─ 2. Recover FCz (online reference)
  ├─ 3. Set channel types (VEOG as EOG)
  ├─ 4. Bad channel detection (amplitude-based)
  │      - Flat: std < 1 µV
  │      - Noisy: std > 100 µV
  │      - Max bad channels: 6
  ├─ 5. Re-reference to mastoids (TP9/TP10)
  ├─ 6. Bandpass filter: 0.1–30 Hz
  ├─ 7. Notch filter: 50 Hz
  ├─ 8. ICA fit (on 1 Hz HP copy)
  │      - Method: Extended Infomax
  │      - Components: 0.999 variance
  ├─ 9. ICLabel classification
  │      - Eye threshold: 0.8
  │      - Artifact threshold: 0.9
  ├─ 10. ICA apply → cleaned continuous
  ├─ 11. Interpolate bad channels (P1, P2 if needed)
  ├─ 12. Epoch: -200 to 800 ms (picture onset)
  ├─ 13. Baseline: -200 to 0 ms
  ├─ 14. Artifact rejection: ±300 µV (relaxed for Sub-003)
  └─ 15. Save preprocessed epochs
```

### 3.4 Exclusion Criteria

#### Bad subjects
- **Criterion:** <50% epochs retained after artifact rejection
- **Current status:** Sub-003 borderline (59%), kept with relaxed threshold

#### Bad channels (per subject)
- **Criterion:** >6 bad channels → flag for review
- **Interpolation:** Applied after ICA

#### Bad epochs
- **Criterion:** Peak-to-peak > 300 µV (standard) or 500 µV (relaxed)

### 3.5 ERP Analysis

#### Time windows

| Component | Window | Electrodes | Expected effect |
|-----------|--------|------------|-----------------|
| P1 | 80–120 ms | O1, O2, Oz | Early visual attention |
| N1 | 140–180 ms | PO7, PO8 | Selective attention |
| P2 | 180–250 ms | Pz, P3, P4 | Feature processing |
| P300 | 300–500 ms | Pz, CPz | Decision/evaluation |

#### Contrasts
1. Scrolling vs Watching (main comparison)
2. Scrolling vs Baseline
3. Watching vs Baseline

### 3.6 Preprocessing Status

| Subject | Preprocessed | Epochs retained | Notes |
|---------|--------------|-----------------|-------|
| 001 | ❌ | — | EEG file linking needed |
| 002 | ✅ | >90% | Good quality |
| 003 | ❌ | — | Pending |
| 004-009 | ❌ | — | Pending |
| 010 | ✅ | >90% | Good quality |
| 011 | ✅ | >90% | Good quality |
| 012 | ✅ | >90% | Good quality |
| 013 | ✅ | >90% | Good quality |
| 014-018 | ❌ | — | New data, pending |

---

## 4. Quality Control Checklist

### 4.1 Behavioral QC
- [ ] All 12 time estimation blocks present
- [ ] All 36 recognition trials present
- [ ] No impossible estimation ratios (<0.1 or >3.0)
- [ ] Recognition accuracy above chance (>50%)

### 4.2 Eye-tracking QC
- [ ] Data loss < 20%
- [ ] Blink rate < 40/min
- [ ] >80% fixations assigned to images
- [ ] No extreme pupil values

### 4.3 EEG QC
- [ ] <6 bad channels
- [ ] >50% epochs retained
- [ ] <3 ICA components excluded (eye/muscle)
- [ ] No drift or discontinuities

---

## 5. Statistical Analysis

### 5.1 Time Estimation
- **Test:** Repeated-measures ANOVA (3 conditions)
- **Post-hoc:** Paired t-tests with Bonferroni correction (α = .017)
- **Effect size:** Cohen's d

### 5.2 Eye Movements
- **Test:** Paired t-tests (Scrolling vs Watching, Scrolling vs Baseline)
- **Effect size:** Cohen's d
- **Correlation:** Pearson r with time estimation

### 5.3 Recognition Memory
- **Test:** Paired t-tests (Scrolling vs Watching)
- **Measure:** Hit rate (sensitivity to OLD items)
- **Effect size:** Cohen's d

### 5.4 ERP
- **Test:** Paired t-tests per component
- **Correction:** Cluster-based permutation (if needed)
- **Brain-behavior:** Correlation with time estimation

---

## 6. File Locations

| Output | Path |
|--------|------|
| Preprocessed behavior | `analysis/output/preprocessed_behavioral.csv` |
| Preprocessed fixations | `analysis/output/preprocessed_fixations.csv` |
| EEG epochs | `analysis/output/sub-XXX/preprocessed-raw.fif` |
| ERP figures | `analysis/output/group_erp.png` |
| Summary stats | `analysis/output/all_subjects_summary.csv` |

---

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `01_preprocess.py` | EEG preprocessing pipeline |
| `02_erp.py` | ERP extraction and analysis |
| `03_frp.py` | Fixation-related potentials |
| `04_timefreq.py` | Time-frequency analysis |
| `05_group_erp_frp_srp.py` | Group-level ERP/FRP/SRP |
| `config.py` | Shared parameters |

---

## 8. Version History

| Date | Change |
|------|--------|
| 2026-03-17 | Initial EEG preprocessing pipeline |
| 2026-03-31 | Updated for N=13 |
| 2026-04-03 | Updated for N=18, added behavioral/ET documentation |
