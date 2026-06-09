# The Attentional Thief

**Self-paced visual exploration compresses subjective time while enhancing visual encoding.**

This repository contains the analysis code, experiment program, and derived
(intermediate) data needed to reproduce the reported statistical results of the
study. Raw EEG and eye-tracking recordings are **not** included (size and
participant privacy); the public reproduction path starts from the derived data
provided here.

---

## Research question

The study tests whether **self-paced control over visual exploration compresses
subjective time** relative to a matched **Watching** condition and a
low-engagement **Baseline**, and whether this behavioral compression co-occurs
with changes in picture-locked ERPs and fixation-related potentials (FRPs).

### Conditions

| Condition    | What participants do                                                   | Purpose                              |
|--------------|------------------------------------------------------------------------|--------------------------------------|
| **Scrolling**| Click to advance through pictures at their own pace                     | Self-paced control / active exploration |
| **Watching** | Watch pictures auto-advance using timings yoked from a Scrolling block | Matched visual stream without control |
| **Baseline** | View one image for the matched block duration                          | Minimal visual change / low engagement |

(Earlier internal names: Scrolling = *active*, Watching = *passive*, Baseline = *constant*.)

**Analysis sample:** N = 23 for behavioral/ERP/FRP analyses (exclusions:
`sub-015`, `sub-016`, `sub-023`, `sub-027`); N = 22 for trial-to-trial
variability analyses (`sub-008` has incomplete cell coverage). The canonical
sample is defined in `code/analysis/config.py`.

---

## Repository layout

```
.
├── data/
│   └── derived/            # Intermediate data — the reproduction entry point
│       ├── behavioral/     # Cell-level subjective-time data + reference/ stats
│       ├── erp/            # Group ERP cluster amplitudes + reference/ stats
│       ├── frp/            # Group FRP cluster amplitudes & contrasts + reference/
│       ├── eyetracking/    # Fixation/saccade/pupil summaries & tests
│       ├── brain_behavior/ # Merged ERP × behavior tables and correlations
│       └── per_subject/    # Per-subject ERP/FRP/ET/TFR summary tables
├── code/
│   ├── preprocessing/      # raw → derived (requires raw data; provided for transparency)
│   ├── analysis/           # repro_*.py run from derived data; 05-09 are full pipeline
│   └── figures/            # Notes on figure-generation limits in this release
├── experiment/             # PsychoPy experiment program (stimuli images not included)
├── figures/                # Notes on manuscript figure availability
└── docs/                   # Pipeline documentation
```

---

## Reproducing the results

```bash
# 1. Install the locked environment (requires uv: https://docs.astral.sh/uv/)
uv sync

# 2. Reproduce the headline behavioral (subjective-time) results from
#    derived data alone. This recomputes the RM-ANOVA, post-hoc t-tests, and
#    descriptives and verifies them against the manuscript reference tables.
uv run python code/analysis/repro_behavioral.py
```

`repro_*.py` scripts read only `data/derived/` and self-verify: each prints a
`[PASS]/[FAIL]` line comparing the recomputed values against the manuscript
reference tables in `data/derived/<measure>/reference/`.

| Script | Reproduces | Status |
|--------|------------|--------|
| `code/analysis/repro_behavioral.py` | Subjective-time RM-ANOVA, post-hoc, descriptives | ✅ verified |
| `code/analysis/repro_erp.py` | ERP cluster-amplitude condition contrasts | ✅ verified |
| `code/analysis/repro_frp.py` | FRP cluster-amplitude condition contrasts | ✅ verified |
| `code/analysis/repro_eyetracking.py` | Eye-tracking descriptives | ✅ verified |

**Provenance / full pipeline.** The numbered scripts in `code/preprocessing/`
(`01`–`04`) and `code/analysis/` (`05`–`09`) document the complete pipeline from
raw EEG/eye-tracking to the derived data. They require the raw dataset (not
shipped) and are included for transparency, not for the from-derived
reproduction.

> **Figure note:** the derived-data reproduction scripts verify the reported
> behavioral, ERP, FRP, and eye-tracking statistics. Grand-average ERP/FRP
> waveform and topomap figure generation requires time × channel evoked data,
> which is not redistributed in this release.

The experiment program under `experiment/` requires a PsychoPy/EyeLink-capable
runtime and the stimulus image set, which are not part of the locked analysis
environment above.

---

## Data dictionary (selected)

Each measure folder ships the **input** tables consumed by the `repro_*` scripts
plus a `reference/` subfolder holding the **manuscript result tables** those
scripts must reproduce.

- `data/derived/behavioral/behavioral_cell_level.csv` — one row per
  subject × condition (`session_type`) × `n_pictures`:
  `estimated_duration_s`, `actual_duration_s`, `estimation_ratio`, `accuracy`,
  `variability_sd_s`, `n_trials`. (Recognition accuracy/criterion exist for the
  `active`/`passive` conditions only; the `constant` condition has no
  old/new recognition test.)
- `data/derived/behavioral/gap_adjusted_cell_level.csv` — cell-level data for
  the gap-adjusted time-discounting analysis.
- `data/derived/behavioral/reference/` — manuscript ANOVA, post-hoc,
  descriptives, and gap-adjusted tables (reproduction targets).
- `data/derived/erp/erp_summary_all.csv` — `sub, condition, window, cluster, amplitude_uV`
  (N = 23 analysis sample).
- `data/derived/frp/frp_summary_all.csv` — `subject, condition, cluster, window, n, mean_amp_uv`.
- `data/derived/brain_behavior/` — merged ERP × behavior tables and correlations.
- `data/derived/per_subject/sub-XXX/` — per-subject ERP/FRP/eye-tracking/TFR summaries.

---

## Citation

If you use this code or data, please cite:

> Msense Lab contributors (2026). *The Attentional Thief: analysis code and derived data*. GitHub. https://github.com/msenselab/The-Attentional-Thief

```bibtex
@misc{msenselab_attentional_thief_2026,
  title        = {The Attentional Thief: analysis code and derived data},
  author       = {{Msense Lab contributors}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/msenselab/The-Attentional-Thief}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

- **Code** (`code/`, `experiment/`): MIT — see [`LICENSE`](LICENSE).
- **Derived data** (`data/derived/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA).

Stimulus images are **not** redistributed here; see `experiment/` for their
source and licensing.
