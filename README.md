# The Attentional Thief

**Self-paced visual exploration compresses subjective time while enhancing visual encoding.**

This repository contains the analysis code, experiment program, and derived
(intermediate) data needed to reproduce the results and figures of the study.
Raw EEG and eye-tracking recordings are **not** included (size and participant
privacy); the pipeline is designed so that all reported statistics and figures
can be regenerated from the derived data provided here.

> ℹ️ **Placeholders to fill in before going public:** authors, paper title,
> journal/preprint DOI, and the BibTeX entry below. Search this file and
> `CITATION.cff` for `TODO`.

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
│   └── figures/            # derived → manuscript figures (TODO)
├── experiment/             # PsychoPy experiment program (stimuli images not included)
├── figures/                # Rendered manuscript figures (outputs of code/figures)
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
| `code/analysis/repro_erp.py` | ERP cluster-amplitude condition contrasts | ⏳ TODO |
| `code/analysis/repro_frp.py` | FRP cluster-amplitude condition contrasts | ⏳ TODO |

**Provenance / full pipeline.** The numbered scripts in `code/preprocessing/`
(`01`–`04`) and `code/analysis/` (`05`–`09`) document the complete pipeline from
raw EEG/eye-tracking to the derived data. They require the raw dataset (not
shipped) and are included for transparency, not for the from-derived
reproduction.

> ⏳ **Not yet derived-runnable:** grand-average ERP/FRP *waveform* and *topomap*
> figures need time × channel evoked data, which is not yet exported as CSV.
> To fully close the figure-reproduction gap, export per-condition group
> evokeds (e.g. as long-format CSV) from the raw pipeline and add figure
> scripts under `code/figures/`.

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

> TODO Authors (YEAR). *TODO Title*. TODO Journal/Preprint. DOI: TODO

```bibtex
@article{TODO_citekey,
  title   = {TODO Title},
  author  = {TODO Authors},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

- **Code** (`code/`, `experiment/`): MIT — see [`LICENSE`](LICENSE).
- **Derived data** (`data/derived/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA).

Stimulus images are **not** redistributed here; see `experiment/` for their
source and licensing.
