"""
Attentional Thief — shared analysis configuration.
All pipeline scripts import from here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root is two levels up from code/analysis/config.py
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'                    # raw subject data (not shipped publicly)
DERIVED_DIR = ROOT / 'data' / 'derived'     # shipped intermediate data — reproduction entry point
OUT_DIR = ROOT / 'outputs'                  # re-run outputs (git-ignored)
DERIVATIVES_DIR = ROOT / 'derivatives'      # preprocessed EEG (not shipped publicly)

# ---------------------------------------------------------------------------
# Canonical analysis sample
# ---------------------------------------------------------------------------
# Current valid multimodal sample for behavioral, eye-tracking, ERP, and FRP
# analyses. Keep this as the single source of truth so rerunning modality-specific
# scripts does not silently change N when new raw subject folders are added.
ANALYSIS_EXCLUDED_SUBJECTS = ('015', '016', '023', '027')
ANALYSIS_SUBJECTS = tuple(
    f'{i:03d}' for i in range(1, 28)
    if f'{i:03d}' not in ANALYSIS_EXCLUDED_SUBJECTS
)
NEW_ANALYSIS_SUBJECTS = ('022', '024', '025', '026')


def normalize_subject_id(sub: str | int) -> str:
    """Return a zero-padded three-digit subject ID."""
    return str(sub).replace('sub-', '').zfill(3)


def analysis_subjects(exclude: list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
    """Canonical valid subject list, with optional additional exclusions."""
    exclude_set = {normalize_subject_id(s) for s in (exclude or [])}
    return [sub for sub in ANALYSIS_SUBJECTS if sub not in exclude_set]


def sub_eeg_dir(sub: str) -> Path:
    """EEG directory — handles nested 'eeg/eeg' layout for newer subjects."""
    standard = DATA_DIR / f'sub-{sub}' / 'eeg'
    nested = standard / 'eeg'
    return nested if nested.is_dir() else standard


def sub_et_dir(sub: str) -> Path:
    """ET directory — handles nested 'eeg/et' layout for newer subjects."""
    standard = DATA_DIR / f'sub-{sub}' / 'et'
    nested = DATA_DIR / f'sub-{sub}' / 'eeg' / 'et'
    return nested if nested.is_dir() else standard


def sub_out_dir(sub: str) -> Path:
    d = OUT_DIR / f'sub-{sub}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def sub_deriv_dir(sub: str) -> Path:
    """Preprocessed EEG output directory (BIDS-like derivatives)."""
    d = DERIVATIVES_DIR / f'sub-{sub}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def eeg_vhdr(sub: str) -> Path:
    """BrainVision header file. Searches eeg dir for matching .vhdr."""
    eeg_dir = sub_eeg_dir(sub)
    # Try standard naming first
    for pattern in [f'{sub}.vhdr', f'0{sub}.vhdr', f'sub-{sub}.vhdr']:
        p = eeg_dir / pattern
        if p.exists():
            return p
    # Fallback: find any .vhdr that isn't eo/ec (resting state)
    vhdrs = [f for f in eeg_dir.glob('*.vhdr')
             if 'eo' not in f.stem.lower() and 'ec' not in f.stem.lower()]
    if vhdrs:
        return vhdrs[0]
    return eeg_dir / f'{sub}.vhdr'  # default (will fail with clear error)


def et_edf(sub: str) -> Path:
    """EyeLink EDF file. Searches et dir for matching .EDF."""
    et_dir = sub_et_dir(sub)
    for pattern in [f'sub{sub}.EDF', f'sub-{sub}.EDF', f'{sub}.EDF']:
        p = et_dir / pattern
        if p.exists():
            return p
    # Fallback: any .EDF
    edfs = list(et_dir.glob('*.EDF')) + list(et_dir.glob('*.edf'))
    if edfs:
        return edfs[0]
    return et_dir / f'sub{sub}.EDF'


# ---------------------------------------------------------------------------
# Trigger / event map
# ---------------------------------------------------------------------------
EVENT_ID = {
    'active/block_start':    10,
    'active/fix_onset':      11,
    'active/pic_onset':      12,
    'active/response':       13,
    'active/block_end':      19,
    'passive/block_start':   20,
    'passive/fix_onset':     21,
    'passive/pic_onset':     22,
    'passive/block_end':     29,
    'constant/block_start':  30,
    'constant/fix_onset':    31,
    'constant/pic_onset':    32,
    'constant/block_end':    39,
    'estimation/onset':      40,
    'estimation/response':   41,
    'verification/onset':    50,
    'verification/response': 51,
}

PIC_ONSET_ID = {
    'active':   12,
    'passive':  22,
    'constant': 32,
}

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
PREPROC = dict(
    # Filtering
    l_freq              = 0.1,      # high-pass (Hz) — preserve slow waves
    h_freq              = 30.0,     # low-pass  (Hz)
    notch_freq          = 50.0,     # power-line noise (EU)
    # Online reference recovery
    online_ref          = 'FCz',    # FCz was hard-wired reference → recover it
    # Re-reference — mastoid (Artyom pipeline)
    reference           = ['TP9', 'TP10'],
    # ICA
    ica_l_freq          = 1.0,      # high-pass for ICA fitting only
    ica_n_components    = None,     # None = full rank; MNE recommended
    ica_random_state    = 42,
    ica_fit_reject      = 250e-6,   # µV — reject threshold during ICA fit
    # ICLabel thresholds (matching Artyom's MATLAB ICLabel settings)
    iclabel_eye_thresh       = 0.8,   # Eye ≥ 0.8 → exclude
    iclabel_artifact_thresh  = 0.9,   # Muscle / Heart / Line Noise / Chan Noise ≥ 0.9 → exclude
    # Channels to protect from bad-channel detection (reference + EOG)
    protect_channels    = ['TP9', 'TP10', 'VEOG'],
    # Bad channel detection (simple amplitude heuristic)
    flat_thresh         = 1e-7,     # V — channel flat if std < this
    noisy_thresh        = 200e-6,   # V — channel noisy if std > this
    # Montage
    montage             = 'standard_1020',
    # QC flags
    max_bad_channels    = 5,        # warn if more channels interpolated
    min_ica_excluded    = 2,        # warn if fewer components excluded
    max_rejection_pct   = 30.0,     # warn if autoreject drops > 30% epochs
)

# ---------------------------------------------------------------------------
# Epoching parameters
# ---------------------------------------------------------------------------
EPOCH = dict(
    tmin            = -0.2,   # s before pic onset
    tmax            =  1.0,   # s after  pic onset (matches Artyom pipeline)
    baseline        = (-0.2, 0.0),
    reject_thresh   = 150e-6, # µV — post-ICA epoch rejection
)

# ---------------------------------------------------------------------------
# Analysis ROIs
# ---------------------------------------------------------------------------
CLUSTERS = {
    'occipital':  ['O1', 'Oz', 'O2'],
    'posterior':  ['PO3', 'PO4', 'POz', 'PO7', 'PO8'],
    'parietal':   ['Pz', 'CPz', 'P3', 'P4'],
    'frontal':    ['F3', 'Fz', 'F4'],
}

ERP_WINDOWS = {
    'P1':   (0.080, 0.130),
    'N1':   (0.140, 0.200),
    'P2':   (0.200, 0.300),
    'P3':   (0.300, 0.500),
    'Late': (0.500, 1.000),
}

# Fixation-related potential windows (fixation-onset locked)
FRP_WINDOWS = {
    'lambda':  (0.060, 0.120),   # lambda response (~80–100 ms)
    'P1_frp':  (0.100, 0.150),   # fixation-locked P1
    'N1_frp':  (0.150, 0.220),   # fixation-locked N1
    'P2_frp':  (0.220, 0.350),   # fixation-locked P2
    'Late_frp':(0.350, 0.600),   # late sustained activity
}

# Saccade-related potential windows (saccade-onset locked)
SRP_WINDOWS = {
    'presac_neg': (-0.100, 0.000),  # presaccadic negativity
    'SP':         (0.000, 0.030),   # spike potential
    'postsac_pos':(0.030, 0.150),   # post-saccadic positivity
    'Late_srp':   (0.150, 0.400),   # late post-saccadic activity
}

# ---------------------------------------------------------------------------
# Time-frequency parameters
# ---------------------------------------------------------------------------
TFR = dict(
    freqs       = (4, 40, 1),    # np.arange(start, stop, step)
    n_cycles_div= 2.0,           # n_cycles = freq / n_cycles_div
    decim       = 4,             # downsample TF output (1000/4 = 250 Hz)
    baseline    = (-0.2, 0.0),
    baseline_mode = 'percent',
    alpha_band  = (8, 12),
    iaf_search  = (7, 13),       # range to search for IAF peak
)

def eeg_ec_vhdr(sub: str) -> Path:
    """Eyes-closed resting-state recording."""
    return sub_eeg_dir(sub) / f'{sub}_EC.vhdr'

def eeg_eo_vhdr(sub: str) -> Path:
    """Eyes-open resting-state recording."""
    return sub_eeg_dir(sub) / f'{sub}_EO.vhdr'

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = {
    'active':   '#111111',
    'passive':  '#d62728',
    'constant': '#1f77b4',
}
