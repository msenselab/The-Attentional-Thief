"""Helper functions and experiment setup."""

import os

from psychopy import core, data, gui
from numpy.random import randint

from .config import CONFIG


# =============================================================================
# Experiment Setup
# =============================================================================

def setup_experiment():
    """Participant dialog and data handler."""
    exp_info = {
        'participant': f"{randint(0, 999):03d}",
    }
    dlg = gui.DlgFromDict(dictionary=exp_info, title='Attention Thief (2025)')
    if not dlg.OK:
        core.quit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Data folder is at project root, not inside experiment/
    script_dir = os.path.dirname(script_dir)

    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    filename = os.path.join(
        data_dir,
        f"attention_thief_ET_sub-{exp_info['participant']}_{data.getDateStr()}"
    )
    exp_handler = data.ExperimentHandler(
        name='attention_thief_2025',
        version='2025.3-ET-dataviewer',
        extraInfo=exp_info,
        dataFileName=filename,
        savePickle=False,
        saveWideText=False
    )
    return exp_info, exp_handler, script_dir
