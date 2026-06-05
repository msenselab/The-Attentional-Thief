"""The Attention Thief — modular experiment package."""

from .config import CONFIG
from .utils import setup_experiment
from .components import setup_window, ExperimentComponents
from .eyetracker import setup_eyetracker, run_calibration, run_validation
from .conditions import generate_sessions, allocate_images
from .trials import TrialHandler
