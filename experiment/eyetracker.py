"""EyeLink eye tracker setup, calibration, and validation."""

from psychopy import visual, event
import psychopy.iohub as io
from psychopy.hardware.eyetracker import EyetrackerCalibration

from .config import CONFIG


def setup_eyetracker(win, session_code):
    """Setup eye tracker using PsychoPy 2025 pattern."""
    ioConfig = {}
    ioConfig['eyetracker.hw.sr_research.eyelink.EyeTracker'] = {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'default_native_data_file_name': session_code,
        'enable_interface_without_connection': False,
        'network_settings': CONFIG['eyelink_ip'],
        'runtime_settings': {
            'sampling_rate': str(CONFIG['sampling_rate']),
            'track_eyes': CONFIG['track_eyes'],
            'sample_filtering': {
                'FILTER_ONLINE': 'FILTER_LEVEL_OFF',
                'FILTER_FILE': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_AREA',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            },
        },
    }

    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = ioServer.getDevice('tracker')

    return ioServer, eyetracker


def run_calibration(win, eyetracker):
    """Run calibration using PsychoPy 2025 EyetrackerCalibration class."""
    calibrationTarget = visual.TargetStim(
        win,
        name='calibrationTarget',
        radius=0.01,
        fillColor='',
        borderColor='black',
        lineWidth=2.0,
        innerRadius=0.0035,
        innerFillColor='green',
        innerBorderColor='black',
        innerLineWidth=2.0,
        colorSpace='rgb',
        units=None
    )

    calibration = EyetrackerCalibration(
        win,
        eyetracker,
        calibrationTarget,
        units=None,
        colorSpace='rgb',
        progressMode='time',
        targetDur=1.5,
        expandScale=1.5,
        targetLayout='NINE_POINTS',
        randomisePos=True,
        textColor='white',
        movementAnimation=True,
        targetDelay=1.0
    )

    calibration.run()
    event.clearEvents()


def run_validation(win, eyetracker):
    """Run validation using PsychoPy 2025 ValidationProcedure."""
    validationTarget = visual.TargetStim(
        win,
        name='validationTarget',
        radius=0.01,
        fillColor='',
        borderColor='black',
        lineWidth=2.0,
        innerRadius=0.0035,
        innerFillColor='green',
        innerBorderColor='black',
        innerLineWidth=2.0,
        colorSpace='rgb',
        units=None
    )

    validation = io.ValidationProcedure(
        win,
        target=validationTarget,
        gaze_cursor='green',
        positions='NINE_POINTS',
        randomize_positions=True,
        expand_scale=1.5,
        target_duration=1.5,
        enable_position_animation=True,
        target_delay=1.0,
        progress_on_key=None,
        text_color='auto',
        show_results_screen=True,
        save_results_screen=False,
        color_space='rgb',
        unit_type=None
    )

    validation.run()
    event.clearEvents()
