"""Experiment configuration and parameters."""

CONFIG = {
    'window_size': [1920, 1080],
    'screen': 0,

    # Design
    'pic_counts': [9, 12, 18],      # all divisible by 3 for category balance
    'n_reps': 4,
    'practice_n_pics': 3,

    # Timing
    'fixation_duration': 1.0,
    'inter_block_pause': 2.5,
    'isi_range': [0.7, 1.0],        # inter-stimulus fixation jitter (uniform, seconds)
    'isi_bg_color': [-0.1361, -0.1361, -0.1361],   # gray background for EEG fixation interval
    'min_pic_duration': 0.8,        # minimum picture display before response accepted

    # Stimulus paths (3 categories: human, social/man-made, nature)
    'human_folder': 'stimuli/human',
    'social_folder': 'stimuli/social',
    'nature_folder': 'stimuli/nature',
    'image_folder': 'stimuli/images',   # legacy fallback

    # Image display
    'image_size': (1.4, 0.8),
    'recognition_image_size': (0.75, 0.43),   # single-image recognition display

    # Time estimation
    'slider_range': [0, 120],
    'slider_step': 0.5,

    # Text sizes
    'text_height': 0.04,
    'title_height': 0.06,

    # Eye tracking
    'eyelink_ip': '100.1.1.1',
    'sampling_rate': 1000,
    'track_eyes': 'RIGHT_EYE',

    # EEG triggers (parallel port)
    'parallel_port': 0xD010,          # port address (e.g., 0x0378), None to disable
    'trigger_duration': 0.005,        # pulse width in seconds
    'triggers': {
        # Condition base codes (8-bit: 0-255)
        'active_base': 10,          # Active blocks: 10-19
        'passive_base': 20,         # Passive blocks: 20-29
        'constant_base': 30,        # Constant blocks: 30-39
        # Within-block events (added to base)
        'block_start': 0,           # 10, 20, 30
        'fixation_onset': 1,        # 11, 21, 31
        'picture_onset': 2,         # 12, 22, 32
        'response': 3,              # 13 (active only)
        'block_end': 9,             # 19, 29, 39
        # Task events (condition-independent)
        'estimation_onset': 40,
        'estimation_response': 41,
        'recognition_onset': 50,
        'recognition_response': 51,
    },
}
