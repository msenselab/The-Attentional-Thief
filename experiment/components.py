"""Visual stimuli and window setup."""

import os
import glob as glob_module

from psychopy import visual

from .config import CONFIG


def setup_window():
    """Create the experiment window."""
    win = visual.Window(
        size=CONFIG['window_size'],
        fullscr=True,
        screen=CONFIG['screen'],
        winType='pyglet',
        allowGUI=False,
        allowStencil=True,
        monitor='testMonitor',
        color='black',
        colorSpace='rgb',
        units='height',
        checkTiming=False
    )
    win.mouseVisible = False
    return win


class ExperimentComponents:
    """Container for all visual components."""

    def __init__(self, win, script_dir, debug=False):
        self.win = win
        self.script_dir = script_dir
        self.debug = debug
        self._init_visual_components()

    def _init_visual_components(self):
        tp = {'font': 'Arial', 'color': 'white', 'colorSpace': 'rgb'}

        self.fixation = visual.ShapeStim(
            win=self.win, vertices='cross', size=(0.03, 0.03),
            lineColor='white', fillColor='white'
        )
        self.instruction_text = visual.TextStim(
            win=self.win, height=CONFIG['text_height'], wrapWidth=1.5,
            pos=(0, 0), **tp
        )
        self.block_intro_text = visual.TextStim(
            win=self.win, height=CONFIG['text_height'], wrapWidth=1.5,
            pos=(0, 0.15), **tp
        )
        self.estimation_prompt = visual.TextStim(
            win=self.win, height=CONFIG['title_height'], wrapWidth=1.5,
            pos=(0, 0.35), text='How long was that (in seconds)?', **tp
        )
        self.estimation_value = visual.TextStim(
            win=self.win, height=0.12, pos=(0, 0.1), **tp
        )
        if self.debug:
            est_hint = 'Hold UP / DOWN to adjust  |  SPACE to confirm'
        else:
            est_hint = 'Scroll wheel to adjust  |  Left click to confirm'
        self.estimation_hint = visual.TextStim(
            win=self.win, height=0.025, pos=(0, -0.05),
            text=est_hint, **tp
        )
        self.reveal_text = visual.TextStim(
            win=self.win, height=CONFIG['text_height'], wrapWidth=1.5,
            pos=(0, 0), **tp
        )
        if self.debug:
            goodbye = 'Experiment completed!\n\nThank you.\n\nPress SPACE to exit.'
        else:
            goodbye = 'Experiment completed!\n\nThank you.\n\nClick mouse to exit.'
        self.goodbye_text = visual.TextStim(
            win=self.win, height=CONFIG['text_height'], pos=(0, 0),
            text=goodbye, **tp
        )
        self.blank = visual.TextStim(win=self.win, text='', pos=(0, 0), **tp)
        self.image_stim = visual.ImageStim(
            win=self.win, pos=(0, 0), size=CONFIG['image_size']
        )
        # ISI fixation for EEG (gray background + white cross)
        self.isi_background = visual.Rect(
            win=self.win, width=2.5, height=2.0,
            pos=(0, 0), fillColor=CONFIG['isi_bg_color'],
            lineColor=CONFIG['isi_bg_color']
        )
        self.isi_fixation = visual.ShapeStim(
            win=self.win, vertices='cross', size=(0.03, 0.03),
            lineColor='white', fillColor='white'
        )
        self.verification_prompt = visual.TextStim(
            win=self.win, height=CONFIG['title_height'], wrapWidth=1.5,
            pos=(0, 0.3), text='Did you see a picture matching this description?', **tp
        )
        self.verification_description = visual.TextStim(
            win=self.win, height=CONFIG['text_height'], wrapWidth=1.4,
            pos=(0, 0.05), **tp
        )
        if self.debug:
            ver_hint = '\u2190 = Yes, I saw it    |    \u2192 = No, I did not'
        else:
            ver_hint = 'Left click = Yes    |    Right click = No'
        self.verification_hint = visual.TextStim(
            win=self.win, height=0.03, pos=(0, -0.25),
            text=ver_hint, **tp
        )

        # Image-based recognition task
        self.recognition_prompt = visual.TextStim(
            win=self.win, height=CONFIG['title_height'], wrapWidth=1.5,
            pos=(0, 0.38), text='Did you see this picture?', **tp
        )
        self.recognition_image = visual.ImageStim(
            win=self.win, size=CONFIG['recognition_image_size'],
            pos=(0, 0.0), interpolate=True
        )
        if self.debug:
            rec_hint = '\u2190 = Old (seen)    |    \u2192 = New (not seen)'
        else:
            rec_hint = 'Left click = Old (seen)    |    Right click = New (not seen)'
        self.recognition_hint = visual.TextStim(
            win=self.win, height=0.03, pos=(0, -0.38),
            text=rec_hint, **tp
        )

    def draw_isi_fixation(self):
        """Draw gray background + white fixation cross for ISI."""
        self.isi_background.draw()
        self.isi_fixation.draw()

    def load_image_pool(self):
        """Load images from human/, social/, nature/ folders.

        Returns dict with 'human', 'social', 'nature' keys (lists of paths),
        as required by conditions.allocate_images().
        """
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        pool = {}
        for cat in ('human', 'social', 'nature'):
            folder = os.path.join(self.script_dir, 'stimuli', cat)
            files = []
            for ext in extensions:
                files.extend(glob_module.glob(os.path.join(folder, ext)))
            files.sort()
            pool[cat] = files
            print(f"Loaded {len(files)} {cat} images")
        return pool
