"""Trial execution logic."""

import os

import numpy as np
from psychopy import core, event
from psychopy.hardware import keyboard

from .config import CONFIG


class TrialHandler:
    """Handles block execution.

    Parameters
    ----------
    components : ExperimentComponents
    exp_handler : psychopy.data.ExperimentHandler
    global_clock : psychopy.core.Clock
    eyetracker : iohub device or None
    debug : bool
        If True, use keyboard input (SPACE/arrows) instead of mouse,
        and skip all eye tracker communication.
    """

    def __init__(self, components, exp_handler, global_clock, eyetracker=None,
                 debug=False):
        self.comp = components
        self.exp = exp_handler
        self.clock = global_clock
        self.tracker = eyetracker
        self.debug = debug

        if debug:
            self._kb = keyboard.Keyboard()
            # Pyglet key-state handler for smooth held-key detection
            from pyglet.window import key as _pkey
            self._pkey = _pkey
            self._key_state = _pkey.KeyStateHandler()
            self.comp.win.winHandle.push_handlers(self._key_state)
        else:
            self._mouse = event.Mouse(win=self.comp.win)

        # EEG parallel port setup
        self._port = None
        if CONFIG['parallel_port'] is not None:
            from psychopy import parallel
            self._port = parallel.ParallelPort(address=CONFIG['parallel_port'])
            self._port.setData(0)

    # ----- EEG trigger helper -----

    def send_trigger(self, code):
        """Send an EEG trigger pulse via parallel port."""
        if self._port is not None:
            self._port.setData(int(code))
            core.wait(CONFIG['trigger_duration'])
            self._port.setData(0)

    # ----- Eye tracker helpers (no-ops when tracker is None) -----

    def send_message(self, text):
        if self.tracker is not None:
            self.tracker.sendMessage(text)

    def start_recording(self):
        if self.tracker is not None:
            self.tracker.setRecordingState(True)

    def stop_recording(self):
        if self.tracker is not None:
            self.tracker.setRecordingState(False)

    # ----- ISI fixation helper -----

    def _present_isi(self, duration, trigger_code=None):
        """Show gray background + fixation cross for the given duration.

        Parameters
        ----------
        duration : float
            Display duration in seconds.
        trigger_code : int or None
            EEG trigger to send on the first frame flip.

        Returns (success, actual_duration).
        """
        first_frame = True
        start = self.clock.getTime()
        while (self.clock.getTime() - start) < duration:
            self.comp.draw_isi_fixation()
            if self._check_escape():
                return False, 0.0
            self.comp.win.flip()
            if first_frame:
                if trigger_code is not None:
                    self.send_trigger(trigger_code)
                first_frame = False
        actual = self.clock.getTime() - start
        return True, actual

    # ----- Block runners -----

    def run_active_block(self, images, rep, session_order, block_n_pics):
        """Run an active block with ISI fixation before each image.

        Returns (success, pic_durations, fix_durations, block_duration).
        """
        n = len(images)
        if n == 0:
            return True, [], [], 0.0

        pic_durations = []
        fix_durations = []
        isi_rng = np.random.default_rng()

        triggers = CONFIG['triggers']
        base = triggers['active_base']

        self.start_recording()
        self.send_message(f"BLOCK_START ACTIVE rep={rep} session={session_order} n_pics={block_n_pics}")
        self.send_trigger(base + triggers['block_start'])

        block_start = None

        if self.debug:
            self._kb.clearEvents()

        for i in range(n):
            # ISI fixation (jittered)
            isi_dur = isi_rng.uniform(*CONFIG['isi_range'])
            if block_start is None:
                block_start = self.clock.getTime()
            self.send_message(f"ISI_ON pic={i+1}/{n} planned_dur={isi_dur:.3f}")
            ok, actual_fix = self._present_isi(
                isi_dur, trigger_code=base + triggers['fixation_onset']
            )
            if not ok:
                self.stop_recording()
                return False, [], [], 0.0
            fix_durations.append(actual_fix)
            self.send_message(f"ISI_OFF actual_dur={actual_fix:.3f}")

            # Show image, wait for advance
            img_name = os.path.basename(images[i])
            img_cat  = os.path.basename(os.path.dirname(images[i]))
            img_rel  = f"stimuli/{img_cat}/{img_name}"
            self.comp.image_stim.image = images[i]

            if self.debug:
                self._kb.clearEvents()
                # First flip with image drawn
                self.comp.image_stim.draw()
                self.comp.win.flip()
                pic_start = self.clock.getTime()
                self.send_trigger(base + triggers['picture_onset'])
                waiting = True
                while waiting:
                    self.comp.image_stim.draw()
                    if self._check_escape():
                        self.stop_recording()
                        return False, [], [], 0.0
                    self.comp.win.flip()
                    if (self.clock.getTime() - pic_start) >= CONFIG['min_pic_duration']:
                        keys = self._kb.getKeys(['space'], waitRelease=False)
                        if keys:
                            self.send_trigger(base + triggers['response'])
                            pic_dur = self.clock.getTime() - pic_start
                            pic_durations.append(pic_dur)
                            waiting = False
                    else:
                        self._kb.getKeys(['space'])  # drain stale presses
            else:
                self._mouse.clickReset()
                while self._mouse.getPressed()[0]:
                    self.comp.win.flip()
                # First flip with image drawn
                self.comp.image_stim.draw()
                self.comp.win.flip()
                pic_start = self.clock.getTime()
                self.send_trigger(base + triggers['picture_onset'])
                self.send_message(f"!V IMGLOAD FILL {img_rel}")
                self.send_message(f"IMAGE_ON {img_name} pic={i+1}/{n}")
                waiting = True
                while waiting:
                    self.comp.image_stim.draw()
                    if self._check_escape():
                        self.stop_recording()
                        return False, [], [], 0.0
                    self.comp.win.flip()
                    if (self.clock.getTime() - pic_start) >= CONFIG['min_pic_duration'] \
                            and self._mouse.getPressed()[0]:
                        self.send_trigger(base + triggers['response'])
                        pic_dur = self.clock.getTime() - pic_start
                        pic_durations.append(pic_dur)
                        self.send_message(f"IMAGE_OFF {img_name} duration={pic_dur:.3f}")
                        self.send_message("!V CLEAR 0 0 0")
                        waiting = False

        block_duration = self.clock.getTime() - block_start
        self.send_trigger(base + triggers['block_end'])
        self.send_message(f"BLOCK_END ACTIVE rep={rep} duration={block_duration:.3f}")
        self.stop_recording()
        return True, pic_durations, fix_durations, block_duration

    def run_passive_block(self, images, yoked_pic_durations, yoked_fix_durations,
                          rep, session_order, block_n_pics):
        """Run a passive block with yoked ISI and picture durations.

        Returns (success, block_duration).
        """
        n = len(images)
        if n == 0:
            return True, 0.0

        triggers = CONFIG['triggers']
        base = triggers['passive_base']

        self.start_recording()
        self.send_message(f"BLOCK_START PASSIVE rep={rep} session={session_order} n_pics={block_n_pics}")
        self.send_trigger(base + triggers['block_start'])

        block_start = None

        for i in range(n):
            # ISI fixation (yoked duration)
            fix_dur = yoked_fix_durations[i]
            if block_start is None:
                block_start = self.clock.getTime()
            self.send_message(f"ISI_ON pic={i+1}/{n} dur={fix_dur:.3f}")
            ok, _ = self._present_isi(
                fix_dur, trigger_code=base + triggers['fixation_onset']
            )
            if not ok:
                self.stop_recording()
                return False, 0.0
            self.send_message("ISI_OFF")

            # Show image for yoked duration
            img_name = os.path.basename(images[i])
            img_cat  = os.path.basename(os.path.dirname(images[i]))
            img_rel  = f"stimuli/{img_cat}/{img_name}"
            self.comp.image_stim.image = images[i]
            self.comp.image_stim.draw()
            self.comp.win.flip()
            pic_start = self.clock.getTime()
            self.send_trigger(base + triggers['picture_onset'])
            self.send_message(f"!V IMGLOAD FILL {img_rel}")
            self.send_message(f"IMAGE_ON {img_name} pic={i+1}/{n} dur={yoked_pic_durations[i]:.3f}")

            while (self.clock.getTime() - pic_start) < yoked_pic_durations[i]:
                self.comp.image_stim.draw()
                if self._check_escape():
                    self.stop_recording()
                    return False, 0.0
                self.comp.win.flip()

            self.send_message(f"IMAGE_OFF {img_name}")
            self.send_message("!V CLEAR 0 0 0")

        block_duration = self.clock.getTime() - block_start
        self.send_trigger(base + triggers['block_end'])
        self.send_message(f"BLOCK_END PASSIVE rep={rep} duration={block_duration:.3f}")
        self.stop_recording()
        return True, block_duration

    def run_constant_block(self, image, total_active_duration,
                           rep, session_order, block_n_pics):
        """Show fixation then one image so total block_duration matches active.

        The image displays for (total_active_duration - fixation_duration)
        so that the full block duration equals the active block.

        Returns (success, block_duration).
        """
        triggers = CONFIG['triggers']
        base = triggers['constant_base']

        self.start_recording()
        self.send_message(f"BLOCK_START CONSTANT rep={rep} session={session_order} n_pics={block_n_pics}")
        self.send_trigger(base + triggers['block_start'])

        # One ISI fixation
        fix_dur = np.random.default_rng().uniform(*CONFIG['isi_range'])
        block_start = self.clock.getTime()
        self.send_message(f"ISI_ON planned_dur={fix_dur:.3f}")
        ok, actual_fix = self._present_isi(
            fix_dur, trigger_code=base + triggers['fixation_onset']
        )
        if not ok:
            self.stop_recording()
            return False, 0.0
        self.send_message(f"ISI_OFF actual_dur={actual_fix:.3f}")

        # Show single image for remaining duration
        image_duration = max(0.0, total_active_duration - actual_fix)
        img_name = os.path.basename(image)
        img_cat  = os.path.basename(os.path.dirname(image))
        img_rel  = f"stimuli/{img_cat}/{img_name}"
        self.comp.image_stim.image = image
        self.comp.image_stim.draw()
        self.comp.win.flip()
        pic_start = self.clock.getTime()
        self.send_trigger(base + triggers['picture_onset'])
        self.send_message(f"!V IMGLOAD FILL {img_rel}")
        self.send_message(f"IMAGE_ON {img_name} constant dur={image_duration:.3f}")

        while (self.clock.getTime() - pic_start) < image_duration:
            self.comp.image_stim.draw()
            if self._check_escape():
                self.stop_recording()
                return False, 0.0
            self.comp.win.flip()

        self.send_message(f"IMAGE_OFF {img_name}")
        self.send_message("!V CLEAR 0 0 0")

        block_duration = self.clock.getTime() - block_start
        self.send_trigger(base + triggers['block_end'])
        self.send_message(f"BLOCK_END CONSTANT rep={rep} duration={block_duration:.3f}")
        self.stop_recording()
        return True, block_duration

    # ----- Time estimation -----

    def run_time_estimation(self):
        lo, hi = CONFIG['slider_range']
        current = 0.0
        start_pos = 0

        self.send_trigger(CONFIG['triggers']['estimation_onset'])
        self.send_message("ESTIMATION_ON")

        if self.debug:
            return self._time_estimation_keyboard(lo, hi, current, start_pos)
        else:
            return self._time_estimation_mouse(lo, hi, current, start_pos)

    def _time_estimation_mouse(self, lo, hi, current, start_pos):
        self._mouse.clickReset()
        while any(self._mouse.getPressed()):
            self.comp.win.flip()
        guard_clock = core.Clock()
        while True:
            scroll = self._mouse.getWheelRel()[1]
            if scroll != 0:
                current = current + scroll * 0.5
                current = max(lo, min(hi, current))
            if guard_clock.getTime() >= 0.5:
                if self._mouse.getPressed()[0]:
                    while self._mouse.getPressed()[0]:
                        self.comp.win.flip()
                    self.send_trigger(CONFIG['triggers']['estimation_response'])
                    snapped = round(current * 2) / 2
                    self.send_message(f"ESTIMATION_RESP estimate={snapped:.1f}")
                    return snapped, start_pos
            snapped = round(current * 2) / 2
            self.comp.estimation_value.text = f"{snapped:.1f} s"
            self.comp.estimation_prompt.draw()
            self.comp.estimation_value.draw()
            self.comp.estimation_hint.draw()
            if self._check_escape():
                return None, None
            self.comp.win.flip()

    def _time_estimation_keyboard(self, lo, hi, current, start_pos):
        """UP/DOWN arrows with acceleration, SPACE to confirm."""
        _pkey = self._pkey
        key_state = self._key_state

        event.clearEvents()
        self._kb.clearEvents()

        hold_clock = core.Clock()
        frame_clock = core.Clock()
        guard_clock = core.Clock()
        prev_dir = None
        applied_initial = False

        while True:
            # Confirmation (with guard to drain stale SPACE)
            keys = self._kb.getKeys(['space', 'return'], waitRelease=False)
            if guard_clock.getTime() >= 0.5:
                for key in keys:
                    if key.name in ('space', 'return'):
                        self.send_trigger(CONFIG['triggers']['estimation_response'])
                        snapped = round(current * 2) / 2
                        self.send_message(f"ESTIMATION_RESP estimate={snapped:.1f}")
                        return snapped, start_pos

            # Smooth held-key adjustment
            cur_dir = None
            if key_state[_pkey.UP]:
                cur_dir = 'up'
            elif key_state[_pkey.DOWN]:
                cur_dir = 'down'

            if cur_dir != prev_dir:
                hold_clock.reset()
                prev_dir = cur_dir
                applied_initial = False

            dt = frame_clock.getTime()
            frame_clock.reset()

            if cur_dir is not None:
                if not applied_initial:
                    delta = float(CONFIG['slider_step'])
                    applied_initial = True
                else:
                    hold_t = hold_clock.getTime()
                    if hold_t < 0.4:
                        delta = 0.0
                    else:
                        speed = min(5.0 + (hold_t - 0.4) * 8.0, 30.0)
                        delta = speed * dt

                if cur_dir == 'up':
                    current = min(hi, current + delta)
                elif cur_dir == 'down':
                    current = max(lo, current - delta)

            snapped = round(current * 2) / 2
            self.comp.estimation_value.text = f"{snapped:.1f} s"
            self.comp.estimation_prompt.draw()
            self.comp.estimation_value.draw()
            self.comp.estimation_hint.draw()
            if self._check_escape():
                return None, None
            self.comp.win.flip()

    def run_image_recognition(self, image_path, is_old):
        """Run image-based recognition trial.
        
        Args:
            image_path: Path to the probe image
            is_old: True if image was shown in block (old), False if foil (new)
        
        Returns:
            dict with image_path, is_old, response ('old'/'new'), correct, rt
            or None if escaped
        """
        import os
        img_name = os.path.basename(image_path)
        probe_type = 'OLD' if is_old else 'NEW'
        
        self.comp.recognition_image.image = image_path
        trial_start = self.clock.getTime()
        guard_clock = core.Clock()

        self.send_trigger(CONFIG['triggers']['recognition_onset'])
        img_cat = os.path.basename(os.path.dirname(image_path))
        img_rel = f"stimuli/{img_cat}/{img_name}"
        self.send_message(f"RECOGNITION_ON {img_name} type={probe_type}")
        self.send_message(f"!V IMGLOAD FILL {img_rel}")

        if self.debug:
            event.clearEvents()
            self._kb.clearEvents()
            while True:
                self.comp.recognition_prompt.draw()
                self.comp.recognition_image.draw()
                self.comp.recognition_hint.draw()
                if self._check_escape():
                    return None
                self.comp.win.flip()
                if guard_clock.getTime() >= 0.5:
                    keys = self._kb.getKeys(['left', 'right'], waitRelease=False)
                    if keys:
                        self.send_trigger(CONFIG['triggers']['recognition_response'])
                        rt = self.clock.getTime() - trial_start
                        response = 'old' if keys[0].name == 'left' else 'new'
                        correct_answer = 'old' if is_old else 'new'
                        correct = (response == correct_answer)
                        self.send_message(f"RECOGNITION_RESP {response} correct={correct} rt={rt:.3f}")
                        self.send_message("!V CLEAR 0 0 0")
                        return {
                            'image_path': image_path,
                            'is_old': is_old,
                            'response': response,
                            'correct': correct,
                            'rt': rt
                        }
        else:
            self._mouse.clickReset()
            while any(self._mouse.getPressed()):
                self.comp.win.flip()
            while True:
                self.comp.recognition_prompt.draw()
                self.comp.recognition_image.draw()
                self.comp.recognition_hint.draw()
                if self._check_escape():
                    return None
                self.comp.win.flip()
                if guard_clock.getTime() >= 0.5:
                    buttons = self._mouse.getPressed()
                    if buttons[0]:  # Left click = Old
                        self.send_trigger(CONFIG['triggers']['recognition_response'])
                        rt = self.clock.getTime() - trial_start
                        while self._mouse.getPressed()[0]:
                            self.comp.win.flip()
                        response = 'old'
                        correct_answer = 'old' if is_old else 'new'
                        correct = (response == correct_answer)
                        self.send_message(f"RECOGNITION_RESP {response} correct={correct} rt={rt:.3f}")
                        self.send_message("!V CLEAR 0 0 0")
                        return {
                            'image_path': image_path,
                            'is_old': is_old,
                            'response': response,
                            'correct': correct,
                            'rt': rt
                        }
                    elif buttons[2]:  # Right click = New
                        self.send_trigger(CONFIG['triggers']['recognition_response'])
                        rt = self.clock.getTime() - trial_start
                        while self._mouse.getPressed()[2]:
                            self.comp.win.flip()
                        response = 'new'
                        correct_answer = 'old' if is_old else 'new'
                        correct = (response == correct_answer)
                        self.send_message(f"RECOGNITION_RESP {response} correct={correct} rt={rt:.3f}")
                        self.send_message("!V CLEAR 0 0 0")
                        return {
                            'image_path': image_path,
                            'is_old': is_old,
                            'response': response,
                            'correct': correct,
                            'rt': rt
                        }

    # ----- Reveal -----

    def show_reveal(self, results):
        by_type = {}
        for r in results:
            bt = r['block_type']
            by_type.setdefault(bt, {'est': [], 'actual': []})
            by_type[bt]['est'].append(r['estimated_duration'])
            by_type[bt]['actual'].append(r['actual_duration'])
        lines = ["\u2014 The Reveal \u2014\n"]
        labels = {'active': 'Active (clicking)', 'passive': 'Passive (watching)',
                  'constant': 'Constant (watching)'}
        for bt in ['active', 'passive', 'constant']:
            if bt in by_type:
                avg_est = np.mean(by_type[bt]['est'])
                avg_act = np.mean(by_type[bt]['actual'])
                lines.append(f"{labels[bt]}:\n  avg. estimate = {avg_est:.1f}s   |   avg. actual = {avg_act:.1f}s")
        dismiss = "Press SPACE to end." if self.debug else "Click mouse to end."
        lines.append(f"\nSame durations. Different engagement.\nYour experience changed how you perceived time.\n\n{dismiss}")
        self.comp.reveal_text.text = '\n'.join(lines)
        return self._wait_for_advance(self.comp.reveal_text)

    # ----- Helpers -----

    def present_stimulus(self, stimulus, duration):
        start = self.clock.getTime()
        while (self.clock.getTime() - start) < duration:
            stimulus.draw()
            if self._check_escape():
                return False
            self.comp.win.flip()
        return True

    def _wait_for_advance(self, stimulus):
        """Show stimulus and wait for SPACE (debug) or mouse click (formal)."""
        if self.debug:
            return self._wait_for_space(stimulus)
        else:
            return self._wait_for_click(stimulus)

    def _wait_for_space(self, stimulus):
        event.clearEvents()
        self._kb.clearEvents()
        guard_clock = core.Clock()
        while True:
            stimulus.draw()
            if self._check_escape():
                return False
            self.comp.win.flip()
            if guard_clock.getTime() < 0.8:
                self._kb.getKeys(['space'])
            else:
                keys = self._kb.getKeys(['space'])
                if keys:
                    return True

    def _wait_for_click(self, stimulus):
        self._mouse.clickReset()
        while any(self._mouse.getPressed()):
            stimulus.draw()
            self.comp.win.flip()
        guard_clock = core.Clock()
        while True:
            stimulus.draw()
            if self._check_escape():
                return False
            self.comp.win.flip()
            if guard_clock.getTime() >= 0.3:
                if self._mouse.getPressed()[0]:
                    while self._mouse.getPressed()[0]:
                        stimulus.draw()
                        self.comp.win.flip()
                    return True

    def _check_escape(self):
        if event.getKeys(keyList=["escape"]):
            return True
        return False
