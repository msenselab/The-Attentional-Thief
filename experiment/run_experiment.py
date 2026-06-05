#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Attention Thief — PsychoPy 2025 + Data Viewer Compatible

Main entry point. Run this script to start the experiment.

Usage:
    python run_experiment.py           # Full experiment with eye tracking
    python run_experiment.py --debug   # Debug mode: no eye tracker, keyboard input

Version: 2025.4-session-design
"""

import os
import sys
import json
import numpy as np
from psychopy import core, data, event, logging, hardware
from psychopy import plugins
plugins.activatePlugins()

from experiment import (
    CONFIG,
    setup_experiment,
    setup_window,
    ExperimentComponents,
    setup_eyetracker,
    run_calibration,
    run_validation,
    generate_sessions,
    allocate_images,
    TrialHandler,
)


def run_experiment(debug=False):
    exp_info, exp_handler, script_dir = setup_experiment()
    win = setup_window()
    global_clock = core.Clock()
    comp = ExperimentComponents(win, script_dir, debug=debug)

    # Eye tracker setup (skipped in debug mode)
    ioServer = None
    eyetracker = None
    if not debug:
        session_code = f"sub{exp_info['participant']}"[:8]
        edf_filename = session_code + ".EDF"
        print("Initializing eye tracker...")
        ioServer, eyetracker = setup_eyetracker(win, session_code)
        print(f"Eye tracker connected. EDF: {edf_filename}")
    else:
        print("DEBUG MODE: eye tracking disabled, using keyboard input")

    trial = TrialHandler(comp, exp_handler, global_clock, eyetracker,
                         debug=debug)

    # Load resources
    full_pool = comp.load_image_pool()
    seed = int(exp_info['participant'])
    rng = np.random.default_rng(seed)

    # Generate session structure and allocate images
    sessions = generate_sessions(seed=seed)
    practice_images, session_images, foil_images = allocate_images(
        full_pool, sessions, CONFIG['practice_n_pics'], seed=seed
    )

    # Yoking data: store active block results by (rep, n_pictures)
    active_data = {}

    all_results = []

    # Recognition task: 36 blocks total, 18 old + 18 new probes
    recognition_is_old = [True] * 18 + [False] * 18
    rng.shuffle(recognition_is_old)
    recognition_idx = 0
    foil_idx = 0  # Track which foil to use next

    # Input-dependent text fragments
    advance = "Press SPACE" if debug else "Click mouse"

    try:
        # === Calibration (eye tracking only) ===
        if not debug:
            comp.instruction_text.text = (
                "Eye Tracking Calibration\n\n"
                "Follow the target with your eyes.\n\n"
                f"{advance} to begin calibration..."
            )
            if not trial._wait_for_advance(comp.instruction_text):
                return

            print("Running calibration...")
            run_calibration(win, eyetracker)
            print("Calibration complete.")

            comp.instruction_text.text = (
                "Validation\n\n"
                "Follow the target again to validate.\n\n"
                f"{advance} to begin validation..."
            )
            if not trial._wait_for_advance(comp.instruction_text):
                return

            print("Running validation...")
            run_validation(win, eyetracker)
            print("Validation complete.")

        trial.send_message("EXPERIMENT_START")

        # === Instructions ===
        comp.instruction_text.text = (
            "The Attention Thief\n\n"
            "You will complete several short viewing blocks.\n\n"
            f"In some blocks, you {advance.lower()} to advance photos.\n"
            "In other blocks, photos advance automatically — just watch.\n"
            "In some blocks, a single photo will appear — just watch.\n\n"
            "After each block, estimate how long it lasted.\n"
            "Pay attention to the pictures!\n\n\n"
            f"{advance} to begin..."
        )
        if not trial._wait_for_advance(comp.instruction_text):
            return

        # === Practice ===
        comp.block_intro_text.pos = (0, 0)
        comp.block_intro_text.text = (
            f"Practice Round\n\n{advance} to advance to the next photo.\n"
            "Pay attention to what you see.\n\nThis is just for practice.\n\n\n"
            f"{advance} when ready..."
        )
        if not trial._wait_for_advance(comp.block_intro_text):
            return
        ok, _ = trial._present_isi(CONFIG['fixation_duration'])
        if not ok:
            return
        trial.send_message("PRACTICE_START")
        success, _, _, _ = trial.run_active_block(
            practice_images, rep=0, session_order=0,
            block_n_pics=CONFIG['practice_n_pics']
        )
        trial.send_message("PRACTICE_END")
        if not success:
            return
        if not trial.present_stimulus(comp.blank, 0.5):
            return
        est_result = trial.run_time_estimation()
        if est_result[0] is None:
            return
        if not trial.present_stimulus(comp.blank, 1.0):
            return

        # === Main experiment ===
        for session in sessions:
            rep = session['rep']
            session_type = session['session_type']
            session_order = session['session_order']

            # Session-level instruction
            if session_type == 'active':
                comp.block_intro_text.pos = (0, 0)
                comp.block_intro_text.text = (
                    f"Next Session — Active Viewing\n\n"
                    f"{advance} to advance to the next photo.\n"
                    "Browse at your own pace.\n"
                    "Pay attention to what you see.\n\n\n"
                    f"{advance} when ready..."
                )
            elif session_type == 'passive':
                comp.block_intro_text.pos = (0, 0)
                comp.block_intro_text.text = (
                    "Next Session — Passive Viewing\n\n"
                    "Photos will appear automatically.\n"
                    "Just watch and pay attention.\n\n\n"
                    f"{advance} when ready..."
                )
            elif session_type == 'constant':
                comp.block_intro_text.pos = (0, 0)
                comp.block_intro_text.text = (
                    "Next Session — Constant Viewing\n\n"
                    "Just watch ONE single image.\n\n\n"
                    f"{advance} when ready..."
                )

            if not trial._wait_for_advance(comp.block_intro_text):
                return

            for block in session['blocks']:
                n_pics = block['n_pictures']
                block_order = block['block_order']
                key = (rep, session_type, n_pics)
                images = session_images[key]

                # Pre-block fixation (grey background, matching ISI)
                ok, _ = trial._present_isi(CONFIG['fixation_duration'])
                if not ok:
                    return

                if session_type == 'active':
                    success, pic_durs, fix_durs, block_dur = trial.run_active_block(
                        images, rep, session_order, n_pics
                    )
                    if not success:
                        return
                    active_data[(rep, n_pics)] = {
                        'pic_durations': pic_durs,
                        'fix_durations': fix_durs,
                        'block_duration': block_dur,
                    }
                    actual_duration = block_dur

                elif session_type == 'passive':
                    yoke = active_data[(rep, n_pics)]
                    # Shuffle with same permutation for pic + fix durations
                    rng_yoke = np.random.default_rng(seed * 1000 + rep * 100 + n_pics)
                    perm = rng_yoke.permutation(n_pics)
                    shuffled_pic = [yoke['pic_durations'][j] for j in perm]
                    shuffled_fix = [yoke['fix_durations'][j] for j in perm]

                    success, block_dur = trial.run_passive_block(
                        images, shuffled_pic, shuffled_fix,
                        rep, session_order, n_pics
                    )
                    if not success:
                        return
                    actual_duration = block_dur

                elif session_type == 'constant':
                    yoke = active_data[(rep, n_pics)]
                    total_active_dur = yoke['block_duration']
                    image = images[0]

                    success, block_dur = trial.run_constant_block(
                        image, total_active_dur, rep, session_order, n_pics
                    )
                    if not success:
                        return
                    actual_duration = block_dur

                # Time estimation
                if not trial.present_stimulus(comp.blank, 0.5):
                    return
                estimate, start_pos = trial.run_time_estimation()
                if estimate is None:
                    return

                # Image recognition (all block types)
                if not trial.present_stimulus(comp.blank, 0.5):
                    return
                is_old = recognition_is_old[recognition_idx]
                recognition_idx += 1
                
                if is_old:
                    # Select random image from the block just shown
                    probe_image = rng.choice(images)
                else:
                    # Use next foil image
                    probe_image = foil_images[foil_idx]
                    foil_idx += 1
                
                recognition_result = trial.run_image_recognition(probe_image, is_old)
                if recognition_result is None:
                    return

                # Record data
                result = {
                    'block_type': session_type,
                    'actual_duration': actual_duration,
                    'estimated_duration': estimate,
                }
                all_results.append(result)

                exp_handler.addData('phase', 'time_estimation')
                exp_handler.addData('rep', rep)
                exp_handler.addData('session_type', session_type)
                exp_handler.addData('session_order', session_order)
                exp_handler.addData('block_order_in_session', block_order)
                exp_handler.addData('n_pictures', n_pics)
                exp_handler.addData('actual_duration', actual_duration)
                exp_handler.addData('estimated_duration', estimate)
                exp_handler.addData('estimation_error', estimate - actual_duration)
                exp_handler.addData('estimation_ratio',
                                   estimate / actual_duration if actual_duration > 0 else np.nan)
                exp_handler.addData('estimation_start_pos', start_pos)

                if session_type == 'active':
                    exp_handler.addData('per_picture_durations',
                                       json.dumps([round(d, 4) for d in pic_durs]))
                    exp_handler.addData('per_fixation_durations',
                                       json.dumps([round(d, 4) for d in fix_durs]))
                elif session_type == 'passive':
                    exp_handler.addData('per_picture_durations',
                                       json.dumps([round(d, 4) for d in shuffled_pic]))
                    exp_handler.addData('per_fixation_durations',
                                       json.dumps([round(d, 4) for d in shuffled_fix]))
                elif session_type == 'constant':
                    exp_handler.addData('constant_image', os.path.basename(images[0]))

                exp_handler.addData('image_files',
                                   json.dumps([os.path.basename(f) for f in images]))

                exp_handler.addData('recognition_image', os.path.basename(recognition_result['image_path']))
                exp_handler.addData('recognition_is_old', recognition_result['is_old'])
                exp_handler.addData('recognition_response', recognition_result['response'])
                exp_handler.addData('recognition_correct', recognition_result['correct'])
                exp_handler.addData('recognition_rt', recognition_result['rt'])

                exp_handler.nextEntry()
                event.clearEvents()

                if not trial.present_stimulus(comp.blank, CONFIG['inter_block_pause']):
                    return
                event.clearEvents()

            # Between-session break (except after last session)
            if session_order < 12:
                # Recalibrate eye tracker after each rep (every 3 sessions)
                if not debug and session_order % 3 == 0:
                    comp.block_intro_text.pos = (0, 0)
                    comp.block_intro_text.text = (
                        f"Round {rep} of 4 complete.\n\n"
                        f"Time for a short break and eye tracker recalibration.\n"
                        f"You can rest your eyes for a moment.\n\n\n"
                        f"{advance} to recalibrate..."
                    )
                    if not trial._wait_for_advance(comp.block_intro_text):
                        return
                    
                    print(f"Running recalibration after round {rep}...")
                    run_calibration(win, eyetracker)
                    print("Recalibration complete.")
                    
                    comp.block_intro_text.text = (
                        f"Recalibration complete!\n\n\n"
                        f"{advance} for the next round..."
                    )
                    if not trial._wait_for_advance(comp.block_intro_text):
                        return
                else:
                    comp.block_intro_text.pos = (0, 0)
                    comp.block_intro_text.text = (
                        f"Session {session_order} of 12 complete.\n\n"
                        f"Take a short break if you like.\n\n\n"
                        f"{advance} for the next session..."
                    )
                    if not trial._wait_for_advance(comp.block_intro_text):
                        return

        trial.send_message("EXPERIMENT_END")
        #trial.show_reveal(all_results)
        trial._wait_for_advance(comp.goodbye_text)

    finally:
        print("\nCleaning up...")
        try:
            exp_handler.saveAsWideText(exp_handler.dataFileName + '.csv')
            print(f"Behavioral data saved: {exp_handler.dataFileName}.csv")
        except Exception as e:
            print(f"Warning saving data: {e}")

        if ioServer is not None:
            try:
                ioServer.quit()
                print("iohub closed")
            except Exception as e:
                print(f"Warning closing iohub: {e}")

        win.close()
        core.quit()


if __name__ == '__main__':
    debug_mode = '--debug' in sys.argv
    run_experiment(debug=debug_mode)
