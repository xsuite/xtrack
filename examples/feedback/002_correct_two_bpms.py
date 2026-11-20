# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Measure the two-BPM response and use it to correct a displaced orbit.
# Example 004 shows what to do when a response direction is weak.

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Model, pickups and correctors
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
BPMS            = ['bpm.s1.f', 'bpm.s4.d']
CORRECTORS      = ['cor.s1', 'cor.s4']
KICK_KNOBS      = ['v_cor_s1', 'v_cor_s4']
PROBE           = 0.2e-6                # Calibration kick [rad]
DISTURBANCE     = 1.0e-6                # Fixed vertical kick at cor.s2 [rad]

########################################
# Orbit feedback
########################################
WINDOW_TURNS    = 344                   # Turns averaged per calculation
N_WINDOWS       = 16
GAIN            = 0.35                  # Fraction of the inferred correction
RESPONSE_TURNS  = (60, 300)             # Corrector time constants [turns]

################################################################################
# Measure how each corrector moves both BPMs
################################################################################

########################################
# Load the ring and install the pickups
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
n_turns     = WINDOW_TURNS * N_WINDOWS
pickups     = ['pickup_1', 'pickup_2']
line.insert([
    env.place(name, xt.BeamStatsMonitor(
        stop_at_turn = n_turns, stats = ['mean_y'],
    ), at = bpm)
    for name, bpm in zip(pickups, BPMS)
], with_progress = False)

########################################
# Scan both correctors with signed kicks
########################################
tw_ref      = line.twiss4d()
reference   = tw_ref['y', BPMS]
line_off    = line.copy()
line_off.build_tracker()
probe_kicks     = np.array([-PROBE, 0.0, PROBE])
probe_offsets   = np.empty((len(probe_kicks), len(BPMS), len(KICK_KNOBS)))

# Launch on the changed closed orbit for each signed probe. The readings
# then give the static response, with no betatron transient from the probe.
for column, knob in enumerate(KICK_KNOBS):
    for step, kick in enumerate(probe_kicks):
        line_off[knob] = kick

        probe = line_off.twiss4d().particle_on_co.copy()
        for name in pickups:
            line_off.get(name).clear()
        line_off.track(probe, num_turns = 1)
        probe_offsets[step, :, column] = [
            line_off.get(name).mean_y[0] for name in pickups
        ] - reference
    line_off[knob] = 0.0
response = (probe_offsets[-1] - probe_offsets[0]) / (2 * PROBE)

########################################
# Check the slopes against linear optics
########################################
# Compare the signed slopes with the closed-orbit formula at nominal optics.
phase_gap       = 2 * np.pi * np.abs(
    tw_ref['muy', BPMS][:, None] - tw_ref['muy', CORRECTORS][None, :]
)
beta_factor     = np.sqrt(
    tw_ref['bety', BPMS][:, None] * tw_ref['bety', CORRECTORS][None, :]
)
half_phase      = np.pi * tw_ref.qy
response_optics = (
    beta_factor * np.cos(half_phase - phase_gap) / (2 * np.sin(half_phase))
)
max_difference  = np.max(np.abs(response - response_optics))

print('Tracked vertical response [m/rad]\n---------------------------------')
print(f'Fixed kick per probe [µrad]: ±{PROBE * 1e6:.3f}')
print(f'BPM plane        {CORRECTORS[0]:>10} {CORRECTORS[1]:>10}')
for bpm, row in zip(BPMS, response):
    print(f'{bpm:<16} {row[0]:+10.2f} {row[1]:+10.2f}')
print(f'Largest difference from linear optics [m/rad]: {max_difference:.2e}')
print('Each corrector moves both BPMs. The loop solves for both kicks.')

########################################
# Plot the two response scans
########################################
scan_fig, scan_axes = plt.subplots(1, 2, sharey = True, figsize = (8, 3.8))
for column, (ax, corrector) in enumerate(zip(scan_axes, CORRECTORS)):
    for row, bpm in enumerate(BPMS):
        predicted   = probe_kicks * response_optics[row, column]
        point,      = ax.plot(
            probe_kicks * 1e6, probe_offsets[:, row, column] * 1e6, 'o',
            label = bpm,
        )
        ax.plot(
            probe_kicks * 1e6, predicted * 1e6, color = point.get_color(),
            zorder = point.get_zorder() - 1,
        )
    ax.set_title(corrector)
    ax.set_xlabel('Calibration kick [µrad]')
scan_axes[0].set_ylabel('Vertical displacement [µm]')
scan_axes[0].legend()
scan_fig.suptitle('Tracked BPM points and linear optics lines')
scan_fig.tight_layout()

################################################################################
# Track the displaced closed orbit
################################################################################

########################################
# Set the common launch and track off
########################################
line['v_cor_s2']        = DISTURBANCE
line_off['v_cor_s2']    = DISTURBANCE

launch          = line.twiss4d().particle_on_co.copy()
particles_off   = launch.copy()
for name in pickups:
    line_off.get(name).clear()
line_off.track(particles_off, num_turns = n_turns)
error_off = np.column_stack(
    [line_off.get(name).mean_y for name in pickups]
) - reference

########################################
# Vary the corrector response time
########################################
turn_time   = line.get_length() / (float(line.particle_ref.beta0[0]) * c)
errors      = {'Feedback off': error_off}
for response_turns in RESPONSE_TURNS:
    ########################################
    # Bind two BPMs and two corrector knobs
    ########################################
    line_on     = line.copy()
    particles   = launch.copy()
    fb          = xt.feedback.Feedback(
        name                    = 'closed_orbit',
        history_length          = WINDOW_TURNS,
        update_turns            = WINDOW_TURNS,
        first_calculation_turn  = WINDOW_TURNS,
    )
    for signal, monitor in zip(('y1', 'y2'), pickups):
        fb.inputs.new(
            name = signal, monitor = monitor, quantity = 'mean_y', unit = 'm',
        )
    correctors = xt.feedback.PhysicalActuator(
        target                          = KICK_KNOBS,
        response_time_constant_seconds  = response_turns * turn_time,
    )
    fb.outputs.new(name = 'kicks', actuator = correctors, unit = 'rad')
    fb.parameters.new(name = 'reference', value = reference, unit = 'm')
    fb.parameters.new(name = 'response', value = response, unit = 'm/rad')
    fb.parameters.new(name = 'gain', value = GAIN)
    fb.state.new(name = 'kicks', initial_value = np.zeros(2), unit = 'rad')

    ########################################
    # Apply the response matrix in tracking
    ########################################
    # Average the pickup to estimate the displaced orbit. The correctors move
    # gradually towards each request to limit excitation of betatron motion.
    fb.program = '''
measured = stack((mean(acquired.y1_history), mean(acquired.y2_history)))
error = measured - parameters.reference
correction = linalg.solve(parameters.response, error)
state.kicks = state.kicks - parameters.gain * correction
outputs.kicks = state.kicks
'''
    line_on.append('orbit_feedback', fb)
    line_on.build_tracker()
    line_on.feedback.enable()
    line_on.track(particles, num_turns = n_turns)
    errors[f'{response_turns}-turn response'] = np.column_stack(
        [line_on.get(name).mean_y for name in pickups]
    ) - reference

################################################################################
# Compare the turn-by-turn errors and the remaining betatron motion
################################################################################
print('\nOrbit error at the two BPMs\n---------------------------')
print(f'Fixed vertical kick at cor.s2 [µrad]: {DISTURBANCE * 1e6:+.3f}')
print(f'Vertical betatron period [turns]: {1 / (tw_ref.qy % 1):.1f}')
print(f'Turns averaged per correction: {WINDOW_TURNS}')
print('All runs start on the same displaced closed orbit.')

########################################
# Compare whole-run and late errors
########################################
headings            = f'{"RMS":>7} {"Late RMS":>8} {"Late p-p":>8}'
bpm_left, bpm_right = [f'{bpm} [µm]' for bpm in BPMS]
print(f'\n{"":19} {bpm_left:^25}   {bpm_right:^25}')
print(f'{"Case":19} {headings}   {headings}')
for label, error in errors.items():
    rms         = np.sqrt(np.mean(error[WINDOW_TURNS:]**2, axis = 0))
    rms_final   = np.sqrt(np.mean(error[-WINDOW_TURNS:]**2, axis = 0))
    pp_final    = np.ptp(error[-WINDOW_TURNS:], axis = 0)
    values      = 1e6 * np.column_stack((rms, rms_final, pp_final))
    left, right = [
        f'{whole:7.3f} {late:8.3f} {pp:8.3f}' for whole, late, pp in values
    ]
    print(f'{label:<19} {left}   {right}')
print('\nRMS uses individual turns after the first request.')
print(f'Late RMS and peak-to-peak span use the last {WINDOW_TURNS} turns.')
print('The slower correctors excite less betatron motion in the undamped ring.')

########################################
# Plot the raw BPM errors
########################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 5.5))
turns       = np.arange(1, n_turns + 1)
for i, (bpm, ax) in enumerate(zip(BPMS, axes)):
    for (label, error), style, color in zip(
        errors.items(), ('--', '-', '-'), ('0.35', 'tab:orange', 'tab:blue'),
    ):
        ax.plot(turns, error[:, i] * 1e6, style, color = color, label = label)
    ax.axvline(WINDOW_TURNS, ls = ':', color = '0.35', label = 'First request')
    ax.set(title = bpm, ylabel = 'Vertical error [µm]')
fig.legend(
    *axes[0].get_legend_handles_labels(), loc = 'upper center', ncol = 4,
)
axes[-1].set_xlabel('Completed turn')
fig.tight_layout(rect = (0, 0, 1, 0.94))
plt.show()
